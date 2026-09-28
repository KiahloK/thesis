import os
import re
import threading
from typing import Any

from dotenv import load_dotenv

_model: Any = None
_tokenizer: Any = None
_model_id = None
_load_lock = threading.Lock()
_generation_lock = threading.Lock()


def _setting(name: str, default: str) -> str:
    return os.environ.get(name, default).strip()


def _get_model(model_id: str) -> tuple[Any, Any]:
    global _model, _tokenizer, _model_id
    if _model is None or _model_id != model_id:
        with _load_lock:
            if _model is None or _model_id != model_id:
                load_dotenv()
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer

                device_map = _setting(
                    "LOCAL_DEVICE_MAP",
                    "auto" if torch.cuda.is_available() else "",
                )

                _tokenizer = AutoTokenizer.from_pretrained(
                    model_id,
                    local_files_only=_setting("HF_HUB_OFFLINE", "0") == "1",
                )
                if _tokenizer.pad_token_id is None and _tokenizer.eos_token_id is not None:
                    _tokenizer.pad_token = _tokenizer.eos_token

                model_kwargs = {
                    "torch_dtype": "auto",
                    "local_files_only": _setting("HF_HUB_OFFLINE", "0") == "1",
                }
                if device_map:
                    model_kwargs["device_map"] = device_map
                _model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
                if not device_map:
                    device = "mps" if torch.backends.mps.is_available() else "cpu"
                    _model.to(device)
                _model.eval()
                _model_id = model_id
    return _model, _tokenizer


def _format_messages(messages: list[dict], tokenizer) -> str:
    return "\n".join(
        f"{message['role'].upper()}: {message.get('content', '')}"
        for message in messages
    ) + "\nASSISTANT:"


def _tokenize_messages(messages: list[dict], tokenizer):
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
    return tokenizer(
        _format_messages(messages, tokenizer),
        return_tensors="pt",
    )


def call_llm_chat(messages: list[dict], model: str) -> tuple[str, dict]:
    """Call the LLM with a pre-built multi-turn message list and return (content, usage).

    usage keys: prompt_tokens, completion_tokens, total_tokens.
    """
    model_instance, tokenizer = _get_model(_setting("LOCAL_MODEL_ID", model))
    inputs = _tokenize_messages(messages, tokenizer)
    device = next(model_instance.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    prompt_tokens = int(inputs["input_ids"].shape[-1])

    do_sample = _setting("LOCAL_DO_SAMPLE", "0") == "1"
    generation_kwargs = {
        "max_new_tokens": int(_setting("LOCAL_MAX_NEW_TOKENS", "1024")),
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = float(_setting("LOCAL_TEMPERATURE", "0.2"))
        generation_kwargs["top_p"] = float(_setting("LOCAL_TOP_P", "0.95"))

    with _generation_lock:
        import torch

        with torch.inference_mode():
            output = model_instance.generate(**inputs, **generation_kwargs)

    completion_tokens = output.shape[-1] - prompt_tokens
    content = tokenizer.decode(output[0, prompt_tokens:], skip_special_tokens=True)
    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": int(completion_tokens),
        "total_tokens": prompt_tokens + int(completion_tokens),
    }
    return content, usage


def call_llm(prompt: str, model: str, instructions: str) -> tuple[str, dict]:
    """Call the LLM with a single-turn prompt and return (content, usage)."""
    messages = []
    if instructions:
        messages.append({"role": "system", "content": instructions})
    messages.append({"role": "user", "content": prompt})
    return call_llm_chat(messages, model)


_CODE_FENCE = re.compile(r"```[\w+-]*[ \t]*\n(.*?)(?:\n```|\Z)", re.DOTALL)


def extract_code(text: str) -> str:
    """Return the Python source from a model response, dropping a markdown code fence (and any
    prose around it) if the model wrapped its answer in one despite the raw-source contract.
    An unterminated fence (output cut off at max_new_tokens) keeps everything after it."""
    match = _CODE_FENCE.search(text)
    return match.group(1).strip() if match else text.strip()


def build_prompt(services: list[str], query: str, prompt_template: str) -> str:
    services_block = "\n---\n".join(services)
    return prompt_template.format(services_block=services_block, query=query)
