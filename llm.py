import os
import re
import threading
import time
import warnings
from typing import Any

from dotenv import load_dotenv

_model: Any = None
_tokenizer: Any = None
_model_id = None
_load_lock = threading.Lock()
# Seconds spent loading the model in this process (for the run timing in summary.json).
MODEL_LOAD_SECONDS: float | None = None
_generation_lock = threading.Lock()


def _setting(name: str, default: str) -> str:
    return os.environ.get(name, default).strip()


def _get_model(model_id: str) -> tuple[Any, Any]:
    global _model, _tokenizer, _model_id, MODEL_LOAD_SECONDS
    if _model is None or _model_id != model_id:
        with _load_lock:
            if _model is None or _model_id != model_id:
                load_started = time.time()
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

                # Opt-in YaRN context extension (e.g. LOCAL_ROPE_SCALING_FACTOR=2 -> 64k tokens for
                # Qwen2.5), needed for large specs like RestBench Spotify. Static YaRN can slightly
                # hurt short prompts, so it stays off unless set.
                rope_factor = float(_setting("LOCAL_ROPE_SCALING_FACTOR", "0") or 0)
                if rope_factor > 1:
                    from transformers import AutoConfig

                    config = AutoConfig.from_pretrained(
                        model_id,
                        local_files_only=model_kwargs["local_files_only"],
                    )
                    original = config.max_position_embeddings
                    config.rope_scaling = {
                        **(config.rope_scaling or {}),  # newer transformers keep rope_theta here
                        "rope_type": "yarn",
                        "factor": rope_factor,
                        "original_max_position_embeddings": original,
                    }
                    config.max_position_embeddings = int(original * rope_factor)
                    model_kwargs["config"] = config

                _model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
                if not device_map:
                    device = "mps" if torch.backends.mps.is_available() else "cpu"
                    _model.to(device)
                _model.eval()
                _model_id = model_id
                MODEL_LOAD_SECONDS = (MODEL_LOAD_SECONDS or 0.0) + time.time() - load_started
    return _model, _tokenizer


def _format_messages(messages: list[dict], tokenizer) -> str:
    return "\n".join(
        f"{message['role'].upper()}: {message.get('content', '')}"
        for message in messages
    ) + "\nASSISTANT:"


# ---------------------------------------------------------------------------
# Reasoning ("thinking") control
# ---------------------------------------------------------------------------
# Reasoning models (e.g. Qwen3-style chat templates with `enable_thinking`) write their reasoning
# between <think> and </think> before the answer. Such templates only offer on/off, so levels are
# realised as a thinking *budget*: the model may reason for at most N tokens; if it has not closed
# its reasoning by then, "</think>" is appended and the answer is generated from there ("budget
# forcing", Muennighoff et al. 2025, s1). The answer always gets its own LOCAL_MAX_NEW_TOKENS.
#
# The level is set per call type, so e.g. refinement can reason while the endpoint check doesn't:
#   LOCAL_REASONING_GENERATE / LOCAL_REASONING_REFINE / LOCAL_REASONING_CHECK = off|low|medium|high
# each falling back to LOCAL_REASONING (default off). LOCAL_ENABLE_THINKING=1 (older runs) = high.
# Budgets per level can be overridden with LOCAL_REASONING_BUDGET_LOW/_MEDIUM/_HIGH (tokens).

REASONING_LEVELS = ("off", "low", "medium", "high")
_DEFAULT_BUDGETS = {"low": 1024, "medium": 4096, "high": 16384}
REASONING_ROLES = ("generate", "refine", "check")


def reasoning_budget(level: str) -> int:
    return int(_setting(f"LOCAL_REASONING_BUDGET_{level.upper()}", str(_DEFAULT_BUDGETS[level])))


def reasoning_level(role: str | None) -> str:
    """Configured reasoning level for a call type (`generate`, `refine`, `check`)."""
    fallback = "high" if _setting("LOCAL_ENABLE_THINKING", "0") == "1" else "off"
    level = _setting("LOCAL_REASONING", fallback).lower() or fallback
    if role:
        level = _setting(f"LOCAL_REASONING_{role.upper()}", level).lower() or level
    if level not in REASONING_LEVELS:
        raise ValueError(f"Reasoning level must be one of {', '.join(REASONING_LEVELS)}, not {level!r}.")
    return level


def reasoning_config() -> dict:
    """Reasoning settings of this process, for a run's summary.json."""
    levels = {role: reasoning_level(role) for role in REASONING_ROLES}
    return {
        "levels": levels,
        "budgets": {level: reasoning_budget(level) for level in set(levels.values()) if level != "off"},
    }


def _supports_thinking(tokenizer) -> bool:
    template = tokenizer.chat_template or ""
    if isinstance(template, dict):  # some tokenizers ship several named templates
        template = " ".join(str(t) for t in template.values())
    return "enable_thinking" in template or "<think>" in template


def _think_end_id(tokenizer) -> int | None:
    token_id = tokenizer.convert_tokens_to_ids("</think>")
    return None if token_id is None or token_id == tokenizer.unk_token_id else int(token_id)


def _eos_ids(model_instance, tokenizer) -> set[int]:
    eos = getattr(model_instance.generation_config, "eos_token_id", None)
    ids = set(eos if isinstance(eos, (list, tuple)) else [eos] if eos is not None else [])
    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)
    return ids


def _split_reasoning(new_ids: list[int], tokenizer) -> tuple[str, str, int]:
    """(reasoning, answer, reasoning_tokens) of generated token ids, split at the last </think>.
    Unclosed reasoning (no </think>) means no answer was produced."""
    end_id = _think_end_id(tokenizer)
    if end_id is not None and end_id in new_ids:
        cut = len(new_ids) - 1 - new_ids[::-1].index(end_id)
        reasoning = tokenizer.decode(new_ids[:cut], skip_special_tokens=True)
        answer = tokenizer.decode(new_ids[cut + 1:], skip_special_tokens=True)
        return reasoning.replace("<think>", "").strip(), answer.strip(), cut
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    if text.lstrip().startswith("<think>"):
        return text.replace("<think>", "").strip(), "", len(new_ids)
    return "", text, 0


def _tokenize_messages(messages: list[dict], tokenizer, enable_thinking: bool = False):
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            # Qwen3-style templates read this flag; templates without it ignore it.
            enable_thinking=enable_thinking,
        )
    return tokenizer(
        _format_messages(messages, tokenizer),
        return_tensors="pt",
    )


def call_llm_chat(messages: list[dict], model: str, role: str | None = None) -> tuple[str, dict]:
    """Call the LLM with a pre-built multi-turn message list and return (content, usage).

    `role` ("generate", "refine", "check") selects the reasoning level (see reasoning_level()).
    `content` never contains the model's reasoning. usage keys: prompt_tokens, completion_tokens,
    total_tokens, reasoning_level (effective), reasoning_tokens, reasoning_forced_stop.
    """
    model_instance, tokenizer = _get_model(_setting("LOCAL_MODEL_ID", model))
    level = reasoning_level(role)
    if level != "off" and not _supports_thinking(tokenizer):
        warnings.warn(f"Reasoning level {level!r} requested, but the model's chat template has no "
                      "thinking mode - generating without reasoning.", stacklevel=2)
        level = "off"

    inputs = _tokenize_messages(messages, tokenizer, enable_thinking=level != "off")
    device = next(model_instance.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    prompt_tokens = int(inputs["input_ids"].shape[-1])

    answer_tokens = int(_setting("LOCAL_MAX_NEW_TOKENS", "1024"))
    do_sample = _setting("LOCAL_DO_SAMPLE", "0") == "1"
    generation_kwargs = {
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = float(_setting("LOCAL_TEMPERATURE", "0.2"))
        generation_kwargs["top_p"] = float(_setting("LOCAL_TOP_P", "0.95"))

    forced_stop = False
    with _generation_lock:
        import torch

        with torch.inference_mode():
            if level == "off":
                output = model_instance.generate(**inputs, max_new_tokens=answer_tokens, **generation_kwargs)
            else:
                # Phase 1: reasoning (and possibly the start of the answer) within the budget.
                output = model_instance.generate(**inputs, max_new_tokens=reasoning_budget(level), **generation_kwargs)
                new_ids = output[0, prompt_tokens:].tolist()
                finished = bool(new_ids) and new_ids[-1] in _eos_ids(model_instance, tokenizer)
                if not finished:
                    end_id = _think_end_id(tokenizer)
                    if end_id is None or end_id not in new_ids:
                        # Budget exhausted while reasoning: close it and let the model answer.
                        forced_stop = True
                        close = tokenizer("\n</think>\n\n", add_special_tokens=False, return_tensors="pt")["input_ids"]
                        output = torch.cat([output, close.to(output.device)], dim=-1)
                    # Phase 2: the answer, with its own budget.
                    output = model_instance.generate(
                        input_ids=output,
                        attention_mask=torch.ones_like(output),
                        max_new_tokens=answer_tokens,
                        **generation_kwargs,
                    )

    new_ids = output[0, prompt_tokens:].tolist()
    _reasoning, content, reasoning_tokens = _split_reasoning(new_ids, tokenizer)
    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": len(new_ids),
        "total_tokens": prompt_tokens + len(new_ids),
        "reasoning_level": level,
        "reasoning_tokens": reasoning_tokens,
        "reasoning_forced_stop": forced_stop,
    }
    return content, usage


def call_llm(prompt: str, model: str, instructions: str, role: str | None = None) -> tuple[str, dict]:
    """Call the LLM with a single-turn prompt and return (content, usage)."""
    messages = []
    if instructions:
        messages.append({"role": "system", "content": instructions})
    messages.append({"role": "user", "content": prompt})
    return call_llm_chat(messages, model, role)


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
