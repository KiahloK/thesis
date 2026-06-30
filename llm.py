import os
from dotenv import load_dotenv

_client = None


def _get_client():
    global _client
    if _client is None:
        load_dotenv()
        from openai import OpenAI
        api_key = os.environ.get("NEBIUS_API_KEY") or os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.tokenfactory.nebius.com/v1/")
        if not api_key:
            raise RuntimeError("No API key found in NEBIUS_API_KEY or OPENAI_API_KEY environment variables")
        _client = OpenAI(base_url=base_url, api_key=api_key)
    return _client


def call_llm(prompt: str, model: str, instructions: str) -> tuple[str, dict]:
    """Call the LLM and return (content, usage).

    usage keys: prompt_tokens, completion_tokens, total_tokens.
    """
    client = _get_client()
    messages = []
    if instructions:
        messages.append({"role": "system", "content": instructions})
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        timeout=120,
    )
    usage = {}
    if response.usage:
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
    content = response.choices[0].message.content or ""
    return content, usage


def build_prompt(services: list[str], query: str, prompt_template: str) -> str:
    services_block = "\n---\n".join(services)
    return prompt_template.format(services_block=services_block, query=query)
