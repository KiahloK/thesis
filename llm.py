import os
from dotenv import load_dotenv


def _get_client():
    """Lazily construct an OpenAI client using environment variables.

    Raises RuntimeError if no API key is found.
    """
    load_dotenv()
    try:
        from openai import OpenAI
    except Exception as e:
        raise

    api_key = os.environ.get("NEBIUS_API_KEY") or os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.tokenfactory.nebius.com/v1/")
    if not api_key:
        raise RuntimeError("No API key found in NEBIUS_API_KEY or OPENAI_API_KEY environment variables")
    return OpenAI(base_url=base_url, api_key=api_key)


def call_llm(prompt: str, model: str, instructions: str) -> str:
    """Call the LLM and return the raw text output.

    The client is created on demand so importing this module doesn't require credentials.
    """
    client = _get_client()
    response = client.responses.create(
        model=model,
        instructions=instructions,
        input=prompt,
    )
    return response.output_text


def build_prompt(services: list[str], query: str, prompt_template: str) -> str:
    services_block = "\n---\n".join(services)
    return prompt_template.format(services_block=services_block, query=query)
