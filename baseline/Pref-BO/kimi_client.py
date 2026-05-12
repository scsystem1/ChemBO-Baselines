from __future__ import annotations

import json
import os
from json import JSONDecodeError, JSONDecoder

from openai import OpenAI


DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "kimi-k2.5-thinking"


def _resolve_model_name(model_name: str | None = None) -> str:
    raw_name = (model_name or os.getenv("PREFBO_LLM_MODEL") or os.getenv("KIMI_MODEL") or DEFAULT_MODEL).strip()
    return raw_name


def _build_client() -> OpenAI:
    api_key = (
        os.getenv("DASHSCOPE_API_KEY")
        or os.getenv("PREFBO_API_KEY")
        or os.getenv("MOONSHOT_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    base_url = os.getenv("PREFBO_BASE_URL", os.getenv("KIMI_BASE_URL", DEFAULT_BASE_URL))
    return OpenAI(api_key=api_key, base_url=base_url)


def call_kimi_chat(user_prompt: str, model_name: str | None = None, system_prompt: str | None = None) -> str:
    client = _build_client()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    response = client.chat.completions.create(
        model=_resolve_model_name(model_name),
        messages=messages,
        max_tokens=4096,
        temperature=1.0,
    )
    return response.choices[0].message.content or ""


def parse_jsonish_response(text: str) -> dict:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        stripped = stripped.replace("json\n", "", 1).strip()
    try:
        return json.loads(stripped)
    except JSONDecodeError:
        decoder = JSONDecoder()
        start = stripped.find("{")
        while start != -1:
            try:
                parsed, _ = decoder.raw_decode(stripped, idx=start)
                if isinstance(parsed, dict):
                    return parsed
            except JSONDecodeError:
                pass
            start = stripped.find("{", start + 1)
        raise
