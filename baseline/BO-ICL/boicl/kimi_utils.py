from __future__ import annotations

import os

from openai import OpenAI


DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "kimi-k2.5-thinking"


def resolve_kimi_model_name(model_name: str | None = None) -> str:
    raw_name = (model_name or os.getenv("BOICL_LLM_MODEL") or DEFAULT_MODEL).strip()
    if raw_name == "kimi-k2.5-thinking":
        return os.getenv("BOICL_KIMI_MODEL_ALIAS", "kimi-k2.5")
    return raw_name


def is_kimi_model(model_name: str | None) -> bool:
    lowered = resolve_kimi_model_name(model_name).lower()
    return lowered.startswith("kimi-")


def kimi_base_url() -> str:
    return os.getenv("BOICL_BASE_URL", os.getenv("KIMI_BASE_URL", DEFAULT_BASE_URL))


def kimi_api_key() -> str | None:
    return (
        os.getenv("BOICL_API_KEY")
        or os.getenv("DASHSCOPE_API_KEY")
        or os.getenv("MOONSHOT_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )


def build_kimi_client() -> OpenAI:
    return OpenAI(
        api_key=kimi_api_key(),
        base_url=kimi_base_url(),
    )
