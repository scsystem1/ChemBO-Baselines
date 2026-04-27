from openai import OpenAI
from typing import Tuple
from src.config import Config
from src.utils.jsonl import add_to_jsonl

config = Config()


def _resolve_model_name() -> str:
    raw_name = (config.KIMI_MODEL_NAME or "kimi-k2.5-thinking").strip()
    if raw_name == "kimi-k2.5-thinking":
        return "kimi-k2.5"
    return raw_name


class KimiClient:
    """Kimi client using an OpenAI-compatible chat completion interface."""

    def __init__(self):
        self.client = OpenAI(
            api_key=config.KIMI_API_KEY,
            base_url=config.KIMI_API_BASE,
        )
        self.messages = []
        self.content = ""
        self.reasoning_content = ""

    def generate(
        self,
        user_prompt: str,
        max_tokens: int = 4096,
        json_output: bool = False,
    ) -> Tuple[str, str]:
        self.messages = []
        if json_output:
            self.messages.append(
                {"role": "system", "content": "response in JSON format"}
            )
        self.messages.append({"role": "user", "content": user_prompt})

        response = self.client.chat.completions.create(
            model=_resolve_model_name(),
            messages=self.messages,
            max_tokens=max_tokens,
            temperature=0.7,
        )
        message = response.choices[0].message
        self.content = message.content or ""
        self.reasoning_content = getattr(message, "reasoning_content", "") or ""

        if self.reasoning_content:
            self.messages.append({"role": "think", "content": self.reasoning_content})
        self.messages.append({"role": "assistant", "content": self.content})
        return (self.content, self.reasoning_content)

    def multi_round_generate(
        self,
        user_prompt: str,
        max_tokens: int = 4096,
    ) -> Tuple[str, str]:
        self.messages.append({"role": "user", "content": user_prompt})

        response = self.client.chat.completions.create(
            model=_resolve_model_name(),
            messages=self.messages,
            max_tokens=max_tokens,
            temperature=0.7,
        )
        message = response.choices[0].message
        self.content = message.content or ""
        self.reasoning_content = getattr(message, "reasoning_content", "") or ""
        if self.reasoning_content:
            self.messages.append({"role": "think", "content": self.reasoning_content})
        self.messages.append({"role": "assistant", "content": self.content})
        return (self.content, self.reasoning_content)

    def save_messages(self, file_path):
        print("Start saving the message data for this round of trials.\n")
        distill_data = {}
        for message in self.messages:
            role = message.get("role")
            content = message.get("content")
            if role == "user":
                distill_data["user"] = content
            elif role == "think":
                distill_data["think"] = content
            elif role == "assistant":
                distill_data["assistant"] = content
        add_to_jsonl(file_path, distill_data)
        print("Save Messages Done!\n")
