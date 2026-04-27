import pytest
from openai import OpenAI
from src.config.config import Config


class TestOpenAI:
    @pytest.fixture
    def client(self):
        return OpenAI(
            base_url=Config().OPENAI_API_BASE, api_key=Config().OPENAI_API_KEY
        )

    @pytest.fixture
    def system_prompt(self):
        return "you are a Geography tutor"

    @pytest.fixture
    def user_prompt(self):
        return """Given the following question and two potential answers: 

                    Question: What's the capital of England

                    Answer 1: It's Paris

                    Answer 2: It's London

                    which is the correct answer"""

    def test_openai_response(self, client, system_prompt, user_prompt):
        completion = client.chat.completions.create(
            model=Config().OPENAI_MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
        )
        result = completion.choices[0].message
        assert result is not None
        assert isinstance(result.content, str)
