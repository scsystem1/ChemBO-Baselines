import unittest
from src.prompts.base import PromptManager


class TestTextPrompt(unittest.TestCase):

    def setUp(self):
        self.prompt_manager = PromptManager()

    def test_get_prompt_valid(self):
        prompt = self.prompt_manager.get("test", "en")
        self.assertEqual(prompt, "hello, I'm {name}, {age} years old")

    def test_format_prompt_valid(self):
        meta_dict = {
            "name": "John",
            "age": 30,
        }
        formatted = self.prompt_manager.format(
            "test",
            **meta_dict,
        )
        self.assertEqual(formatted, "hello, I'm John, 30 years old")

        formatted = self.prompt_manager.format(
            "test",
            lang="zh",
            **meta_dict,
        )
        self.assertEqual(formatted, "你好，我是John，30岁")


if __name__ == "__main__":
    unittest.main()
