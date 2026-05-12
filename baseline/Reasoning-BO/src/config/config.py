#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
@DATE: 2025-02-27 16:06:39
@File: src/app/config.py
@IDE: vscode
@Description:
    Store environment variables
"""

import os
from dotenv import load_dotenv

# 系统环境变量优先级 > .env 配置的值
load_dotenv('.env', override=False)


class Config:
    def __init__(self):
        self.OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME")
        self.KIMI_API_KEY = (
            os.getenv("DASHSCOPE_API_KEY")
            or os.getenv("REASONINGBO_API_KEY")
            or os.getenv("MOONSHOT_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        self.KIMI_API_BASE = os.getenv(
            "REASONINGBO_BASE_URL",
            os.getenv("KIMI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        )
        self.KIMI_MODEL_NAME = os.getenv(
            "REASONINGBO_LLM_MODEL",
            os.getenv("KIMI_MODEL", "kimi-k2.5-thinking"),
        )
        self.DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
        self.DEEPSEEK_API_BASE = os.getenv("DEEPSEEK_API_BASE")
        self.DEEPSEEK_MODEL_NAME = os.getenv("DEEPSEEK_MODEL_NAME")
        self.QWQ_API_KEY = os.getenv('QWQ_API_KEY')
        self.QWQ_API_BASE = os.getenv('QWQ_API_BASE')
        self.QWQ_MODEL_NAME = os.getenv('QWQ_MODEL_NAME')
        self.NOTES_AGENT = os.getenv('NOTES_AGENT')
        self.NEO4J_URL = os.getenv("NEO4J_URL")
        self.NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
        self.NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
        self.MILVUS_URL = os.getenv("MILVUS_URL")
