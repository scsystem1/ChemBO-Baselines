from src.bo.reasoner.base import BaseReasoner
from src.llms.kimi import KimiClient


class KimiReasoner(BaseReasoner):
    def __init__(self, exp_config_path: str, result_dir: str):
        super().__init__(exp_config_path, result_dir)
        self.client = KimiClient()
