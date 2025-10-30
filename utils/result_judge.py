from llm.llm_server import ChatBot
from utils.my_api import MyApi
import numpy as np

class ResultJudge:
    def __init__(self, chain_modal = "Deepseek"):
        self.api_key = "sk-GRmKqS2w6MLY8gjlo3Sxz0LzxIEYkcSfA8IAVSB5rVKJTbEI"#"sk-fvXDuNXlHri5Mh02WA4HWpFHPFxktboNCEDU1wD1jGfidpqh"
        self.chain_modal = chain_modal
        if chain_modal == "Deepseek":
            self.modal = ChatBot()
        elif self.chain_modal == "gpt":
            self.modal = MyApi(self.api_key)
        elif self.chain_modal == "deepseek-api":
            self.modal = MyApi("sk-fa9c6c9c60ee4296ac9dbda8b86ad503", model = "deepseek-chat", url = "https://api.deepseek.com/chat/completions")
        else:
            print("未识别的模型类型.")
            return None
    
    def judge(self, prompt):
        #print(prompt)
        response = self.modal.talk_new(prompt)
       # print(response)
        return response