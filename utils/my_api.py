# -*- coding: utf-8 -*-

import json
from datetime import datetime as dt
from os import getenv
from requests import Response
import requests
from certifi import where


class MyApi:
    DEFAULT_SYSTEM_PROMPT = "You are a QA analyst."
    TITLE_PROMPT = "Generate a brief title for our conversation."

    def __init__(
        self,
        api_key,
        model="gpt-4o",
        url = "https://xiaoai.plus/v1/chat/completions",
      #  url="https://api.pumpkinaigc.online/v1/chat/completions",
        proxy= "",
    ):
        self.api_key = api_key
        self.system_prompt = getenv("API_SYSTEM_PROMPT", self.DEFAULT_SYSTEM_PROMPT)
        self.messages = self.init_message()
        self.conversation_id = 0
        self.model_slug = model
        self.url = url
        self.req_kwargs = {
            "proxies": {
                "http": proxy,
                "https": proxy,
            }
            if proxy
            else None,
            "verify": where(),
            "timeout": 600,
            "allow_redirects": False,
        }

    def init_message(self):
        messages = [{"role": "system", "content": self.DEFAULT_SYSTEM_PROMPT}]

        return messages

    def __get_headers(self, api_key):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        return headers

    def talk(self, prompt):
        user_prompt = {"role": "user", "content": prompt}
        self.messages.append(user_prompt)
        data = {
            "model": self.model_slug,
            "messages": self.messages,
        }

        max_retry = 10
        try_num = 0
        while try_num < max_retry:
            try_num += 1
            try:
                resp = requests.post(
                    url=self.url,
                    headers=self.__get_headers(self.api_key),
                    data=json.dumps(data),
                    **self.req_kwargs,
                )

                is_error, text = self.deal_reply(resp.status_code, json.loads(resp.text), 1)
                if not is_error:
                    print(text)
                    continue
                else:
                    return text
            except Exception as e:
                print(text)

        return "llm server error"
    
    def talk_new(self, prompt):
        user_prompt = {"role": "user", "content": prompt}
        tmpMessages = self.messages.copy()
        tmpMessages.append(user_prompt)
        data = {
            "model": self.model_slug,
            "messages": tmpMessages,
        }

        max_retry = 10
        try_num = 0
        while try_num < max_retry:
            try_num += 1
            try:
                resp = requests.post(
                    url=self.url,
                    headers=self.__get_headers(self.api_key),
                    data=json.dumps(data),
                    **self.req_kwargs,
                )

                is_error, text = self.deal_reply(resp.status_code, json.loads(resp.text), 0)
                if not is_error:
                    print(text)
                    continue
                else:
                    return text
            except Exception as e:
                print(e)

        return "llm server error"
    
    def deal_reply(self, status, response, flag):
        if status != 200:
            print(f"访问API出现异常，建议查询当前情况：\n\n异常key: {self.api_key}")
            print(f"响应内容: {response}")
            return False, response

        choice = response["choices"][0]
        
        if "message" in choice:
            text = choice["message"].get("content", "")
        elif "delta" in choice:
            text = choice["delta"].get("content", "")
        else:
            return False, response

        if flag == 1:
            self.messages.append({"role": "assistant", "content": text})

        return True, text
