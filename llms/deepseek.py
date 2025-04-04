from langchain.chat_models.base import SimpleChatModel
from langchain.schema import ChatMessage, HumanMessage, AIMessage, SystemMessage
from typing import List
from llms.secrete import get_api_key
import requests

class DeepSeekLLM(SimpleChatModel):
    api_key: str = get_api_key("deepseek-api-key")
    url: str = "https://api.deepseek.com/chat/completions"  # ✅ DeepSeek 正确路径
    model: str = "deepseek-chat"

    def _call(self, messages: List[ChatMessage], **kwargs) -> str:
        api_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            elif isinstance(msg, SystemMessage):
                role = "system"
            else:
                continue  # 跳过 FunctionMessage 等暂不支持的
            api_messages.append({"role": role, "content": msg.content})  # ✅ 这行是关键

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": api_messages,
            "temperature": 0.7,
            "stream": False  # 明确关掉 stream，便于调试
        }

        response = requests.post(self.url, headers=headers, json=payload)
        response.raise_for_status()  # 报错就抛异常
        return response.json()["choices"][0]["message"]["content"]

    def _llm_type(self) -> str:
        return self.model