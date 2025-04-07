from langchain.chat_models.base import SimpleChatModel
from langchain.schema import ChatMessage, HumanMessage, AIMessage, SystemMessage
from typing import List
import requests
from ai_gateway_py_sdk.hmac_auth.client import HmacAuthClient
from llms.secrete import get_api_key

class DeepSeekLLM(SimpleChatModel):
    access_key: str = get_api_key("bili-access-key")
    secret_key: str = get_api_key("bili-secret-key")
    url: str = "http://ai-gateway.bilibili.co/v1/chat/completions"
    model: str = "deepseek-r1"
    source: str = "ali"

    def _call(self, messages: List[ChatMessage], **kwargs) -> str:
        # 构造符合 deepseek API 格式的 messages
        api_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            elif isinstance(msg, SystemMessage):
                role = "system"
            else:
                continue
            api_messages.append({"role": role, "content": msg.content})

        # 初始化 HMAC 鉴权客户端
        client = HmacAuthClient(
            access_key=self.access_key,
            secret_key=self.secret_key,
        )

        # 签名生成 headers
        headers = client.get_headers(
            method="POST",
            uri="/v1/chat/completions",
            query_params={"source": self.source, "model": self.model},
        )
        headers["Content-Type"] = "application/json"

        payload = {
            "messages": api_messages,
            "stream": False,
            "model": self.model
        }

        response = requests.post(
            url=self.url,
            headers=headers,
            params={"source": self.source, "model": self.model},
            json=payload,
        )
        response.raise_for_status()

        return response.json()["choices"][0]["message"]["content"]

    def _llm_type(self) -> str:
        return self.model