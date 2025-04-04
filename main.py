# main.py
import os
import sys

import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

from llms.deepseek import DeepSeekLLM  # ✅ 当前用 DeepSeek
# from llms.bili_custom import BilibiliChatLLM  # ✅ 后面切换公司 LLM 时改这里


@st.cache_resource
def build_vector_store():
    loader = TextLoader("files/example.md", encoding="utf-8")
    docs = loader.load()
    chunks = CharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
    return FAISS.from_documents(chunks, embeddings)


def get_api_key(filepath="api.txt") -> str:
    if not os.path.exists(filepath):
        sys.exit(f"❌ API key file '{filepath}' not found.")

    with open(filepath, "r", encoding="utf-8") as f:
        key = f.read().strip()
        if not key:
            sys.exit("❌ API key is empty.")
        return key


if __name__ == '__main__':
    st.set_page_config(page_title="RAG Demo", layout="wide")
    st.title("📚 RAG 本地知识库问答")

    db = build_vector_store()
    retriever = db.as_retriever(search_kwargs={"k": 3})
    llm = DeepSeekLLM(api_key=get_api_key())  # 👈 填你的 DeepSeek Key
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    query = st.text_input("请输入你的问题")
    if query:
        with st.spinner("思考中..."):
            result = qa.run(query)
        st.success(result)
