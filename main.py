# main.py
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import RetrievalQA

from llms.deepseek import DeepSeekLLM  # ✅ 当前用 DeepSeek
# from llms.bili_deepseek import DeepSeekLLM  # ✅ 后面切换公司 LLM 时改这里


@st.cache_resource
def build_vector_store():
    loader = TextLoader("files/example.md", encoding="utf-8")
    docs = loader.load()
    chunks = CharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
    return FAISS.from_documents(chunks, embeddings)

if "messages" not in st.session_state:
    st.session_state.messages = []  # ✅


if __name__ == '__main__':
    st.set_page_config(page_title="直播交易知识库", layout="wide")
    st.title("📚 直播交易知识库")

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    history = StreamlitChatMessageHistory()
    memory = ConversationBufferMemory(
        chat_memory=history, return_messages=True, memory_key="chat_history", output_key="output"
    )


    db = build_vector_store()
    retriever = db.as_retriever(search_kwargs={"k": 3})
    llm = DeepSeekLLM()
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    query = st.chat_input("请输入你的问题")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        st.chat_message("user").write(query)
        # 获取回答
        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                result = qa.run(query)
                st.markdown(result)
        st.session_state.messages.append({"role": "assistant", "content": result})