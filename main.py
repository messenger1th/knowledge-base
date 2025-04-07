import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from langchain.schema import Document

from llms.bili_deepseek import DeepSeekLLM  # 你的 DeepSeek 接口

# -----------------------
# ✅ 页面设置
# -----------------------
st.set_page_config(page_title="📚 直播交易知识库", layout="wide")
st.title("📚 直播交易知识库")

# -----------------------
# ✅ Session 初始化
# -----------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "uploaded_files_changed" not in st.session_state:
    st.session_state.uploaded_files_changed = False

if "last_uploaded_files" not in st.session_state:
    st.session_state.last_uploaded_files = []

# -----------------------
# ✅ 上传文件区域
# -----------------------
uploaded_files = st.file_uploader(
    "📎 上传文件（支持 .txt / .md）",
    type=["txt", "md"],
    accept_multiple_files=True
)

# 判断是否需要更新向量库
if uploaded_files:
    filenames = [f.name for f in uploaded_files]
    if filenames != st.session_state.last_uploaded_files:
        st.session_state.uploaded_files_changed = True
        st.session_state.last_uploaded_files = filenames
else:
    st.session_state.last_uploaded_files = []
    st.session_state.uploaded_files_changed = False

# -----------------------
# ✅ 构建/更新向量库
# -----------------------
if st.session_state.vector_store is None or st.session_state.uploaded_files_changed:
    with st.spinner("🔄 正在构建知识库..."):
        splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
        docs = []

        if uploaded_files:
            for uploaded_file in uploaded_files:
                content = uploaded_file.read().decode("utf-8")
                doc = Document(page_content=content, metadata={"filename": uploaded_file.name})
                docs.append(doc)
        else:
            loader = TextLoader("files/example.md", encoding="utf-8")
            docs = loader.load()

        chunks = splitter.split_documents(docs)
        if st.session_state.vector_store is None:
            st.session_state.vector_store = FAISS.from_documents(chunks, embeddings)
        else:
            st.session_state.vector_store.add_documents(chunks)
        st.session_state.uploaded_files_changed = False

    st.success("✅ 知识库已更新！")

# -----------------------
# ✅ 展示历史对话
# -----------------------
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# -----------------------
# ✅ 构建 QA chain
# -----------------------
history = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(
    chat_memory=history, return_messages=True, memory_key="chat_history", output_key="output"
)

retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
llm = DeepSeekLLM()
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# -----------------------
# ✅ 聊天输入与回复
# -----------------------
query = st.chat_input("请输入你的问题")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    st.chat_message("user").write(query)

    with st.chat_message("assistant"):
        with st.spinner("🤖 思考中..."):
            result = qa.run(query)
            st.markdown(result)

    st.session_state.messages.append({"role": "assistant", "content": result})
