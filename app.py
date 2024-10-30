import streamlit as st
from llama_index.core import (
    Settings,
    StorageContext,
    load_index_from_storage,
    Document,
    SimpleDirectoryReader,
    ServiceContext,
    VectorStoreIndex
)
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.node_parser import SentenceSplitter
import os


# Set OpenAI API key
# openai_key = st.secrets["OPENAI_API_KEY"]
# openai_key = st.text_input('OPENAI_API_KEY')

# os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"]
google_api_key = st.secrets["GOOGLE_API_KEY"]

# Define index directory
INDEX_DIR = "./index"

# Streamlit app setup
st.title("Chat with Theatron AI")

# Initialize the embedding model
# embedding_model = OpenAIEmbedding(api_key=openai_key)
# embedding_model = OpenAIEmbedding()
embedding_model = GeminiEmbedding()

# Set LLM (OpenAI in this case)
# Settings.llm = OpenAI(
#     # model="gpt-4o-mini", embed_model=embedding_model, api_key=openai_key
#     model="gpt-4o-mini", embed_model=embedding_model,
# )

Settings.llm = Gemini(model='models/gemini-1.5-flash', google_api_key=google_api_key)

Settings.embed_model = embedding_model

# # Load the index from storage
# storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
# index = load_index_from_storage(storage_context)


@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="โปรดรอสักครู่ตอนนี้ระบบกำลังทำประมวณผล"):
        reader = SimpleDirectoryReader(input_dir="./data_update", recursive=True)
        docs = reader.load_data()

        embed_model = GeminiEmbedding()
        llm = Gemini(model='models/gemini-1.5-flash', google_api_key=google_api_key)

        # service_context = ServiceContext.from_defaults(llm=llm, embed_model=embedding_model)
        # Parse documents into nodes
        node_parser = SentenceSplitter(chunk_size=256, chunk_overlap=20)
        nodes = node_parser.get_nodes_from_documents(docs)
        # storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
        # index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)
        # index = VectorStoreIndex.from_documents(docs)
        index = VectorStoreIndex(nodes, show_progress=True)
        return index

index = load_data()



# Cache the chat engine to maintain the session state
@st.cache_resource
def get_chat_engine():
    return index.as_chat_engine(chat_mode="context", streaming=True, similarity_top_k=10)


chat_engine = get_chat_engine()

# Start message
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "มีอะไรให้ฉันช่วยไหม"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


# Chat Loop
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    response = chat_engine.stream_chat(prompt)
    st.chat_message("assistant").write(response.response_gen)
    st.session_state.messages.append(
        {"role": "assistant", "content": response.response}
    )
