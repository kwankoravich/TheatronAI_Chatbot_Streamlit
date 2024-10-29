import streamlit as st
from llama_index.core import (
    Settings,
    StorageContext,
    load_index_from_storage,
)
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding


# Set OpenAI API key
# openai_key = st.secrets["OPENAI_API_KEY"]
openai_key = st.text_input('OPENAI_API_KEY')

# Define index directory
INDEX_DIR = "./index"

# Streamlit app setup
st.title("Chat with Llama Index")

# Initialize the embedding model
embedding_model = OpenAIEmbedding(api_key=openai_key)

# Set LLM (OpenAI in this case)
Settings.llm = OpenAI(
    model="gpt-4o-mini", embed_model=embedding_model, api_key=openai_key
)
Settings.embed_model = embedding_model

# Load the index from storage
storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
index = load_index_from_storage(storage_context)


# Cache the chat engine to maintain the session state
@st.cache_resource
def get_chat_engine():
    return index.as_chat_engine(streaming=True, similarity_top_k=10)


chat_engine = get_chat_engine()

# Start message
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "How can I help you?"}
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
