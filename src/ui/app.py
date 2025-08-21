import os
import streamlit as st
from dotenv import load_dotenv
from src.agents.agent import MultiToolAgent

load_dotenv()

st.set_page_config(page_title="RAG Multi-Tool Agent", page_icon="🤖")

@st.cache_resource
def get_agent():
    together_key = os.getenv("TOGETHER_API_KEY")
    cohere_key = os.getenv("COHERE_API_KEY")
    return MultiToolAgent(together_key, cohere_key, "data/document.txt")

# Initialize
st.title("🤖 VAIA RAG Multi-Tool Agent")
agent = get_agent()

# Initialize chat
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I can answer questions, summarize content, or extract data as JSON. What would you like to know?"}
    ]

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        content = message["content"]
        st.markdown(content)

# Chat input
if prompt := st.chat_input("Ask me anything..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Generate response
    with st.spinner("Processing..."):
        response = agent.query(prompt)
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    st.rerun()