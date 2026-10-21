import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# ---------- Page setup ----------
st.set_page_config(page_title="Groq Chatbot", page_icon="🤖")
st.title("🤖 Groq + LangChain Chatbot")

# ---------- .env aur API key ----------
load_dotenv()
if not os.getenv("GROQ_API_KEY"):
    st.error("GROQ_API_KEY .env file mein nahi mili!")
    st.stop()

# ---------- Settings ----------
MODEL = "openai/gpt-oss-120b"
TEMPERATURE = 0.4
MAX_HISTORY = 10
SESSION_ID = "my-chat-session"

# ---------- LangChain setup ----------
model = ChatGroq(
    model=MODEL,
    temperature=TEMPERATURE,
    api_key=os.getenv("GROQ_API_KEY"),
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Answer clearly and simply. Use Hinglish when the user speaks Hinglish."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

chain = prompt | model

# LangChain ki history browser session mein rakhte hain
if "lc_store" not in st.session_state:
    st.session_state.lc_store = {}

def get_session_history(session_id: str):
    store = st.session_state.lc_store
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    history = store[session_id]
    if len(history.messages) > MAX_HISTORY:
        history.messages[:] = history.messages[-MAX_HISTORY:]
    return history

chatbot = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# ---------- Screen par purane messages dikhao ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ---------- Naya message lo aur jawab do ----------
user_input = st.chat_input("Kuch likhiye...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Soch raha hoon..."):
            response = chatbot.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": SESSION_ID}},
            )
        st.write(response.content)

    st.session_state.messages.append({"role": "assistant", "content": response.content})