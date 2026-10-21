
import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Step 1: Load environment variables from .env
load_dotenv()

# Step 2: Check whether the Groq API key exists
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY .env file mein nahi mili!")

# Step 3: Model settings
MODEL = "openai/gpt-oss-120b"
TEMPERATURE = 0.4
MAX_HISTORY = 10

# Step 4: Initialize the Groq chat model through LangChain
model = ChatGroq(
    model=MODEL,
    temperature=TEMPERATURE,
    api_key=os.getenv("GROQ_API_KEY")
)

# Step 5: Define chatbot instructions and message structure
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful AI assistant. "
        "Answer clearly and simply. "
        "Use Hinglish when the user speaks Hinglish."
    ),

    # Insert previous messages into the prompt
    MessagesPlaceholder(variable_name="history"),

    # Insert the current user message
    ("human", "{input}")
])

# Step 6: Connect the prompt with the Groq model
chain = prompt | model

# Step 7: Store chat histories by session ID
chat_histories = {}

def get_session_history(session_id: str):
    # Create a new history if this session does not exist
    if session_id not in chat_histories:
        chat_histories[session_id] = InMemoryChatMessageHistory()

    history = chat_histories[session_id]

    # Keep only the most recent MAX_HISTORY messages
    if len(history.messages) > MAX_HISTORY:
        history.messages[:] = history.messages[-MAX_HISTORY:]

    return history

# Step 8: Automatically load and save conversation messages
chatbot = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

# Step 9: Start the chatbot
def main():
    print("Groq + LangChain Chatbot Started!")
    print("Type 'exit' to stop.\n")

    session_id = "my-chat-session"

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Bot: Goodbye bhai!")
            break

        if not user_input:
            continue

        try:
            # Send the message with its conversation history
            response = chatbot.invoke(
                {"input": user_input},
                config={
                    "configurable": {
                        "session_id": session_id
                    }
                }
            )

            # Display the assistant's response
            print(f"Bot: {response.content}\n")

        except Exception as e:
            print(f"Error: {e}\n")

if __name__ == "__main__":
    main()
