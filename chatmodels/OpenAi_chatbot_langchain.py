import os
from dotenv import load_dotenv

# Langchain import
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

# check whether the api key exists
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY .env file main nahi mili!")

# Initialize the openAi chat model through langchain
model = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0.4
)


prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful AI assistant. "
        "Answer clearly and simply. "
        "Use Hinglish when the user speaks    Hinglish."
    ),

    # Previous conversation history
    MessagesPlaceholder(variable_name="history"),

    # Current user message
    ("human", "{input}")
])


# combine the prompt and model into a chain
chain = prompt | model


# create an in-memory store for conversation histories
store = {}
MAX_MESSAGES = 10

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()

    history = store[session_id]

    if len(history.messages) > MAX_MESSAGES:
        history.messages = history.messages[-MAX_MESSAGES:]  # agar error aaye to niche wala use karein

    return history  
# step 7 add automatic message-history management to the chain
chatbot = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

# Step 8: Start the chatbot
print("LangChain Chatbot Started!")
print("Type 'exit' to stop.\n")

session_id = "my-chat-session"

while True:
    user_input = input("You: ").strip()

    if user_input.lower() in ["exit", "quit", "bye"]:
        print("Bhagwan: Goodbye bhai!")
        break

    if not user_input:
        continue

    try:
        # Step 9: Invoke the chain with the current message
        response = chatbot.invoke(
            {"input": user_input},
            config={
                "configurable": {
                    "session_id": session_id
                }
            }
        )

        # Step 10: Display the AI response
        print(f"Bot: {response.content}\n")

    except Exception as e:
        print(f"Error: {e}\n")
