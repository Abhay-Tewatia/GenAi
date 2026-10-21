import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# api key check
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY .env file mein nahi mili!")

# openai  client bnao
client = OpenAI(api_key=api_key)

# settings
MODEL = "gpt-4.1-mini"
TEMPERATURE=0.4
MAX_HISTORY=10

SYSTEM_PROMPT = (
    "You are a helpful AI assistant. "
    "Answer clearly and simply. "
    "Use Hinglish when the user speaks Hinglish."
)

# Chat history Store
chat_histories = {}
def get_history(session_id):
    if session_id not in chat_histories:
        chat_histories[session_id] = []
    return chat_histories[session_id]
    
# history ko chota rakho
def trim_history(history):
    # Sirf last MAX_HISTORY messages rakho
    if len(history) > MAX_HISTORY:
        del history[:-MAX_HISTORY]

# bot se jawab mangne ka function
def ask_bot(session_id, user_input):
    history = get_history(session_id)

    # (a) User ka message history mein add karo
    history.append({"role": "user", "content": user_input})
    trim_history(history)

    # (b) System prompt + purani history ek saath banao
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

    try:
        # (c) OpenAI ko request bhejo
        response = client.chat.completions.create(
            model=MODEL,
            temperature=TEMPERATURE,
            messages=messages,
        )
    except Exception:
        # Error aaye to user ka last message hata do, taaki history kharab na ho
        history.pop()
        raise

    # (d) Jawab nikalo
    reply = response.choices[0].message.content

    # (e) Bot ka jawab history mein save karo
    history.append({"role": "assistant", "content": reply})
    trim_history(history)

    return reply


# ---------- STEP 8: Main chat loop ----------
def main():
    print("OpenAI Chatbot Started!")
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
            reply = ask_bot(session_id, user_input)
            print(f"Bot: {reply}\n")
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()