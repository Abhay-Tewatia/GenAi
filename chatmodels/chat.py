# load variables from dotenv file
import os
from groq import Groq
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
load_dotenv()

# select model
my_api_key = os.getenv("GROQ_API_KEY")
if not my_api_key:
    raise ValueError("GROQ_API_KEY .env file mein nahi mili.")

model = init_chat_model(
    "openai/gpt-oss-120b",
    model_provider="groq",
    api_key = my_api_key,
    temperature=1,
    max_tokens = 500
  )
response = model.invoke("write a short poen for my gf ")

print(response.content)
# print(model)
