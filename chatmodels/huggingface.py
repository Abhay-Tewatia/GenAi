from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
    temperature=0.7,
    task="text-generation",
  max_new_tokens=1024,
)
model = ChatHuggingFace(llm=llm)

response = model.invoke("Why do parrots talk?")
print(response.content)

# payment requires
