
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Different texts that we want to convert into vectors
texts = [
    "Machine learning allows computers to learn from data.",
    "Deep learning uses neural networks with multiple layers.",
    "I enjoy playing cricket on weekends.",
    "Natural language processing helps computers understand text."
]

# Generate embeddings for all texts in one API request
response = client.embeddings.create(
    model="text-embedding-3-small",
    input=texts
)

# Each response item contains the embedding for one input text
for item in response.data:
    text = texts[item.index]
    vector = item.embedding

    print("\nText:", text)
    print("Vector dimensions:", len(vector))
    print("First 3 values:", vector[:3])

# Print the number of input tokens reported by the API
print("\nTotal input tokens:", response.usage.total_tokens)
