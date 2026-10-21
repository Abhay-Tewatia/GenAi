
# Import os to read environment variables
import os

# Load variables stored in the .env file
from dotenv import load_dotenv

# OpenAI Python SDK
from openai import OpenAI

# Load the .env file
load_dotenv()

# Read the API key securely from the environment
api_key = os.getenv("OPENAI_API_KEY")

# Stop if the API key is missing
if not api_key:
    raise ValueError(
        "OPENAI_API_KEY nahi mili. Apni .env file check karo."
    )

# Create the OpenAI API client
client = OpenAI(api_key=api_key)

# This is the text we want to convert into an embedding
text = "I love learning machine learning and artificial intelligence."

# Generate the embedding using OpenAI's embedding model
response = client.embeddings.create(
    model="text-embedding-3-small",
    input=text
)

# Extract the numerical vector from the API response
vector = response.data[0].embedding

# Print the results
print("Original text:", text)
print("Vector dimensions:", len(vector))
print("First 5 vector values:", vector[:5])
