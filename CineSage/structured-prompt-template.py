import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List, Optional
load_dotenv()

# api key check
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY .env file mein nahi mili!")


# step 1:  Define the output structure
class MovieInfo(BaseModel):
    movie_name: str = Field(description="Name of the movie")
    director: str = Field(description="Movie director")
    release_year: int = Field(description="Release year")
    genre: str = Field(description="Movie genre")
    main_cast: list[str] = Field(description="Main actors")
    summary: str = Field(description="Short movie summary")
    
# step 2: Initialize the model
model = ChatOpenAI(
    model="gpt-4.1-mini",
    api_key=api_key,
    temperature=0.4
)

# Step 4: Enable structured output
structured_model = model.with_structured_output(MovieInfo)

# Step 5: Take paragraph input from user
description = input("Enter the movie description:\n")

# Step 6: Extract movie information
response = structured_model.invoke(
    "Extract movie information from this description:\n" + description
)
# Step 7: Display output
print("\nMovie Information")
print("-----------------")
print(response.model_dump_json(indent=2))