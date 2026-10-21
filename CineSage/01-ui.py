
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Page setup
st.set_page_config(page_title="Movie Information Extractor")
st.title("🎬 Movie Information Extractor")

# API key check
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("OPENAI_API_KEY .env file mein nahi mili!")
    st.stop()

model = ChatOpenAI(
    model="gpt-4.1-mini",
    api_key=api_key,
    temperature=0.4
)

prompt = ChatPromptTemplate.from_messages([
    ("system",
     """
     You are a movie information extraction assistant.

     Your task is to extract the movie summary and important
     information from the movie description provided by the user.

     Extract the following details:
     1. Movie Name
     2. Director
     3. Release Year
     4. Genre
     5. Main Cast
     6. Movie Summary
     7. IMDb Rating (if provided)
     8. Music Composer (if provided)
     9. Key Themes
     10. Why the movie is popular or notable

     Instructions:
     - Use only information available in the given description.
     - Do not invent facts or ratings.
     - If information is missing, write "Not provided".
     - Write the summary in clear, simple English.
     - Keep the summary between 80 and 120 words.
     - Present the output in a well-organized format.
     """),
    (
        "human",
        "Extract the movie information from this description:\n\n"
        "{movie_description}"
    )
])

# Connect prompt with model
chain = prompt | model

# Take paragraph input from the user
movie_description = st.text_area(
    "Enter the movie description paragraph:",
    height=200
)

# Generate output
if st.button("Extract Movie Information"):
    if movie_description.strip():
        with st.spinner("Extracting movie information..."):
            response = chain.invoke({
                "movie_description": movie_description
            })

        st.subheader("Movie Information and Summary")
        st.write(response.content)
    else:
        st.warning("Please enter a movie description.")
