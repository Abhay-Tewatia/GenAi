import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from operator import itemgetter

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY .env file mein nahi mili!")

model = ChatOpenAI(
    model="gpt-4.1-mini",
    api_key=api_key,
    temperature=0.4,
)
parser = StrOutputParser()

explain_prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in simple words."
)

example_prompt = ChatPromptTemplate.from_template(
    "Give one real-world example of {topic}."
)

explain_chain = explain_prompt | model | parser
example_chain = example_prompt | model | parser

full_chain = RunnableParallel(
    topic=RunnablePassthrough() | itemgetter("topic"),
    explanation={"topic": itemgetter("topic")} | explain_chain,
    example={"topic": itemgetter("topic")} | example_chain,
)

response = full_chain.invoke({"topic": "Generative AI"})


print("Topic      :", response["topic"])
print("\nExplanation:\n", response["explanation"])
print("\nExample    :\n", response["example"])