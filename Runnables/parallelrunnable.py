import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda

load_dotenv()

# api key check
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY .env file mein nahi mili!")

# model
model = ChatOpenAI(
    model="gpt-4.1-mini",
    api_key=api_key,
    temperature=0.4
)

parser = StrOutputParser()

# chain 1: Explanation
prompt1 = ChatPromptTemplate.from_template(
    "Explain {topic} in simple words."
)
chain1 = prompt1 | model|parser

# Chain 2: Example

prompt2 = ChatPromptTemplate.from_template(
    "Give one real-world example of {topic}."
)

chain2 = prompt2 | model | parser

parallel_chain = RunnableParallel(
    explanation=RunnableLambda(
        lambda x: chain1.invoke({"topic": x["topic1"]})
    ),

    example=RunnableLambda(
        lambda x: chain2.invoke({"topic": x["topic2"]})
    )
)

response = parallel_chain.invoke({
    "topic1": "Generative AI",
    "topic2": "Machine Learning"
})

print(response)