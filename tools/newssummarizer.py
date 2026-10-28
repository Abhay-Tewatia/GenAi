import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

search_tool = TavilySearchResults(max_result=5)

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0.3
)

query = "Latest AI news"
search_results = search_tool.invoke(query)

prompt = ChatPromptTemplate.from_messages([
    ("system", """
    You are a news summarizer.
    Summarize the news in exactly 5 short bullet points.
    Each bullet point must be concise and easy to understand.
    Do not include any introduction or conclusion.
    """),
    ("human", "Summarize these search results:\n{news}")
])

parser = StrOutputParser()

chain = prompt | llm | parser

response = chain.invoke({
    "news": search_results
})

print(response)
print(search_tool.description)
print(search_tool.name)
print(search_tool.args)