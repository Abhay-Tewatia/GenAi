import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
load_dotenv()



# api key check
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY .env file mein nahi mili!")


model = ChatOpenAI(
    model="gpt-4.1-mini",
    api_key=api_key,
    temperature=0.4
)

parser = JsonOutputParser()

# Step 1: ChatPromptTemplate banana
prompt  = ChatPromptTemplate.from_messages([
    ( "system",
        "You are a helpful AI assistant.\n"
        "Answer the user question in JSON format.\n"
        "Return only valid JSON without markdown.\n"
        "{format_instructions}"),
    ("human", "{user_input}")
])

# add format instructions
prompt = prompt.partial(
    format_instructions = parser.get_format_instructions()
)

# create chain

chain = prompt | model | parser

# invoke chain

response = chain.invoke({
    "user_input":"Explain Generative AI"
})

# step 6 print output
print(response)