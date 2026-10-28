from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults

@tool
def get_greeting(name :str)->str:
    """Generate a greeting message for a user"""
    return f"Hello {name}, Welcome to the AI world"

result = get_greeting.invoke({f"name": "abhay"})
print(result)
