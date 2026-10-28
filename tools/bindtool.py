# Import required libraries
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# -----------------------------------
# 1. Create a Custom Tool
# -----------------------------------

@tool
def sentence_length(sentence: str) -> int:
    """Find the number of words in a sentence."""

    # Split the sentence into words and count them
    return len(sentence.split())


# -----------------------------------
# 2. Initialize OpenAI Model
# -----------------------------------

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
)


# -----------------------------------
# 3. Bind the Tool with OpenAI
# -----------------------------------

# Make the custom tool available to the OpenAI model
llm_with_tools = llm.bind_tools([sentence_length])


# -----------------------------------
# 4. Define User Query
# -----------------------------------

query = """
Find the word count of this sentence:
Artificial intelligence is changing the world
"""


# -----------------------------------
# 5. Create Messages
# -----------------------------------

messages = [
    SystemMessage(
        content=(
            "You are a helpful assistant. "
            "Use the sentence_length tool when the user asks for a word count."
        )
    ),
    HumanMessage(content=query)
]


# -----------------------------------
# 6. Call the OpenAI Model
# -----------------------------------

# The model decides whether to call the custom tool
response = llm_with_tools.invoke(messages)

# Print the tool call requested by the model
print("Tool Calls:")
print(response.tool_calls)


# -----------------------------------
# 7. Execute the Tool
# -----------------------------------

# Loop through the tool calls requested by the model
for tool_call in response.tool_calls:

    # Check whether the requested tool is sentence_length
    if tool_call["name"] == "sentence_length":

        # Execute the custom tool with the provided arguments
        result = sentence_length.invoke(tool_call["args"])

        # Add the model's tool call message to conversation history
        messages.append(response)

        # Add the tool execution result to conversation history
        messages.append(
            ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"]
            )
        )


# -----------------------------------
# 8. Generate Final Response
# -----------------------------------

# Send the tool result back to OpenAI to generate the final answer
final_response = llm.invoke(messages)

# Print the final response
print("\nFinal Response:")
print(final_response.content)