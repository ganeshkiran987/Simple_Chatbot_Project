# Import necessary Langchain components
from langchain_openai import ChatOpenAI

from langchain.chains.conversation.base import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
# openai_api_key =  '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'

# if not openai_api_key or openai_api_key =='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':
if not openai_api_key or openai_api_key == os.getenv("OPENAI_API_KEY"):

    print("WARNING: OpenAI API key is not set. Please set it as an environment variable 'OPENAI_API_KEY' or replace 'YOUR_OPENAI_API_KEY' in the script.")
    print("The chatbot might not function correctly without a valid API key.")

llm = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo")

memory = ConversationBufferMemory()

template = """The following is a friendly conversation between a human and an AI.
The AI is talkative and provides lots of specific details from its context.
If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:
{history}
Human: {input}
AI:"""

prompt = PromptTemplate(input_variables=["history", "input"], template=template)

# --- Create the Conversation Chain ---
# The ConversationChain combines the LLM, memory, and prompt to manage the conversation flow.
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    prompt=prompt,
    verbose=True # Set to True to see the internal workings of the chain (useful for debugging)
)

# --- Chatbot Interaction Loop ---
print("Welcome to the Langchain Chatbot! Type 'exit' to end the conversation.")

while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        print("Goodbye!")
        break
    
    try:
        # Predict the AI's response based on the current input and conversation history
        ai_response = conversation.predict(input=user_input)
        print(f"AI: {ai_response}")
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure your OpenAI API key is correctly set and you have an active internet connection.")

