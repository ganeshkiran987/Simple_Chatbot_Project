import streamlit as st
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.chains.conversation.base import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# --- Configuration ---
# This line MUST be the first Streamlit command
st.set_page_config(page_title="Langchain Chatbot", page_icon="🤖")

# Load environment variables from .env file
load_dotenv()

# Get the OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

# Check if the API key is set. This warning will be displayed in the Streamlit app.
if not openai_api_key:
    st.warning("OpenAI API key is not found in your environment variables. "
               "Please set it as 'OPENAI_API_KEY' in a .env file or directly in your environment. "
               "The chatbot might not function correctly without a valid API key.")
    # If the API key is not set, we can stop the app or disable functionality
    st.stop() # Stop execution if API key is missing

# --- Initialize Langchain Components (cached for performance) ---
# Use st.cache_resource to avoid re-initializing LLM and memory on every rerun
# This is crucial for maintaining conversation history across Streamlit reruns.
@st.cache_resource
def initialize_chatbot():
    """
    Initializes and caches the Langchain components for the chatbot.
    """
    # Initialize the Language Model (LLM)
    # Using the loaded openai_api_key for ChatOpenAI
    llm = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo", openai_api_key=openai_api_key)

    # Set up Conversation Memory
    # Streamlit's session_state is used to persist memory across reruns
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory()

    # Define the Prompt Template
    template = """The following is a friendly conversation between a human and an AI.
The AI is talkative and provides lots of specific details from its context.
If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:
{history}
Human: {input}
AI:"""
    prompt = PromptTemplate(input_variables=["history", "input"], template=template)

    # Create the Conversation Chain
    conversation = ConversationChain(
        llm=llm,
        memory=st.session_state.memory, # Use the memory from session_state
        prompt=prompt,
        verbose=True # Set to True to see the internal workings in the console where Streamlit runs
    )
    return conversation

# Initialize the chatbot components
conversation_chain = initialize_chatbot()

# --- Streamlit UI ---

st.title("🤖 Langchain Chatbot")
st.markdown("Ask me anything! I'll remember our conversation.")

# Display chat messages from history on app rerun
# Initialize chat history in session_state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What would you like to ask?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Predict the AI's response based on the current input and conversation history
                ai_response = conversation_chain.predict(input=prompt)
                st.markdown(ai_response)
                # Add AI response to chat history
                st.session_state.messages.append({"role": "assistant", "content": ai_response})
            except Exception as e:
                error_message = f"An error occurred: {e}. Please check your OpenAI API key and internet connection."
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})

# Optional: Clear chat history button
if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    st.session_state.memory = ConversationBufferMemory() # Reset memory as well
    st.experimental_rerun() # Rerun the app to clear the display