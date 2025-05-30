OPENAI_API_KEY = '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'

from langchain_openai import OpenAI

llm = OpenAI(temperature=0.6, openai_api_key=OPENAI_API_KEY)
name = llm.invoke("I want to open a restaurant for Indian food, please suggest some good names")
print(name)
