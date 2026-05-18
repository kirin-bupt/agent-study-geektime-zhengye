import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# 加载环境变量
load_dotenv()

messages = [
    SystemMessage(content="Translate the following from English into Chinese:"),
    HumanMessage(content="Welcome to LLM application development!"),
]

model = ChatOpenAI(model="gpt-4o-mini")
stream = model.stream(messages)
for chunk in stream:
    print(chunk.content, end="|", flush=True)