import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# 加载环境变量
load_dotenv()

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "Translate the following from English into Chinese:"),
        ("user", "{text}"),
    ]
)

model = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()

chain = prompt_template|model|parser

stream = chain.invoke({"text": "Welcome to my LLM application development!"})
print(stream)
# for chunk in stream:
#     print(chunk.content, end="|", flush=True)
