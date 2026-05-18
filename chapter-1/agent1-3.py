import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# 加载环境变量
load_dotenv()

# prompt_template = ChatPromptTemplate.from_messages(
#     [
#         ("system", "Translate the following from English into Chinese:"),
#         ("user", "{text}"),
#     ]
# )
class work(BaseModel):
    title: str = Field(description="The title of the work")
    decription: str = Field(description="A brief description of the work")


model = ChatOpenAI(model="gpt-4o-mini")
parser = JsonOutputParser(pydantic_object=work)

prompt = PromptTemplate(
    template="列举3部{author}的作品。\n{format_instructions}",
    input_variables=["author"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

chain = prompt | model | parser

stream = chain.invoke({"author": "鲁迅"})
print(stream)
# for chunk in stream:
#     print(chunk.content, end="|", flush=True)
