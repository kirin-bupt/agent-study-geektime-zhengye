from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

chat_model = ChatOpenAI(model="gpt-4o-mini")

while True:
    user_input = input("You:> ")
    if user_input.lower() == "exit":
        break
    stream = chat_model.stream([HumanMessage(content=user_input)])
    for chunk in stream:
        print(chunk.content, end="", flush=True)
    print()
