from langchain_openai import ChatOpenAI
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)

from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

store = {}

chat_model = ChatOpenAI(model="gpt-4o-mini")

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你现在扮演孔子的角色，尽量按照孔子的风格回复，不要出现‘子曰’",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


runnable_with_history = RunnableWithMessageHistory(
    prompt | chat_model, get_session_history
)

config: RunnableConfig = {"configurable": {"session_id": "session-1"}}

while True:
    user_input = input("You:> ")
    if user_input.lower() == "exit":
        break
    stream = runnable_with_history.stream(
        {"messages": [HumanMessage(content=user_input)]}, config=config
    )
    for chunk in stream:
        print(chunk.content, end="", flush=True)
    print()
