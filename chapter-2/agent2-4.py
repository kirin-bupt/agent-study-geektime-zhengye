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
import tiktoken
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    SystemMessage,
    trim_messages,
)
from typing import List

load_dotenv()

store = {}

chat_model = ChatOpenAI(model="gpt-4o-mini")

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你现在扮演杨幂的角色，尽量按照杨幂的风格回复",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


def str_token_counter(text: str) -> int:
    enc = tiktoken.get_encoding("o200k_base")
    return len(enc.encode(text))


def tiktoken_counter(messages: List[BaseMessage]) -> int:
    num_tokens = 3
    tokens_per_message = 3
    tokens_per_name = 1
    for msg in messages:
        if isinstance(msg, HumanMessage):
            role = "user"
        elif isinstance(msg, AIMessage):
            role = "assistant"
        elif isinstance(msg, ToolMessage):
            role = "tool"
        elif isinstance(msg, SystemMessage):
            role = "system"
        else:
            raise ValueError(f"Unsupported messages type {msg.__class__}")

        content = str(msg.content) if msg.content is not None else ""
        num_tokens += (
            tokens_per_message + str_token_counter(role) + str_token_counter(content)
        )
        if msg.name:
            num_tokens += tokens_per_name + str_token_counter(msg.name)
    return num_tokens


trimmer = trim_messages(
    max_tokens=4096,
    strategy="last",
    token_counter=tiktoken_counter,
    include_system=True,
)


runnable_with_history = RunnableWithMessageHistory(
    trimmer | prompt | chat_model, get_session_history
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
