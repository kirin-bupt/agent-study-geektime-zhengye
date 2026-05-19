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
import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
vectorstore = Chroma(
    collection_name="ai_learning",
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
    persist_directory=os.path.join(SCRIPT_DIR, "vectordb"),
)
retriever = vectorstore.as_retriever(search_type="similarity")

store = {}

chat_model = ChatOpenAI(model="gpt-4o-mini")

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
            Context: {context}""",
        ),
        MessagesPlaceholder(variable_name="history"),
        (
            "human",
            "{question}",
        ),
    ]
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


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

context = itemgetter("question") | retriever | format_docs
first_step = RunnablePassthrough.assign(context=context)
chain = first_step | prompt | trimmer | chat_model

runnable_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)

config: RunnableConfig = {"configurable": {"session_id": "session-1"}}

while True:
    user_input = input("You:> ")
    if user_input.lower() == "exit":
        break
    stream = runnable_with_history.stream({"question": user_input}, config=config)
    for chunk in stream:
        print(chunk.content, end="", flush=True)
    print()
