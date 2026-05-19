from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from dotenv import load_dotenv

import os

# 修复：显式加载项目根目录的 .env
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
load_dotenv()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

loader = TextLoader(os.path.join(SCRIPT_DIR, "geektime-article-821470.txt"))
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma(
    collection_name="ai_learning",
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
    persist_directory=os.path.join(SCRIPT_DIR, "vectordb"),
)
vectorstore.add_documents(splits)

documents = vectorstore.similarity_search("专栏的作者是谁？")
print(documents)
