from langchain.document_loaders import WebBaseLoader
from bs4.filter import SoupStrainer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

loader = WebBaseLoader(
    web_paths=("https://www.greatfrontend.com/front-end-interview-playbook/system-design",),
    # bs_kwargs=dict(
    #     parse_only=SoupStrainer(
    #         class_=("article-content", "article-title")
    #     )
    # ),
)

blog_docs = loader.load()

# Split the documents into chunks
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000, 
    chunk_overlap=50
)
splits = text_splitter.split_documents(blog_docs[0:1])

import asyncio
import threading

try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

import time
embed_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", async_client=False)

batch_size = 5
all_embeddings = []
for i in range(0, len(splits), batch_size):
    batch = splits[i:i+batch_size]
    texts = [doc.page_content for doc in batch]
    batch_emb = embed_model.embed_documents(texts)
    all_embeddings.extend(batch_emb)
    time.sleep(1)  # delay giữa các batch

# Index the chunks in a Chroma vector store
vectorstore = Chroma.from_documents(documents=splits, 
                                    embedding=embed_model)

# Create our retriever
retriever = vectorstore.as_retriever()