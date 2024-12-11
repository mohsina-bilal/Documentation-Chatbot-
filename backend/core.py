# For retrieval augmentation code
from typing import List, Dict
from dotenv import load_dotenv
import os
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


INDEX_MAPPING = {
    "text-embedding-3-small": "langchain-doc-index",
    "text-embedding-3-large": "langchain-doc-index-2"
}

def run_llm(query : str, chat_history: List[Dict[str, any]] = [], embedding_model: str = "text-embedding-3-small", temperature: float = 0.0, verbose: bool = False, ):

    index_name = INDEX_MAPPING.get(embedding_model)
    embeddings = OpenAIEmbeddings(model=embedding_model)
    docsearch = PineconeVectorStore(index_name= index_name, embedding = embeddings)
    chat = ChatOpenAI(verbose = verbose, temperature = temperature)

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    # creating a chain that wraps together the chat model and the retrieved documents as per the query
    stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)

    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
    history_aware_retriever = create_history_aware_retriever(
        llm = chat, retriever = docsearch.as_retriever(), prompt = rephrase_prompt
    )

    #building another chain that basically implements the concept of RAG
    # handles query interpretation, document retrieval and response generation
    qa = create_retrieval_chain(
        retriever = history_aware_retriever, combine_docs_chain= stuff_documents_chain
    )

    result = qa.invoke(input = {"input": query, "chat_history": chat_history})
    new_result = {
        "query" : result["input"],
        "result": result["answer"],
        "source_documents": result["context"]
    }
    return new_result



