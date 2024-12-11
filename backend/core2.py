from typing import List, Dict, Any
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

INDEX_MAPPING = {
    "text-embedding-3-small": "langchain-doc-index",
    "text-embedding-3-large": "langchain-doc-index-2"
}
def run_llm2(query: str,
            chat_history: List[Dict[str, Any]] = [],
            embedding_model: str = "text-embedding-3-small",
            model_name: str = "gpt-3.5-turbo",
            temperature: float = 0.0,
            verbose: bool = False) -> Dict[str, Any]:
    # Select the correct index based on the embedding model
    index_name = INDEX_MAPPING.get(embedding_model)

    # Initialize embeddings and the vector store
    embeddings = OpenAIEmbeddings(model=embedding_model)
    docsearch = PineconeVectorStore(index_name=index_name, embedding=embeddings)

    # Set up the chat model with the chosen model_name
    chat = ChatOpenAI(model_name=model_name, verbose=verbose, temperature=temperature)

    # Load retrieval and rephrase prompts from LangChain Hub
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)

    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")
    history_aware_retriever = create_history_aware_retriever(
        llm=chat,
        retriever=docsearch.as_retriever(),
        prompt=rephrase_prompt
    )

    # Set up the retrieval chain
    qa = create_retrieval_chain(
        retriever=history_aware_retriever,
        combine_docs_chain=stuff_documents_chain
    )

    # Run the query through the retrieval chain
    result = qa.invoke(input={"input": query, "chat_history": chat_history})

    # Format and return the result
    new_result = {
        "query": result["input"],
        "result": result["answer"],
        "source_documents": result["context"]
    }
    return new_result