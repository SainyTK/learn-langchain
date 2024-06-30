from typing import List
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Notes
# 1. VectorStore objects do not subclass Runnable, so can't be integrated into LangChain
# 2. Retrievers wrap VectorStore objects to Runnable

load_dotenv(override=True)

documents = [
    Document(
        page_content="Dogs are great companions, known for their loyalty and friendliness.",
        metadata={"source": "mammal-pets-doc"}
    ),
    Document(
        page_content="Cats are independent pets that often enjoy their own space.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Goldfish are popular pets for beginners, requiring relatively simple care.",
        metadata={"source": "fish-pets-doc"},
    ),
    Document(
        page_content="Parrots are intelligent birds capable of mimicking human speech.",
        metadata={"source": "bird-pets-doc"},
    ),
    Document(
        page_content="Rabbits are social animals that need plenty of space to hop around.",
        metadata={"source": "mammal-pets-doc"},
    ),
]

vectorstore = Chroma.from_documents(
    documents,
    embedding=OpenAIEmbeddings(),
)

def vector_store_search():
    result = vectorstore.similarity_search("cat")
    print(result)
    
def vector_store_embedding_search():
    embedding = OpenAIEmbeddings().embed_query("cat")
    result = vectorstore.similarity_search_by_vector(embedding)
    print(result)
    
def retriever_search():
    retriever = RunnableLambda(vectorstore.similarity_search).bind(k=1)
    result = retriever.batch(["cat", "shark"])
    print(result)
    
def vector_store_retriever_search():
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 1},
    )
    result = retriever.batch(["cat", "shark"])
    print(result)
    
def rag_chain_query():
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
    
    message = """
    Answer this question using the provided context only.
    
    {question}
    
    Context:
    {context}
    """
    
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 1},
    )
    
    prompt = ChatPromptTemplate.from_messages([("human", message)])
    rag_chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm
    response = rag_chain.invoke("tell me about cats")
    print(response.content)

# vector_store_search()
# vector_store_embedding_search()
# retriever_search()
# vector_store_retriever_search()
rag_chain_query()