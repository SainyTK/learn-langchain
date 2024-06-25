from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from fastapi import FastAPI
from langserve import add_routes
import uvicorn


load_dotenv(override=True)
model = ChatOpenAI(model="gpt-4o")
parser = StrOutputParser()


prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "Translate the following into {language}:"),
        ("user", "{text}")
    ]
)

chain = prompt_template | model | parser

app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description="A simple API server using LangChain's Runnable interfaces"
)

add_routes(
    app,
    chain,
    path="/chain"
)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)