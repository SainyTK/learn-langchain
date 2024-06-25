from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv(override=True)
model = ChatOpenAI(model="gpt-4o")
parser = StrOutputParser()

def simple_chain(): 
    messages = [
        SystemMessage(content="Translate the following from English into Italian"),
        HumanMessage(content="hi!")
    ]
    
    # This is how we chain model and parser together
    chain = model | parser
    result = chain.invoke(messages)
    print(result)

def prompt_chain():
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "Translate the following into {language}:"),
            ("user", "{text}")
        ]
    )
    
    chain = prompt_template | model | parser
    result = chain.invoke({"language": "italian", "text": "hi"})
    print(result)


# simple_chain()
prompt_chain()