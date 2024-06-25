from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, trim_messages
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_message_histories import ChatMessageHistory

from operator import itemgetter

load_dotenv(override=True)

# Persistent database to store mapping of session id and conversation
store = {}
# Configuration to select chat session
config = {"configurable": {"session_id": "abc2"}}

model = ChatOpenAI(model="gpt-3.5-turbo")
prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
        "You are a helpful assistant. Answer all questions to the best of your ability in {language}."),
        MessagesPlaceholder(variable_name="messages")
    ]
)

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def conversation_chain():
    chain = prompt | model

    with_message_history = RunnableWithMessageHistory(chain, get_session_history, input_messages_key="messages")

    messages = ["Hi! I'm Bob", "What's my name?"]

    for message in messages:
        response = with_message_history.invoke(
            {"messages": [HumanMessage(content=message)], "language": "Spanish"},
            config=config
        )

        print(response.content)

def trimmer_chain():
    trimmer = trim_messages(
        max_tokens=65,
        strategy="last",
        token_counter=model,
        include_system=True,
        allow_partial=False,
        start_on="human"
    )
    
    messages = [
        SystemMessage(content="you're a good assistant."),
        HumanMessage(content="hi! I'm Bob"),
        AIMessage(content="hi!"),
        HumanMessage(content="I like vanilla ice creame"),
        AIMessage(content="nice"),
        HumanMessage(content="whats 2 + 2"),
        AIMessage(content="4"),
        HumanMessage(content="thanks"),
        AIMessage(content="no problem!"),
        HumanMessage(content="having fun?"),
        AIMessage(content="yes!")
    ]
    
    chain = (
        RunnablePassthrough.assign(messages=itemgetter("messages") | trimmer)
        | prompt
        | model
    )
    
    with_message_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="messages"
    )
    
    # Stream output instead of invoke
    for result in with_message_history.stream(
        {
            "messages": messages + [HumanMessage(content="What's my name?")],
            "language": "English"
        },
        config=config
    ):
        print(result.content, end="|")
        
# conversation_chain()
trimmer_chain()