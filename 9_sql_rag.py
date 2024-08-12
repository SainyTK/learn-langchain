# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
# https://python.langchain.com/v0.2/docs/tutorials/sql_qa/
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import create_sql_query_chain
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_community.agent_toolkits import SQLDatabaseToolkit

# %%
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

# %%
import ast
import re

# %%
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# %%
load_dotenv(override=True)

# %%
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
db = SQLDatabase.from_uri("sqlite:///data/Chinook.db")

# %%
# Note: This doesn't work as the AI always produces incorrect syntax SQL
def direct_connect():
    chain = create_sql_query_chain(llm, db)
    query = chain.invoke({"question": "How many employees are there"})
    print(query)
    
    formatted_query = query.replace("```", "").replace("sql", "").replace("SQL:", "").replace("SQLQuery:", "")

    result = db.run(formatted_query)
    print(db.run(result)) 

# %%
# Note: This doesn't work as the AI always produces incorrect syntax SQL
def sql_chain_connect():
    execute_query = QuerySQLDataBaseTool(db=db)
    write_query = create_sql_query_chain(llm, db)
    chain = write_query | execute_query
    response = chain.invoke({"question": "How many employees are there."})
    print(response)

# %%
# Note: This works well. There is a tool to check to correct SQL syntax (by AI itself, though)
def agent_db_connect():
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()
    
    SQL_PREFIX = """You are an agent designed to interact with a SQL database.
        Given an input question, create a syntactically correct SQLite query to run, then look at the results of the query and return the answer.
        Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 5 results.
        You can order the results by a relevant column to return the most interesting examples in the database.
        Never query for all the columns from a specific table, only ask for the relevant columns given the question.
        You have access to tools for interacting with the database.
        Only use the below tools. Only use the information returned by the below tools to construct your final answer.
        You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

        DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

        To start you should ALWAYS look at the tables in the database to see what you can query.
        Do NOT skip this step.
        Then you should query the schema of the most relevant tables."""

    system_message = SystemMessage(content=SQL_PREFIX)
    
    agent_executor = create_react_agent(llm, tools, messages_modifier=system_message)
    
    for s in agent_executor.stream(
        {"messages": [HumanMessage(content="Which country's customers spent the most?")]}
    ):
        print(s)
        print("----")

# %%
def query_as_list(db, query):
    res = db.run(query)
    res = [el for sub in ast.literal_eval(res) for el in sub if el]
    res = [re.sub(r"\b\d+\b", "", string).strip() for string in res]
    return list(set(res))

# %%
def dealing_with_proper_nouns():
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()
    
    artists = query_as_list(db, "SELECT Name FROM Artist")
    albums = query_as_list(db, "SELECT Title FROM Album")
    vector_db = FAISS.from_texts(artists + albums, OpenAIEmbeddings())
    retriever = vector_db.as_retriever(search_kwargs={"k": 5})
    description = """Use to look up values to filter on. Input is an approximate spelling of the proper noun, output is \
        valid proper nouns. Use the noun most similar to the search."""
    retriever_tool = create_retriever_tool(
        retriever,
        name="search_proper_nouns",
        description=description,
    )
    
    tools.append(retriever_tool)
    
    system = """You are an agent designed to interact with a SQL database.
        Given an input question, create a syntactically correct SQLite query to run, then look at the results of the query and return the answer.
        Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 5 results.
        You can order the results by a relevant column to return the most interesting examples in the database.
        Never query for all the columns from a specific table, only ask for the relevant columns given the question.
        You have access to tools for interacting with the database.
        Only use the given tools. Only use the information returned by the tools to construct your final answer.
        You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

        DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

        You have access to the following tables: {table_names}

        If you need to filter on a proper noun, you must ALWAYS first look up the filter value using the "search_proper_nouns" tool!
        Do not try to guess at the proper name - use this function to find similar ones.""".format(
            table_names=db.get_usable_table_names()
        )
    system_message = SystemMessage(content=system)
    agent = create_react_agent(llm, tools, messages_modifier=system_message)
    for s in agent.stream(
        {"messages": [HumanMessage(content="Which country's customers spent the most?")]}
    ):
        print(s)
        print("----")

# %%
# direct_connect()
# sql_chain_connect()
# agent_db_connect()
dealing_with_proper_nouns()
