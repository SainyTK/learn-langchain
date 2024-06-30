from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
import asyncio
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv(override=True)

model = ChatOpenAI(model="gpt-4o")
search = TavilySearchResults(max_results=2)
memory = SqliteSaver.from_conn_string(":memory:")

def invoke_search():
    search_results = search.invoke("what is the weather in SF")
    print(search_results)

def use_tools_with_llm(query: str):
    tools = [search]
    model_with_tools = model.bind_tools(tools)
    response = model_with_tools.invoke([HumanMessage(content=query)])
    print(f"ContentString: {response.content}")
    print(f"ToolCalls: {response.tool_calls}")
   
def use_agent(query: str):
    tools = [search]
    agent_executor = create_react_agent(model, tools) 
    response = agent_executor.invoke({"messages": [HumanMessage(content=query)]})
    print(response["messages"])

def use_agent_stream(query: str):
    tools = [search]
    agent_executor = create_react_agent(model, tools) 
    for chunk in agent_executor.stream({"messages": [HumanMessage(content=query)]}):
        print(chunk)
        print("----")
    
async def use_agent_stream_events(query: str):
    tools = [search]
    agent_executor = create_react_agent(model, tools) 
    async for event in agent_executor.astream_events(
        {"messages": [HumanMessage(content=query)]},
        version="v1"
    ):
        kind = event["event"]
        if kind == "on_chain_start":
            if (
                event["name"] == "Agent"
            ):  # Was assigned when creating the agent with `.with_config({"run_name": "Agent"})`
                print(
                    f"Starting agent: {event['name']} with input: {event['data'].get('input')}"
                )
        elif kind == "on_chain_end":
            if (
                event["name"] == "Agent"
            ):  # Was assigned when creating the agent with `.with_config({"run_name": "Agent"})`
                print()
                print("--")
                print(
                    f"Done agent: {event['name']} with output: {event['data'].get('output')['output']}"
                )
        if kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                # Empty content in the context of OpenAI means
                # that the model is asking for a tool to be invoked.
                # So we only print non-empty content
                print(content, end="|")
        elif kind == "on_tool_start":
            print("--")
            print(
                f"Starting tool: {event['name']} with inputs: {event['data'].get('input')}"
            )
        elif kind == "on_tool_end":
            print(f"Done tool: {event['name']}")
            print(f"Tool output was: {event['data'].get('output')}")
            print("--")

def agent_with_memory(query: str):
    tools = [search]
    agent_executor = create_react_agent(model, tools, checkpointer=memory)
    
    config = {"configurable": {"thread_id": "abc123"}}
    
    for chunk in agent_executor.stream(
        {"messages": [HumanMessage(content=query)]}, config
    ):
        print(chunk)
        print("----")

# use_tools_with_llm("Hi!") # LLM determines to not use tools
# use_tools_with_llm("What's the weather in SF?") # LLM determines to search weather info in SF
# use_agent("hi!") # Invoke llm with agent chain by not using tools
# use_agent("Whats the weather in sf?") # Invoke llm with agent chain and use tools
# use_agent_stream("Whats the weather in sf?")
# asyncio.run(use_agent_stream_events("Whats the weather in sf?"))

agent_with_memory("hi im bob!") 
agent_with_memory("whats my name?")
agent_with_memory("please search how much my name is used in America?")