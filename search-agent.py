import os

from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from tavily import TavilyClient

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


@tool
def search(query: str) -> str:
    """Search the web for the query
    Args:
        query: The query to search for
    Returns:
        The search results
    """
    return tavily_client.search(query)


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# tools = [search]
tools = [TavilySearch(max_results=5)]

agent = create_agent(llm, tools)


def main():
    result = agent.invoke(
        {
            "messages": "search for 3 job postings for an AI engineer using langchain in the unique Bay Area    on LinkedIn and list their details"
        }
    )
    print(result)


if __name__ == "__main__":
    main()
