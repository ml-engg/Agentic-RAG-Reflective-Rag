from langchain_community.tools.tavily_search import TavilySearchResults
import define_os_env



def web_search(query: str, k: int):
    """
    Run a web search using Tavily and return top-k results.
    Each result contains title, URL, and snippet.
    """
    search_tool = TavilySearchResults(k=k)
    results = search_tool.invoke({"query": query})

    return results

if __name__ == "__main__":
    query = "what is langchain ?"
    output = web_search(query, k=3)
    print(output)