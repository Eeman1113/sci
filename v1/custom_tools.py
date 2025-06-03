from crewai.tools import BaseTool
from duckduckgo_search import DDGS
from typing import List, Dict, Any, Type
from langchain_core.pydantic_v1 import BaseModel, Field

# Input schema for the search tool
class SearchToolInput(BaseModel):
    query: str = Field(description="The search query to be executed.")
    max_results: int = Field(default=3, description="Maximum number of search results to return.")

class DuckDuckGoSearchTool(BaseTool):
    name: str = "DuckDuckGo Search"
    description: str = (
        "A tool for performing web searches using DuckDuckGo. "
        "Useful for finding information, articles, and data on various topics. "
        "Input should be a search query string."
    )
    args_schema: Type[BaseModel] = SearchToolInput

    def _run(self, query: str, max_results: int = 3) -> List[Dict[str, str]]:
        """
        Executes a DuckDuckGo search and returns a list of results.
        Each result is a dictionary with 'title', 'href', and 'body' (snippet).
        """
        results_list = []
        try:
            with DDGS() as ddgs:
                search_results = ddgs.text(query, max_results=max_results)
                if search_results:
                    for r in search_results:
                        results_list.append({
                            "title": r.get('title', 'N/A'),
                            "href": r.get('href', 'N/A'),
                            "snippet": r.get('body', 'N/A')
                        })
                return results_list
        except Exception as e:
            return [{"error": f"DuckDuckGo search failed: {str(e)}"}]

# You could add more tools here, e.g., a tool to read a URL's content,
# a file writing tool, etc.
# For now, we'll keep it simple with the search tool.
# LangChain community tools can also be wrapped or used directly by agents.

# Example of a placeholder tool for fetching content from a URL
class WebPageContentFetcherTool(BaseTool):
    name: str = "Web Page Content Fetcher"
    description: str = (
        "Fetches the main textual content from a given URL. "
        "Input should be a valid URL string."
    )
    # args_schema: Type[BaseModel] = # Define input schema if needed, e.g., {"url": "string"}

    def _run(self, url: str) -> str:
        """
        Placeholder for fetching web page content.
        In a real implementation, use libraries like BeautifulSoup, requests, or LangChain's WebBaseLoader.
        """
        # This is a simplified placeholder.
        # A robust implementation would handle various errors, content types, etc.
        if not url or not url.startswith(('http://', 'https://')):
            return "Error: Invalid URL provided."
        try:
            # Simulating content fetching
            # from langchain_community.document_loaders import WebBaseLoader
            # loader = WebBaseLoader(url)
            # docs = loader.load()
            # if docs:
            #     return docs[0].page_content[:5000] # Limit content length
            # return "Error: Could not fetch content."
            return f"Successfully fetched content for {url} (simulated - first 200 chars). This is a placeholder. Implement actual fetching."
        except Exception as e:
            return f"Error fetching content from {url}: {str(e)}"

# Instantiate tools
search_tool = DuckDuckGoSearchTool()
web_fetcher_tool = WebPageContentFetcherTool()

# List of tools to be easily imported
all_tools = [search_tool, web_fetcher_tool]

if __name__ == '__main__':
    # Example usage:
    # tool = DuckDuckGoSearchTool()
    # results = tool.run(query="latest advancements in AI", max_results=2)
    # for res in results:
    #     print(f"Title: {res['title']}\nLink: {res['href']}\nSnippet: {res['snippet']}\n---")

    # fetch_results = web_fetcher_tool.run(url="https://www.example.com")
    # print(fetch_results)
    pass
