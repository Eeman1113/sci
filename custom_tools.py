from crewai_tools import BaseTool
from duckduckgo_search import DDGS
from typing import List, Dict, Any, Type
from langchain_core.pydantic_v1 import BaseModel, Field
import requests
from bs4 import BeautifulSoup

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

class WebPageContentFetcherToolInput(BaseModel):
    url: str = Field(description="The URL of the web page to fetch content from.")

class WebPageContentFetcherTool(BaseTool):
    name: str = "Web Page Content Fetcher"
    description: str = (
        "Fetches and extracts the main textual content from a given web page URL. "
        "Input must be a valid URL string."
    )
    args_schema: Type[BaseModel] = WebPageContentFetcherToolInput

    def _run(self, url: str) -> str:
        """
        Fetches web page content using requests and parses it with BeautifulSoup
        to extract meaningful text.
        """
        if not url or not url.startswith(('http://', 'https://')):
            return "Error: Invalid URL provided. Must start with http:// or https://."

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        try:
            response = requests.get(url, headers=headers, timeout=15) # Increased timeout

            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')

                # Remove script and style elements
                for script_or_style in soup(["script", "style"]):
                    script_or_style.decompose()

                text_parts = []

                # Try common main content tags
                # Order of preference for main content tags
                main_content_selectors = [
                    "article", "main",
                    "div[class*='content']", "div[id*='content']",
                    "div[class*='main']", "div[id*='main']",
                    "div[role='main']"
                ]

                found_main_content = False
                for selector in main_content_selectors:
                    main_element = soup.select_one(selector)
                    if main_element:
                        text_parts.append(main_element.get_text(separator=' ', strip=True))
                        found_main_content = True
                        break # Found preferred main content

                if not found_main_content:
                    # Fallback: try to get specific text blocks like paragraphs and headings
                    # from the body if no main content container is identified.
                    body_text = soup.body
                    if body_text:
                        paragraphs = body_text.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'li'])
                        if paragraphs:
                             for p_or_h in paragraphs:
                                text_parts.append(p_or_h.get_text(separator=' ', strip=True))
                        else: # If no p/h tags, get all text from body (less ideal)
                            text_parts.append(body_text.get_text(separator=' ', strip=True))
                    else: # If no body tag, something is very wrong
                        return "Error: Could not find body content in the page."

                full_text = " ".join(text_parts)

                # Clean up excessive whitespace that might remain
                full_text = ' '.join(full_text.split())

                if not full_text.strip():
                    return f"Error: No meaningful text content found on the page {url} after parsing."

                # Return a substantial portion (e.g., up to 8000 characters)
                max_length = 8000
                return full_text[:max_length] if len(full_text) > max_length else full_text

            else:
                return f"Error: Failed to fetch URL {url}. Status code: {response.status_code}."

        except requests.exceptions.Timeout:
            return f"Error: Request timed out while trying to fetch URL {url}."
        except requests.exceptions.RequestException as e:
            return f"Error: An exception occurred while fetching URL {url}: {str(e)}."
        except Exception as e:
            return f"Error: An unexpected error occurred while processing URL {url}: {str(e)}."

# Instantiate tools
search_tool = DuckDuckGoSearchTool()
web_fetcher_tool = WebPageContentFetcherTool()

# List of tools to be easily imported
all_tools = [search_tool, web_fetcher_tool]

if __name__ == '__main__':
    # Example usage for DuckDuckGoSearchTool:
    # search_results = search_tool.run(query="latest advancements in AI", max_results=2)
    # if "error" in search_results[0]:
    #     print(search_results[0]["error"])
    # else:
    #     for res in search_results:
    #         print(f"Title: {res['title']}\nLink: {res['href']}\nSnippet: {res['snippet']}\n---")

    # Example usage for WebPageContentFetcherTool:
    # Test with a known working URL
    # test_url = "https://www.wired.com/story/what-is-generative-ai/" # Choose a real article URL
    # print(f"\nFetching content from: {test_url}")
    # fetched_content = web_fetcher_tool.run(url=test_url)
    # print(f"Fetched Content (first 500 chars):\n{fetched_content[:500]}...")
    # print(f"\nLength of fetched content: {len(fetched_content)}")

    # Test with a non-existent or problematic URL
    # test_error_url = "http://thisurldoesnotexist12345.com"
    # print(f"\nFetching content from (error test): {test_error_url}")
    # error_content = web_fetcher_tool.run(url=test_error_url)
    # print(f"Error Content:\n{error_content}")

    # Test with an invalid URL format
    # invalid_url = "not_a_url"
    # print(f"\nFetching content from (invalid format test): {invalid_url}")
    # invalid_content = web_fetcher_tool.run(url=invalid_url)
    # print(f"Invalid URL Content:\n{invalid_content}")
    pass

[end of custom_tools.py]
