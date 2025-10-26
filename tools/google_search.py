from langchain.tools import Tool
import os
import requests

def google_search(input: str) -> str:
    """A tool that simulates a Google search and returns top results."""
    
    try:
        response = requests.get(
            "https://www.googleapis.com/customsearch/v1",
            params={
                "q": input,
                "key": os.getenv("GOOGLE_API_KEY"),
                "cx": os.getenv("GOOGLE_SEARCH_ENGINE_ID"),
                "num": 1
            }
        )
        data = response.json()
        # Extract and return the top search result summary
        return data.get("items", [])[0].get("snippet", "No results found.")
    except Exception as e:
        return f"Google search error: {str(e)}"

google_search_tool = Tool(
    name="google_search",
    func=google_search,
    description="Search the web using Google and return the top summary from the results."
)
