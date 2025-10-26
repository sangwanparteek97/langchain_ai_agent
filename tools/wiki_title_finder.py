from langchain.tools import Tool
import wikipedia as wiki

def wiki_title_finder(input: str) -> str:
    """A tool that finds Wikipedia article titles based on a query."""
    
    results = wiki.search(input)
    return ", ".join(results) if results else "No matching Wikipedia article found."

wiki_title_finder_tool = Tool(
    name="wiki_title_finder",
    func=wiki_title_finder,
    description="Find related Wikipedia article page titles based on a query."
)
