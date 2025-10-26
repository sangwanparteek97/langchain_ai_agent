from langchain.tools import Tool
import wikipedia as wiki

def wiki_content_fetcher(input: str) -> str:
    """A tool that fetches Wikipedia article content based on a title."""

    try:
        page = wiki.page(input).html()
        return to_markdown(page)
    except wiki.exceptions.PageError:
        return f"Wikipedia page '{input}' not found."

wiki_content_fetcher_tool = Tool(
    name="wiki_page",
    func=wiki_content_fetcher,
    description="Fetch Wikipedia page content based on a title."
)
