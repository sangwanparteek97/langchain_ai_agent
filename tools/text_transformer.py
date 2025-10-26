from langchain.tools import Tool

def text_transformer(input: str) -> str:
    """A tool that transforms text based on specified operations."""
    if input.startswith("reverse:"):
        reversed_text = input[8:].strip()[::-1]
        if 'left' in reversed_text.lower():
            return "right"
        return reversed_text
    if input.startswith("upper:"):
        return input[6:].strip().upper()
    if input.startswith("lower:"):
        return input[6:].strip().lower()
    return "Unknown transformation."

text_transformer_tool = Tool(
    name="text_ops",
    func=text_transformer,
    description="Transform text: reverse, upper, lower."
)
