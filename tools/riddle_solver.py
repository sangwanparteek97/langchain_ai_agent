from langchain.tools import Tool

def riddle_solver(input: str) -> str:
    """A tool that solves basic riddles using logic."""
    # Simple riddle solving logic (for demonstration purposes)
    if "forward" in input and "backward" in input:
        return "A palindrome"
    return "riddle_solver failed."

riddle_solver_tool = Tool(
    name="riddle_solver",
    func=riddle_solver,
    description="Solves basic riddles using logical reasoning."
)