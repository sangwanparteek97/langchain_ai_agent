from langchain.tools import Tool

def math_solver(input: str) -> str:
    """A tool that safely evaluates basic math expressions."""
    try:
        # Evaluate the math expression safely
        return str(eval(input, {"__builtins__": {}}))
    except Exception as e:
        return f"Math error: {e}"

math_solver_tool = Tool(
    name="math_solver",
    func=math_solver,
    description="Safely evaluates the basic math expressions."
)