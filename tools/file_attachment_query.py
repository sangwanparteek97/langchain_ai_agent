from langchain.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI
import requests
import os

def file_attachment_query(task_id: str, query: str) -> str:
    """A tool that processes file attachment queries."""

    file_url = f"https://agents-course-unit4-scoring.hf.space/files/{task_id}"
    file_response = requests.get(file_url)
    if file_response.status_code != 200:
        return f"Error downloading file with task_id {task_id}: {file_response.status_code} - {file_response.text}"
    
    file_data = file_response.content
    # TODO: Change the model selection dynamic.
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.0,
        api_key=os.getenv("GOOGLE_API_KEY"))
    
    messages = [
        SystemMessage(content="You are a helpful file analysis assistant."),
        HumanMessage(
            content=[
                {"type": "text", "text": f"Analyze this file and answer: {user_query}"},
                {"type": "file", "data": file_data, "mime_type": "application/octet-stream"}
            ]
        )
    ]
    response = llm.invoke(messages)
    return getattr(response, "text", str(response))

file_attachment_query_tool = Tool(
    name="run_query_on_file_attachment",
    func=file_attachment_query,
    description="Downloads file attached in the user prompt, adds it to the context, and runs the query on it.",
    input_schema={
        "task_id": {
            "type": "string",
            "description": "The unique identifier for the task associated with the file attachment, used to download the correct file.",
            "nullable": True
        },
        "query": {
            "type": "string",
            "description": "The query to be executed on the file attachment content."
        }
    },
    output_schema={
        "type": "string",
        "description": "The result of the query executed on the file attachment content."
    }
)
