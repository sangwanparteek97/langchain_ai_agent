import os
import requests
from langchain.tools import Tool

def gemini_video_qa(video_url: str, user_query: str) -> str:
    """Analyze video content and answer questions using Gemini."""
    model_name = "gemini-1.5-flash"

    req = {
        "model": f"models/{model_name}",
        "contents": [{
            "parts": [
                {"fileData": {"fileUri": video_url}},
                {"text": f"Please watch the video and answer the question: {user_query}"}
            ]
        }]
    }

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model_name}:generateContent?key={os.getenv('GOOGLE_API_KEY')}"
    )

    try:
        res = requests.post(url, json=req, headers={"Content-Type": "application/json"})
        if res.status_code != 200:
            return f"Video error {res.status_code}: {res.text}"

        data = res.json()
        parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        return "".join([p.get("text", "") for p in parts]).strip()

    except Exception as e:
        return f"[ERROR] GeminiVideoQATool failed: {str(e)}"


gemini_video_tool = Tool(
    name="video_inspector",
    description="Analyze video content to answer questions using Gemini. Inputs: video_url, user_query.",
    func=lambda x: gemini_video_qa(**x)
)