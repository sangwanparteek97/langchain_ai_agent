from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEndpoint
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models import ChatAnthropic
from langchain_community.chat_models import ChatGrok
from langchain_community.chat_models import ChatDeepSeek

# Define supported providers
AVAILABLE_MODELS = {
    "openai": {
        "model": "gpt-4o-mini",
        "client": ChatOpenAI,
        "params": {"temperature": 0},
    },
    "huggingface": {
        "model": "Qwen/Qwen2.5-Coder-32B-Instruct",
        "client": HuggingFaceEndpoint,
        "params": {"temperature": 0},
    },
    "gemini": {
        "model": "gemini-2.0-flash",
        "client": ChatGoogleGenerativeAI,
        "params": {"temperature": 0},
    },
    "grok": {
        "model": "qwen-qwq-32b",
        "client": ChatGrok,
        "params": {"temperature": 0},
    },
    "deepseek": {
        "model": "deepseek-coder",
        "client": ChatDeepSeek,
        "params": {"temperature": 0},
    },
}

# Choose provider dynamically here
PROVIDER = "huggingface"  # Change this to "huggingface", "gemini", "grok", or "deepseek"

def get_llm(PROVIDER=PROVIDER):
    config = AVAILABLE_MODELS[PROVIDER]
    model_class = config["client"]
    return model_class(model=config["model"], **config["params"])
