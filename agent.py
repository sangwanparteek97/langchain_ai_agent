import os
import pandas as pd
from rich.table import Table
from rich.console import Console
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from config import get_llm
from prompt_template import gaia_prompt

from tools.file_attachment_query import file_attachment_query_tool
from tools.math_solver import math_solver_tool
from tools.google_search import google_search_tool
from tools.gemini_video_qa import gemini_video_qa_tool
from tools.riddle_solver import riddle_solver_tool
from tools.text_transformer import text_transformer_tool
from tools.wiki_content_fetcher import wiki_content_fetcher_tool
from tools.wiki_title_finder import wiki_title_finder_tool


class LangChainGAIAAgent:
    def __init__(self, provider="deepseek"):
        print("LangChain GAIA Agent initialized.")

        # Select model (config.py handles provider switching)
        if provider == "huggingface":
            llm = ChatHuggingFace(
                llm = HuggingFaceEndpoint(
                    url="https://api-inference.huggingface.co/models/Meta-DeepLearning/llama-2-7b-chat-hf",
                    temperature=0
                )
            )
        else:
            self.llm = get_llm(provider)

        # Register all tools
        self.tools = [
            file_attachment_query_tool,
            math_solver_tool,
            google_search_tool,
            gemini_video_qa_tool,
            riddle_solver_tool,
            text_transformer_tool,
            wiki_content_fetcher_tool,
            wiki_title_finder_tool,
        ]

        # Combines rules with LangChain tool orchestration
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", gaia_prompt.template),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

        # Optional memory (multi-turn conversations)
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # Create tool-calling agent directly
        self.agent = create_tool_calling_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt
        )

        # Wrap in AgentExecutor (LangChain runtime)
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True
        )

        print("GAIA Agent ready with all tools and system rules.\n")

    def __call__(self, question: str) -> str:
        """
        Call the agent like a function.
        """
        print(f"Received question (first 50 chars): {question[:50]}...")
        try:
            response = self.agent_executor.invoke({"input": question})
            result = response.get("output", "").strip()
            return result
        except Exception as e:
            return f"[ERROR] {str(e)}"

    def evaluate_random_questions(self, csv_path: str, sample_size: int = 3, show_steps: bool = True):
        """
        Evaluate GAIA benchmark questions from CSV.
        CSV must contain: 'question', 'answer', (optional) 'taskid'
        """
        df = pd.read_csv(csv_path)
        if not {"question", "answer"}.issubset(df.columns):
            print("CSV must contain 'question' and 'answer' columns.")
            print("Found columns:", df.columns.tolist())
            return

        samples = df.sample(n=sample_size)
        records = []
        correct_count = 0

        for _, row in samples.iterrows():
            taskid = str(row.get("taskid", "")).strip()
            question = row["question"].strip()
            expected = str(row["answer"]).strip()

            query = f"taskid: {taskid}, question: {question}" if taskid else question
            agent_answer = self(query).strip()

            is_correct = (expected == agent_answer)
            correct_count += is_correct
            records.append((question, expected, agent_answer, "✓" if is_correct else "✗"))

            if show_steps:
                print("---")
                print(f"Question: {question}")
                print(f"Expected: {expected}")
                print(f"Agent: {agent_answer}")
                print(f"Correct: {is_correct}")

        # Pretty print summary
        console = Console()
        table = Table(show_lines=True)
        table.add_column("Question", overflow="fold")
        table.add_column("Expected")
        table.add_column("Agent")
        table.add_column("Correct")

        for question, expected, agent_ans, correct in records:
            table.add_row(question, expected, agent_ans, correct)

        console.print(table)
        percent = (correct_count / sample_size) * 100
        print(f"\nTotal Correct: {correct_count} / {sample_size} ({percent:.2f}%)")
