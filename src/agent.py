import os
import sys
from dotenv import load_dotenv
import tomllib
import threading

# langchain
from langchain_deepseek import ChatDeepSeek
from langchain_community.retrievers import TavilySearchAPIRetriever

# my modules
from rag_graph import RagGraph
from rag_retriever import RagRetriever
from llm_processor import LLMProcessor


class Agent:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                # Check if the instance is already created
                # If not, create a new instance
                cls._instance = super(Agent, cls).__new__(cls)

        # If the instance is already created, return it
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized") and self._initialized:
            return

        self._initialized = True

        load_dotenv()
        self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        self.linkup_api_key = os.getenv("LINKUP_API_KEY")

        with open("config/config.toml", "rb") as f:
            config = tomllib.load(f)
            if not config:
                raise ValueError("Error: config.toml file is empty")

        self.embed_model = config.get("huggingface", {}).get("embed_model")
        self.reranker_model = config.get("huggingface", {}).get("reranker_model")
        self.files_directory = config.get("vector", {}).get("files_directory")
        self.persist_directory = config.get("vector", {}).get("persist_directory")
        self.collection_name = config.get("vector", {}).get("collection_name")
        self.deepseek_llm_model = config.get("deepseek", {}).get("model")
        self.deepseek_llm_temperature = config.get("deepseek", {}).get(
            "temperature", 0.7
        )
        self.deepseek_llm_max_tokens = config.get("deepseek", {}).get(
            "max_tokens", 2048
        )
        self.web_search_num = config.get("retriever", {}).get("web_search_num", 2)
        self.chat_agent_name = config.get("chat", {}).get("chat_agent_name")
        self._validate_config()

        self.llm = ChatDeepSeek(
            api_key=self.deepseek_api_key,
            model=self.deepseek_llm_model,
            temperature=self.deepseek_llm_temperature,
            max_tokens=self.deepseek_llm_max_tokens,
            timeout=None,
            top_p=0.9,
            frequency_penalty=0.7,
            presence_penalty=0.5,
            max_retries=3,
            streaming=True,
        )

        self.llm_processor = LLMProcessor(llm=self.llm)

        self.rag_retriever = RagRetriever(
            embed_model=self.embed_model,
            reranker_model=self.reranker_model,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name,
        )
        self.rag_retriever.load_index()

        self.web_retriever = TavilySearchAPIRetriever(
            api_key=self.tavily_api_key, k=self.web_search_num
        )

        self.rag_graph = RagGraph(
            llm_processor=self.llm_processor,
            rag_retriever=self.rag_retriever,
            web_retriever=self.web_retriever,
        )

        self.rag_app = self.rag_graph.compile()

    def _validate_config(self):
        if not self.persist_directory:
            raise ValueError("Error: persist_directory is not defined")

        if not self.files_directory:
            raise ValueError("Error: files_directory is not defined")

        if not self.embed_model:
            raise ValueError("Error: model_name is not defined")

    def get_app(self):
        return self.rag_app
