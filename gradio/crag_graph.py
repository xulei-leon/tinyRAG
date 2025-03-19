# Description: This file contains the RAG graph for the CRAG project.
#
# The RAG graph is a state machine that defines the workflow of the CRAG project.
# The graph is defined by nodes and edges. Each node is a function that takes a state
# and returns a new state. The edges define the transitions between the nodes based on
# the state of the system.
#
# The RAG graph is defined using the langgraph library, which provides a simple way to
# define state machines in Python.
#

################################################################################
# Imports
###############################################################################
import os
import logging
import operator
from typing import List, Literal, Annotated, Optional, Union, Any
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import tomllib

# langchain
from langchain import hub
from langchain.chains import LLMChain
from langchain_deepseek import ChatDeepSeek
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnablePassthrough,
    RunnableSerializable,
)
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_chroma import Chroma
from langchain_text_splitters import TokenTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, UnstructuredHTMLLoader
from langchain.schema import Document
from langchain_community.retrievers import TavilySearchAPIRetriever
from langchain_huggingface import HuggingFaceEmbeddings

from langgraph.graph import START, END, MessagesState, StateGraph
from langgraph.checkpoint.memory import MemorySaver, InMemorySaver


# my modules
from crag_retriever import CragRetriever


################################################################################
# RAG state
################################################################################
class RagState(TypedDict):
    question: str
    answer: str
    retrieves: List[Document]
    retrieves_relevant: List[Document]
    retrieves_weak: List[Document]
    web_searchs: List[Document]
    # documents: Annotated[list, operator.add]


################################################################################
# RAG Graph
################################################################################
class CragGraph:
    def __init__(self, retriever: CragRetriever = None):
        self.graph = None
        self.llm = None
        self.retriever = retriever

        self.__init_config()
        self.__init_llm()

    def compile(self) -> StateGraph:
        if not self.graph:
            self.graph = self.__build_graph()
        return self.graph.compile()

    def invoke(self, state: RagState) -> RagState:
        if not self.graph:
            self.graph = self.__build_graph()
        return self.graph.invoke(state)

    #
    # Internal functions
    #
    def __init_config(self):
        load_dotenv()
        self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
        self.silicon_api_key = os.getenv("SILICON_API_KEY")
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        self.linkup_api_key = os.getenv("LINKUP_API_KEY")

        with open("config.toml", "rb") as f:
            config_data = tomllib.load(f)
            self.deepseek_llm_model = config_data.get("deepseek", {}).get("model")
            self.deepseek_llm_temperature = config_data.get("deepseek", {}).get(
                "temperature"
            )
            self.deepseek_llm_max_tokens = config_data.get("deepseek", {}).get(
                "max_tokens"
            )

    def __init_llm(self):
        self.llm = ChatDeepSeek(
            model=self.deepseek_llm_model,
            temperature=self.deepseek_llm_temperature,
            max_tokens=self.deepseek_llm_max_tokens,
            timeout=None,
            top_p=0.9,
            frequency_penalty=0.7,
            presence_penalty=0.5,
            max_retries=3,
            api_key=self.deepseek_api_key,
        )

    def __build_graph(self) -> StateGraph:
        # Create Graph
        workflow = StateGraph(RagState)

        # Add nodes
        workflow.add_node("rewrite_qestion", self.__node_rewrite_qestion)
        workflow.add_node("retrieve", self.__node_retrieve)
        workflow.add_node("retrieve_grade", self.__node_retrieve_grade)
        workflow.add_node("web_search", self.__node_web_search)
        workflow.add_node("generate", self.__node_generate)

        # Add edges
        workflow.add_edge(START, "rewrite_qestion")
        workflow.add_edge("rewrite_qestion", "retrieve")

        # retrieve
        workflow.add_conditional_edges(
            "retrieve",
            self.__condition_retrieve,
            {
                "success": "retrieve_grade",
                "failure": "web_search",
            },
        )

        workflow.add_conditional_edges(
            "retrieve_grade",
            self.__condition_retrieve_grade,
            {
                "Relevant": "generate",
                "Weak": "web_search",
                "Irrelevant": "web_search",
            },
        )

        # web_search
        workflow.add_edge("web_search", "generate")

        # generate
        workflow.add_edge("generate", END)

        return workflow

    #
    # Nodes functions
    #
    @staticmethod
    def __node_rewrite_qestion(state: RagState) -> RagState:
        question = state["question"]
        better_question = None

        # Re-write question
        # better_question = question_rewriter.invoke({"question": question})
        if not better_question:
            better_question = question

        updated_state = state.copy()
        updated_state.update({"question": better_question})
        return updated_state

    @staticmethod
    def __node_retrieve(state: RagState) -> RagState:
        question = state["question"]
        retrieves = []

        # Retrieval
        # retrieves = retriever.invoke(question)

        updated_state = state.copy()
        updated_state.update({"retrieves": retrieves})
        return updated_state

    @staticmethod
    def __node_retrieve_grade(state: RagState) -> RagState:
        question = state["question"]
        retrieves = state["retrieves"]
        relevant_docs = []
        weak_docs = []

        # Score each doc
        for doc in retrieves:
            # grade = retrieval_grader.invoke(
            #    {"question": question, "document": doc.page_content}
            # )

            if grade:
                score = grade.score
            else:
                score = 0.5

            if score > 0.7:
                relevant_docs.append(doc)
            elif score >= 0.5:
                weak_docs.append(doc)

        updated_state = state.copy()
        updated_state.update(
            {"retrieves_relevant": relevant_docs, "retrieves_weak": weak_docs}
        )
        return updated_state

    @staticmethod
    def __node_web_search(state: RagState) -> RagState:
        question = state["question"]

        # Web search
        web_results = []
        # web_results = web_search_tool.invoke({"query": question})

        updated_state = state.copy()
        updated_state.update({"web_searchs": web_results})
        return updated_state

    @staticmethod
    def __node_generate(state: RagState) -> RagState:
        question = state["question"]
        retrieves_relevant = state.get("retrieves_relevant", None)
        retrieves_weak = state.get("retrieves_weak", None)
        web_searchs = state.get("web_searchs", None)

        # generation
        generation = "This is the answer!"

        updated_state = state.copy()
        updated_state.update({"answer": generation})
        return updated_state

    #
    # Edges conditional functions
    #
    @staticmethod
    def __condition_retrieve(state: RagState) -> str:
        retrieves = state["retrieves"]

        if retrieves:
            return "success"
        else:
            return "failure"

    @staticmethod
    def __condition_retrieve_grade(state: RagState) -> str:
        retrieves_relevant = state["retrieves_relevant"] or None
        retrieves_weak = state["retrieves_weak"] or None

        if retrieves_relevant:
            return "Relevant"
        elif retrieves_weak:
            return "Weak"
        else:
            return "Irrelevant"
