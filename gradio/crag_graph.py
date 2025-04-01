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
################################################################################
import os
import logging
import operator
from typing import List, Literal, Annotated, Optional, Union, Any
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import tomllib

# import langchain
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
from rag_retriever import RagRetriever
from llm_processor import LLMProcessor

# TODO list
# Parallel processing of multiple retrieve


#
# RAG state
#
class RagState(TypedDict):
    question: str
    answer: str
    rag_retrieves: List[Document]
    web_retrieves: List[Document]
    # documents: Annotated[list, operator.add]
    completed: Annotated[list, operator.add]


#
# RAG Graph
#
class CragGraph:

    def __init__(
        self,
        rag_retriever: RagRetriever,
        web_retriever: TavilySearchAPIRetriever,
        llm_processor: LLMProcessor,
    ):
        load_dotenv()
        with open("config.toml", "rb") as f:
            config_data = tomllib.load(f)
            self.search_result_num = config_data.get("retriever", {}).get(
                "search_result_num", 3
            )
            self.score_relevant = config_data.get("retriever", {}).get(
                "rerank_score_relevant", 0.55
            )

        self.llm_processor = llm_processor
        self.rag_retriever = rag_retriever
        self.web_retriever = web_retriever
        self.graph = None
        self.logger = logging.getLogger(__name__)

    def compile(self) -> StateGraph:
        if not self.graph:
            self.graph = self.__build_graph()
        return self.graph.compile()

    def invoke(self, state: RagState) -> RagState:
        if not self.graph:
            self.graph = self.__build_graph()
        return self.graph.invoke(state)

    ############################################################################
    ## Internal functions
    ############################################################################
    def __build_graph(self) -> StateGraph:
        # Create Graph
        graph = StateGraph(RagState)

        # Add nodes
        graph.add_node("rewrite_qestion", self.__node_rewrite_question)
        graph.add_node("rag_retrieve", self.__node_rag_retrieve)
        graph.add_node("rag_retrieve_grade", self.__node_rag_retrieve_grade)
        graph.add_node("rag_retrieve_finish", self.__node_rag_retrieve_finish)
        graph.add_node("web_retrieve", self.__node_web_retrieve)
        graph.add_node("generate_answer", self.__node_generate_answer)

        # Add edges
        graph.add_edge(START, "rewrite_qestion")

        # RAG retrieve
        graph.add_edge("rewrite_qestion", "rag_retrieve")
        graph.add_conditional_edges(
            "rag_retrieve",
            self.__condition_retrieve,
            {
                "success": "rag_retrieve_grade",
                "failure": "rag_retrieve_finish",
            },
        )
        graph.add_edge("rag_retrieve_grade", "rag_retrieve_finish")
        graph.add_conditional_edges(
            "rag_retrieve_finish",
            self.__condition_complete,
            {
                "success": "generate_answer",
                "failure": END,
            },
        )
        # web retrieve
        graph.add_edge("rewrite_qestion", "web_retrieve")
        graph.add_conditional_edges(
            "web_retrieve",
            self.__condition_complete,
            {
                "success": "generate_answer",
                "failure": END,
            },
        )

        # generate answer
        graph.add_edge("generate_answer", END)

        return graph

    ############################################################################
    ## Nodes functions
    ############################################################################
    def __node_rewrite_question(self, state: RagState) -> RagState:
        question = state["question"]
        print(f"[rewrite_question] question: {question}")

        # Re-write question
        rewrite_question = self.llm_processor.rewrite_question(question=question)
        if not rewrite_question:
            rewrite_question = question

        print(f"[rewrite_question] rewite question: {rewrite_question}")
        return {"question": rewrite_question}

    def __node_rag_retrieve(self, state: RagState) -> RagState:
        question = state["question"]

        # Retrieval
        # rag_retrieves = self.rag_retriever.query_rerank(question)
        rag_retrieves = self.rag_retriever.query(question)
        print(f"[rag_retrieve] rag retrieve number: {len(rag_retrieves)}")

        return {"rag_retrieves": rag_retrieves}

    def __node_rag_retrieve_grade(self, state: RagState) -> RagState:
        question = state["question"]
        rag_retrieves = state["rag_retrieves"]
        content_size_min = 50

        # Filter relevant retrieves
        relevants_with_score = []
        for doc in rag_retrieves:
            print(f"=== RAG retrieve score === ")
            print(doc.page_content[:200])
            score = self.rag_retriever.query_score(
                question=question, context=doc.page_content
            )

            if len(doc.page_content) < content_size_min:
                score = score * 0.8

            if score < self.score_relevant:
                print(f"[rag_retrieve_grade] relevant score: {score} is low")
                continue

            print(f"[rag_retrieve_grade] relevant score: {score}")
            relevants_with_score.append((doc, score))

        # sort relevant retrieves by score in descending order
        if len(relevants_with_score) > 0:
            relevants_with_score.sort(key=lambda doc: doc[1], reverse=True)
            rag_retrieves = [
                doc[0] for doc in relevants_with_score[: self.search_result_num]
            ]

        return {"rag_retrieves": rag_retrieves}

    def __node_rag_retrieve_finish(self, state: RagState) -> RagState:
        new_completed = state["completed"].copy()
        new_completed.append("rag")
        return {"completed": new_completed}

    def __node_web_retrieve(self, state: RagState) -> RagState:
        question = state["question"]
        new_completed = state["completed"].copy()

        # Web search
        web_retrieves = self.web_retriever.invoke(question)
        for doc in web_retrieves:
            print("=== web retrieve === ")
            print(doc.page_content[:200])

        new_completed.append("web")
        return {"web_retrieves": web_retrieves, "completed": new_completed}

    def __node_generate_answer(self, state: RagState) -> RagState:
        question = state["question"]
        rag_retrieves = state.get("rag_retrieves", None)
        web_retrieves = state.get("web_retrieves", None)
        web_retrieves = web_retrieves[:1]  # now only use one web content

        rag_context = ""
        rag_num = 0
        if rag_retrieves:
            rag_num = len(rag_retrieves)
            rag_context += "\n".join(doc.page_content for doc in rag_retrieves)

        web_context = ""
        web_num = 0
        if web_retrieves:
            web_num = len(web_retrieves)
            web_context += "\n".join([doc.page_content for doc in web_retrieves])

        # Generation
        generation = self.llm_processor.generate_answer(
            question=question,
            rag_context=rag_context,
            web_context=web_context,
        )

        print(f"[generate_answer] question: {question}")
        print(f"[generate_answer] {rag_num} rag_retrieves")
        print(f"[generate_answer] {web_num} web_retrieves")

        print(f"[generate_answer] answer: {generation}")

        return {"answer": generation}

    ############################################################################
    ## Edges conditional functions
    ############################################################################
    def __condition_retrieve(self, state: RagState) -> str:
        if len(state["rag_retrieves"]) > 0:
            return "success"
        else:
            print("[condition_retrieve]: failure")
            return "failure"

    def __condition_complete(self, state: RagState) -> str:
        if {"rag", "web"}.issubset(state["completed"]):
            print("[condition_complete]: success")
            return "success"
        else:
            print("[condition_complete]: failure")
            return "failure"
