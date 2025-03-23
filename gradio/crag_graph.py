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
    rag_retrieves_relevant: List[Document]
    rag_retrieves_weak: List[Document]
    web_retrieves: List[Document]
    # documents: Annotated[list, operator.add]


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
        self.llm_processor = llm_processor
        self.rag_retriever = rag_retriever
        self.web_retriever = web_retriever
        self.graph = None

        self.search_type = "score_threshold"
        self.score_relevant = 0.7
        self.score_weak = 0.5
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
        graph.add_node("web_retrieve", self.__node_web_retrieve)
        graph.add_node("generate_answer", self.__node_generate_answer)

        # Add edges
        graph.add_edge(START, "rewrite_qestion")
        graph.add_edge("rewrite_qestion", "rag_retrieve")

        # RAG retrieve
        graph.add_conditional_edges(
            "rag_retrieve",
            self.__condition_retrieve,
            {
                "success": "rag_retrieve_grade",
                "failure": "web_retrieve",
            },
        )

        graph.add_conditional_edges(
            "rag_retrieve_grade",
            self.__condition_retrieve_grade,
            {
                "Relevant": "generate_answer",
                "Weak": "web_retrieve",
                "Irrelevant": "web_retrieve",
            },
        )

        # web retrieve
        graph.add_edge("web_retrieve", "generate_answer")

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
        updated_state = state.copy()
        updated_state.update({"question": rewrite_question})
        return updated_state

    def __node_rag_retrieve(self, state: RagState) -> RagState:
        question = state["question"]

        # Retrieval
        rag_retrieves = self.rag_retriever.invoke(question)
        print(f"[rag_retrieve] rag retrieve number: {len(rag_retrieves)}")

        updated_state = state.copy()
        updated_state.update({"rag_retrieves": rag_retrieves})
        return updated_state

    def __node_rag_retrieve_grade(self, state: RagState) -> RagState:
        question = state["question"]
        rag_retrieves = state["rag_retrieves"]
        rag_retrieves_relevant = []
        rag_retrieves_weak = []
        content_size_min = 50

        # Score each retrieve document
        for index, document in enumerate(rag_retrieves):
            print(f"=== RAG retrieve [{index}] grade === ")
            print(document.page_content)

            if len(document.page_content) < content_size_min:
                print("Warning: skip RAG retrieves content too less.")
                continue

            if self.search_type == "score_threshold":
                score = self.score_relevant
            else:
                score = self.llm_processor.grade_relevance(
                    question=question, context=document.page_content
                )

            if score >= self.score_relevant:
                print(f"RAG retrieves relevant score {score}.")
                rag_retrieves_relevant.append(document)
            elif score >= self.score_weak:
                print(f"RAG retrieves weak score {score}.")
                rag_retrieves_weak.append(document)
            else:
                print(f"Warning: skip RAG retrieves content score {score}.")

        updated_state = state.copy()
        updated_state.update(
            {
                "rag_retrieves": [],
                "rag_retrieves_relevant": rag_retrieves_relevant,
                "rag_retrieves_weak": rag_retrieves_weak,
            }
        )
        return updated_state

    def __node_web_retrieve(self, state: RagState) -> RagState:
        question = state["question"]

        # Web search
        web_retrieves = self.web_retriever.invoke(question)
        for doc in web_retrieves:
            print("=== web retrieve === ")
            print(doc.page_content)

        updated_state = state.copy()
        updated_state.update({"web_retrieves": web_retrieves})
        return updated_state

    def __node_generate_answer(self, state: RagState) -> RagState:
        question = state["question"]
        rag_retrieves_relevant = state.get("rag_retrieves_relevant", None)
        rag_retrieves_weak = state.get("rag_retrieves_weak", None)
        web_retrieves = state.get("web_retrieves", None)

        context = ""
        if rag_retrieves_relevant:
            context += "\n".join(doc.page_content for doc in rag_retrieves_relevant)

        if rag_retrieves_weak:
            context += "\n".join([doc.page_content for doc in rag_retrieves_weak])

        if web_retrieves:
            context += "\n".join([doc.page_content for doc in web_retrieves])

        # Generation
        generation = self.llm_processor.generate_answer(
            question=question, context=context
        )
        if not generation:
            generation = context

        print(f"[generate_answer] answer: {generation}")

        updated_state = state.copy()
        updated_state.update({"answer": generation})
        return updated_state

    ############################################################################
    ## Edges conditional functions
    ############################################################################
    def __condition_retrieve(self, state: RagState) -> str:
        rag_retrieves = state["rag_retrieves"]

        if len(rag_retrieves) > 0:
            return "success"
        else:
            print("[condition_retrieve]: failure")
            return "failure"

    def __condition_retrieve_grade(self, state: RagState) -> str:
        rag_retrieves_relevant = state["rag_retrieves_relevant"] or []
        rag_retrieves_weak = state["rag_retrieves_weak"] or []

        if len(rag_retrieves_relevant) > 0:
            print(f"[condition_retrieve_grade] Relevant number: {len(rag_retrieves_relevant)}")
            return "Relevant"
        elif len(rag_retrieves_weak):
            print(f"[condition_retrieve_grade] Weak number: {len(rag_retrieves_weak)}")
            return "Weak"
        else:
            print(f"[condition_retrieve_grade] Irrelevant")
            return "Irrelevant"
