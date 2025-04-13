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


def summary_reducer(n):
    """Returns a reducer function that keeps only the last n elements."""

    def reducer(old, new):
        return (old or [])[-n:] + (new or [])[-n:]

    return reducer


with open("config/config.toml", "rb") as f:
    config_data = tomllib.load(f)
    chat_history_count = config_data.get("chat", {}).get("chat_history_count", 3)


#
# RAG state
#
class RagState(TypedDict):
    question: str
    answer: str
    rag_retrieves: List[Document]
    rag_completed: str
    web_retrieves: List[Document]
    web_completed: str
    thinking: Annotated[str, operator.add]
    summary: Annotated[
        list, summary_reducer(chat_history_count - 1)
    ]  # save summary for history


#
# RAG Graph
#
class RagGraph:

    def __init__(
        self,
        rag_retriever: RagRetriever,
        web_retriever: TavilySearchAPIRetriever,
        llm_processor: LLMProcessor,
    ):
        # for langsmith token
        load_dotenv()

        with open("config/config.toml", "rb") as f:
            config_data = tomllib.load(f)
            self.search_result_num = config_data.get("retriever", {}).get(
                "search_result_num", 3
            )
            self.rerank_score_relevant = float(
                config_data.get("retriever", {}).get("rerank_score_relevant", 0.55)
            )
            self.rerank_score_enable = config_data.get("retriever", {}).get(
                "rerank_score_enable", "off"
            )
            self.chat_agent_name = config_data.get("chat", {}).get("chat_agent_name")

        self.llm_processor = llm_processor
        self.rag_retriever = rag_retriever
        self.web_retriever = web_retriever
        self.memory = MemorySaver()
        self.graph = None
        self.logger = logging.getLogger(__name__)

    def compile(self) -> StateGraph:
        if not self.graph:
            self.graph = self.__build_graph()
        return self.graph.compile(checkpointer=self.memory)

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
        graph.add_node("start", self.__node_start)
        graph.add_node("rewrite_qestion_start", self.__node_rewrite_question_start)
        graph.add_node("rewrite_qestion", self.__node_rewrite_question)

        graph.add_node("rag_retrieve_start", self.__node_rag_retrieve_start)
        graph.add_node("rag_retrieve", self.__node_rag_retrieve)
        graph.add_node("rag_retrieve_grade_start", self.__node_rag_retrieve_grade_start)
        graph.add_node("rag_retrieve_grade", self.__node_rag_retrieve_grade)
        graph.add_node("rag_retrieve_finish", self.__node_rag_retrieve_finish)

        graph.add_node("web_retrieve_start", self.__node_web_retrieve_start)
        graph.add_node("web_retrieve", self.__node_web_retrieve)

        graph.add_node("generate_answer_start", self.__node_generate_answer_start)
        graph.add_node("generate_answer", self.__node_generate_answer)

        # Add edges
        graph.add_edge(START, "start")
        graph.add_edge("start", "rewrite_qestion_start")
        graph.add_edge("rewrite_qestion_start", "rewrite_qestion")

        # RAG retrieve
        graph.add_edge("rewrite_qestion", "rag_retrieve_start")
        graph.add_edge("rag_retrieve_start", "rag_retrieve")
        graph.add_conditional_edges(
            "rag_retrieve",
            self.__condition_retrieve,
            {
                "success": "rag_retrieve_grade_start",
                "failure": "rag_retrieve_finish",
            },
        )

        graph.add_edge("rag_retrieve_grade_start", "rag_retrieve_grade")
        graph.add_edge("rag_retrieve_grade", "rag_retrieve_finish")
        graph.add_conditional_edges(
            "rag_retrieve_finish",
            self.__condition_complete,
            {
                "success": "generate_answer_start",
                "failure": END,
            },
        )

        # web retrieve
        graph.add_edge("rewrite_qestion", "web_retrieve_start")
        graph.add_edge("web_retrieve_start", "web_retrieve")
        graph.add_conditional_edges(
            "web_retrieve",
            self.__condition_complete,
            {
                "success": "generate_answer_start",
                "failure": END,
            },
        )

        # generate answer
        graph.add_edge("generate_answer_start", "generate_answer")
        graph.add_edge("generate_answer", END)

        return graph

    ############################################################################
    ## Nodes functions
    ############################################################################
    def __node_start(self, state: RagState) -> RagState:
        if state.get("summary"):
            print(f"[start] {len(state['summary'])} historical summary")

        thinking = f"💡 你好，我是{self.chat_agent_name}，我现在对你的问题进行专业的分析。请稍后..."
        new_state = {
            "answer": "",
            "rag_retrieves": [],
            "rag_completed": "",
            "web_retrieves": [],
            "web_completed": "",
            "thinking": thinking,
        }
        return new_state

    def __node_rewrite_question_start(self, state: RagState) -> RagState:
        question = state.get("question")

        if not question:
            question = "请介绍保健产品对中老年人身体健康的好处有哪些。"

        question = question.strip()
        if len(question) > 200:
            question = question[:200]

        print(f"[rewrite_question_start] question: {question}")

        thinking = (
            "📝 正在分析问题...\n"
            # "优化问题是为了更好地理解和回答你的问题。\n"
            # f"原问题: {question}\n"
            # "请稍后..."
        )

        new_state = {"thinking": thinking, "question": question}
        return new_state

    def __node_rewrite_question(self, state: RagState) -> RagState:
        question = state.get("question")
        print(f"[rewrite_question] question: {question}")

        # Re-write question
        rewrite_question = self.llm_processor.rewrite_question(question=question)
        if not rewrite_question:
            rewrite_question = question

        print(f"[rewrite_question] rewite question: {rewrite_question}")

        thinking = f"📝 优化后问题: {rewrite_question}"
        new_state = {
            # "thinking": thinking,
            "question": rewrite_question,
        }
        return new_state

    def __node_rag_retrieve_start(self, state: RagState) -> RagState:
        thinking = "🔍 正在检索专业资料和产品。请稍后..."
        # new_state = {"thinking": thinking}
        new_state = {}
        return new_state

    def __node_rag_retrieve(self, state: RagState) -> RagState:
        question = state.get("question")

        # Retrieval
        # rag_retrieves = self.rag_retriever.query_rerank(question)
        rag_retrieves = self.rag_retriever.query(question)
        print(f"[rag_retrieve] rag retrieve number: {len(rag_retrieves)}")

        thinking = f"🔍 已经检索到{len(rag_retrieves)}份产品资料。"
        new_state = {
            # "thinking": thinking,
            "rag_retrieves": rag_retrieves,
        }
        return new_state

    def __node_rag_retrieve_grade_start(self, state: RagState) -> RagState:
        if self.rerank_score_enable == "on":
            thinking = "📚 正在分析检索资料。请稍后..."
            new_state = {"thinking": thinking}
        else:
            new_state = {}

        return new_state

    def __node_rag_retrieve_grade(self, state: RagState) -> RagState:
        question = state.get("question")
        rag_retrieves = state.get("rag_retrieves", [])
        content_size_min = 50

        # Filter relevant retrieves
        relevants_with_score = []
        for doc in rag_retrieves:
            print(f"=== RAG retrieve score === ")
            # print(doc.page_content[:80])
            print(f"retrieve doc len {len(doc.page_content)}")

            if self.rerank_score_enable == "on":
                score = self.rag_retriever.query_score(
                    question=question, context=doc.page_content
                )
            else:
                score = self.rerank_score_relevant

            if len(doc.page_content) < content_size_min:
                score = score * 0.8

            score = round(score, 2)
            if score < self.rerank_score_relevant:
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

        if self.rerank_score_enable == "on":
            thinking = f"📚 已经分析有{len(rag_retrieves)}份资料与您的问题相关。"
            new_state = {
                # "thinking": thinking,
                "rag_retrieves": rag_retrieves,
            }
        else:
            new_state = {"rag_retrieves": rag_retrieves}

        return new_state

    def __node_rag_retrieve_finish(self, state: RagState) -> RagState:
        return {"rag_completed": "completed"}

    def __node_web_retrieve_start(self, state: RagState) -> RagState:
        thinking = "🌐 正在检索最新数据，请稍后..."
        new_state = {"thinking": thinking}
        return new_state

    def __node_web_retrieve(self, state: RagState) -> RagState:
        question = state.get("question")

        # Web search
        web_retrieves = self.web_retriever.invoke(question)
        for doc in web_retrieves:
            print("=== web retrieve === ")
            # print(doc.page_content[:50])

        thinking = f"🌐 已经检索到{len(web_retrieves)}份最新数据。"
        new_state = {
            # "thinking": thinking,
            "web_retrieves": web_retrieves,
            "web_completed": "completed",
        }
        return new_state

    def __node_generate_answer_start(self, state: RagState) -> RagState:
        thinking = "💡 正在生成答案。请稍后..."
        new_state = {"thinking": thinking}
        return new_state

    def __node_generate_answer(self, state: RagState) -> RagState:
        question = state.get("question")
        rag_retrieves = state.get("rag_retrieves", [])
        web_retrieves = state.get("web_retrieves", [])
        summary_history = state.get("summary", [])

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

        # Generate summary
        history_num = 0
        if summary_history:
            history_num = len(summary_history)
            history = "\n\n---\n\n".join(summary_history)
        else:
            history = ""

        # Generation
        generation = self.llm_processor.generate_answer(
            question=question,
            rag_context=rag_context,
            web_context=web_context,
            history=history,
        )

        print(f"[generate_answer] question: {question}")
        print(f"[generate_answer] {rag_num} rag_retrieves")
        print(f"[generate_answer] {web_num} web_retrieves")
        print(f"[generate_answer] {history_num} summary_history")

        print(f"[generate_answer] answer: {generation}")

        thinking = f"下面是{self.chat_agent_name}的回答："
        summary = f"User question:\n{question}\n\nAI answer:\n{generation}"
        new_state = {
            "thinking": thinking,
            "answer": generation,
            "summary": [summary],
        }
        return new_state

    ############################################################################
    ## Edges conditional functions
    ############################################################################
    def __condition_retrieve(self, state: RagState) -> str:
        if state.get("rag_retrieves"):
            return "success"
        else:
            print("[condition_retrieve]: failure")
            return "failure"

    def __condition_complete(self, state: RagState) -> str:
        if (
            state.get("rag_completed") == "completed"
            and state.get("web_completed") == "completed"
        ):
            print("[condition_complete]: success")
            return "success"
        else:
            print("[condition_complete]: failure")
            return "failure"
