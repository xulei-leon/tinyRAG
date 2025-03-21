# Description: This file contains the LLMProcessor class that provides methods to interact with a Language Model (LLM)
# to perform various tasks such as rewriting questions, generating content, and grading context relevance.
#

################################################################################
# Imports
################################################################################
from typing import List, Dict
import logging

# import langchain
from langchain_deepseek import ChatDeepSeek
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate


class LLMProcessor:
    """
    A class to interact with a Language Model (LLM).
    """

    def __init__(self, llm: ChatDeepSeek):
        self.llm = llm

    def rewrite_question(self, question: str) -> str:
        llm = self.llm
        json_key = "rewrite_question"
        system_role = "Role: Question Optimization"
        system_instruction = self.__system_instruction()
        system_response_format = self.__system_response_format(json_key=json_key)
        user_instruction = "Rewrite this question to a better version that is optimized for vectorstore retrieval."

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"{system_role}\n{system_instruction}\n{system_response_format}",
                ),
                (
                    "human",
                    f"{user_instruction}\n\nOriginal question:\n{{question}}",
                ),
            ]
        )

        print("=== rewrite_question prompt template ===\n")
        prompt.pretty_print()

        rewriter = prompt | llm | JsonOutputParser()
        try:
            doc = rewriter.invoke({"question": question})
            rewrite_result = doc[json_key]
        except KeyError:
            logging.error("KeyError: 'refined_question' not found in response")
            rewrite_result = question

        return rewrite_result or question

    ################################################################################
    ### Retrieval content re-writer
    ################################################################################
    def rewrite_retrieval(self, question: str, context: str) -> str:
        llm = self.llm
        json_key = "rewrite_retrieval"
        system_role = "Role: Documents Optimization"
        system_instruction = self.__system_instruction()
        system_response_format = self.__system_response_format(json_key=json_key)
        user_instruction = "Convert the retrieved document into a more optimised version, rewrite the parts of the document that are relevant to the user's problem, and output them."

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"{system_role}\n{system_instruction}\n{system_response_format}",
                ),
                (
                    "human",
                    f"{user_instruction}\n\nUser question:\n{{question}}\n\nOriginal retrieved document:\n\n {{document}}",
                ),
            ]
        )

        print("=== rewrite_retrieval prompt template ===\n")
        prompt.pretty_print()

        rewriter = prompt | llm | JsonOutputParser()
        try:
            doc = rewriter.invoke({"question": question, "document": context})
            rewrite_result = doc[json_key]
        except KeyError:
            logging.error("KeyError: 'refined_question' not found in response")
            rewrite_result = context

        return rewrite_result or context

    def generate_answer(self, question: str, context: str) -> str:
        llm = self.llm
        json_key = "rewrite_document"
        system_role = "Role: Generate answer"
        system_instruction = self.__system_instruction()
        system_response_format = self.__system_response_format(json_key=json_key)
        user_instruction = "Based on the following context, answer the user's question."

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"{system_role}\n{system_instruction}\n{system_response_format}",
                ),
                (
                    "human",
                    f"{user_instruction}\n\nUser question:\n{{question}}\n\nnContext:\n\n {{context}}",
                ),
            ]
        )

        print("=== rewrite_document prompt template ===\n")
        prompt.pretty_print()

        rewriter = prompt | llm | JsonOutputParser()
        try:
            doc = rewriter.invoke({"question": question, "context": context})
            rewrite_result = doc[json_key]
        except KeyError:
            logging.error("KeyError: 'refined_question' not found in response")
            rewrite_result = context

        return rewrite_result or context

    def grade_relevance(self, question: str, context: str) -> float:
        return 0.8

    @staticmethod
    def __system_instruction() -> str:
        return (
            "Respond to the request:\n"
            "1. Direct response to core content\n"
            "2. Disable examples/extended descriptions\n"
            "3. Use simple sentence structure\n"
            "4. Omit non-critical details\n\n"
            "Current Scenario: Rapid Response Mode"
        )

    @staticmethod
    def __system_response_format(json_key: str) -> str:
        return f"**Format Requirement**:\n- Return a JSON object.\n- Key: {json_key}\n- Value: The improved document"
