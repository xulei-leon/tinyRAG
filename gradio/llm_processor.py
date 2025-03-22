# Description: This file contains the LLMProcessor class that provides methods to interact with a Language Model (LLM)
# to perform various tasks such as rewriting questions, generating content, and grading context relevance.
#

################################################################################
# Imports
################################################################################
from typing import List, Dict
import logging
from pydantic import BaseModel, Field, validator

# import langchain
from langchain_deepseek import ChatDeepSeek
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate


class GradeSchema(BaseModel):
    score: float = Field(description="Score between 0.0-1.0")
    score: float = Field(
        default=0.0,
        description=("Score between 0.0-1.0"),
        examples=[0.5],
        ge=0.0,
        le=1.0,
    )


# TODO list
# Add chinese prompt to answer.
# answer prompt:
# -- Use differentiation for different types of context
# -- Generate answers by user's gender, age, etc.
class LLMProcessor:
    """
    A class to interact with a Language Model (LLM).
    """

    def __init__(self, llm: ChatDeepSeek):
        self.llm = llm

    ################################################################################
    ### Question re-writer
    ################################################################################
    def rewrite_question(self, question: str) -> str:
        llm = self.llm
        json_key = "rewrite_question"
        system_role = "Role: Question Optimization"
        system_instruction = self.__text_instruction()
        system_response_format = self.__json_response_format(json_key=json_key)
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

        # print("=== rewrite_question prompt template ===\n")
        # prompt.pretty_print()

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
        system_role = "Role: Documents Optimization Specialist"
        system_instruction = self.__text_instruction()
        system_response_format = self.__json_response_format(json_key=json_key)
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

        # print("=== rewrite_retrieval prompt template ===\n")
        # prompt.pretty_print()

        rewriter = prompt | llm | JsonOutputParser()
        try:
            doc = rewriter.invoke({"question": question, "document": context})
            rewrite_result = doc[json_key]
        except KeyError:
            logging.error(f"KeyError: {json_key} not found in response")
            rewrite_result = context

        return rewrite_result or context

    ################################################################################
    ### Generate answer
    ################################################################################
    def generate_answer(self, question: str, context: str) -> str:
        llm = self.llm
        json_key = "rewrite_document"
        system_role = "Role: Generate Answer Specialist"
        system_instruction = self.__text_instruction()
        system_response_format = self.__json_response_format(json_key=json_key)
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

        # print("=== rewrite_document prompt template ===\n")
        # prompt.pretty_print()

        rewriter = prompt | llm | JsonOutputParser()
        try:
            doc = rewriter.invoke({"question": question, "context": context})
            rewrite_result = doc[json_key]
        except KeyError:
            logging.error(f"KeyError: {json_key} not found in response")
            rewrite_result = context

        return rewrite_result or context

    ################################################################################
    ### Grade context relevance
    ################################################################################
    def grade_relevance(self, question: str, context: str) -> float:
        return self.__grade_context(
            question=question, context=context, grader_type="relevance"
        )

    ################################################################################
    ### Internal methods
    ################################################################################
    def __grade_context(self, question: str, context: str, grader_type: str) -> float:
        llm = self.llm

        system_role = "Role: Scoring Expert"
        system_instruction = self.__score_system_instruction(grader_type)
        system_response_format = self.__score_response_format()
        user_instruction = self.__score_user_instruction(grader_type)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"{system_role}\n{system_instruction}\n{system_response_format}",
                ),
                (
                    "human",
                    f"{user_instruction}\n\nUser question:\n{{question}}\n\nDocument:\n\n {{context}}",
                ),
            ]
        )

        # print("=== grade_relevance prompt template ===\n")
        # prompt.pretty_print()

        structured_llm = llm.with_structured_output(GradeSchema)
        grader = prompt | structured_llm
        try:
            grade = grader.invoke({"question": question, "context": context})
            score = grade.score
        except KeyError:
            logging.error("KeyError: score not found in grade")
            score = 7

        return score

    @staticmethod
    def __text_instruction() -> str:
        return (
            "Respond to the request:\n"
            "1. Direct response to core content\n"
            "2. Disable examples/extended descriptions\n"
            "3. Use simple sentence structure\n"
            "4. Omit non-critical details\n\n"
            "Current Scenario: Rapid Response Mode"
        )

    @staticmethod
    def __json_response_format(json_key: str) -> str:
        return f"**Format Requirement**:\n- Return a JSON object.\n- Key: {json_key}\n- Value: The improved document"

    @staticmethod
    def __score_system_instruction(type: str) -> str:
        instructions = {
            "relevance": "relevance of the document to the question",
            "answer": "answer addresses the question",
        }

        instruction = instructions[type]
        return (
            "Guidelines:\n"
            f"Please assess the {instruction} based on the following criteria:\n"
            "Scoring Criteria:\n"
            "0.0: Irrelevant/no answer\n"
            "0.3: Only keywords/no relevance\n"
            "0.5: Relevant keywords, partial answer\n"
            "0.7: Mostly relevant, insufficient support\n"
            "0.9: Highly relevant, detailed evidence\n"
            "1.0: Perfect solution/answer"
        )

    @staticmethod
    def __score_user_instruction(type: str) -> str:
        instructions = {
            "relevance": "Rrelevance document",
            "answer": "Aanswer",
        }

        return instructions[type]

    @staticmethod
    def __score_response_format() -> str:
        return f"Please return a score between 0.0-1.0"
