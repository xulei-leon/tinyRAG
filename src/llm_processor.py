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

# my modules
from llm_prompt import LLMPrompt


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

        system_role = LLMPrompt.rewrite_question_role()
        system_instruction = LLMPrompt.rewrite_question_output()
        system_response_format = LLMPrompt.json_output_format(json_key=json_key)
        user_instruction = LLMPrompt.rewrite_question_instruction()

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"{system_role}\n\n{system_instruction}\n{system_response_format}",
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
            logging.error(f"KeyError: {json_key} not found in response")
            rewrite_result = question

        return rewrite_result or question

    ################################################################################
    ### Generate answer
    ################################################################################
    def generate_answer(
        self, question: str, rag_context: str, web_context: str, history: str
    ) -> str:
        llm = self.llm
        json_key = "answer"

        system_role = LLMPrompt.generate_answer_role()
        system_instruction = LLMPrompt.generate_answer_output()
        system_response_format = LLMPrompt.json_output_format(json_key=json_key)
        user_profile = LLMPrompt.user_profile()
        user_instruction = LLMPrompt.generate_answer_instruction()

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"{system_role}\n{system_instruction}\n{system_response_format}",
                ),
                (
                    "human",
                    (
                        f"{user_profile}\n\n"
                        f"{user_instruction}\n\n"
                        f"## User question:\n{{question}}\n\n"
                        f"## RAG Retrieve Context:\n{{rag_context}}\n\n"
                        f"## Web Search Context:\n{{web_context}}\n\n"
                        f"## Historical User Questions and Answers:\n{{history}}"
                    ),
                ),
            ]
        )

        # print("=== rewrite_document prompt template ===\n")
        # prompt.pretty_print()
        answer_result = None
        try:
            rewriter = prompt | llm | JsonOutputParser()
            result = rewriter.invoke(
                {
                    "question": question,
                    "rag_context": rag_context,
                    "web_context": web_context,
                    "history": history,
                }
            )

            if type(result[json_key]) is str:
                answer_result = result[json_key]

        except KeyError:
            print(f"LLM answer KeyError: key {json_key} not found in result")
        except Exception as e:
            print(f"LLM answer except: {e}")

        return answer_result or web_context
