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

        system_role = "## Role: Question Optimization"
        system_instruction = self.__output_instruction()
        system_response_format = self.__json_output_format(json_key=json_key)
        user_instruction = (
            "Rewrite, optimise and extend the questions users ask, based on the personal information they provide.\n"
            "This helps us to understand their needs better and provide more detailed responses.\n"
            "You can add more details to the question, but do not change the original meaning.\n"
            "Please answer questions from users in Chinese.\n"
        )

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
    ### Retrieval content re-writer
    ################################################################################
    def rewrite_retrieval(self, question: str, context: str) -> str:
        llm = self.llm
        json_key = "rewrite_retrieval"

        system_role = "## Role: Documents Optimization Specialist"
        system_instruction = self.__output_instruction()
        system_response_format = self.__json_output_format(json_key=json_key)
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
    def generate_answer(self, question: str, rag_context: str, web_context: str) -> str:
        llm = self.llm
        json_key = "answer"

        system_role = (
            "## Role: Healthcare Consultant\n"
            "Your main goal is to provide solutions that are relevant to the user's personality, while also staying professional.\n"
        )
        system_instruction = self.__output_instruction()
        system_response_format = self.__json_output_format(json_key=json_key)
        user_profile = self.__user_profile()
        user_instruction = (
            "Answer the question based on the user's profile and the database context and web context provided.\n"
            "1. Please focus on the database context to answer the question, but also refer to the web context.\n"
            "2. If the database context does not provide enough information or is not relevant to the question, please answer based on web context and your knowledge.\n"
            "3. Use a formal but friendly tone, and the simple and clear language.\n"
            "4. Use the user's profile to adjust your tone and communication style.\n"
            "5. The recommended length of the output string is between 50 and 200 characters.\n"
            "6. Please answer questions from users in Chinese.\n"
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"{system_role}\n{system_instruction}\n{system_response_format}",
                ),
                (
                    "human",
                    f"{user_profile}\n\n{user_instruction}\n\n## User question:\n{{question}}\n\nn## Database Context:\n\n {{rag_context}}\n\n## Web Context:\n\n {{web_context}}",
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
                }
            )

            if type(result[json_key]) is str:
                answer_result = result[json_key]

        except KeyError:
            print(f"LLM answer KeyError: key {json_key} not found in result")
        except Exception as e:
            print(f"LLM answer except: {e}")

        return answer_result or web_context

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

        system_role = "## Role: Scoring Expert"
        system_instruction = self.__score_system_instruction(grader_type)
        system_response_format = self.__score_response_format()
        user_instruction = self.__score_user_instruction(grader_type)

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"{system_role}\n\n{system_instruction}\n\n{system_response_format}",
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

    @staticmethod
    def __output_instruction() -> str:
        return (
            "## Output instruction\n"
            "1. Follow the instructions strictly\n"
            "2. Use simple sentence structure\n"
            "3. Use clear and concise language\n"
            "4. Avoid unnecessary details\n"
            "5. Avoid excessive examples\n"
            "6. Avoid excessive explanations\n"
            "7. Respond quickly and accurately\n"
        )

    @staticmethod
    def __json_output_format(json_key: str) -> str:
        return (
            "## Output Format Description\n"
            "- Output a strict JSON format object.\n"
            "- Note: Do not include any other text or explanation.\n"
            f"- Key: {json_key}\n"
            "- Value: Must a text string.\n"
        )

    @staticmethod
    def __user_profile() -> str:
        return (
            "## User Profile\n"
            "Age: 55 to 65 years\n"
            "Sex: Male\n"
            "Income: Good\n"
            "Job: Retired\n"
            "Lifestyle: Moderately active, enjoys walking. Active in the community. Follows doctor's advice on health.\n"
            "Health problems: Occasional insomnia, low energy\n"
            "Health goals: Maintain good health. Improve sleep quality and maintain an active lifestyle.\n"
            "Seeking Information: Trusted source of information, willing to listen to family and friends who share their health care experiences.\n"
        )
