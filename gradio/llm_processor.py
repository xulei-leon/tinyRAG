from typing import List, Dict

from langchain_deepseek import ChatDeepSeek


class LLMProcessor:
    """
    A class to interact with a Language Model (LLM) for various tasks such as
    rewriting questions, generating content based on retrieved results, and
    rating context content based on user questions.
    """

    def __init__(self, llm:ChatDeepSeek):
        self.llm = llm

    def rewrite_question(self, question: str) -> str:
        prompt = f"Rewrite the following question for clarity:\n\n{question}"
        response = self.llm.invoke(prompt)
        return response.strip()

    def generate_content(self, question: str, retrieved_results: List[Dict]) -> str:
        context = "\n".join([result['content'] for result in retrieved_results])
        prompt = f"Based on the following context, answer the question:\n\nContext:\n{context}\n\nQuestion:\n{question}"
        response = self.llm.invoke(prompt)
        return response.strip()

    def context_grade(self, question: str, context: str) -> float:
        prompt = f"Rate the relevance of the following context to the question on a scale of 0 to 1:\n\nQuestion:\n{question}\n\nContext:\n{context}"
        response = self.llm.invoke(prompt)
        try:
            score = float(response.strip())
            return max(0.0, min(1.0, score))  # Ensure the score is between 0 and 1
        except ValueError:
            return 0.0  # Default to 0 if the response is not a valid number