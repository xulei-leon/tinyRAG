import tomllib
import json
import logging


class LLMPrompt:

    def __init__(self):
        self.health_prescription = self.__load_health_prescription()

    @classmethod
    def json_output_format(cls, json_key: str) -> str:
        return (
            "## Output Format Description\n"
            "- Output a strict JSON format object.\n"
            "- Note: Do not include any other text or explanation.\n"
            f"- Key: {json_key}\n"
            "- Value: Must a text string.\n"
        )

    @classmethod
    def user_profile(cls) -> str:
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

    ################################################################################
    ### rewrite_question
    ################################################################################
    @classmethod
    def rewrite_question_role(cls) -> str:
        return "## Role: Question Optimization"

    @classmethod
    def rewrite_question_instruction(cls) -> str:
        return (
            "Rewrite, optimise and extend the questions users ask, based on the personal information they provide.\n"
            "This helps us to understand their needs better and provide more detailed responses.\n"
            "You can add more details to the question, but do not change the original meaning.\n"
            "Please answer questions from users in Chinese.\n"
        )

    @classmethod
    def rewrite_question_output(cls) -> str:
        return (
            "## Output Rules\n"
            "1. Follow the instructions strictly\n"
            "2. Use simple sentence structure\n"
            "3. Use clear and concise language\n"
            "4. Avoid unnecessary details\n"
            "5. Avoid excessive examples\n"
            "6. Avoid excessive explanations\n"
            "7. Respond quickly and accurately\n"
        )

    ################################################################################
    ### Generate answer
    ################################################################################
    @classmethod
    def generate_answer_role(cls) -> str:
        return (
            "## Role: Healthcare Consultant\n"
            "Your main goal is to provide solutions that are relevant to the user's personality, while also staying professional."
        )

    @classmethod
    def generate_answer_instruction(cls) -> str:
        return (
            "Answer the question based on the user's profile and the database context and web context provided.\n"
            "1. Please base on the 'RAG Retrieve Context' to answer the question, but also refer to the 'Web Search Context'.\n"
            "2. If the 'RAG Retrieve Context' does not provide enough information or is not relevant to the question, please answer based on your knowledge.\n"
            "3. Please also refer to the information in the 'Historical User Questions and Answers'.\n"
            "4. Use the user's profile to adjust your tone and communication style. Use a formal but friendly tone, and the simple and clear language\n"
        )

    @classmethod
    def generate_answer_output(cls) -> str:
        return (
            "## Output Rules\n"
            "1.  **Recommendation:** Clearly state whether you recommend the product based on the user's question and available information."
            "2.  **Reason for Recommendation:** Explain *why* you are recommending this specific product. Connect it directly to the user's implied needs or explicit statements in their question.\n"
            "3.  **Product Advantages:** List the key advantages and features of the product that make it suitable for the user.\n"
            "4.  **Relevance to Customer:** Explicitly explain how the product relates to the customer's potential needs, preferences, or any information they have provided (even implicitly).\n"
            "5.  **Logic and Reasoning:** Briefly explain the logical steps you took to arrive at your recommendation. Explain *why* the advantages you listed are important and how they address the user's potential needs. This will help the user understand your reasoning and build trust in your recommendations.\n"
            "6. The recommended length of the output string is between 100 and 300 characters.\n"
            "7. Please answer questions from users in Chinese.\n"
        )

    def generate_answer_prescription(self) -> str:
        health_prescription = self.health_prescription
        if not health_prescription:
            return "None"

        prescription_text = ""
        for key, value in health_prescription.items():
            prescription_text += (
                "---\n"
                f"Health problems: [{key}]\n"
                f"Recommended health supplements: {value}\n"
            )

        return prescription_text

    #################################################################################
    ### Internal mothods
    #################################################################################
    @classmethod
    def __load_health_prescription(cls):
        health_prescription_file = None
        with open("config/config.toml", "rb") as f:
            config_data = tomllib.load(f)
            health_prescription_file = config_data.get("database", {}).get(
                "health_prescription_file", None
            )
        if not health_prescription_file:
            return None

        file_r = None
        try:
            with open(health_prescription_file, "r", encoding="utf-8") as f:
                file_r = f.read()
        except FileNotFoundError:
            logging.error(
                f"Health prescription file not found: {health_prescription_file}"
            )
            return None
        except IOError as e:
            logging.error(f"Error reading health prescription file: {e}")
            return None
            file_r = f.read()
        if not file_r:
            logging.error("Failed to load health prescription file.")
            return None

        try:
            health_prescription = json.loads(file_r)
        except json.JSONDecodeError:
            logging.error("Failed to parse health prescription file.")
            return None

        logging.info("Health prescription file loaded successfully.")
        return health_prescription
