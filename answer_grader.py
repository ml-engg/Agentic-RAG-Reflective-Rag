from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import data_injest_pipeline.config as config

# ---------------------- Answer Checker ----------------------
class AnswerCheck(BaseModel):
    valid: bool = Field(
        description="True if the answer addresses the question, False otherwise"
    )
    reasoning: str = Field(
        description="One short sentence explaining why"
    )

def answer_checker(question: str, answer: str) -> AnswerCheck:
    llm = ChatOpenAI(
        temperature=0,
        model_name="gpt-3.5-turbo",
        openai_api_key=config.secrets['OPENAI_API_KEY']
    )
    structured_checker = llm.with_structured_output(AnswerCheck)

    prompt = f"""
    You are an answer validator.

    Question:
    {question}

    Answer:
    {answer}

    Decide if the answer addresses the question.
    """
    return structured_checker.invoke(prompt)

if __name__ == "__main__":
    question = "What is LangChain?"

    # ✅ Correct answer
    # answer1 = "LangChain is a framework for building applications powered by large language models."
    # result1 = answer_checker(question, answer1)
    # print("Test 1 - Valid Answer")
    # print("Answer:", answer1)
    # print("Valid:", result1.valid)
    # print("Reasoning:", result1.reasoning)
    # print("-" * 50)

     # ❌ Wrong / unrelated answer
    # answer2 = "It is a type of blockchain technology used for cryptocurrencies."
    # result2 = answer_checker(question, answer2)
    # print("Test 2 - Invalid Answer")
    # print("Answer:", answer2)
    # print("Valid:", result2.valid)
    # print("Reasoning:", result2.reasoning)
    # print("-" * 50)
