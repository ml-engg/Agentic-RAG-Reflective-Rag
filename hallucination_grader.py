from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import data_injest_pipeline.config as config

class HallucinationGrade(BaseModel):
    hallucination: bool = Field(
        description="True if the answer contains information not supported by context, False otherwise"
    )
    reasoning: str = Field(
        description="One short sentence explaining why it is or is not hallucinated"
    )

def hallucination_grader(answer: str, context : str) -> HallucinationGrade:
    """
    Check if the answer generated is from context
    """
    llm = ChatOpenAI(temperature=0, model_name = 'gpt-3.5-turbo', openai_api_key=config.secrets['OPENAI_API_KEY'])
    structured_h_grader = llm.with_structured_output(HallucinationGrade)

    prompt = f"""
    You are a hallucination checker.

    Context:
    {context}

    Answer:
    {answer}

    Decide if the answer introduces information not present in the context.
    """
    return structured_h_grader.invoke(prompt)

if __name__ == "__main__":
    # Example 1: Answer fully supported by context
    context = """
    LangChain is a framework for developing applications powered by language models.
    It provides tools for chaining together LLMs with external data sources.
    """
    answer = "LangChain is a framework created by Elon Musk to improve chatbots."
    
    grade = hallucination_grader(answer, context)
    print("✅ Example 1")
    print("Answer:", answer)
    print("Hallucination:", grade.hallucination)
    print("Reasoning:", grade.reasoning)
    print()