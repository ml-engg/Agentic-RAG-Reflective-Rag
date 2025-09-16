from langchain_openai import ChatOpenAI
from langchain.schema import Document
import data_injest_pipeline.config as config
import os 
from pydantic import BaseModel, Field


os.environ["PINECONE_API_KEY"] = config.secrets['PINECONE_API_KEY']

class GradeDocs(BaseModel):
    binary_score: bool = Field(description="Answer the question 'yes' or 'no'")
    reasoning: str = Field(description="One short sentence explaning the decision")


def grade_documents(question: str, retreived_docs: list[Document]) -> list[dict]:
    llm = ChatOpenAI(temperature=0, model_name = 'gpt-3.5-turbo', openai_api_key=config.secrets['OPENAI_API_KEY'])
    structued_llm_grader = llm.with_structured_output(GradeDocs)
    
    graded = []
    for doc in retreived_docs:
        grading_prompt = f""" You are a strict grader.
        Question: {question}
        Document: {doc.page_content}

        Does the document address the question?
        """

        result = structued_llm_grader.invoke(grading_prompt)

        graded.append({
            "document": doc.page_content,
            "binary_score": result.binary_score,
            "reasoning": result.reasoning
        })

    return graded

def final_grader(graded_docs: list[dict]) -> dict:
    """
    Aggregates all doc grades and decides if at least one document is relevant.
    """
    any_relevant = any(doc["binary_score"] for doc in graded_docs)

    return {
        "at_least_one_relevant": any_relevant,
        "relevant_count": sum(1 for doc in graded_docs if doc["binary_score"]),
        "total_docs": len(graded_docs)
    }


if __name__ == "__main__":
    # Example usage after retrieval
    from retrieve import retrieve  # import your earlier retrieve function
    
    query = "what is langchain ?"
    docs = retrieve(query, None)  # This currently returns a string
    # In practice, modify retrieve() to return list[Document] for grading
    
    results = grade_documents(query, docs)
    for r in results:
        print(r)
    
    final_result = final_grader(results)
    print("\nFinal Grader Output:")
    print(final_result)

