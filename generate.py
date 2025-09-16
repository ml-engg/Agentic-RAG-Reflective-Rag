from langchain_openai import ChatOpenAI
from retrieve import retrieve
from grading_docs import grade_documents, final_grader
from web_search  import web_search
import data_injest_pipeline.config as config
from hallucination_grader import hallucination_grader
from answer_grader import answer_checker

import sqlite3

DB_PATH = "chat_history.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chat_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        role TEXT,
        message TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()
    conn.close()

def save_message(user_id: str, role: str, message: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO chat_history (user_id, role, message) VALUES (?, ?, ?)",
        (user_id, role, message)
    )
    conn.commit()
    conn.close()

def get_chat_history(user_id: str, limit: int = 10):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT role, message
        FROM chat_history
        WHERE user_id = ?
        ORDER BY id DESC
        LIMIT ?
    """, (user_id, limit))
    rows = cursor.fetchall()
    conn.close()

    # Return as conversation string
    messages = [f"{r[0]}: {r[1]}" for r in rows][::-1]
    return "\n".join(messages)


def generate(user_id: str, query: str) -> dict:

    # Step 0: Get chat history
    chat_history = get_chat_history(user_id)

    ## step 1:  retrieve from vector database
    docs = retrieve(query=query)

    ## step 2: grade documents
    docs_to_grade = grade_documents(query, docs)
    final_result = final_grader(docs_to_grade)

    # Step 3: Choose source
    llm = ChatOpenAI(temperature=0, model_name = 'gpt-3.5-turbo', openai_api_key=config.secrets['OPENAI_API_KEY'])
    
    if final_result["at_least_one_relevant"]:
        # Use relevant docs
        relevant_docs = [doc["document"] for doc in docs_to_grade if doc["binary_score"]]
        context = "\n\n".join(relevant_docs)

        prompt = f"""
        You are answering based on retrieved documents and prior chat history.

        Question: {query}

        Chat History:
        {chat_history}

        Retrieved Context:
        {context}

        Provide a clear, concise answer.
        """
        response = llm.invoke(prompt)

        hallucination_flag = hallucination_grader(response.content, context)
        if not hallucination_flag.hallucination:
            ans_check = answer_checker(query, response.content)
            if ans_check.valid:
                save_message(user_id, "user", query)
                save_message(user_id, "assistant", response.content)
                return {"source": "vector_store", "answer": response.content}
            else:
                print("⚠️ Answer did not address question → fallback to web search")
        else:
            print("⚠️ Hallucination detected from retrieval → fallback to web search")

    for attempt in range(1):
        # Fallback to web search
        web_results = web_search(query, k=3)
        snippets = "\n\n".join([r.get("snippet", "") for r in web_results])

        prompt = f"""
        You are answering based on web search results and prior chat history.

        Question: {query}

        Chat History:
        {chat_history}

        Web Search Results:
        {snippets}

        Provide a clear, concise answer.
        """
        response = llm.invoke(prompt)

        # Step 4: Hallucination check for web search
        hallucination_flag = hallucination_grader(response.content, snippets)
        if not hallucination_flag.hallucination:
            ans_check = answer_checker(query, response.content)
            if ans_check.valid:
                save_message(user_id, "user", query)
                save_message(user_id, "assistant", response.content)
                return {"source": f"web_search_attempt_{attempt+1}", "answer": response.content}

        print(f"⚠️ Hallucination detected or invalid answer from web search (attempt {attempt+1}) → retrying...")

    # If all retries failed
    # Step 5: Failure case
    fallback_msg = "I could not find a reliable answer, even after web search."
    save_message(user_id, "user", query)
    save_message(user_id, "assistant", fallback_msg)

    return {"source": "web_search", "answer": fallback_msg}


if __name__ == "__main__":
    init_db()
    user_id = "user123"
    query = "what is langchain ?"
    chat_history = "User previously asked about langchain"
    result = generate(query, user_id)
    print(f"Source: {result['source']}\nAnswer:\n{result['answer']}")



    





