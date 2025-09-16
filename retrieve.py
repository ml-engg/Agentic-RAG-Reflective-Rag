from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
import data_injest_pipeline.config as config
import os 

os.environ["PINECONE_API_KEY"] = config.secrets['PINECONE_API_KEY']

query_template = """
You are retrieving relevant factual knowledge from the scraped documents.
These documents were collected from langchain website
Focus only on retrieving factual evidence, not generating answers.
Always prioritize objective statements over opinions.

Focus only on retrieving factual evidence, not generating answers.

Current Question:
{user_query}

"""

def retrieve(query:str) -> str:
    embedding_model = OpenAIEmbeddings(openai_api_key = config.secrets['OPENAI_API_KEY'])
    vector_store = PineconeVectorStore(index_name= config.secrets['PINECONE_INDEX_NAME'], embedding=embedding_model)
    final_query = query_template.format(user_query = query)
    results = vector_store.similarity_search(final_query, k = 4)
    combined_results = "\n\n".join([doc.page_content for doc in results])
    return results

if __name__ == "__main__":
    res = retrieve('what is langchain ?', None)
    print(res)





