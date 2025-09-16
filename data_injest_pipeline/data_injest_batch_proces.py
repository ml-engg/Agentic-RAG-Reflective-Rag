import os
from langchain_pinecone import Pinecone
from langchain_tavily import TavilyCrawl
from langchain_tavily import TavilyExtract
import json
import os 
import config
from langchain.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.schema import Document

os.environ["PINECONE_API_KEY"] = config.secrets['PINECONE_API_KEY']


def get_file_name(fl):
    with open(fl, "r") as file:
        data = json.load(file)
    
    link = data['site']
    return link

def web_scrap(link):
    extract_tool = TavilyCrawl(extract_depth = "basic", 
                               max_depth = 5,
                               instructions="Documentation pages",
                               max_breadth = 10, 
                               format = 'text', 
                               tavily_api_key = config.secrets['TAVILY_API_KEY'])
    reponses = extract_tool.invoke({"url": link})
    results = reponses['results']
    for item in results:
        raw_extract = item['raw_content']
    
    return raw_extract

def create_chunks(text):
    documents = [Document(page_content=text)]

    print("...splitting")
    splitter = CharacterTextSplitter(chunk_size = 200, chunk_overlap = 50)
    texts = splitter.split_documents(documents)
    print(f"created {len(texts)} chunks")

    print("...injesting")
    embedding_model = OpenAIEmbeddings(openai_api_key = config.secrets['OPENAI_API_KEY'])
    index = config.secrets['PINECONE_INDEX_NAME']
    docsearch = PineconeVectorStore.from_documents(texts,embedding_model, index_name=index)

    print("finish")


if __name__ == "__main__":
    link_name = get_file_name("web.json")
    res = web_scrap(link_name)
    create_chunks(res)
