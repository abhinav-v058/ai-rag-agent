import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

load_dotenv()

def main():
    print("Hello from ai-rag-agent!")
    loader = TextLoader("/Users/verma/lab/ai/ai-rag-agent/artemis.txt")
    documents = loader.load()

    print(f"Loaded {len(documents)} documents.")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separator="\n")
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks.")
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    PineconeVectorStore.from_documents(texts, embeddings, index_name=os.getenv("INDEX_NAME"))
    print("Documents embedded and stored in Pinecone.")
    
if __name__ == "__main__":
    main()
