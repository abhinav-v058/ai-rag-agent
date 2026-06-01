import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from operator import itemgetter

load_dotenv()

print("Hello from ai-rag-agent retrival component!")

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(temperature=0.7, model="gpt-5.2", openai_api_key=os.getenv("OPENAI_API_KEY"))
vector_store = PineconeVectorStore(
    index_name=os.getenv("INDEX_NAME"),
    embedding=embeddings
)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

prompt_template = ChatPromptTemplate.from_template(
    """Answer the following question based on the following context:
{context}
Question: {question}    
Provide a detailed answer. 
"""
)

def retrieve_without_lcel(query: str):
    relevant_docs = retriever.invoke(query)
    context = format_doc(relevant_docs)
    prompt = prompt_template.format_messages(context=context, question=query)
    return llm.invoke(prompt)

def create_retrival_chain_with_lcel():
    """Create a retrieval chain with LCEL (Language Chain Execution Language)
    Returns:
        A retrieval chain that can be invoked with {"question": "..."}.

    Advantages over the non-LCEL version:

    """
    retrieval_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_doc
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return retrieval_chain
    
def format_doc(docs):
    return "\n".join([f"Document {i+1}: {doc.page_content}" for i, doc in enumerate(docs)])

if __name__ == "__main__":
    print("Retrieving relevant documents...")
    query = "What is artemis?"
    #answer = retrieve_without_lcel(query)
    retrieval_chain = create_retrival_chain_with_lcel()
    answer = retrieval_chain.invoke({"question": query})
    print("Answer:")
    print(answer)
    # result_raw = llm.invoke([HumanMessage(content=query)])
    #print("Raw LLM response:")
    #print(result_raw)