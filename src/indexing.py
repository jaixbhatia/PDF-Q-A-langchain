
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import OpenAI
from langchain_pinecone import PineconeVectorStore

def indexing(pdf_path, k):
    docs, embeddings = load_and_split(pdf_path)
    retriever = store(docs, embeddings, k)
    return retriever

"""
Load: First we need to load our data. This is done with DocumentLoaders.
Using OpenAIEmbeddings and Facebook AI Similarity Search (Faiss)

Split: Text splitters break large Documents into smaller chunks.
    This is useful both for indexing data and for passing it in to a model,
    since large chunks are harder to search over and won’t fit in a model’s finite context window.
"""
def load_and_split(pdf_path):
    loader = PyPDFLoader(pdf_path)
    texts = loader.load_and_split()
    embeddings = OpenAIEmbeddings()
    return texts, embeddings

    """
    doc_search = document_search.similarity_search("tenant rights", k=2)
    print(doc_search[0].page_content)

    outputs information on page 28: Tenant’s basic legal rights
    """  

"""
Store: We need somewhere to store and index our splits, so that they can later be searched over. 
Using pinecone
"""
def store(texts, embeddings, k):
    vectorstore = FAISS.from_documents(texts, embeddings) # langchain_community.vectorstores.faiss.FAISS object
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
    return retriever
