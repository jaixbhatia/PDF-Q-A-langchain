from langchain_openai import ChatOpenAI
from langchain import hub
from dotenv import find_dotenv, load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from indexing import indexing

load_dotenv(find_dotenv())

def generate(pdf_path, query, k):
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
    retriever = indexing(pdf_path, k)

    template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use three sentences maximum and keep the answer as concise as possible.
    Always say "thanks for asking!" at the end of the answer.

    {context}

    Question: {question}

    Helpful Answer:"""

    custom_prompt = PromptTemplate.from_template(template)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | custom_prompt
        | llm
        | StrOutputParser()
    )

    response = chain.invoke(query)
    print(response)
   
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)