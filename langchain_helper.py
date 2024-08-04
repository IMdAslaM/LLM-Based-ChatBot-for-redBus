import os
from dotenv import load_dotenv #to access api key in .env file
load_dotenv() #it will set api key as operating system's env variable
from langchain.llms import GooglePalm
from langchain_community.embeddings import HuggingFaceInstructEmbeddings #this is to do a vector embedding
#from InstructorEmbedding import INSTRUCTOR
#from torch import Tensor, device
from langchain.vectorstores import FAISS #FAISS is a vector database
from langchain.document_loaders import CSVLoader
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

llm = GooglePalm(google_api_key=os.environ["GOOGLE_API_KEY"],temperature=0.4)
print(llm('who is virat kohli?'))

instructor_embeddings = HuggingFaceInstructEmbeddings()
vectordb_file_path="faiss_index"

def create_vector_db():
#Creating a vector database using FAISS
    source_column='ï»¿question'
    loader= CSVLoader(file_path=r"redBus.csv", source_column=source_column.strip())
    docs=loader.load()
    vectordb=FAISS.from_documents(documents=docs,embedding=instructor_embeddings)
    vectordb.save_local(vectordb_file_path)#saving a vectordb in a file to access it later

def get_qa_chain():
    #loading vector database from local
    vectordb= FAISS.load_local(vectordb_file_path, instructor_embeddings,allow_dangerous_deserialization=True) 

    retriever = vectordb.as_retriever() #used to pick similar looking embeddings from vectordb

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""


    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        input_key="query",
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt":PROMPT})
    return chain

if __name__ == "__main__":
    #create_vector_db() #we have to create Vector database when we are updating the CSV file, Since already we created Vectordb no need to create once more
    chain= get_qa_chain()
    # print(chain("do you provide refund for cancellation?"))

