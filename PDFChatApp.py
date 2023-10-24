from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
import os
from vars import openAI_key
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
import shutil
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

def main():

    current_directory = os.getcwd()

    st.title("Talk to your PDF")

    c = st.empty()

    # Uploading File
    pdf = st.file_uploader("Upload PDF", type="pdf")

    if pdf is not None:
        tmp_location = os.path.join(f'{current_directory}/pdf', pdf.name)
        with open(tmp_location, "wb") as f:
            f.write(pdf.read())

        c.write("Uploading PDF...")
        
        loader = PyPDFLoader(tmp_location)
        data = loader.load()


        # Split file and upload embeddings to vector store
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 200
        )
        chunks = text_splitter.split_documents(data)

        embeddings = OpenAIEmbeddings()
        DB_DIRECTORY = f"{current_directory}/vectorStoreDB"

        if os.path.exists(DB_DIRECTORY):
            shutil.rmtree(DB_DIRECTORY)

        total_length = len(chunks)
        batch_size = 64 

        # Batch upload is for limitations with my OpenAIEmbeddings, can do regular loading 
        for batch_start in range(0, total_length, batch_size):
            batch_end = min(batch_start + batch_size, total_length)
            batch_texts = chunks[batch_start:batch_end]
            Chroma.from_documents(documents=batch_texts, embedding=embeddings, persist_directory=DB_DIRECTORY)
            print(f"Inserted {batch_end}/{total_length} chunks")

        

        vectordb = Chroma(persist_directory=DB_DIRECTORY, embedding_function=embeddings)
        retriever = vectordb.as_retriever()

        query = st.text_input("Talk to PDF:")
        if query:
            llm = OpenAI(temperature=0.6)

            # Create PromptTemplate
            template = """
            Use the following document to answer the question.
            Document: {context}
            Question: {question}
            """
            prompt = PromptTemplate(
                input_variables=["context", "question"],
                template=template
            )

            # Create and run chain
            chain_type_kwargs = {"prompt": prompt}
            qa = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                chain_type_kwargs=chain_type_kwargs,
                return_source_documents=True,
            )
            result = qa({"query": query})

            print(result)

            st.write(result["result"])

            pages = [i.metadata["page"] for i in result["source_documents"]]

            result_string = ', '.join(map(lambda i: str(i + 1), pages))
            st.write("Pages: ", result_string)


if __name__ == "__main__":
    main()
