from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import os
from PyPDF2 import PdfReader
from vars import openAI_key
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
import streamlit as st


def main():
    os.environ["OPENAI_API_KEY"] = openAI_key
    st.title("Talk to your PDF")

    pdf = st.file_uploader("Upload PDF", type="pdf")

    if pdf is not None:
        # Have to use PdfReader to save memory
            # Can spare memory to get metadata - https://stackoverflow.com/questions/76675978/how-can-i-use-langchain-document-loaders-pypdfloader-for-pdf-documents-uploade
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 200
        )
        chunks = text_splitter.split_text(text)
        # st.write(chunks)

        embeddings = OpenAIEmbeddings()

        vs = Chroma.from_texts(chunks, embedding=embeddings)
        # st.write(vs.get(include=["embeddings"]))

        query = st.text_input("Talk to PDF:")
        if query:

            llm = OpenAI(temperature=0.6)

            prompt_template = """
            Use the following pieces of context to respond to the statement/question.
            If you don't know the answer, say you don't know, don't try to make up an answer.

            {context}

            Statement/Question: {question}
            """
            prompt = PromptTemplate(
                input_variables=["context", "question"],
                template=prompt_template,

            )

            chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=prompt)

            docs = vs.similarity_search(query=query)
            answer = chain({"input_documents": docs, "question": query}, return_only_outputs=True)

            st.write(answer["output_text"])


if __name__ == "__main__":
    main()