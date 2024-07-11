#pip install pymupdf langchain openai faiss-cpu

#pip install -U langchain-community

import fitz  # PyMuPDF
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings

class PDFTextRetriever:
    def __init__(self, openai_api_key=None) -> None:
        os.environ["OPENAI_API_KEY"] = openai_api_key
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.db = None
        self.chain = None
        self.embeddings = OpenAIEmbeddings()  # Initialize the embeddings model

    def extract_text_from_pdf(self, pdf_path):
        document = fitz.open(pdf_path)
        text = ""
        for page_num in range(len(document)):
            page = document.load_page(page_num)
            text += page.get_text()
        return text

    def ingest(self, pdf_path: str) -> str:
        text = self.extract_text_from_pdf(pdf_path)
        if not text:
            return "Failed to extract text from the PDF."

        documents = [Document(page_content=text, metadata={"source": pdf_path})]
        splitted_documents = self.text_splitter.split_documents(documents)
        self.db = Chroma.from_documents(splitted_documents, self.embeddings).as_retriever()
        self.chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")
        return "Success"

    def ask(self, question: str) -> str:
        if self.db is None:
            return "Please add some data to the AI assistant first."
        else:
            docs = self.db.similarity_search(question, k=3)
            context = "\n".join([doc.page_content for doc in docs])
            response = self.chain.run(input_documents=docs, question=question)
            return response

    def clear_data(self):
        self.db = None
        self.chain = None

