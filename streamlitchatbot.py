import streamlit as st
from pdf_vector_store import PDFTextRetriever

def main():
    st.title("PDF Text Retriever and Vector Database")

    openai_api_key = st.text_input("OpenAI API Key", type="password")
    if openai_api_key:
        retriever = PDFTextRetriever(openai_api_key=openai_api_key)

        st.subheader("Ingest PDF")
        pdf_file = st.file_uploader("Upload PDF", type="pdf")
        if pdf_file:
            with open("temp.pdf", "wb") as f:
                f.write(pdf_file.read())
            if st.button("Ingest"):
                with st.spinner("Ingesting..."):
                    result = retriever.ingest("temp.pdf")
                    if result == "Success":
                        st.success("PDF ingested successfully.")
                    else:
                        st.error(result)

        st.subheader("Ask a Question")
        question = st.text_input("Enter your question")
        if st.button("Ask"):
            if retriever.db is None:
                st.warning("Please ingest data first.")
            else:
                with st.spinner("Generating answer..."):
                    answer = retriever.ask(question)
                    st.write(answer)

        if st.button("Clear Data"):
            retriever.clear_data()
            st.success("Data cleared successfully.")

if __name__ == "__main__":
    main()
