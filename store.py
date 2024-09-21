from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from pathlib import Path
import streamlit as st


class Store:
    def __init__(self) -> None:
        self.TEMP_DIR = Path("temp")
        self.TEMP_DIR.mkdir(exist_ok=True)

        self.splitter = RecursiveCharacterTextSplitter(chunk_size=450, chunk_overlap=20)
        embeddings = AzureOpenAIEmbeddings(
            azure_deployment=st.secrets.embedding.deployment_name,
            model=st.secrets.embedding.model_name,
            api_key=st.secrets.embedding.api_key,
            azure_endpoint=st.secrets.embedding.azure_endpoint,
            openai_api_type=st.secrets.embedding.api_type,
            chunk_size=1,
        )

        self.db = Chroma(embedding_function=embeddings, persist_directory="./chroma")

    def add_file(self, uploaded_file):
        files_uploaded = self.get_uploaded_files()
        if uploaded_file.name in files_uploaded:
            return False, "File already uploaded"

        file_path = self.TEMP_DIR / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        loader = PyPDFLoader(file_path)
        pages = loader.load()
        documents = self.splitter.split_documents(pages)

        if len(documents) == 0:
            return False, "Unable to split documents"

        ids = [f"{uploaded_file.name}_{i}" for i in range(len(documents))]

        self.db.add_documents(documents=documents, ids=ids)

        return True, "File uploaded successfully"

    def get_uploaded_files(self):
        return set(
            [
                m.get("file_path").split("\\")[1]
                for m in self.db._collection.get()["metadatas"]
            ]
        )

    def similarity_search(self, query):
        docs = self.db.similarity_search(query, k=3)
        context = ""
        for doc in docs:
            source = doc.metadata.get("source").split("\\")[1]
            context += f"""
                content: {doc.page_content}
                source of context: {source}
            """
        return context


@st.cache_resource
def get_store():
    return Store()
