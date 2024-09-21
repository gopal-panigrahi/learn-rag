import streamlit as st
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


class Bot:
    def __init__(self, store):
        self.store = store
        self.llm = AzureChatOpenAI(
            model=st.secrets.model.model_name,
            api_key=st.secrets.model.api_key,
            openapi_api_type=st.secrets.model.api_type,
            azure_endpoint=st.secrets.model.azure_endpoint,
            api_version=st.secrets.model.api_version,
            azure_deployment=st.secrets.model.deployment_name,
        )

    def get_response(self, query):
        context = self.store.similarity_search(query)
        template = f"""
            You are a helpful assisstant. Answer the following user query using the given context.
            You only have access to the given context and answer only from the context.
            If you don't know any answer then just reply with "I don't know", don't make up your answers.

            context: {context}

            user query: {query} 

            Provide the response in the following format:
            Answer: <<answer>>

            Source: <<source of context>>
        """

        prompt = ChatPromptTemplate.from_template(template)

        chain = prompt | self.llm | StrOutputParser()

        return chain.stream({"context": context, "query": query})


@st.cache_resource
def get_bot(_store):
    return Bot(_store)
