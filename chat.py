import os
import streamlit as st
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain, BaseConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from langchain_community.callbacks import get_openai_callback
# configurações streamlit 
st.set_page_config(page_title="Chatbot RAG com FAISS")

# configurações do Azure
AZURE_OPENAI_API_KEY = st.secrets["AZURE_OPENAI_API_KEY"]
AZURE_OPENAI_ENDPOINT =  st.secrets["AZURE_OPENAI_ENDPOINT"]
AZURE_DEPLOYMENT_NAME = st.secrets["AZURE_DEPLOYMENT_NAME"]
AZURE_MODEL_NAME = st.secrets["AZURE_MODEL_NAME"]
AZURE_API_VERSION= st.secrets["AZURE_API_VERSION"]

AZURE_DEPLOYMENT_EMBEDDING_ENDPOINT =  st.secrets["AZURE_DEPLOYMENT_EMBEDDING_ENDPOINT"]
AZURE_DEPLOYMENT_EMBEDDING_NAME = st.secrets["AZURE_DEPLOYMENT_EMBEDDING_NAME"]
AZURE_EMBEDDING_MODEL_NAME = st.secrets["AZURE_EMBEDDING_MODEL_NAME"]
AZURE_EMBEDDING_API_VERSION= st.secrets["AZURE_EMBEDDING_API_VERSION"]
MEMORY_KEY = 'memory_key'
CHAIN_KEY='chain_key'

# Inicialização do modelo LLM
@st.cache_resource
def initialize_llm():
    """ Initialize the llm model from Azure OpenAI """
    return AzureChatOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        azure_deployment=AZURE_DEPLOYMENT_NAME,
        model=AZURE_MODEL_NAME,
        api_version=AZURE_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        openai_api_type='azure'    
    )
    
# Configure FAISS Vector Store
@st.cache_resource
def initialize_vector_store():
    """ Configure the FAISS like a vector database using local documents """
    embeddings = AzureOpenAIEmbeddings(
        api_key=AZURE_OPENAI_API_KEY,
        azure_deployment=AZURE_DEPLOYMENT_EMBEDDING_NAME,
        model=AZURE_EMBEDDING_MODEL_NAME,
        azure_endpoint=AZURE_DEPLOYMENT_EMBEDDING_ENDPOINT,
        api_version=AZURE_EMBEDDING_API_VERSION,
        openai_api_type='azure',
        chunk_size=1
    )
    
    # load documents
    loader = TextLoader('data/documents.txt')
    documents = loader.load()
    
    # divide text in lesser parts
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(documents)
    
    # create the FAISS Vector Store
    return FAISS.from_documents(split_docs, embeddings)

# Generate Prompt Template
@st.cache_resource
def generate_prompt_template():
    PROMPT = """
        Use as informações recuperadas do banco vetorial para responder à pergunta do usuário.
    Se não encontrar informações relevantes, diga que não sabe.
    Contexto:
    {context}
    ---
    Histórico da conversa:
    {chat_history}
    Pergunta do usuário:
    {question}                                                  
    """
    return ChatPromptTemplate.from_template(template=PROMPT)

# retrieve memory to history chain
@st.cache_resource 
def retrive_memory():
    if MEMORY_KEY not in st.session_state:
        st.session_state[MEMORY_KEY] = ConversationBufferMemory(
            return_messages=True,
            memory_key='chat_history',
            input_key='question',
            output_key='answer'
        )
    memory = st.session_state[MEMORY_KEY]
    return memory

# Initialize conversation chain
@st.cache_resource
def initialize_chat_chain():
    """ Initialize the retrieve chain and generate response """
    
    vector_store = initialize_vector_store()
    llm = initialize_llm()
    memory = retrive_memory()
    prompt = generate_prompt_template()
    retriever = vector_store.as_retriever(search_type='similarity', k=10)
    
    chat_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=memory,
        retriever=retriever,
        return_source_documents=True,
        combine_docs_chain_kwargs={'prompt': prompt},
        verbose=True
    ) 
    
    st.session_state[CHAIN_KEY] = chat_chain
    
def main():
    
    # Initialize chatbot
    if CHAIN_KEY not in st.session_state:
        initialize_chat_chain()
    
    memory = st.session_state[MEMORY_KEY]
    messages = memory.load_memory_variables({})['chat_history']
    
    
    # Streamlit interface

    st.title('Chatbot Python Specialist')
    #container to show chat message style
    container = st.container()
    for message in messages:
        chat = container.chat_message(message.type)
        chat.markdown(message.content)

    # User input
    user_input = st.chat_input('Inform your question:', key='user_input')

    if user_input:
        chat = container.chat_message('human')
        chat.markdown(user_input)
        chat = container.chat_message('bot') 
        chat.markdown('thinking...')
        
        chain: BaseConversationalRetrievalChain = st.session_state[CHAIN_KEY]
        with get_openai_callback() as cb:
            chain.invoke({'question': user_input})

        print(f'Total tokens: {cb.total_tokens}')
        print(f'Prompt Tokens: {cb.prompt_tokens}')
        print(f'Completion Tokens: {cb.completion_tokens}')
        print(f'Total Cost (USD): {cb.total_cost}')
        
        st.rerun() 

if __name__ == '__main__':
    main()

            