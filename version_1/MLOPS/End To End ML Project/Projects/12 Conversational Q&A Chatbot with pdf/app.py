import streamlit as st
from langchain_classic.chains import create_history_aware_retriever,create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_ollama import OllamaEmbeddings,ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_core.runnables import RunnableWithMessageHistory
import uvicorn

embedding=OllamaEmbeddings(model="llama3.1:8b")
llm=ChatOllama(model="phi4:latest")


st.title("Conversational RAG with pdf upload and chat history")
st.write("Upload pdf and chat with content")
session_id=st.text_input("Session ID",value="default_session")
if 'store' not in st.session_state:
    st.session_state.store={}

uploaded_files=st.file_uploader("Choose a pdf file",type="pdf",accept_multiple_files=True)

if uploaded_files:
    documents=[]
    for uploaded_file in uploaded_files:
        temp_pdf=f"./temp.pdf"
        with open(temp_pdf,"wb") as fl:
            fl.write(uploaded_file.getvalue())
            file_name=uploaded_file.name
        loader=PyPDFLoader(temp_pdf)
        docs=loader.load()
        documents.extend(docs)
    

    text_splitter=RecursiveCharacterTextSplitter(chunk_size=5000,chunk_overlap=200)
    splits=text_splitter.split_documents(documents)
    vector_store=Chroma.from_documents(documents=splits,embedding=embedding)
    retriever=vector_store.as_retriever()



    contextualize_q_system_prompt=(
        "Given a chat history and the latest user question"
        "which might reference context in the chat history"
        "formulate a standalone question which can be understood"
        "without the chat history. Do NOT answer the question,"
        "just reformulate it if needed and otherwise return it as is "
    )

    contextualize_q_prompt=ChatPromptTemplate.from_messages(
        [
            ("system",contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human","{input}")
        ]
    )

    history_aware_retriever=create_history_aware_retriever(llm,retriever,contextualize_q_prompt)
    system_prompt=(
        "You are an assistant for question answering tasks."
        "use the following pieces of retrieved context to answer"
        "the question. If you don't know the answer, say that you"
        "don't know. Use three sentences maximum and keep answer concise\n\n"
        "{context}"
    )


    qa_prompt=ChatPromptTemplate.from_messages([
        ("system",system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human","{input}")
    ])


    question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)
    rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)


    def get_session_history(session:str)->BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id]=ChatMessageHistory()
        return st.session_state.store[session_id]

    conversational_rag_chain=RunnableWithMessageHistory(
        rag_chain,get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )


    user_input=st.text_input("Ask your question")
    if user_input:
        session_history=get_session_history(session_id)
        res=conversational_rag_chain.invoke({"input":user_input},config={"configurable":{"session_id":session_id}})
        st.write(st.session_state.store)
        st.write("AI:",res['answer'])
        st.write("Chat History:",session_history.messages)

