import streamlit as st
from dotenv import load_dotenv
from utils import transcribe_audio
from streamlit_mic_recorder import mic_recorder
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate

def create_retriever_chain(vectorstore):
    """
    Loads a conversational retrieval chain from a vectorstore.

    Given a vectorstore, loads a conversational retrieval chain using the following steps:

    1. Loads a retriever from the vectorstore using FAISS.
    2. Loads an LLM from the OpenAI API.
    3. Creates a PromptTemplate using a template string.
    4. Creates a ConversationalRetrievalChain from the retriever and LLM.

    Args:
        vectorstore (FAISS): The vector store containing the document embeddings.

    Returns:
        ConversationalRetrievalChain: The conversational retrieval chain.
    """
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5})
    llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

    template = """
    You are an AI research assistant specializing in research_field.  
    Your task is to answer questions about the research papers.  

    Use the following context from the paper to provide an accurate response in markdown format (highly structured):  
    {context}  

    Question: {question}  

    Answer the question strictly based on the provided context. If the context is insufficient, state that more information is needed.
    """

    prompt = PromptTemplate(
        input_variables=["context", "question"], 
        template=template
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt, "document_variable_name": "context"},
    )


def main():
    st.set_page_config(page_title="RAG Using Pinecone 🚀", page_icon="🌲")
    st.title("🗣️ Voice & Text Input Chatbot")
    
    load_dotenv()
    if "vector_db" not in st.session_state:
        index_name = 'rag-pinecone'
        st.session_state.vector_db = PineconeVectorStore(embedding=OpenAIEmbeddings(),
                                                         index_name=index_name)
    if "qa_system" not in st.session_state:
        st.session_state.qa_system = create_retriever_chain(st.session_state.vector_db)
        
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    col1, col2 = st.columns([4, 1])  # Adjust ratio as needed

    with col1:
        user_question = st.chat_input("Type your question here...")

    with col2:
        audio_data = mic_recorder(
            start_prompt="🎤",
            stop_prompt="⏹",
            key="voice"
        )

    if audio_data:
        user_question = transcribe_audio(audio_data["bytes"])
        audio_data.clear()

    
    if user_question:
        with st.spinner("Processing your question...", show_time=True):
            result = st.session_state.qa_system.invoke(
                {
                    "question": user_question,
                    "chat_history": st.session_state.chat_history,
                }
            )
        st.session_state.chat_history.append((user_question, result["answer"]))
        
        # Display chat history
        for chat in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(chat[0])

            with st.chat_message("assistant"):
                st.write(chat[1].replace("```markdown", "").replace("```", "").strip())
    
        with st.expander("References from the Paper", icon="📂"):
            for i, doc in enumerate(result["source_documents"], 1):
                st.subheader(f"Reference num: {i+1}")
                st.write(f"{doc.page_content}")
            

if __name__ == "__main__":
    main()