import streamlit as st
from dotenv import load_dotenv
from utils import transcribe_audio, get_response
from streamlit_mic_recorder import mic_recorder

def main():
    st.set_page_config(page_title="RAG Using Pinecone 🚀", page_icon="🌲")
    st.title("🗣️ Voice & Text Input Chatbot")
    
    load_dotenv()
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

    if user_question:
        pass
    elif audio_data:
        user_question = transcribe_audio(audio_data["bytes"])

    if user_question:
        with st.spinner("Processing your question...", show_time=True):
            result = get_response(user_query=user_question,
                                  chat_history=st.session_state.chat_history)
            
            
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