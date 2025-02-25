from dotenv import load_dotenv
from fastapi import HTTPException, FastAPI, UploadFile, File
from typing import List
from utils import transcribe_audio, get_response

# Load environment variables
load_dotenv(override=True)

# Initialize FastAPI app
app = FastAPI(
    debug=True,
    title="RAG Chatbot Using Pinecone",
    description="A chatbot for E-Commerce websites using RAG and Pinecone.",
    version="1.0.0"
)

# In-memory chat history
chat_history: List[tuple] = []

@app.get("/")
async def root():
    """
    Simple root endpoint to check if the API is running.
    """
    return {"message": "Hello World"}


@app.post("/response/text")
async def response_text(user_query: str):
    """
    Handles text-based queries.
    
    - Stores chat history in memory.
    - Calls RAG system to get response.
    """
    try:
        result = get_response(user_query=user_query, chat_history=chat_history)
        
        # Store in chat history
        chat_history.append((user_query, result["answer"]))

        return {"result": result, "chat_history": chat_history}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/response/voice")
async def response_voice(audio: UploadFile = File(...)):
    """
    Handles voice-based queries.
    
    - Transcribes the audio to text.
    - Passes transcribed text to the chatbot.
    - Stores chat history in memory.
    """
    try:
        # Read audio bytes
        audio_bytes = await audio.read()
        
        # Transcribe the audio
        transcribed_text = transcribe_audio(audio_bytes)
        
        # Get chatbot response
        result = get_response(user_query=transcribed_text, chat_history=chat_history)
        
        # Store in chat history
        chat_history.append((transcribed_text, result["answer"]))

        return {
            "transcribed_text": transcribed_text,
            "result": result,
            "chat_history": chat_history
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
