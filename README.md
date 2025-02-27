# RAG Using Pinecone 🚀

## Overview
This project implements a **Retrieval-Augmented Generation (RAG) chatbot** using **Pinecone** for vector storage and **OpenAI's GPT-4o** for answering user queries. The chatbot allows both **text and voice input**, utilizing **Streamlit** for the UI, **FastAPI** for the backend, and **LangChain** for the conversational retrieval chain.

## Features
- **Text & Voice Input:** Users can type or use voice recordings for questions.
- **Conversational Memory:** Maintains chat history for better context-aware responses.
- **Pinecone Vector Database:** Stores and retrieves research papers efficiently.
- **OpenAI GPT-4o Integration:** Provides intelligent answers.
- **Streamlit UI:** User-friendly interface for interaction.
- **FastAPI Backend:** Handles API requests efficiently.
- **Document Processing:** Converts and chunks research documents for indexing.

## Installation
### Prerequisites
Ensure you have Python installed (>=3.8).

### Setup
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```
2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up environment variables:
   - Copy `.env.example` to `.env`:
     ```bash
     cp .env.example .env
     ```
   - Fill in the required API keys (PINECONE, OpenAI, etc.).

## Usage
### Running the Backend
Start the FastAPI server:
```bash
uvicorn server:app --reload
```
This starts the API, which can be accessed at `http://localhost:8000`.

### Running the Frontend
Start the Streamlit app:
```bash
streamlit run main.py
```
This launches a web UI where users can interact with the chatbot.

### Processing Documents
To create a **vector database** from documents:
```python
from utils import create_vector_store
create_vector_store("path_to_documents", "rag-pinecone")
```

## Code Structure
```
📂 Project Root
├── .gitignore               # Ignore unnecessary files
├── .env.example             # Example environment variables
├── requirements.txt         # Dependencies
├── main.py                  # Streamlit app entry point
├── server.py                # FastAPI backend
├── utils.py                 # Utility functions (vector database, document processing, transcription)
└── README.md                # Project documentation
```

## Key Components
### `main.py`
- Loads the **Pinecone Vector Store**.
- Sends queries to the FastAPI backend.
- Handles **text and voice inputs**.
- Displays **chat history and references**.

### `server.py`
- FastAPI backend to handle requests.
- Routes for **text and voice queries**.
- Stores **chat history** in memory.

### `utils.py`
- **Transcribes audio** using **Groq's Whisper model**.
- **Processes documents** and creates Pinecone vector stores.
- **Manages Pinecone indexes**.
- **Handles API communication** with the chatbot.

### `requirements.txt`
Contains necessary dependencies such as:
- `streamlit`, `fastapi`, `uvicorn`, `pinecone`, `langchain`, `groq`, `openai`, `dotenv`.

## Environment Variables
Set up the `.env` file with:
```
PINECONE_API_KEY=<your_api_key>
OPENAI_API_KEY=<your_api_key>
GROQ_API_KEY=<your_api_key>
```  

## Future Improvements
- **Enhanced UI/UX** for a more interactive experience.
- **Multilingual Support** for non-English queries.
- **Improved Speech Recognition** using fine-tuned models.

## License
This project is licensed under the **MIT License**.

## Author
Developed by **Osama Abo-Bakr** 🚀.