import os
from groq import Groq
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from docling.document_converter import DocumentConverter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate


load_dotenv()

def create_index(index_name, vect_length=1536):
    """
    Create an index in Pinecone for storing vectors.

    This function deletes all existing indexes and creates a new index
    if it does not already exist. The index is created with the specified
    name and vector length, using the 'cosine' similarity metric.

    Args:
        index_name (str): The name of the index to create.
        vect_length (int, optional): The dimensionality of the vectors. Defaults to 1536.
    """

    try:
        print('Deleting all indexes')
        _ = [pinecone.delete_index(name=index_name['name']) for index_name in pinecone.list_indexes()]
    except Exception as e:
        print('Error In Deleting Indexes: {}'.format(e))
        
    if index_name not in pinecone.list_indexes():
        print('Creating Index: {}'.format(index_name))
        pinecone.create_index(
            name=index_name,
            dimension=vect_length,
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
        print('Done Creating Index: {}'.format(index_name))
        
def loading_data(paths):
    """
    Loads and processes text data from documents in a specified directory.

    This function reads all files in the given directory, converts them into markdown format,
    and splits the text into smaller chunks for further processing.

    Args:
        paths (str): The path to the directory containing the documents.

    Returns:
        list: A list of text chunks after processing and splitting.

    Functionality:
    - Iterates through all files in the given directory.
    - Converts each file to markdown format using `DocumentConverter`.
    - Concatenates the converted text with file names as headers.
    - Splits the final text into chunks using `RecursiveCharacterTextSplitter` with a 
      chunk size of 1000 characters and an overlap of 200 characters.
    """
    converter = DocumentConverter()
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, 
                                              chunk_overlap=200)
    files_list = os.listdir(paths)
    final_data = ""
    
    print("🚩 Start Converting Data")
    for file in files_list:
        file_path = os.path.join(paths, file)
        if os.path.exists(file_path):
            local_doc = converter.convert(file_path)
            result = local_doc.document.export_to_markdown()
            
            final_data += "## " + file + result + "\n\n" + "---" * 20
    print("🏁 Finish Converting Data")
    print("✈️ Start Splitting Data Into Chunks...")
    final_data = splitter.split_text(final_data)
    print("💫 Finish Splitting Data Into Chunks...")
    return final_data


def create_vector_store(data_paths, index_name):
    """
    Creates a vector store in Pinecone from the provided data.

    This function loads data from the specified paths, generates embeddings
    using OpenAIEmbeddings, and creates a Pinecone vector store with the given
    index name.

    Args:
        data_paths (str): The path or paths to the data files.
        index_name (str): The name of the index for the Pinecone vector store.

    Returns:
        PineconeVectorStore: A vector store object created in Pinecone.
    """    
    data = loading_data(data_paths)
    embedding = OpenAIEmbeddings()
    print("🚀 Starting to Create Vector-Database (Pinecone)")
    vector_store = PineconeVectorStore.from_texts(data, embedding, index_name=index_name)
    print("✅ Finish create Vector-Database (Pinecone)")
    return vector_store


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

def get_response(user_query, chat_history):
    index_name = 'rag-pinecone'
    vector_db = PineconeVectorStore(embedding=OpenAIEmbeddings(), index_name=index_name)
    load_qa_chain = create_retriever_chain(vectorstore=vector_db)
    result = load_qa_chain.invoke(
                {
                    "question": user_query,
                    "chat_history": chat_history,
                }
            )
    
    return result
        

def transcribe_audio(audio_bytes):
    client = Groq()
    temp_filename = "temp_audio.wav"
    
    with open(temp_filename, "wb") as f:
        f.write(audio_bytes)
    
    # Open and transcribe
    with open(temp_filename, "rb") as file:
        transcription = client.audio.transcriptions.create(
            file=("audio.wav", file.read()),
            # file=audio_bytes,
            model="whisper-large-v3-turbo",
            response_format="json",
            language="en",
            temperature=0.0
        )
    os.remove(temp_filename)
    
    return transcription.text



pinecone = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
index_name = 'rag-pinecone'