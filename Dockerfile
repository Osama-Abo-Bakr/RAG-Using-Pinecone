# Use official Python image as a parent image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the Streamlit default port
EXPOSE 8501

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PINECONE_API_KEY='pcsk_2v5qt9_EBogTceGY5577v8E1B8ZvoTSGr3Fi4T5bTHTGVzemuwSFURKnktmvCBEHGzZdnD'
ENV GROQ_API_KEY="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"
ENV OPENAI_API_KEY="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"

# Run the Streamlit application
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]