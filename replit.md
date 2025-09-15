# LinkedRAG

## Overview

This is a Streamlit-based RAG (Retrieval-Augmented Generation) chatbot application that allows users to upload their LinkedIn connections data and interact with it through natural language queries. The system processes CSV files exported from LinkedIn, creates vector embeddings of the connection information, and uses OpenAI's GPT models to answer questions about the user's professional network. Users can ask questions like "Who works at Microsoft?" or "Show me connections in the tech industry" and get relevant responses based on their actual LinkedIn connections data.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Streamlit Framework**: Web-based interface with sidebar for file upload and main area for chat interaction
- **Session State Management**: Maintains chat history, processed data, vector store, and chatbot instances across user sessions
- **Reactive UI Components**: File uploader, data preview expander, and chat interface with message history

### Backend Architecture
- **Service-Oriented Design**: Modular architecture with separate services for data processing, vector operations, and chatbot functionality
- **DataProcessor Service**: Handles CSV file loading, validation, and data cleaning with support for multiple encodings
- **VectorStore Service**: Manages FAISS-based vector similarity search with OpenAI embeddings
- **ChatBot Service**: Implements RAG pipeline using OpenAI GPT models for natural language responses

### Data Processing Pipeline
- **CSV Validation**: Ensures required columns (First Name, Last Name) are present with optional fields (Company, Position, Email, etc.)
- **Data Cleaning**: Removes empty rows, fills NaN values, and standardizes text formatting
- **Embedding Generation**: Creates searchable text representations and converts to vector embeddings using OpenAI's text-embedding-3-large model
- **FAISS Indexing**: Builds efficient similarity search index for fast retrieval

### RAG Implementation
- **Query Analysis**: Processes user questions to determine optimal search strategy
- **Similarity Search**: Uses FAISS to find most relevant connections based on vector similarity
- **Context Generation**: Formats retrieved connection data for GPT model consumption
