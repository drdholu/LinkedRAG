import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from services.data_processor import DataProcessor
from services.vector_store import VectorStore
from services.chatbot import ChatBot
from utils.helpers import format_connection_display

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'connections_data' not in st.session_state:
    st.session_state.connections_data = None
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = None

load_dotenv()

st.set_page_config(
    page_title="LinkedIn Connections RAG Chatbot",
    page_icon="💼",
    layout="wide"
)

st.title("💼 LinkedIn Connections RAG Chatbot")
st.markdown("Upload your LinkedIn connections data and ask questions about your network!")

# Sidebar for data upload and management
with st.sidebar:
    st.header("📊 Data Management")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload LinkedIn Connections CSV",
        type=['csv'],
        help="Export your LinkedIn connections as CSV from LinkedIn"
    )
    
    if uploaded_file is not None:
        try:
            # Process uploaded file
            data_processor = DataProcessor()
            df = data_processor.load_csv(uploaded_file)
            
            st.success(f"✅ Loaded {len(df)} connections")
            
            # Display data preview
            with st.expander("📋 Data Preview"):
                st.dataframe(df.head(), use_container_width=True)
            
            # Embedding mode selection
            embedding_mode = st.selectbox(
                "🔧 Embedding Mode",
                ["openai", "mock"],
                index=0 if os.getenv("EMBEDDINGS_MODE", "openai") == "openai" else 1,
                help="Choose OpenAI for production or Mock for testing without API quota"
            )
            
            # Process data button
            if st.button("🔄 Process Data & Create Embeddings", type="primary"):
                # Set embedding mode environment variable
                os.environ["EMBEDDINGS_MODE"] = embedding_mode
                
                with st.spinner("Processing connections and creating embeddings..."):
                    try:
                        # Process and create embeddings
                        processed_data = data_processor.process_connections(df)
                        
                        # Initialize vector store
                        vector_store = VectorStore()
                        
                        if embedding_mode == "mock":
                            st.info("ℹ️ Using mock embeddings for testing. Results will be simulated.")
                        
                        vector_store.create_embeddings(processed_data)
                        
                        # Verify vector store was successfully initialized
                        if vector_store.index is None or not processed_data:
                            st.error("❌ Failed to process data. Please check your CSV file and try again.")
                            st.stop()
                        
                        # Initialize chatbot
                        chatbot = ChatBot(vector_store)
                        
                        # Store in session state
                        st.session_state.connections_data = processed_data
                        st.session_state.vector_store = vector_store
                        st.session_state.chatbot = chatbot
                        
                        if embedding_mode == "mock":
                            st.success("✅ Data processed with mock embeddings! You can now test the chatbot functionality.")
                        else:
                            st.success("✅ Data processed and embeddings created!")
                        st.rerun()
                        
                    except Exception as e:
                        error_msg = str(e)
                        if "insufficient_quota" in error_msg.lower():
                            st.error("❌ OpenAI API Quota Exhausted")
                            st.markdown("""
                            **Your OpenAI API key has insufficient quota.** Here's what you can do:
                            
                            1. **Check your OpenAI billing**: Visit [OpenAI Platform](https://platform.openai.com/account/billing) to add credits
                            2. **Try Mock Mode**: Select "mock" from the dropdown above to test the app without using OpenAI
                            3. **Rotate API Key**: Use a different OpenAI API key with available quota
                            
                            Mock mode will simulate embeddings and let you test all features except the actual AI responses.
                            """)
                        else:
                            st.error(f"❌ Error processing data: {error_msg}")
        
        except Exception as e:
            st.error(f"❌ Error loading CSV: {str(e)}")
    
    # Display connection stats if data is loaded
    if st.session_state.connections_data is not None:
        st.header("📈 Network Stats")
        data = st.session_state.connections_data
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Connections", len(data))
        with col2:
            companies = [conn.get('company', 'Unknown') for conn in data]
            unique_companies = len(set([c for c in companies if c and c != 'Unknown']))
            st.metric("Companies", unique_companies)

# Main content area
if st.session_state.chatbot is None:
    st.info("👆 Please upload your LinkedIn connections CSV file and process the data to start chatting!")
    
    # Show example CSV format
    st.subheader("📝 Expected CSV Format")
    st.markdown("""
    Your LinkedIn connections CSV should contain columns like:
    - **First Name**: Contact's first name
    - **Last Name**: Contact's last name 
    - **Email Address**: Contact's email (if available)
    - **Company**: Current company
    - **Position**: Job title/position
    - **Connected On**: Date connected
    - **URL**: LinkedIn profile URL
    
    You can export this data from LinkedIn by going to Settings & Privacy > Data Privacy > Get a copy of your data.
    """)

else:
    # Chat interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Chat with your Network")
        
        # Pre-built query templates
        st.markdown("**Quick Questions:**")
        query_templates = [
            "Who in my network is currently hiring?",
            "Who works at Google?",
            "Who has experience in artificial intelligence?",
            "Who are my connections in San Francisco?",
            "Who works in software engineering?",
            "Who are my most recent connections?",
            "Who works at startups?",
            "Who has marketing experience?"
        ]
        
        cols = st.columns(4)
        for i, template in enumerate(query_templates):
            with cols[i % 4]:
                if st.button(template, key=f"template_{i}"):
                    st.session_state.current_query = template
        
        # Chat input
        user_query = st.chat_input("Ask about your LinkedIn connections...")
        
        # Handle template selection
        if 'current_query' in st.session_state:
            user_query = st.session_state.current_query
            del st.session_state.current_query
        
        if user_query:
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            
            # Get chatbot response
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.chatbot.get_response(user_query)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Error getting response: {str(e)}")
        
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    with col2:
        st.subheader("👥 Your Connections")
        
        # Search connections
        search_term = st.text_input("🔍 Search connections", placeholder="Search by name, company, or position...")
        
        # Filter connections based on search
        connections_to_show = st.session_state.connections_data or []
        if search_term and connections_to_show:
            connections_to_show = [
                conn for conn in connections_to_show
                if search_term.lower() in str(conn).lower()
            ]
        
        # Display connections
        if connections_to_show:
            st.markdown(f"**Showing {len(connections_to_show)} connections**")
            
            # Pagination
            connections_per_page = 10
            total_pages = (len(connections_to_show) + connections_per_page - 1) // connections_per_page
            
            if total_pages > 1:
                page = st.selectbox("Page", range(1, total_pages + 1)) - 1
            else:
                page = 0
            
            start_idx = page * connections_per_page
            end_idx = min(start_idx + connections_per_page, len(connections_to_show))
            
            for conn in connections_to_show[start_idx:end_idx]:
                with st.container():
                    st.markdown(format_connection_display(conn))
                    st.divider()
        else:
            st.info("No connections found matching your search.")

# Footer
st.markdown("---")
st.markdown(
    "💡 **Tip:** This chatbot uses RAG (Retrieval-Augmented Generation) to find relevant connections "
    "and provide accurate answers about your LinkedIn network."
)
