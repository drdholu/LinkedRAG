import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from services.data_processor import DataProcessor
from services.vector_store import VectorStore
from services.chatbot import ChatBot
from utils.helpers import format_connection_display, setup_logger, log_error_with_context, create_user_friendly_message

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

# Initialize logger
logger = setup_logger("LinkedRAG", level=os.getenv("LOG_LEVEL", "INFO"))
logger.info("LinkedRAG application started")

st.set_page_config(
    page_title="LinkedRAG",
    page_icon="💼",
    layout="wide"
)

# Custom CSS for better chat UI
st.markdown("""
<style>
    /* Make chat messages more readable */
    .stChatMessage {
        margin-bottom: 1rem;
    }
    
    /* Better spacing for chat input */
    .stChatInput {
        margin-top: 1rem;
    }
    
    /* Improve suggestion buttons */
    .stButton > button {
        height: auto;
        white-space: normal;
        text-align: left;
        padding: 0.5rem 1rem;
    }
    
    /* Better contrast for conversation history */
    .conversation-header {
        background-color: #f0f2f6;
        padding: 0.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Try to auto-load previous index and data
if st.session_state.vector_store is None:
    try:
        logger.info("Attempting to auto-load previous session data")
        _vs = VectorStore()
        if _vs.load_from_disk():
            st.session_state.vector_store = _vs
            st.session_state.connections_data = _vs.connections_data
            _chatbot = ChatBot(_vs)
            st.session_state.chatbot = _chatbot
            logger.info(f"Successfully auto-loaded previous session data with backend: {_chatbot.chat_mode}")
        else:
            logger.info("No previous session data found to auto-load")
    except Exception as e:
        logger.warning(f"Failed to auto-load previous session: {str(e)}")
        # Don't show error to user for auto-load failures

st.title("💼 LinkedRAG")
st.markdown("Your AI assistant for exploring and connecting with your LinkedIn network! Upload your connections and start chatting about who you know.")

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
            logger.info(f"Processing uploaded file: {uploaded_file.name}")
            # Process uploaded file
            data_processor = DataProcessor()
            df = data_processor.load_csv(uploaded_file)

            logger.info(f"Successfully loaded {len(df)} connections from CSV")
            st.success(f"✅ Loaded {len(df)} connections")

            # Display data preview
            with st.expander("📋 Data Preview"):
                st.dataframe(df.head(), width='stretch')
            
            # Embedding mode selection
            embedding_modes = ["openai", "ollama", "mock"]
            default_mode = os.getenv("EMBEDDINGS_MODE", "openai")
            default_index = embedding_modes.index(default_mode) if default_mode in embedding_modes else 0
            embedding_mode = st.selectbox(
                "🔧 Embedding Mode",
                embedding_modes,
                index=default_index,
                help="Choose OpenAI, local Ollama, or Mock for testing"
            )
            
            # Process data button
            if st.button("🔄 Process Data & Create Embeddings", type="primary"):
                logger.info(f"Starting data processing with embedding mode: {embedding_mode}")
                # Set embedding mode and chat backend environment variables
                os.environ["EMBEDDINGS_MODE"] = embedding_mode
                if embedding_mode == "ollama":
                    os.environ["CHAT_MODE"] = "ollama"
                elif embedding_mode == "openai":
                    os.environ["CHAT_MODE"] = "openai"

                with st.spinner("Processing connections and creating embeddings..."):
                    try:
                        logger.info("Processing connections data")
                        # Process and create embeddings
                        processed_data = data_processor.process_connections(df)
                        logger.info(f"Processed {len(processed_data)} connections")

                        # Initialize vector store
                        vector_store = VectorStore()

                        if embedding_mode == "mock":
                            st.info("ℹ️ Using mock embeddings for testing. Results will be simulated.")
                        elif embedding_mode == "ollama":
                            st.info("🖥️ Using local Ollama for embeddings and chat. Ensure Ollama is running and models are pulled.")

                        logger.info("Creating embeddings")
                        vector_store.create_embeddings(processed_data)

                        # Verify vector store was successfully initialized
                        if vector_store.index is None or not processed_data:
                            error_msg = "Failed to process data. Please check your CSV file and try again."
                            logger.error(error_msg)
                            st.error(f"❌ {error_msg}")
                            st.stop()

                        # Initialize chatbot
                        chatbot = ChatBot(vector_store)
                        logger.info(f"Chatbot initialized successfully with backend: {chatbot.chat_mode}")

                        # Store in session state
                        st.session_state.connections_data = processed_data
                        st.session_state.vector_store = vector_store
                        st.session_state.chatbot = chatbot

                        if embedding_mode == "mock":
                            st.success("✅ Data processed with mock embeddings! You can now test the chatbot functionality.")
                        else:
                            st.success("✅ Data processed and embeddings created!")
                        logger.info("Data processing completed successfully")
                        st.rerun()

                    except Exception as e:
                        logger.error(f"Error during data processing: {str(e)}", exc_info=True)
                        error_info = create_user_friendly_message(e, "processing LinkedIn connections data")

                        # Display user-friendly error
                        st.error(f"❌ {error_info['title']}")
                        st.markdown(f"**{error_info['message']}")

                        # Show remediation steps
                        if error_info['remediation']:
                            st.markdown("**What you can do:**")
                            for step in error_info['remediation']:
                                st.markdown(step)

        except Exception as e:
            logger.error(f"Error loading CSV file: {str(e)}", exc_info=True)
            error_info = create_user_friendly_message(e, "loading CSV file")

            st.error(f"❌ {error_info['title']}")
            st.markdown(f"**{error_info['message']}")

            if error_info['remediation']:
                st.markdown("**What you can do:**")
                for step in error_info['remediation']:
                    st.markdown(step)
    
    # Display connection stats and filters if data is loaded
    if st.session_state.connections_data is not None:
        # Show chat status in sidebar too
        if st.session_state.chatbot:
            backend = st.session_state.chatbot.chat_mode.upper()
            if backend == "OLLAMA":
                st.success(f"✅ Chat Ready: Local LLM")
            elif backend == "OPENAI":
                st.info(f"✅ Chat Ready: OpenAI")
            else:
                st.warning(f"✅ Chat Ready: Mock Mode")
        
        st.header("📈 Network Stats")
        data = st.session_state.connections_data
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Connections", len(data))
        with col2:
            companies = [conn.get('company', 'Unknown') for conn in data]
            unique_companies = len(set([c for c in companies if c and c != 'Unknown']))
            st.metric("Companies", unique_companies)
        
        # Advanced Filters
        st.header("🔍 Filters")
        
        # Company filter
        all_companies = sorted(set([conn.get('company', '') for conn in data if conn.get('company', '').strip()]))
        selected_companies = st.multiselect(
            "Filter by Company",
            options=all_companies,
            default=[],
            help="Select companies to filter connections"
        )
        
        # Position/Role filter
        all_positions = sorted(set([conn.get('position', '') for conn in data if conn.get('position', '').strip()]))
        position_filter = st.selectbox(
            "Filter by Position Type",
            options=["All"] + ["Manager/Director", "Engineer/Developer", "Sales/Marketing", "Other"],
            help="Filter by general position category"
        )
        
        # Date range filter
        st.subheader("Connection Date Range")
        date_filter = st.selectbox(
            "Show connections from",
            options=["All time", "Last year", "Last 6 months", "Last 3 months", "Last month"]
        )
        
        # Store filters in session state for use in connection display
        st.session_state.filters = {
            'companies': selected_companies,
            'position_type': position_filter,
            'date_range': date_filter
        }
        
        # Export functionality
        st.header("📤 Export")
        if st.button("📄 Export All Connections to CSV", help="Download all your connections as CSV"):
            try:
                import pandas as pd
                df = pd.DataFrame(data)
                csv_data = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name="linkedin_connections.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Export failed: {str(e)}")

# Main content area
if st.session_state.chatbot is None:
    st.info("👆 Ready to get started? Upload your LinkedIn connections CSV file above and I'll help you explore your network!")
    
    # Show example CSV format
    st.subheader("📝 What Data Do I Need?")
    st.markdown("""
    I work best with your LinkedIn connections CSV export, which should include:
    - **First Name** & **Last Name**: Your connections' names
    - **Email Address**: Contact emails (when available)
    - **Company**: Where they work
    - **Position**: Their job titles
    - **Connected On**: When you connected
    - **URL**: Their LinkedIn profile links
    
    **How to get this:** Go to LinkedIn → Settings & Privacy → Data Privacy → "Get a copy of your data" → Select "Connections" and export as CSV.
    
    Once you upload this, I'll learn about your entire network and be ready to answer questions like "Who do I know at Google?" or "Which of my connections might be hiring?"
    """)

else:
    # Main interface with tabs
    tab1, tab2 = st.tabs(["💬 Chat with Network", "👥 Browse Connections"])
    
    with tab1:
        st.subheader("💬 Chat with Your Network Assistant")
        
        # Show backend status
        if st.session_state.chatbot:
            backend = st.session_state.chatbot.chat_mode.upper()
            if backend == "OLLAMA":
                st.success(f"🖥️ Using Local LLM ({backend})")
            elif backend == "OPENAI":
                st.info(f"☁️ Using Cloud LLM ({backend})")
            else:
                st.warning(f"🧪 Using Mock Mode ({backend})")
        
        # Dynamic suggested queries from chatbot
        st.markdown("**💡 Great questions to get started:**")
        st.markdown("*Just click on any question below, or type your own!*")

        try:
            suggested_queries = st.session_state.chatbot.get_suggested_queries()
            logger.debug(f"Generated {len(suggested_queries)} dynamic query suggestions")

            if suggested_queries and len(suggested_queries) > 0:
                st.info(f"🎯 Perfect! I've analyzed your {len(st.session_state.connections_data)} connections and here are some questions I think you'll find interesting:")
            else:
                logger.info("No dynamic suggestions available, using fallback queries")
                # Fallback to context-aware static templates
                suggested_queries = [
                    "Who in my network might be hiring?",
                    "Show me who I've connected with recently",
                    "Find engineers and developers I know",
                    "Who has marketing or sales experience?"
                ]
        except Exception as e:
            logger.warning(f"Error getting dynamic suggestions: {str(e)}")
            # Fallback to context-aware static templates
            suggested_queries = [
                "Who in my network might be hiring?",
                "Show me who I've connected with recently",
                "Find engineers and developers I know",
                "Who has marketing or sales experience?"
            ]

        # Better organized suggestions with categories
        with st.container():
            # Display main suggestions in a more compact way
            suggestions_to_show = suggested_queries[:6]  # Show fewer but better organized
            
            # Create 2 columns for suggestions
            col1, col2 = st.columns(2)
            
            for i, template in enumerate(suggestions_to_show):
                target_col = col1 if i % 2 == 0 else col2
                with target_col:
                    if st.button(f"💡 {template}", key=f"template_{i}", use_container_width=True, help="Click to ask this question"):
                        st.session_state.current_query = template
                        st.rerun()

        # More suggestions in a cleaner expander
        if len(suggested_queries) > 6:
            with st.expander("🔍 More Question Ideas", expanded=False):
                remaining_suggestions = suggested_queries[6:12]  # Show up to 6 more
                
                for i, template in enumerate(remaining_suggestions):
                    if st.button(f"• {template}", key=f"template_extra_{i+6}", use_container_width=True):
                        st.session_state.current_query = template
                        st.rerun()
                        
                if len(suggested_queries) > 12:
                    st.caption(f"💡 And {len(suggested_queries) - 12} more possibilities! Just ask naturally.")
        
        st.markdown("")  # Add some spacing
        
        # Chat input at the top
        user_query = st.chat_input("Ask me anything about your network... Who do you want to connect with?", key="main_chat_input")
        
        # Display chat history with newest messages at the top (if any)
        if st.session_state.chat_history:
            # Clear chat button (small, unobtrusive)
            col1, col2 = st.columns([5, 1])
            with col2:
                if st.button("🗑️ Clear", help="Clear conversation history", key="clear_chat"):
                    st.session_state.chat_history = []
                    st.rerun()
            
            st.markdown("---")
            
            # Chat container with better spacing
            with st.container():
                # Group messages into conversation pairs (user question + assistant response)
                conversation_pairs = []
                for i in range(0, len(st.session_state.chat_history), 2):
                    if i + 1 < len(st.session_state.chat_history):
                        # Complete pair (user + assistant)
                        user_msg = st.session_state.chat_history[i]
                        assistant_msg = st.session_state.chat_history[i + 1]
                        conversation_pairs.append((user_msg, assistant_msg))
                    else:
                        # Only user message (response pending)
                        user_msg = st.session_state.chat_history[i]
                        conversation_pairs.append((user_msg, None))
                
                # Display pairs in reverse order (newest conversation first)
                for user_msg, assistant_msg in reversed(conversation_pairs):
                    # Show user message first
                    with st.chat_message("user"):
                        st.markdown(f"**You:** {user_msg['content']}")
                    
                    # Then show assistant response (if it exists)
                    if assistant_msg:
                        with st.chat_message("assistant"):
                            st.markdown(assistant_msg['content'])
                    
                    # Add spacing between conversation pairs
                    st.markdown("")
        
        # Handle template selection
        if 'current_query' in st.session_state:
            user_query = st.session_state.current_query
            del st.session_state.current_query
        
        if user_query:
            logger.info(f"Processing chat query: {user_query[:100]}...")
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_query})

            # Get chatbot response with chat history context
            with st.spinner("🤔 Thinking..."):
                try:
                    # Pass chat history for better context
                    response = st.session_state.chatbot.get_response(user_query, st.session_state.chat_history)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    logger.info("Chatbot response generated successfully with history context")
                    
                    # Force a rerun to show the new message immediately
                    st.rerun()
                    
                except Exception as e:
                    logger.error(f"Error getting chatbot response: {str(e)}", exc_info=True)
                    error_info = create_user_friendly_message(e, "generating chatbot response")

                    st.error(f"❌ {error_info['title']}")
                    st.markdown(f"**{error_info['message']}")

                    if error_info['remediation']:
                        st.markdown("**What you can do:**")
                        for step in error_info['remediation']:
                            st.markdown(step)
    
    with tab2:
        st.markdown("### 🔍 Explore Your Professional Network")
        st.markdown("Browse, search, and filter through your LinkedIn connections")
        
        # Top controls bar
        control_col1, control_col2, control_col3 = st.columns([2, 1, 1])
        with control_col1:
            search_term = st.text_input("🔍 Search by name, company, or position", placeholder="Type to search...")
        with control_col2:
            sort_by = st.selectbox("📊 Sort by", ["Name", "Company", "Date Connected", "Position"], key="sort_connections")
        with control_col3:
            view_mode = st.selectbox("👁️ View", ["Table", "Cards", "List"], key="view_mode")
        
        # Apply filters and search
        connections_to_show = st.session_state.connections_data or []
        
        # Apply sidebar filters if they exist
        if hasattr(st.session_state, 'filters'):
            filters = st.session_state.filters
            
            # Company filter
            if filters['companies']:
                connections_to_show = [
                    conn for conn in connections_to_show 
                    if conn.get('company', '') in filters['companies']
                ]
            
            # Position type filter
            if filters['position_type'] != "All":
                position_keywords = {
                    "Manager/Director": ["manager", "director", "head", "lead", "vp", "vice president", "ceo", "cto", "cfo"],
                    "Engineer/Developer": ["engineer", "developer", "programmer", "architect", "software"],
                    "Sales/Marketing": ["sales", "marketing", "business development", "account", "marketing"]
                }
                keywords = position_keywords.get(filters['position_type'], [])
                if keywords:
                    connections_to_show = [
                        conn for conn in connections_to_show
                        if any(keyword in conn.get('position', '').lower() for keyword in keywords)
                    ]
            
            # Date range filter
            if filters['date_range'] != "All time":
                from datetime import datetime, timedelta
                cutoff_days = {
                    "Last month": 30,
                    "Last 3 months": 90,
                    "Last 6 months": 180,
                    "Last year": 365
                }.get(filters['date_range'], 0)
                
                if cutoff_days:
                    cutoff_date = (datetime.now() - timedelta(days=cutoff_days)).strftime('%Y-%m-%d')
                    connections_to_show = [
                        conn for conn in connections_to_show
                        if conn.get('connected_on', '0000-00-00') >= cutoff_date
                    ]
        
        # Apply text search
        if search_term and connections_to_show:
            search_lower = search_term.lower()
            connections_to_show = [
                conn for conn in connections_to_show
                if (search_lower in conn.get('full_name', '').lower() or
                    search_lower in conn.get('company', '').lower() or
                    search_lower in conn.get('position', '').lower())
            ]
        
        # Sort connections
        if connections_to_show:
            if sort_by == "Name":
                connections_to_show.sort(key=lambda x: x.get('full_name', '').lower())
            elif sort_by == "Company":
                connections_to_show.sort(key=lambda x: x.get('company', '').lower())
            elif sort_by == "Date Connected":
                connections_to_show.sort(key=lambda x: x.get('connected_on', '0000-00-00'), reverse=True)
            elif sort_by == "Position":
                connections_to_show.sort(key=lambda x: x.get('position', '').lower())
        
        # Results summary and export
        if connections_to_show:
            # Results summary bar
            summary_col1, summary_col2 = st.columns([2, 1])
            with summary_col1:
                st.markdown(f"**Found {len(connections_to_show)} connections** out of {len(st.session_state.connections_data or [])}")
            with summary_col2:
                # Export filtered results
                if len(connections_to_show) < len(st.session_state.connections_data or []):
                    if st.button("📄 Export Results", help="Export current filtered connections"):
                        try:
                            import pandas as pd
                            df = pd.DataFrame(connections_to_show)
                            csv_data = df.to_csv(index=False)
                            st.download_button(
                                label="Download CSV",
                                data=csv_data,
                                file_name="filtered_connections.csv",
                                mime="text/csv",
                                key="download_filtered"
                            )
                        except Exception as e:
                            st.error(f"Export failed: {str(e)}")
            
            st.divider()
            
            # Pagination controls
            connections_per_page = 12 if view_mode == "Cards" else 20
            total_pages = (len(connections_to_show) + connections_per_page - 1) // connections_per_page
            
            if total_pages > 1:
                page_col1, page_col2, page_col3 = st.columns([1, 2, 1])
                with page_col2:
                    page = st.selectbox("📄 Page", range(1, total_pages + 1), key="page_selector") - 1
            else:
                page = 0
            
            start_idx = page * connections_per_page
            end_idx = min(start_idx + connections_per_page, len(connections_to_show))
            current_page_connections = connections_to_show[start_idx:end_idx]
            
            # Display connections based on view mode
            if view_mode == "Cards":
                # Card view - 3 columns
                for i in range(0, len(current_page_connections), 3):
                    cols = st.columns(3)
                    for j, col in enumerate(cols):
                        if i + j < len(current_page_connections):
                            conn = current_page_connections[i + j]
                            with col:
                                with st.container():
                                    st.markdown(f"**{conn.get('full_name', 'Unknown')}**")
                                    if conn.get('position'):
                                        st.markdown(f"*{conn.get('position')}*")
                                    if conn.get('company'):
                                        st.markdown(f"🏢 {conn.get('company')}")
                                    if conn.get('connected_on'):
                                        st.markdown(f"🤝 {conn.get('connected_on')}")
                                    if conn.get('profile_url'):
                                        st.markdown(f"[LinkedIn Profile]({conn.get('profile_url')})")
                                    st.markdown("---")
                                    
            elif view_mode == "Table":
                # Table view
                import pandas as pd
                display_df = pd.DataFrame(current_page_connections)
                # Select and rename columns for display
                if not display_df.empty:
                    display_columns = []
                    column_mapping = {}
                    if 'full_name' in display_df.columns:
                        display_columns.append('full_name')
                        column_mapping['full_name'] = 'Name'
                    if 'company' in display_df.columns:
                        display_columns.append('company')
                        column_mapping['company'] = 'Company'
                    if 'position' in display_df.columns:
                        display_columns.append('position')
                        column_mapping['position'] = 'Position'
                    if 'connected_on' in display_df.columns:
                        display_columns.append('connected_on')
                        column_mapping['connected_on'] = 'Connected'
                    
                    if display_columns:
                        table_df = display_df[display_columns].rename(columns=column_mapping)
                        st.dataframe(table_df, width='stretch', hide_index=True)
                    
            else:  # List view (default)
                for conn in current_page_connections:
                    with st.container():
                        # Create more compact list display
                        name = conn.get('full_name', 'Unknown')
                        company = conn.get('company', '')
                        position = conn.get('position', '')
                        connected_on = conn.get('connected_on', '')
                        profile_url = conn.get('profile_url', '')
                        
                        # Highlight search terms if applicable
                        if search_term:
                            from utils.helpers import highlight_search_term
                            name = highlight_search_term(name, search_term)
                            company = highlight_search_term(company, search_term)
                            position = highlight_search_term(position, search_term)
                        
                        # Build compact display
                        display_parts = [f"**{name}**"]
                        if position and company:
                            display_parts.append(f"*{position}* at **{company}**")
                        elif position:
                            display_parts.append(f"*{position}*")
                        elif company:
                            display_parts.append(f"**{company}**")
                        
                        if connected_on:
                            display_parts.append(f"🤝 Connected: {connected_on}")
                        if profile_url:
                            display_parts.append(f"[LinkedIn Profile]({profile_url})")
                        
                        st.markdown("  \n".join(display_parts))
                        st.divider()
                        
        else:
            st.info("🔍 No connections found matching your search and filters. Try adjusting your search terms or filters in the sidebar.")

# Footer
st.markdown("---")
st.markdown(
    "💡 **How it works:** I use smart search technology (RAG) to understand your questions and find exactly the right people in your network. "
    "Think of me as your personal LinkedIn network assistant with perfect memory!"
)
