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
    page_title="LinkedRAG",
    page_icon="💼",
    layout="wide"
)

# Try to auto-load previous index and data
if st.session_state.vector_store is None:
    try:
        _vs = VectorStore()
        if _vs.load_from_disk():
            st.session_state.vector_store = _vs
            st.session_state.connections_data = _vs.connections_data
            st.session_state.chatbot = ChatBot(_vs)
    except Exception:
        pass

st.title("💼 LinkedRAG")
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
                # Set embedding mode and chat backend environment variables
                os.environ["EMBEDDINGS_MODE"] = embedding_mode
                if embedding_mode == "ollama":
                    os.environ["CHAT_MODE"] = "ollama"
                elif embedding_mode == "openai":
                    os.environ["CHAT_MODE"] = "openai"
                
                with st.spinner("Processing connections and creating embeddings..."):
                    try:
                        # Process and create embeddings
                        processed_data = data_processor.process_connections(df)
                        
                        # Initialize vector store
                        vector_store = VectorStore()
                        
                        if embedding_mode == "mock":
                            st.info("ℹ️ Using mock embeddings for testing. Results will be simulated.")
                        elif embedding_mode == "ollama":
                            st.info("🖥️ Using local Ollama for embeddings and chat. Ensure Ollama is running and models are pulled.")
                        
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
    
    # Display connection stats and filters if data is loaded
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
    # Main interface with tabs
    tab1, tab2 = st.tabs(["💬 Chat with Network", "👥 Browse Connections"])
    
    with tab1:
        st.subheader("💬 Ask Questions About Your Network")
        
        # Dynamic suggested queries from chatbot
        st.markdown("**Quick Questions:**")
        try:
            suggested_queries = st.session_state.chatbot.get_suggested_queries()
            if not suggested_queries:
                # Fallback to static templates
                suggested_queries = [
                    "Who in my network is currently hiring?",
                    "Who works at startups?",
                    "Who has marketing experience?"
                ]
        except:
            suggested_queries = [
                "Who in my network is currently hiring?",
                "Who works at startups?", 
                "Who has marketing experience?"
            ]
        
        cols = st.columns(min(4, len(suggested_queries)))
        for i, template in enumerate(suggested_queries[:8]):  # Limit to 8 suggestions
            with cols[i % len(cols)]:
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
                        st.dataframe(table_df, use_container_width=True, hide_index=True)
                    
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
    "💡 **Tip:** This chatbot uses RAG (Retrieval-Augmented Generation) to find relevant connections "
    "and provide accurate answers about your LinkedIn network."
)
