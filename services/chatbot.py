import os
from openai import OpenAI
from typing import List, Dict, Any
import json
import re
import requests
import logging
from utils.helpers import setup_logger

class ChatBot:
    """RAG-powered chatbot for querying LinkedIn connections"""
    
    def __init__(self, vector_store):
        # Initialize logger
        self.logger = setup_logger("LinkedRAG.ChatBot", level=os.getenv("LOG_LEVEL", "INFO"))

        # Chat backends: ollama (local), openai, mock
        # Auto-detect local LLM first, then fall back to other options
        self.chat_mode = self._detect_best_backend()
        self.logger.info(f"✅ ChatBot initialized with backend: {self.chat_mode.upper()}")
        
        self.vector_store = vector_store
        self.use_mock = self.chat_mode == 'mock'
        
        # Ollama config (local LLM)
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip('/')
        self.ollama_model = os.getenv("OLLAMA_CHAT_MODEL", "llama3.1")
        
        # OpenAI config (only initialize if needed)
        if self.chat_mode == 'openai':
            # the newest OpenAI model is "gpt-5" which was released August 7, 2025.
            # do not change this unless explicitly requested by the user
            self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.model = "gpt-5"
        else:
            self.openai_client = None
            self.model = None
        
        # System prompt for the chatbot
        self.system_prompt = """You are a knowledgeable and friendly personal LinkedIn network assistant. You speak conversationally and naturally, like a trusted colleague who knows their professional network inside and out.

        Your personality:
        - Warm, helpful, and enthusiastic about professional networking
        - You remember previous conversations and build upon them naturally
        - You provide insights and suggestions, not just raw data
        - You're proactive in offering follow-up information that might be useful

        IMPORTANT: Your response should be based on the CONTEXT provided to you:

        1. If you receive NETWORK CONNECTION DATA in the context:
           - Answer about their LinkedIn network and connections
           - Start responses naturally ("Looking at your network...", "I found...", "You have several...")
           - Present information conversationally, weaving details into natural sentences
           - Reference previous conversation when relevant
           - Offer insights about their network and suggest networking opportunities
           - Use their actual connection data - no fabrication

        2. If you receive ASSISTANT CONTEXT (about your capabilities):
           - Answer questions about what you are and what you can do
           - Explain your role as their LinkedIn network assistant
           - Be warm and welcoming in your introduction
           - Suggest ways they can explore their network

        3. If the context indicates NO MATCHES were found for a network query:
           - Be encouraging and helpful
           - Suggest alternative ways to search their network
           - Offer to help them explore their connections differently

        Always respond naturally and conversationally. You're their professional network advisor with perfect memory."""

        # Centralized patterns and keyword sets for reuse
        self.company_patterns = [
            r'works? at (.+?)(?:\?|$|\.)',
            r'from (.+?)(?:\?|$|\.)',
            r'at (.+?)(?:\?|$|\.)',
            r'in (.+?)(?:\?|$|\.)'
        ]
        # Pre-compile regex patterns once
        self._company_regexes = [re.compile(p) for p in self.company_patterns]

        self.hiring_keywords = {"hiring", "recruit", "recruiting", "recruiter", "job", "jobs", "opening", "openings", "position", "positions"}
        self.role_keywords_engineering = {"engineer", "software", "developer", "programming", "architect"}
        self.business_keywords = {"marketing", "sales", "business"}
        self.hiring_positions = {"manager", "director", "recruiter", "hr", "talent", "people"}

        # Heuristics disabled: retrieval is strictly embeddings-based
        self.enable_heuristics = False

    def _detect_best_backend(self) -> str:
        """Auto-detect the best available backend, prioritizing local LLM"""
        # Check if user explicitly set a mode
        explicit_mode = os.getenv("CHAT_MODE")
        if explicit_mode:
            self.logger.info(f"Using explicitly set chat mode: {explicit_mode}")
            return explicit_mode.lower()
        
        # Try to detect Ollama (local LLM) first
        try:
            ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip('/')
            response = requests.get(f"{ollama_url}/api/tags", timeout=3)
            if response.status_code == 200:
                self.logger.info("Local Ollama server detected - using local LLM")
                return "ollama"
        except Exception:
            self.logger.debug("Ollama not detected locally")
        
        # Check if OpenAI API key is available
        if os.getenv("OPENAI_API_KEY"):
            self.logger.info("OpenAI API key found - using OpenAI")
            return "openai"
        
        # Fall back to mock mode
        self.logger.info("No LLM backend available - using mock mode")
        return "mock"

    def _contains_any(self, text_lower: str, keywords: set) -> bool:
        return any(kw in text_lower for kw in keywords)

    def _extract_company_from_query(self, query_lower: str) -> str:
        """Extract a probable company name from the query using shared patterns."""
        for regex in self._company_regexes:
            match = regex.search(query_lower)
            if match:
                company_name = match.group(1).strip()
                company_name = re.sub(r'\b(company|corp|inc|ltd|llc)\b', '', company_name).strip()
                if company_name:
                    return company_name
        return ""

    def _filter_connections_by_company(self, connections, company_lower: str):
        return [c for c in connections if company_lower in c.get('company', '').lower()]

    def _filter_connections_by_positions(self, connections, positions: set):
        return [c for c in connections if any(pos in c.get('position', '').lower() for pos in positions)]
    
    def get_response(self, user_query: str, chat_history: List[Dict[str, str]] = None) -> str:
        """Get chatbot response using RAG with conversation context - LLM handles everything"""
        try:
            self.logger.info(f"Processing chat query with history context: {len(chat_history or [])} previous messages")

            # Determine if this looks like a network query or a meta/general query
            if self._is_likely_network_query(user_query):
                # Search for relevant connections
                search_results = self._retrieve_connections(user_query)
                if search_results:
                    # Create context from search results
                    context = self._create_context(search_results, user_query)
                    # Generate response with network context
                    response = self._generate_response(user_query, context, search_results, chat_history)
                else:
                    # No connections found, but still let LLM handle it with network context
                    context = self._create_empty_network_context()
                    response = self._generate_response(user_query, context, [], chat_history)
            else:
                # For meta/general queries, provide assistant context instead of network context
                context = self._create_assistant_context()
                response = self._generate_response(user_query, context, [], chat_history)

            self.logger.info("Chatbot response generated with history context")
            return response

        except Exception as e:
            self.logger.error(f"Error in get_response: {str(e)}", exc_info=True)
            return f"I'm sorry, I encountered an error while processing your question: {str(e)}"
    
    def _is_likely_network_query(self, query: str) -> bool:
        """Simple heuristic to determine if query is about the network vs about the assistant"""
        query_lower = query.lower().strip()
        
        # Clear network indicators
        network_indicators = [
            "who works", "who do I know", "who has", "find people", "show me people",
            "connections", "network", "hiring", "recruiter", "manager", "engineer",
            "company", "startup", "works at", "experience in", "people in"
        ]
        
        # Clear meta/assistant indicators  
        meta_indicators = [
            "who are you", "what are you", "what can you do", "how do you work",
            "help", "instructions", "capabilities", "introduce yourself"
        ]
        
        # Check for meta indicators first
        if any(indicator in query_lower for indicator in meta_indicators):
            return False
            
        # Check for network indicators
        if any(indicator in query_lower for indicator in network_indicators):
            return True
            
        # For greetings and thanks, treat as meta
        greeting_indicators = ["hello", "hi", "hey", "thank", "bye", "goodbye"]
        if any(indicator in query_lower for indicator in greeting_indicators):
            return False
            
        # Default to network query for anything else
        return True

    def _create_assistant_context(self) -> str:
        """Create context about the assistant for meta queries"""
        stats = self.vector_store.get_connection_stats()
        total_connections = stats.get('total_connections', 0) if stats else 0
        
        context = f"""You are a helpful LinkedIn network assistant. Here's what you should know about yourself:

ABOUT YOU:
- You are a personal LinkedIn network assistant that helps users explore their professional connections
- You have access to the user's LinkedIn network data ({total_connections} connections)
- You can search, analyze, and provide insights about their professional network
- You speak conversationally and naturally, like a trusted colleague
- You help with networking, job searching, reconnecting, and discovering opportunities

YOUR CAPABILITIES:
- Smart search through LinkedIn connections by company, role, skills, experience
- Identify hiring opportunities and recruiters in their network  
- Help reconnect with old colleagues and find interesting people to reach out to
- Provide insights and analysis about their professional network
- Answer questions naturally using their actual connection data

IMPORTANT: If someone asks who you are, what you do, or about your capabilities, explain this in a warm, conversational way. Don't search their network data for these types of questions."""
        
        return context

    def _create_empty_network_context(self) -> str:
        """Create context when no network connections are found"""
        stats = self.vector_store.get_connection_stats()
        total_connections = stats.get('total_connections', 0) if stats else 0
        
        context = f"""The user asked about their LinkedIn network, but no specific connections were found matching their query.

NETWORK CONTEXT:
- Total connections in their network: {total_connections}
- No specific people matched their current search query
- They are asking about their professional network connections

Please respond helpfully by acknowledging that you didn't find specific matches, but offer to help them search differently or suggest related ways to explore their network. Be encouraging and provide alternative search suggestions based on their network size."""
        
        return context

    def _retrieve_connections(self, query: str) -> List[Dict[str, Any]]:
        """Primary retrieval path: embeddings-first similarity search on the full knowledge base.
        """
        # Embeddings-based retrieval
        results = self.vector_store.similarity_search(query, k=10)
        relevant_results = [(conn, score) for conn, score in results if score > 0.2]
        if relevant_results:
            return [conn for conn, _ in relevant_results]
        return []

    def _get_relevant_connections(self, query: str) -> List[Dict[str, Any]]:
        """Deprecated: kept for backward compatibility; use _retrieve_connections instead."""
        return self._retrieve_connections(query)
    
    def _create_context(self, connections: List[Dict[str, Any]], query: str) -> str:
        """Create natural, conversational context from relevant connections"""
        if not connections:
            return "No relevant connections found in your network."
        
        context_parts = ["Here are the most relevant people from your LinkedIn network for this query:\n"]
        
        for i, conn in enumerate(connections[:15], 1):  # Limit to top 15 results
            # Create a more natural, flowing description of each connection
            name = conn['full_name']
            company = conn.get('company', '')
            position = conn.get('position', '')
            connected_on = conn.get('connected_on', '')
            email = conn.get('email', '')
            
            # Build a natural description
            description = f"{name}"
            
            if position and company:
                description += f" works as {position} at {company}"
            elif position:
                description += f" works as {position}"
            elif company:
                description += f" works at {company}"
            
            if connected_on:
                description += f" (connected {connected_on})"
            
            if email:
                description += f" [Email: {email}]"
            
            context_parts.append(f"• {description}")
        
        # Add some helpful context about the network
        context_parts.append(f"\nTotal connections shown: {len(connections[:15])}")
        if len(connections) > 15:
            context_parts.append(f"Note: Showing top 15 most relevant out of {len(connections)} matches")
        
        return "\n".join(context_parts)
    
    def _generate_mock_response(self, query: str, connections: List[Dict[str, Any]], chat_history: List[Dict[str, str]] = None) -> str:
        """Generate mock response by simulating LLM behavior with simple templates"""
        # In mock mode, we still use simple template responses, but they should be more natural
        # This is just for testing when APIs aren't available
        
        if not connections:
            return "I couldn't find anyone in your network matching that query, but I'd be happy to help you search in a different way. What specific aspect of your network would you like to explore?"

        # Simple mock response for demonstration
        if len(connections) == 1:
            conn = connections[0]
            name = conn['full_name']
            company = conn.get('company', '')
            position = conn.get('position', '')
            
            if company and position:
                return f"I found {name} in your network - they work as {position} at {company}. They might be exactly who you're looking for!"
            else:
                return f"I found {name} in your network who could be relevant to your query."
        else:
            names = [c['full_name'] for c in connections[:2]]
            return f"Looking through your network, I found several people including {' and '.join(names)} who seem relevant. Would you like me to tell you more about any of them?"

    def _generate_mock_assistant_response(self, query: str) -> str:
        """Generate mock response for assistant/meta queries"""
        query_lower = query.lower()
        
        if any(phrase in query_lower for phrase in ["who are you", "what are you"]):
            return "Hi! I'm your personal LinkedIn network assistant. I help you explore and search through your professional connections. You can ask me things like 'Who works at Google?' or 'Which of my connections might be hiring?' - just talk to me naturally!"
        elif "what can you do" in query_lower:
            return "I can help you explore your LinkedIn network in many ways! I can find people by company, role, or skills. I can identify hiring opportunities, help you reconnect with old colleagues, and provide insights about your professional network. Just ask me naturally about your connections!"
        elif any(phrase in query_lower for phrase in ["hello", "hi", "hey"]):
            return "Hello! I'm excited to help you explore your LinkedIn network. What would you like to know about your connections?"
        else:
            return "I'm your LinkedIn network assistant! I'm here to help you discover and explore your professional connections. What would you like to know about your network?"
    
    def _generate_response(self, query: str, context: str, connections: List[Dict[str, Any]], chat_history: List[Dict[str, str]] = None) -> str:
        """Generate response using selected backend with chat history context and append citations."""
        self.logger.debug(f"Generating response with {len(chat_history or [])} chat history messages")

        # If using mock mode, generate deterministic response based on context type
        if self.use_mock:
            if "You are a helpful LinkedIn network assistant" in context:
                # This is assistant context - handle meta queries
                answer = self._generate_mock_assistant_response(query)
            else:
                # This is network context 
                answer = self._generate_mock_response(query, connections, chat_history)
            citations = self._format_citations(connections)
            return f"{answer}\n\n{citations}" if citations else answer

        # Build messages with conversation context
        messages = [{"role": "system", "content": self.system_prompt}]

        # Add conversation history for context (limit to last 10 exchanges to avoid token limits)
        if chat_history:
            history_limit = 10
            recent_history = chat_history[-history_limit:] if len(chat_history) > history_limit else chat_history

            for msg in recent_history:
                messages.append({"role": msg["role"], "content": msg["content"]})

        # Add current context and question - let LLM determine how to respond based on context type
        messages.append({"role": "user", "content": f"Context:\n{context}\n\nUser Question: {query}"})

        if self.chat_mode == 'ollama':
            try:
                answer = self._generate_with_ollama(messages)
                citations = self._format_citations(connections)
                return f"{answer}\n\n{citations}" if citations else answer
            except Exception as e:
                # Fallback to mock on failure
                if connections:
                    return self._generate_mock_response(query, connections, chat_history)
                raise Exception(f"Error generating response with Ollama: {str(e)}")

        # Default: OpenAI (if available)
        if self.openai_client:
            try:
                response = self.openai_client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=1000,
                    temperature=0.7
                )
                answer = response.choices[0].message.content
                citations = self._format_citations(connections)
                return f"{answer}\n\n{citations}" if citations else answer
            except Exception as e:
                # Fall back to mock on OpenAI failure
                self.logger.warning(f"OpenAI failed, falling back to mock: {str(e)}")
                if "You are a helpful LinkedIn network assistant" in context:
                    answer = self._generate_mock_assistant_response(query)
                else:
                    answer = self._generate_mock_response(query, connections, chat_history)
                citations = self._format_citations(connections)
                return f"{answer}\n\n{citations}" if citations else answer
        else:
            # No OpenAI client available, use mock mode
            if "You are a helpful LinkedIn network assistant" in context:
                answer = self._generate_mock_assistant_response(query)
            else:
                answer = self._generate_mock_response(query, connections, chat_history)
            citations = self._format_citations(connections)
            return f"{answer}\n\n{citations}" if citations else answer

    def _generate_with_ollama(self, messages: List[Dict[str, str]]) -> str:
        """Generate chat response using local Ollama server."""
        url = f"{self.ollama_base_url}/api/chat"
        payload = {
            "model": self.ollama_model,
            "messages": messages,
            "stream": False  # Get single response instead of streaming
        }
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        
        try:
            data = resp.json()
            # Check for the expected response format
            if isinstance(data, dict) and 'message' in data:
                if isinstance(data['message'], dict) and 'content' in data['message']:
                    return data['message']['content']
            
            # Fallback for other possible formats
            if isinstance(data, dict) and 'response' in data:
                return data['response']
                
            self.logger.error(f"Unexpected Ollama response format: {data}")
            raise Exception(f"Unexpected Ollama response format: {type(data)}")
            
        except Exception as e:
            self.logger.error(f"Error parsing Ollama response: {str(e)}")
            self.logger.error(f"Raw response text: {resp.text[:500]}...")
            raise Exception(f"Error parsing Ollama response: {str(e)}")

    def _format_citations(self, connections: List[Dict[str, Any]]) -> str:
        if not connections:
            return ""
        lines = ["Sources:"]
        for conn in connections[:5]:
            name = conn.get('full_name', 'Unknown')
            company = conn.get('company', '')
            url = conn.get('profile_url', '')
            if url:
                lines.append(f"- {name} — {company} — {url}")
            else:
                lines.append(f"- {name} — {company}")
        return "\n".join(lines)
    

    def _extract_topic_from_query(self, query: str) -> str:
        """Extract the main topic from a user query for context"""
        query_lower = query.lower()

        # Company mentions
        company_name = self._extract_company_from_query(query_lower)
        if company_name:
            return company_name

        # Role/skill mentions
        if self._contains_any(query_lower, {"engineer", "developer", "manager", "director"}):
            return "technical roles"
        if self._contains_any(query_lower, self.business_keywords):
            return "business roles"
        if self._contains_any(query_lower, self.hiring_keywords):
            return "hiring opportunities"

        return ""
    
    def get_suggested_queries(self) -> List[str]:
        """Get natural, conversational suggested queries based on the data"""
        if not self.vector_store.connections_data:
            return []

        self.logger.debug("Generating dynamic query suggestions")

        suggestions = []

        try:
            # Get network statistics
            stats = self.vector_store.get_connection_stats()
            companies = self.vector_store.get_all_companies()

            # Hiring-related queries (more conversational)
            if stats.get('total_connections', 0) > 0:
                suggestions.append("Who in my network might be hiring right now?")
                suggestions.append("Can you find recruiters and hiring managers I know?")

            # Recent connections (more natural)
            suggestions.append("Show me who I've connected with recently")
            suggestions.append("Who are my newest connections?")

            # Company-specific queries (more conversational)
            if companies:
                top_companies = companies[:2]  # Reduce to avoid overwhelming
                for company in top_companies:
                    suggestions.append(f"Who do I know at {company}?")
                suggestions.append("Which companies have the most of my connections?")

            # Role-based queries (more natural language)
            suggestions.append("Find engineers and developers in my network")
            suggestions.append("Who are the managers and leaders I know?")
            suggestions.append("Show me people in sales and marketing")
            suggestions.append("Who has experience in product management?")

            # Industry and experience-based
            suggestions.append("Who works at startups I should know about?")
            suggestions.append("Find people with consulting experience")

            # Size-based queries (if large network)
            if stats.get('total_connections', 0) > 50:
                suggestions.append("Who haven't I talked to in a while?")
                suggestions.append("Show me connections who might have changed jobs")

            # Career-focused suggestions
            suggestions.append("Who could give me career advice in tech?")
            suggestions.append("Find founders and entrepreneurs I know")

            self.logger.debug(f"Generated {len(suggestions)} dynamic suggestions")
            return suggestions[:12]  # Limit to 12 suggestions

        except Exception as e:
            self.logger.warning(f"Error generating suggestions: {str(e)}")
            # Return basic fallback suggestions with natural language
            return [
                "Who in my network might be hiring?",
                "Show me my most recent connections",
                "Find engineers and developers I know",
                "Who has marketing or sales experience?",
                "Which companies do most of my connections work at?",
                "Who are the managers and directors in my network?"
            ]
