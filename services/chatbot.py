import os
from openai import OpenAI
from typing import List, Dict, Any
import json
import re
import requests

class ChatBot:
    """RAG-powered chatbot for querying LinkedIn connections"""
    
    def __init__(self, vector_store):
        # Chat backends: openai, ollama, mock
        self.chat_mode = os.getenv("CHAT_MODE", "openai").lower()
        # the newest OpenAI model is "gpt-5" which was released August 7, 2025.
        # do not change this unless explicitly requested by the user
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.vector_store = vector_store
        self.model = "gpt-5"
        self.use_mock = self.chat_mode == 'mock'
        # Ollama config
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip('/')
        self.ollama_model = os.getenv("OLLAMA_CHAT_MODEL", "llama3.1")
        
        # System prompt for the chatbot
        self.system_prompt = """You are a helpful assistant that answers questions about LinkedIn connections.
        You will be provided with relevant connection information from a user's LinkedIn network.
        
        Instructions:
        1. Answer strictly based on the provided connection data (do not invent facts).
        2. Return specific names, companies, positions, and connected dates when available.
        3. If asking about hiring, prioritize roles like manager, director, recruiter, HR.
        4. For company queries, list the top matches. For general queries, list the most relevant connections.
        5. Output up to 5 results as a concise bulleted list, one per line: "Name — Company — Position (Connected: YYYY-MM-DD)".
        6. If no relevant connections are found, clearly say so.
        7. Do not include information outside the provided context.
        """
    
    def get_response(self, user_query: str) -> str:
        """Get chatbot response using RAG"""
        try:
            # Analyze query to determine search strategy
            search_results = self._get_relevant_connections(user_query)
            
            if not search_results:
                return self._handle_no_results(user_query)
            
            # Create context from search results
            context = self._create_context(search_results, user_query)
            
            # Generate response (include connections for citations)
            response = self._generate_response(user_query, context, search_results)
            
            return response
            
        except Exception as e:
            return f"I'm sorry, I encountered an error while processing your question: {str(e)}"
    
    def _get_relevant_connections(self, query: str) -> List[Dict[str, Any]]:
        """Get relevant connections based on the query"""
        query_lower = query.lower()
        
        # Check if it's a company-specific query
        company_patterns = [
            r'works? at (.+?)(?:\?|$|\.)',
            r'from (.+?)(?:\?|$|\.)',
            r'at (.+?)(?:\?|$|\.)',
            r'in (.+?)(?:\?|$|\.)' 
        ]
        
        for pattern in company_patterns:
            match = re.search(pattern, query_lower)
            if match:
                company_name = match.group(1).strip()
                # Remove common words that might interfere
                company_name = re.sub(r'\b(company|corp|inc|ltd|llc)\b', '', company_name).strip()
                if company_name:
                    company_results = self.vector_store.search_by_company(company_name, k=15)
                    if company_results:
                        return company_results
        
        # Check for hiring-related queries
        if any(keyword in query_lower for keyword in ['hiring', 'recruit', 'job', 'position', 'opening']):
            # Search for connections that might be hiring
            hiring_results = self.vector_store.similarity_search(
                "hiring recruiting talent acquisition hr human resources manager director", k=10
            )
            relevant_results = [(conn, score) for conn, score in hiring_results if score > 0.3]
            if relevant_results:
                return [conn for conn, _ in relevant_results]
        
        # General similarity search
        search_results = self.vector_store.similarity_search(query, k=10)
        
        # Filter results by relevance score
        relevant_results = [(conn, score) for conn, score in search_results if score > 0.2]
        
        return [conn for conn, _ in relevant_results]
    
    def _create_context(self, connections: List[Dict[str, Any]], query: str) -> str:
        """Create context string from relevant connections"""
        if not connections:
            return "No relevant connections found."
        
        context_parts = ["Here are the relevant connections from your LinkedIn network:\n"]
        
        for i, conn in enumerate(connections[:15], 1):  # Limit to top 15 results
            context_parts.append(f"{i}. **{conn['full_name']}**")
            
            if conn.get('company'):
                context_parts.append(f"   - Company: {conn['company']}")
            
            if conn.get('position'):
                context_parts.append(f"   - Position: {conn['position']}")
            
            if conn.get('email'):
                context_parts.append(f"   - Email: {conn['email']}")
            
            if conn.get('connected_on'):
                context_parts.append(f"   - Connected: {conn['connected_on']}")
            
            context_parts.append("")  # Empty line for spacing
        
        return "\n".join(context_parts)
    
    def _generate_mock_response(self, query: str, connections: List[Dict[str, Any]]) -> str:
        """Generate deterministic mock response based on connections"""
        query_lower = query.lower()
        
        if not connections:
            return "I couldn't find any connections matching your query in the sample data."
        
        # Company-specific queries
        if "google" in query_lower:
            google_conns = [c for c in connections if "google" in c.get('company', '').lower()]
            if google_conns:
                conn = google_conns[0]
                return f"I found {conn['full_name']} who works at {conn['company']} as a {conn.get('position', 'team member')}. They were connected on {conn.get('connected_on', 'an unknown date')}."
        
        if "microsoft" in query_lower:
            ms_conns = [c for c in connections if "microsoft" in c.get('company', '').lower()]
            if ms_conns:
                conn = ms_conns[0]
                return f"Yes! {conn['full_name']} works at {conn['company']} as a {conn.get('position', 'team member')}."
        
        # Hiring queries
        if any(keyword in query_lower for keyword in ['hiring', 'recruit', 'job']):
            hiring_positions = ['manager', 'director', 'recruiter', 'hr']
            hiring_conns = [c for c in connections if any(pos in c.get('position', '').lower() for pos in hiring_positions)]
            if hiring_conns:
                conn = hiring_conns[0]
                return f"Based on their role, {conn['full_name']} at {conn.get('company', 'their company')} ({conn.get('position', 'their position')}) might be involved in hiring decisions. You could reach out to them."
            return "I don't see any obvious hiring managers in your immediate connections, but you could reach out to engineering managers or directors."
        
        # Engineering/software queries
        if any(keyword in query_lower for keyword in ['engineer', 'software', 'developer', 'programming']):
            eng_conns = [c for c in connections if any(term in c.get('position', '').lower() for term in ['engineer', 'developer', 'software'])]
            if eng_conns:
                names = [c['full_name'] for c in eng_conns[:3]]
                companies = [c.get('company', 'Unknown') for c in eng_conns[:3]]
                return f"I found several software engineers in your network: {', '.join(names)}. They work at {', '.join(set(companies))} respectively."
        
        # General response with first few connections
        if len(connections) >= 2:
            conn1, conn2 = connections[0], connections[1]
            return f"Based on your connections, I found {conn1['full_name']} at {conn1.get('company', 'their company')} and {conn2['full_name']} at {conn2.get('company', 'their company')} who might be relevant to your query."
        
        conn = connections[0]
        return f"I found {conn['full_name']} at {conn.get('company', 'their company')} who might be relevant to your query."
    
    def _generate_response(self, query: str, context: str, connections: List[Dict[str, Any]]) -> str:
        """Generate response using selected backend and append citations."""
        # If using mock mode, generate deterministic response
        if self.use_mock:
            answer = self._generate_mock_response(query, connections)
            citations = self._format_citations(connections)
            return f"{answer}\n\n{citations}" if citations else answer

        # Build messages once
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]

        if self.chat_mode == 'ollama':
            try:
                answer = self._generate_with_ollama(messages)
                citations = self._format_citations(connections)
                return f"{answer}\n\n{citations}" if citations else answer
            except Exception as e:
                # Fallback to mock on failure
                if connections:
                    return self._generate_mock_response(query, connections)
                raise Exception(f"Error generating response with Ollama: {str(e)}")
        
        # Default: OpenAI
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
            if connections:
                return self._generate_mock_response(query, connections)
            raise Exception(f"Error generating response: {str(e)}")

    def _generate_with_ollama(self, messages: List[Dict[str, str]]) -> str:
        """Generate chat response using local Ollama server."""
        url = f"{self.ollama_base_url}/api/chat"
        payload = {
            "model": self.ollama_model,
            "messages": messages
        }
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        # Ollama chat may return either a streaming format (not used here) or final message
        if isinstance(data, dict):
            # Non-streaming final response
            if 'message' in data and isinstance(data['message'], dict) and 'content' in data['message']:
                return data['message']['content']
            if 'response' in data:
                return data['response']
        raise Exception("Unexpected Ollama response format")

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
    
    def _handle_no_results(self, query: str) -> str:
        """Handle cases where no relevant connections are found"""
        # Get some general stats to provide helpful info
        try:
            stats = self.vector_store.get_connection_stats()
            companies = self.vector_store.get_all_companies()
            
            response_parts = [
                "I couldn't find specific connections that match your query."
            ]
            
            if stats:
                response_parts.append(f"\nYour network includes {stats['total_connections']} connections across {stats['total_companies']} companies.")
                
                if stats.get('top_companies'):
                    top_companies = [company for company, count in stats['top_companies'][:5]]
                    response_parts.append(f"Your top companies include: {', '.join(top_companies)}.")
            
            response_parts.append("\nTry asking about:")
            response_parts.append("- Specific companies: 'Who works at Google?'")
            response_parts.append("- Job roles: 'Who works in software engineering?'")
            response_parts.append("- General questions: 'Who is hiring?' or 'Show me recent connections'")
            
            return "\n".join(response_parts)
            
        except Exception:
            return "I couldn't find any connections matching your query. Please try rephrasing your question or check if your data has been processed correctly."
    
    def get_suggested_queries(self) -> List[str]:
        """Get suggested queries based on the data"""
        if not self.vector_store.connections_data:
            return []
        
        suggestions = [
            "Who in my network is currently hiring?",
            "Show me my most recent connections",
            "Who works in technology?",
            "Who are my connections in marketing?"
        ]
        
        # Add company-specific suggestions
        companies = self.vector_store.get_all_companies()
        if companies:
            top_companies = companies[:3]
            for company in top_companies:
                suggestions.append(f"Who works at {company}?")
        
        return suggestions
