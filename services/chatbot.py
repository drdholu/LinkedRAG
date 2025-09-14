import os
from openai import OpenAI
from typing import List, Dict, Any
import json
import re

class ChatBot:
    """RAG-powered chatbot for querying LinkedIn connections"""
    
    def __init__(self, vector_store):
        # the newest OpenAI model is "gpt-5" which was released August 7, 2025.
        # do not change this unless explicitly requested by the user
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.vector_store = vector_store
        self.model = "gpt-5"
        self.use_mock = getattr(vector_store, 'embeddings_mode', 'openai') == 'mock'
        
        # System prompt for the chatbot
        self.system_prompt = """You are a helpful assistant that answers questions about LinkedIn connections.
        You will be provided with relevant connection information from a user's LinkedIn network.
        
        Instructions:
        1. Answer questions based only on the provided connection data
        2. Be specific and mention names, companies, and positions when relevant
        3. If asking about hiring, look for keywords like "hiring", "recruiting", "open positions" in positions/companies
        4. For company searches, find exact or partial matches
        5. For skill-based searches, look at job titles and positions
        6. If no relevant connections are found, say so clearly
        7. Format your responses in a helpful, conversational manner
        8. Always provide specific names and details when available
        
        Connection data format:
        - full_name: Person's full name
        - company: Current company
        - position: Job title/position
        - email: Email address (if available)
        - connected_on: When you connected (if available)
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
            
            # Generate response
            response = self._generate_response(user_query, context)
            
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
    
    def _generate_response(self, query: str, context: str) -> str:
        """Generate response using OpenAI or mock"""
        # If using mock mode, generate deterministic response
        if self.use_mock:
            # Extract connections from the context for mock response
            relevant_connections = self._get_relevant_connections(query)
            return self._generate_mock_response(query, relevant_connections)
        
        # Use OpenAI for response generation
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=1000,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            # Fallback to mock response if OpenAI fails
            relevant_connections = self._get_relevant_connections(query)
            if relevant_connections:
                return self._generate_mock_response(query, relevant_connections)
            raise Exception(f"Error generating response: {str(e)}")
    
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
