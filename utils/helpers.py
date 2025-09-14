import streamlit as st
from typing import Dict, Any
import re

def format_connection_display(connection: Dict[str, Any]) -> str:
    """Format connection data for display in the UI"""
    display_parts = []
    
    # Name (bold)
    name = connection.get('full_name', 'Unknown')
    display_parts.append(f"**{name}**")
    
    # Company and Position
    company = connection.get('company', '')
    position = connection.get('position', '')
    
    if position and company:
        display_parts.append(f"*{position}* at **{company}**")
    elif position:
        display_parts.append(f"*{position}*")
    elif company:
        display_parts.append(f"**{company}**")
    
    # Email (if available)
    email = connection.get('email', '')
    if email and '@' in email:
        display_parts.append(f"📧 {email}")
    
    # Connected date
    connected_on = connection.get('connected_on', '')
    if connected_on:
        display_parts.append(f"🤝 Connected: {connected_on}")
    
    # Profile URL
    profile_url = connection.get('profile_url', '')
    if profile_url and 'linkedin.com' in profile_url:
        display_parts.append(f"🔗 [LinkedIn Profile]({profile_url})")
    
    return '  \n'.join(display_parts)

def extract_company_from_query(query: str) -> str:
    """Extract company name from user query"""
    query_lower = query.lower()
    
    # Common patterns for company mentions
    patterns = [
        r'works? at (.+?)(?:\?|$|\.)',
        r'from (.+?)(?:\?|$|\.)',
        r'at (.+?)(?:\?|$|\.)',
        r'in (.+?)(?:\?|$|\.)' 
    ]
    
    for pattern in patterns:
        match = re.search(pattern, query_lower)
        if match:
            company_name = match.group(1).strip()
            # Clean up common suffixes
            company_name = re.sub(r'\b(company|corp|inc|ltd|llc)\b', '', company_name).strip()
            return company_name
    
    return ''

def highlight_search_term(text: str, search_term: str) -> str:
    """Highlight search terms in text for display"""
    if not search_term or not text:
        return text
    
    # Use case-insensitive highlighting
    pattern = re.compile(re.escape(search_term), re.IGNORECASE)
    return pattern.sub(lambda m: f"**{m.group()}**", text)

def validate_email(email: str) -> bool:
    """Simple email validation"""
    if not email:
        return False
    
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(email_pattern, email))

def clean_company_name(company: str) -> str:
    """Clean and standardize company names"""
    if not company:
        return ''
    
    # Remove common corporate suffixes
    company = re.sub(r'\b(inc|corp|company|ltd|llc|co)\b\.?', '', company, flags=re.IGNORECASE)
    
    # Remove extra whitespace
    company = ' '.join(company.split())
    
    return company.strip()

def get_query_intent(query: str) -> str:
    """Determine the intent of the user query"""
    query_lower = query.lower()
    
    if any(keyword in query_lower for keyword in ['hiring', 'recruit', 'job', 'position', 'opening']):
        return 'hiring'
    elif any(keyword in query_lower for keyword in ['works at', 'from', 'company']):
        return 'company'
    elif any(keyword in query_lower for keyword in ['skill', 'experience', 'engineer', 'developer', 'manager']):
        return 'skills'
    elif any(keyword in query_lower for keyword in ['recent', 'new', 'latest']):
        return 'recent'
    elif any(keyword in query_lower for keyword in ['location', 'city', 'area']):
        return 'location'
    else:
        return 'general'

def format_stats_display(stats: Dict[str, Any]) -> str:
    """Format network statistics for display"""
    if not stats:
        return "No statistics available."
    
    parts = []
    
    total = stats.get('total_connections', 0)
    parts.append(f"**Total Connections:** {total}")
    
    companies = stats.get('total_companies', 0)
    parts.append(f"**Companies:** {companies}")
    
    with_email = stats.get('connections_with_email', 0)
    if total > 0:
        email_percent = (with_email / total) * 100
        parts.append(f"**With Email:** {with_email} ({email_percent:.1f}%)")
    
    with_position = stats.get('connections_with_position', 0)
    if total > 0:
        position_percent = (with_position / total) * 100
        parts.append(f"**With Position:** {with_position} ({position_percent:.1f}%)")
    
    top_companies = stats.get('top_companies', [])
    if top_companies:
        parts.append("**Top Companies:**")
        for company, count in top_companies[:5]:
            parts.append(f"  • {company}: {count}")
    
    return '\n'.join(parts)
