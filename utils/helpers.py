import streamlit as st
from typing import Dict, Any
import re
import logging
import sys
from datetime import datetime
import traceback
import os

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

# Logging and Error Handling Utilities
def setup_logger(name: str = "LinkedRAG", level: str = "INFO") -> logging.Logger:
    """Set up a logger with both file and console output"""
    logger = logging.getLogger(name)

    # Avoid duplicate handlers if logger already exists
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper()))

    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )

    # File handler (append mode)
    log_file = f"logs/linkedrag_{datetime.now().strftime('%Y%m%d')}.log"
    os.makedirs('logs', exist_ok=True)

    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def log_error_with_context(logger: logging.Logger, error: Exception, context: str = "") -> str:
    """Log an error with full context and return user-friendly message"""
    error_id = datetime.now().strftime("%H%M%S")

    # Log detailed error information
    logger.error(f"Error ID: {error_id} - {context}")
    logger.error(f"Error type: {type(error).__name__}")
    logger.error(f"Error message: {str(error)}")
    logger.error(f"Traceback: {traceback.format_exc()}")

    # Return user-friendly error message
    error_msg = str(error).lower()

    if "insufficient_quota" in error_msg or "quota" in error_msg:
        return f"❌ API quota exceeded. Please check your billing or try mock mode."
    elif "rate limit" in error_msg:
        return f"❌ Rate limit reached. Please wait a moment and try again."
    elif "connection" in error_msg or "network" in error_msg:
        return f"❌ Network connection issue. Please check your internet connection."
    elif "file" in error_msg and "not found" in error_msg:
        return f"❌ File not found. Please check the file path and try again."
    elif "csv" in error_msg:
        return f"❌ CSV file error. Please check your file format and try again."
    else:
        return f"❌ An unexpected error occurred (ID: {error_id}). Please try again or check the logs."

def create_user_friendly_message(error: Exception, context: str = "") -> Dict[str, str]:
    """Create a user-friendly error message with remediation steps"""
    error_msg = str(error).lower()

    if "insufficient_quota" in error_msg:
        return {
            "title": "OpenAI API Quota Exhausted",
            "message": "Your OpenAI API key has insufficient quota.",
            "remediation": [
                "1. **Check your billing**: Visit [OpenAI Platform](https://platform.openai.com/account/billing) to add credits",
                "2. **Try Mock Mode**: Select 'mock' from the embedding mode dropdown",
                "3. **Rotate API Key**: Use a different OpenAI API key with available quota"
            ]
        }
    elif "rate limit" in error_msg:
        return {
            "title": "Rate Limit Reached",
            "message": "Too many requests. Please wait before trying again.",
            "remediation": [
                "1. **Wait a few minutes** before retrying",
                "2. **Switch to Mock Mode** for testing without API calls",
                "3. **Check your API usage** on the OpenAI dashboard"
            ]
        }
    elif "connection" in error_msg or "network" in error_msg:
        return {
            "title": "Connection Error",
            "message": "Unable to connect to external services.",
            "remediation": [
                "1. **Check your internet connection**",
                "2. **Verify API keys** are correctly set in your .env file",
                "3. **Try Mock Mode** to test without external dependencies"
            ]
        }
    else:
        return {
            "title": "Unexpected Error",
            "message": f"An error occurred: {str(error)}",
            "remediation": [
                "1. **Try again** in a few moments",
                "2. **Check the logs** for detailed error information",
                "3. **Restart the application** if problems persist"
            ]
        }

def log_function_call(logger: logging.Logger, func_name: str, args: tuple = None, kwargs: dict = None):
    """Decorator to log function calls"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.debug(f"Calling {func_name} with args={args}, kwargs={kwargs}")
            try:
                result = func(*args, **kwargs)
                logger.debug(f"{func_name} completed successfully")
                return result
            except Exception as e:
                logger.error(f"{func_name} failed: {str(e)}")
                raise
        return wrapper
    return decorator
