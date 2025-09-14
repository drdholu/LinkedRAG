import pandas as pd
import streamlit as st
from typing import List, Dict, Any
import re

class DataProcessor:
    """Handles loading and processing of LinkedIn connections data"""
    
    def __init__(self):
        self.required_columns = ['First Name', 'Last Name']
        self.optional_columns = ['Company', 'Position', 'Email Address', 'Connected On', 'URL']
    
    def load_csv(self, uploaded_file) -> pd.DataFrame:
        """Load CSV file and validate basic structure"""
        try:
            df = pd.read_csv(uploaded_file, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(uploaded_file, encoding='latin-1')
            except Exception as e:
                raise Exception(f"Could not read CSV file. Please ensure it's a valid CSV: {str(e)}")
        
        # Validate required columns
        missing_columns = [col for col in self.required_columns if col not in df.columns]
        if missing_columns:
            raise Exception(f"Missing required columns: {missing_columns}. Available columns: {list(df.columns)}")
        
        # Clean data
        df = self._clean_dataframe(df)
        
        return df
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize the dataframe"""
        # Remove rows where both first and last name are empty
        df = df.dropna(subset=['First Name', 'Last Name'], how='all')
        
        # Fill NaN values with empty strings for text columns
        text_columns = ['First Name', 'Last Name', 'Company', 'Position', 'Email Address']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].fillna('')
        
        # Remove extra whitespace
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        
        return df
    
    def process_connections(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Process connections data into structured format for embeddings"""
        connections = []
        
        for _, row in df.iterrows():
            connection = self._create_connection_profile(row)
            if connection:  # Only add if profile is valid
                connections.append(connection)
        
        return connections
    
    def _create_connection_profile(self, row: pd.Series) -> Dict[str, Any]:
        """Create a structured connection profile from a row"""
        # Extract basic info
        first_name = str(row.get('First Name', '')).strip()
        last_name = str(row.get('Last Name', '')).strip()
        
        # Skip if no name
        if not first_name and not last_name:
            return None
        
        full_name = f"{first_name} {last_name}".strip()
        
        # Extract other information
        company = str(row.get('Company', '')).strip()
        position = str(row.get('Position', '')).strip()
        email = str(row.get('Email Address', '')).strip()
        connected_on = str(row.get('Connected On', '')).strip()
        profile_url = str(row.get('URL', '')).strip()
        
        # Create searchable text for embeddings
        searchable_text = self._create_searchable_text(
            full_name, company, position, email
        )
        
        connection = {
            'id': f"{first_name}_{last_name}_{company}".replace(' ', '_').lower(),
            'full_name': full_name,
            'first_name': first_name,
            'last_name': last_name,
            'company': company if company and company != 'nan' else '',
            'position': position if position and position != 'nan' else '',
            'email': email if email and email != 'nan' and '@' in email else '',
            'connected_on': connected_on if connected_on and connected_on != 'nan' else '',
            'profile_url': profile_url if profile_url and profile_url != 'nan' else '',
            'searchable_text': searchable_text
        }
        
        return connection
    
    def _create_searchable_text(self, name: str, company: str, position: str, email: str) -> str:
        """Create comprehensive searchable text for vector embeddings"""
        components = []
        
        if name:
            components.append(f"Name: {name}")
        
        if company:
            components.append(f"Company: {company}")
        
        if position:
            components.append(f"Position: {position}")
            # Extract keywords from position
            keywords = self._extract_keywords_from_position(position)
            if keywords:
                components.append(f"Skills/Keywords: {', '.join(keywords)}")
        
        if email:
            # Extract domain for company inference if company is missing
            if not company and '@' in email:
                domain = email.split('@')[1]
                if domain and '.' in domain:
                    company_from_email = domain.split('.')[0]
                    components.append(f"Email domain: {company_from_email}")
        
        return ' | '.join(components)
    
    def _extract_keywords_from_position(self, position: str) -> List[str]:
        """Extract relevant keywords from job position"""
        if not position:
            return []
        
        # Common tech and business keywords
        keywords = []
        position_lower = position.lower()
        
        # Technology keywords
        tech_keywords = [
            'software', 'engineer', 'developer', 'programmer', 'architect',
            'data', 'scientist', 'analyst', 'machine learning', 'ai', 'artificial intelligence',
            'python', 'java', 'javascript', 'react', 'node', 'aws', 'cloud',
            'devops', 'security', 'frontend', 'backend', 'fullstack', 'mobile',
            'ios', 'android', 'web', 'database', 'sql', 'api'
        ]
        
        # Business keywords
        business_keywords = [
            'manager', 'director', 'vp', 'vice president', 'ceo', 'cto', 'cfo',
            'marketing', 'sales', 'business', 'product', 'design', 'ux', 'ui',
            'consultant', 'strategy', 'operations', 'finance', 'hr', 'recruiting',
            'startup', 'founder', 'entrepreneur'
        ]
        
        all_keywords = tech_keywords + business_keywords
        
        for keyword in all_keywords:
            if keyword in position_lower:
                keywords.append(keyword)
        
        return list(set(keywords))  # Remove duplicates
