import pandas as pd
import streamlit as st
from typing import List, Dict, Any
import re
from datetime import datetime

class DataProcessor:
    """Handles loading and processing of LinkedIn connections data"""
    
    def __init__(self):
        self.required_columns = ['First Name', 'Last Name']
        self.optional_columns = ['Company', 'Position', 'Email Address', 'Connected On', 'URL']
        # Aliases to auto-map common LinkedIn and custom export column names to canonical names
        self.column_aliases = {
            'First Name': ['first name', 'firstname', 'first_name', 'given name', 'given_name'],
            'Last Name': ['last name', 'lastname', 'last_name', 'surname', 'family name', 'family_name'],
            'Company': ['company', 'company name', 'company_name', 'organization', 'current company', 'employer'],
            'Position': ['position', 'job title', 'jobtitle', 'title', 'role'],
            'Email Address': ['email address', 'email', 'email_address', 'e-mail', 'e-mail address'],
            'Connected On': ['connected on', 'connected_on', 'date connected', 'connected date', 'connected', 'date'],
            'URL': ['url', 'profile url', 'profile_url', 'linkedin url', 'linkedin profile url', 'public profile url']
        }
    
    def load_csv(self, uploaded_file) -> pd.DataFrame:
        """Load CSV file and validate basic structure"""
        try:
            df = pd.read_csv(uploaded_file, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(uploaded_file, encoding='latin-1')
            except Exception as e:
                raise Exception(f"Could not read CSV file. Please ensure it's a valid CSV: {str(e)}")
        
        # Auto-map columns to canonical schema
        df = self._map_columns(df)

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
        
        # Fill NaN values with empty strings for text columns (use .loc to avoid SettingWithCopyWarning)
        text_columns = ['First Name', 'Last Name', 'Company', 'Position', 'Email Address']
        for col in text_columns:
            if col in df.columns:
                df.loc[:, col] = df[col].fillna('')
        
        # Remove extra whitespace using .loc
        for col in text_columns:
            if col in df.columns:
                df.loc[:, col] = df[col].astype(str).str.strip()

        # Normalize Connected On date to ISO (YYYY-MM-DD) if present
        if 'Connected On' in df.columns:
            df.loc[:, 'Connected On'] = self._parse_dates_series(df['Connected On'])
        
        return df

    def _map_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map various possible input column names to the canonical schema used by the app"""
        if df is None or df.empty:
            return df
        
        # Build reverse lookup: lowercase alias -> canonical
        alias_to_canonical = {}
        for canonical, aliases in self.column_aliases.items():
            alias_to_canonical[canonical.lower()] = canonical
            for alias in aliases:
                alias_to_canonical[alias.lower()] = canonical
        
        rename_map = {}
        for col in list(df.columns):
            key = str(col).strip().lower()
            if key in alias_to_canonical:
                target = alias_to_canonical[key]
                if col != target:
                    # Avoid overwriting an existing canonical column: prefer existing canonical
                    if target not in df.columns:
                        rename_map[col] = target
        
        if rename_map:
            df = df.rename(columns=rename_map)
        
        return df

    def _parse_dates_series(self, series: pd.Series) -> pd.Series:
        """Parse a pandas Series of dates into ISO strings (YYYY-MM-DD). Leaves unparseable as ''."""
        # Common LinkedIn date formats and general fallbacks
        known_formats = [
            '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%b %d, %Y', '%B %d, %Y',
            '%d %b %Y', '%d %B %Y'
        ]
        parsed = []
        for val in series.fillna(''):
            s = str(val).strip()
            if not s or s.lower() in ['nan', 'none', 'null']:
                parsed.append('')
                continue
            dt = None
            # Try known formats
            for fmt in known_formats:
                try:
                    dt = datetime.strptime(s, fmt)
                    break
                except Exception:
                    continue
            # Try pandas parser as fallback
            if dt is None:
                try:
                    dt = pd.to_datetime(s, errors='coerce')
                    if pd.isna(dt):
                        dt = None
                except Exception:
                    dt = None
            parsed.append(dt.strftime('%Y-%m-%d') if dt else '')
        return pd.Series(parsed, index=series.index)

    def sort_connections_by_recent(self, connections: List[Dict[str, Any]], limit: int = 10) -> List[Dict[str, Any]]:
        """Return most recent connections based on Connected On ISO date."""
        def parse(c: Dict[str, Any]) -> int:
            d = c.get('connected_on', '')
            try:
                return int(datetime.strptime(d, '%Y-%m-%d').timestamp())
            except Exception:
                return 0
        return sorted(connections, key=parse, reverse=True)[:max(0, limit)]
    
    def process_connections(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Process connections data into structured format for embeddings"""
        connections = []
        seen_ids = set()
        
        for _, row in df.iterrows():
            connection = self._create_connection_profile(row)
            if connection:  # Only add if profile is valid
                # Deduplicate by stable id
                if connection['id'] in seen_ids:
                    continue
                seen_ids.add(connection['id'])
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
        
        stable_id = self._generate_stable_id(first_name, last_name, email, company, profile_url)

        connection = {
            'id': stable_id,
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

    def _generate_stable_id(self, first_name: str, last_name: str, email: str, company: str, profile_url: str) -> str:
        """Generate a stable unique id using strongest available identifiers."""
        base_parts = []
        if email and '@' in email:
            base_parts.append(email.lower())
        if profile_url:
            base_parts.append(profile_url.lower())
        # Fallback to name+company
        name_company = f"{first_name} {last_name} {company}".strip().lower()
        if name_company:
            base_parts.append(name_company)
        base = '|'.join(base_parts)
        # Hash to keep id compact and filesystem-safe
        import hashlib
        digest = hashlib.md5(base.encode('utf-8')).hexdigest()[:16]
        # Human-readable prefix
        prefix = f"{first_name}_{last_name}".strip().replace(' ', '_').lower() or 'conn'
        return f"{prefix}_{digest}"
    
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
