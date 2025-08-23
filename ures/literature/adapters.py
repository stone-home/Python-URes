#!/usr/bin/env python3
"""
Database Adapters
Abstract base class and concrete implementations for different academic databases.
All adapters inherit from DatabaseAdapter and provide unified interface.
"""

import time
import json
import urllib.request
import urllib.parse
import urllib.error
import xml.etree.ElementTree as ET
import re
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
import logging
from datetime import datetime
from ures.literature.paper import Paper, PaperFormatter


class QueryParser:
    """Parse complex Boolean search queries and convert to database-specific formats."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def parse_boolean_query(self, query: str) -> Dict[str, Any]:
        """Parse Boolean query into structured format."""
        query = query.strip()

        # Extract quoted phrases first
        quoted_phrases = re.findall(r'"([^"]*)"', query)

        # Replace quoted phrases with placeholders to protect them
        temp_query = query
        for i, phrase in enumerate(quoted_phrases):
            temp_query = temp_query.replace(f'"{phrase}"', f"__PHRASE_{i}__")

        # Extract parenthetical groups
        groups = self._extract_parenthetical_groups(temp_query)

        # If no groups found, try to parse the whole query as a single group
        if not groups:
            if " OR " in temp_query:
                terms = [term.strip() for term in temp_query.split(" OR ")]
                groups.append({"type": "OR", "terms": terms})
            elif " AND " in temp_query:
                terms = [term.strip() for term in temp_query.split(" AND ")]
                groups.append({"type": "AND", "terms": terms})
            else:
                groups.append({"type": "SINGLE", "terms": [temp_query.strip()]})

        # Restore quoted phrases in terms
        for group in groups:
            for j, term in enumerate(group["terms"]):
                for i, phrase in enumerate(quoted_phrases):
                    term = term.replace(f"__PHRASE_{i}__", f'"{phrase}"')
                group["terms"][j] = term

        return {
            "quoted_phrases": quoted_phrases,
            "groups": groups,
            "original_query": query,
        }

    def _extract_parenthetical_groups(self, query: str) -> List[Dict]:
        """Extract and parse parenthetical groups."""
        groups = []

        # Find all parenthetical expressions - handle nested parentheses
        paren_count = 0
        start_pos = -1

        for i, char in enumerate(query):
            if char == "(":
                if paren_count == 0:
                    start_pos = i
                paren_count += 1
            elif char == ")":
                paren_count -= 1
                if paren_count == 0 and start_pos != -1:
                    group_content = query[start_pos + 1 : i]
                    group = self._parse_group_content(group_content)
                    if group:
                        groups.append(group)

        # If no parentheses found, look for top-level AND operations
        if not groups:
            and_parts = query.split(" AND ")
            if len(and_parts) > 1:
                for part in and_parts:
                    part = part.strip()
                    if part:
                        group = self._parse_group_content(part)
                        if group:
                            groups.append(group)

        return groups

    def _parse_group_content(self, content: str) -> Dict:
        """Parse the content within a group."""
        content = content.strip()

        if " OR " in content:
            terms = [term.strip() for term in content.split(" OR ")]
            return {"type": "OR", "terms": [t for t in terms if t]}
        elif " AND " in content:
            terms = [term.strip() for term in content.split(" AND ")]
            return {"type": "AND", "terms": [t for t in terms if t]}
        else:
            if content:
                return {"type": "SINGLE", "terms": [content]}

        return None

    def to_arxiv_query(self, parsed_query: Dict) -> str:
        """Convert parsed query to arXiv format."""
        terms = []

        for group in parsed_query["groups"]:
            if group["type"] == "OR":
                # Clean quotes from terms for arXiv
                clean_terms = [term.replace('"', "") for term in group["terms"]]
                group_terms = " OR ".join([f'all:"{term}"' for term in clean_terms])
                terms.append(f"({group_terms})")
            elif group["type"] == "AND":
                clean_terms = [term.replace('"', "") for term in group["terms"]]
                group_terms = " AND ".join([f'all:"{term}"' for term in clean_terms])
                terms.append(f"({group_terms})")
            else:
                clean_term = group["terms"][0].replace('"', "")
                terms.append(f'all:"{clean_term}"')

        final_query = " AND ".join(terms)
        self.logger.debug(f"Converted to arXiv query: {final_query}")
        return final_query

    def to_ieee_query(self, parsed_query: Dict) -> str:
        """Convert parsed query to IEEE format."""
        terms = []

        for group in parsed_query["groups"]:
            if group["type"] == "OR":
                group_terms = " OR ".join(group["terms"])
                terms.append(f"({group_terms})")
            elif group["type"] == "AND":
                group_terms = " AND ".join(group["terms"])
                terms.append(f"({group_terms})")
            else:
                terms.append(group["terms"][0])

        final_query = " AND ".join(terms)
        self.logger.debug(f"Converted to IEEE query: {final_query}")
        return final_query

    def to_simple_query(self, parsed_query: Dict) -> str:
        """Convert to simple query for basic APIs."""
        all_terms = []
        for group in parsed_query["groups"]:
            for term in group["terms"]:
                # Remove quotes and clean up
                clean_term = term.replace('"', "").strip()
                if clean_term and clean_term not in all_terms:
                    all_terms.append(clean_term)

        simple_query = " ".join(all_terms)
        self.logger.debug(f"Converted to simple query: {simple_query}")
        return simple_query


class DatabaseAdapter(ABC):
    """Abstract base class for all database adapters."""

    def __init__(self, rate_limit: float = 1.0, api_key: str = None):
        """
        Initialize database adapter.

        Args:
                rate_limit: Requests per second limit
                api_key: API key for the database (if required)
        """
        self.rate_limit = rate_limit
        self.api_key = api_key
        self.last_request = 0
        self.logger = logging.getLogger(self.__class__.__name__)
        self.query_parser = QueryParser()

        # Statistics
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_papers_found": 0,
            "last_request_time": None,
            "rate_limit_hits": 0,
        }

    def _rate_limit_wait(self):
        """Implement rate limiting based on configured rate."""
        elapsed = time.time() - self.last_request
        min_interval = 1.0 / self.rate_limit

        if elapsed < min_interval:
            wait_time = min_interval - elapsed
            self.logger.debug(f"Rate limiting: waiting {wait_time:.2f}s")
            time.sleep(wait_time)
            self.stats["rate_limit_hits"] += 1

        self.last_request = time.time()
        self.stats["last_request_time"] = datetime.now().isoformat()

    def _log_request(self, success: bool, papers_found: int = 0):
        """Log request statistics."""
        self.stats["total_requests"] += 1
        if success:
            self.stats["successful_requests"] += 1
            self.stats["total_papers_found"] += papers_found
        else:
            self.stats["failed_requests"] += 1

    @abstractmethod
    def search(self, query: str, max_results: int = 100, **kwargs) -> List[Paper]:
        """
        Search the database for papers.

        Args:
                query: Search query (supports Boolean operations)
                max_results: Maximum number of results to return
                **kwargs: Database-specific parameters

        Returns:
                List[Paper]: List of found papers
        """
        pass

    @abstractmethod
    def get_database_name(self) -> str:
        """Get the name of this database."""
        pass

    def get_stats(self) -> Dict[str, Any]:
        """Get adapter statistics."""
        return self.stats.copy()

    def reset_stats(self):
        """Reset adapter statistics."""
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_papers_found": 0,
            "last_request_time": None,
            "rate_limit_hits": 0,
        }

    def is_available(self) -> bool:
        """Check if the database adapter is available and configured."""
        return True  # Override in subclasses if needed

    def validate_query(self, query: str) -> bool:
        """Validate if the query is supported by this adapter."""
        return bool(query and query.strip())

    def preprocess_query(self, query: str) -> str:
        """Preprocess query for this specific database."""
        return query.strip()


class ArxivAdapter(DatabaseAdapter):
    """Adapter for arXiv API with Boolean query support."""

    def __init__(self, rate_limit: float = 3.0):
        super().__init__(rate_limit=rate_limit)
        self.base_url = "http://export.arxiv.org/api/query"

    def get_database_name(self) -> str:
        return "arxiv"

    def search(
        self, query: str, max_results: int = 100, categories: List[str] = None
    ) -> List[Paper]:
        """Search arXiv with Boolean query support."""
        self._rate_limit_wait()

        try:
            # Parse Boolean query
            parsed_query = self.query_parser.parse_boolean_query(query)
            self.logger.info(f"Parsed query: {parsed_query}")

            arxiv_query = self.query_parser.to_arxiv_query(parsed_query)
            self.logger.info(f"ArXiv query: {arxiv_query}")

            # Add category filter if specified
            if categories:
                cat_filter = " OR ".join([f"cat:{cat}" for cat in categories])
                arxiv_query = f"({arxiv_query}) AND ({cat_filter})"
                self.logger.info(f"ArXiv query with categories: {arxiv_query}")

            params = {
                "search_query": arxiv_query,
                "start": 0,
                "max_results": max_results,
                "sortBy": "relevance",
                "sortOrder": "descending",
            }

            url = f"{self.base_url}?{urllib.parse.urlencode(params)}"
            self.logger.debug(f"ArXiv URL: {url}")

            with urllib.request.urlopen(url, timeout=30) as response:
                xml_data = response.read().decode("utf-8")

            papers = self._parse_arxiv_response(xml_data)
            self.logger.info(f"ArXiv returned {len(papers)} papers")
            self._log_request(success=True, papers_found=len(papers))
            return papers

        except Exception as e:
            self.logger.error(f"ArXiv search failed: {e}")
            self._log_request(success=False)
            return []

    def _parse_arxiv_response(self, xml_data: str) -> List[Paper]:
        """Parse arXiv XML response and return standardized Paper objects."""
        papers = []

        try:
            root = ET.fromstring(xml_data)

            for entry in root.findall(".//{http://www.w3.org/2005/Atom}entry"):
                entry_data = {}

                title_elem = entry.find(".//{http://www.w3.org/2005/Atom}title")
                entry_data["title"] = title_elem.text if title_elem is not None else ""

                authors = []
                for author in entry.findall(".//{http://www.w3.org/2005/Atom}author"):
                    name_elem = author.find(".//{http://www.w3.org/2005/Atom}name")
                    if name_elem is not None:
                        authors.append({"name": name_elem.text.strip()})
                entry_data["authors"] = authors

                summary_elem = entry.find(".//{http://www.w3.org/2005/Atom}summary")
                entry_data["summary"] = (
                    summary_elem.text if summary_elem is not None else ""
                )

                id_elem = entry.find(".//{http://www.w3.org/2005/Atom}id")
                if id_elem is not None:
                    entry_data["id"] = id_elem.text

                for link in entry.findall(".//{http://www.w3.org/2005/Atom}link"):
                    if link.get("title") == "pdf":
                        entry_data["pdf_url"] = link.get("href", "")
                        break

                categories = []
                for category in entry.findall(
                    ".//{http://arxiv.org/schemas/atom}primary_category"
                ):
                    term = category.get("term", "")
                    if term:
                        categories.append(term)
                for category in entry.findall(
                    ".//{http://arxiv.org/schemas/atom}category"
                ):
                    term = category.get("term", "")
                    if term and term not in categories:
                        categories.append(term)
                entry_data["categories"] = categories

                paper = PaperFormatter.format_arxiv_paper(entry_data)
                papers.append(paper)

        except Exception as e:
            self.logger.error(f"Error parsing arXiv response: {e}")

        return papers

    def is_available(self) -> bool:
        """ArXiv is always available (no API key required)."""
        return True


class IEEEAdapter(DatabaseAdapter):
    """Adapter for IEEE Xplore API with Boolean query support."""

    def __init__(self, api_key: str, rate_limit: float = 100.0):
        super().__init__(rate_limit=rate_limit, api_key=api_key)
        self.base_url = "http://ieeexploreapi.ieee.org/api/v1/search/articles"

    def get_database_name(self) -> str:
        return "ieee"

    def is_available(self) -> bool:
        """IEEE requires API key."""
        return bool(self.api_key)

    def _rate_limit_wait(self):
        """IEEE has hourly rate limits."""
        elapsed = time.time() - self.last_request
        if elapsed < (3600.0 / self.rate_limit):
            time.sleep((3600.0 / self.rate_limit) - elapsed)
        self.last_request = time.time()

    def search(
        self, query: str, max_results: int = 100, year_min: int = None
    ) -> List[Paper]:
        """Search IEEE Xplore with Boolean query support."""
        if not self.is_available():
            self.logger.warning("IEEE API key not provided")
            return []

        self._rate_limit_wait()

        try:
            parsed_query = self.query_parser.parse_boolean_query(query)
            ieee_query = self.query_parser.to_ieee_query(parsed_query)

            params = {
                "apikey": self.api_key,
                "querytext": ieee_query,
                "max_records": min(max_results, 200),
                "start_record": 1,
                "sort_field": "publication_year",
                "sort_order": "desc",
            }

            if year_min:
                params["start_year"] = year_min

            url = f"{self.base_url}?{urllib.parse.urlencode(params)}"
            with urllib.request.urlopen(url, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))

            papers = []
            for article in data.get("articles", []):
                paper = PaperFormatter.format_ieee_paper(article)
                papers.append(paper)

            self._log_request(success=True, papers_found=len(papers))
            return papers

        except Exception as e:
            self.logger.error(f"IEEE search failed: {e}")
            self._log_request(success=False)
            return []


class ElsevierAdapter(DatabaseAdapter):
    """Adapter for Elsevier/ScienceDirect API."""

    def __init__(self, api_key: str, rate_limit: float = 100.0):
        super().__init__(rate_limit=rate_limit, api_key=api_key)
        self.base_url = "https://api.elsevier.com/content/search/sciencedirect"

    def get_database_name(self) -> str:
        return "elsevier"

    def is_available(self) -> bool:
        """Elsevier requires API key."""
        return bool(self.api_key)

    def search(self, query: str, max_results: int = 100) -> List[Paper]:
        """Search Elsevier ScienceDirect."""
        if not self.is_available():
            self.logger.warning("Elsevier API key not provided")
            return []

        self._rate_limit_wait()

        try:
            parsed_query = self.query_parser.parse_boolean_query(query)
            simple_query = self.query_parser.to_simple_query(parsed_query)

            headers = {"X-ELS-APIKey": self.api_key, "Accept": "application/json"}

            params = {"query": simple_query, "count": min(max_results, 100)}

            url = f"{self.base_url}?{urllib.parse.urlencode(params)}"
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))

            papers = []
            for entry in data.get("search-results", {}).get("entry", []):
                paper = PaperFormatter.format_elsevier_paper(entry)
                papers.append(paper)

            self._log_request(success=True, papers_found=len(papers))
            return papers

        except Exception as e:
            self.logger.error(f"Elsevier search failed: {e}")
            self._log_request(success=False)
            return []


class SpringerAdapter(DatabaseAdapter):
    """Adapter for Springer Nature API."""

    def __init__(self, api_key: str, rate_limit: float = 100.0):
        super().__init__(rate_limit=rate_limit, api_key=api_key)
        self.base_url = "http://api.springernature.com/meta/v1/json"

    def get_database_name(self) -> str:
        return "springer"

    def is_available(self) -> bool:
        """Springer requires API key."""
        return bool(self.api_key)

    def search(
        self, query: str, max_results: int = 100, year_min: int = None
    ) -> List[Paper]:
        """Search Springer Nature."""
        if not self.is_available():
            self.logger.warning("Springer API key not provided")
            return []

        self._rate_limit_wait()

        try:
            parsed_query = self.query_parser.parse_boolean_query(query)
            simple_query = self.query_parser.to_simple_query(parsed_query)

            if year_min:
                simple_query += f" year:{year_min}-{datetime.now().year}"

            params = {
                "api_key": self.api_key,
                "q": simple_query,
                "s": min(max_results, 100),
                "p": 1,
            }

            url = f"{self.base_url}?{urllib.parse.urlencode(params)}"
            with urllib.request.urlopen(url, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))

            papers = []
            for record in data.get("records", []):
                paper = PaperFormatter.format_springer_paper(record)
                papers.append(paper)

            self._log_request(success=True, papers_found=len(papers))
            return papers

        except Exception as e:
            self.logger.error(f"Springer search failed: {e}")
            self._log_request(success=False)
            return []


class WileyAdapter(DatabaseAdapter):
    """Adapter for Wiley Online Library API."""

    def __init__(self, api_key: str, rate_limit: float = 100.0):
        super().__init__(rate_limit=rate_limit, api_key=api_key)
        self.base_url = "https://api.wiley.com/onlinelibrary/tdm/v1/articles"

    def get_database_name(self) -> str:
        return "wiley"

    def is_available(self) -> bool:
        """Wiley requires API key."""
        return bool(self.api_key)

    def search(self, query: str, max_results: int = 100) -> List[Paper]:
        """Search Wiley Online Library."""
        if not self.is_available():
            self.logger.warning("Wiley API key not provided")
            return []

        self._rate_limit_wait()

        try:
            parsed_query = self.query_parser.parse_boolean_query(query)
            simple_query = self.query_parser.to_simple_query(parsed_query)

            headers = {
                "Wiley-TDM-Client-Token": self.api_key,
                "Accept": "application/json",
            }

            params = {"query": simple_query, "count": min(max_results, 100)}

            url = f"{self.base_url}?{urllib.parse.urlencode(params)}"
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))

            papers = []
            for item in data.get("items", []):
                paper = PaperFormatter.format_wiley_paper(item)
                papers.append(paper)

            self._log_request(success=True, papers_found=len(papers))
            return papers

        except Exception as e:
            self.logger.error(f"Wiley search failed: {e}")
            self._log_request(success=False)
            return []


class ACMAdapter(DatabaseAdapter):
    """Adapter for ACM Digital Library."""

    def __init__(self, rate_limit: float = 100.0):
        super().__init__(rate_limit=rate_limit)
        self.base_url = "https://dl.acm.org/action/doSearch"

    def get_database_name(self) -> str:
        return "acm"

    def is_available(self) -> bool:
        """ACM doesn't have a public API, requires web scraping."""
        return False  # Disabled for now

    def search(self, query: str, max_results: int = 100) -> List[Paper]:
        """Search ACM Digital Library (limited scraping approach)."""
        self.logger.warning(
            "ACM Digital Library requires web scraping - limited functionality"
        )
        # ACM doesn't have a public API, would require careful web scraping
        # For now, return empty list to avoid blocking issues
        self._log_request(success=False)
        return []


class GoogleScholarAdapter(DatabaseAdapter):
    """Limited Google Scholar adapter."""

    def __init__(self, rate_limit: float = 1.0):
        super().__init__(rate_limit=rate_limit)

    def get_database_name(self) -> str:
        return "google_scholar"

    def is_available(self) -> bool:
        """Google Scholar blocks automated access."""
        return False  # Disabled for now to avoid blocking

    def search(
        self, query: str, max_results: int = 20, year_min: int = None
    ) -> List[Paper]:
        """Limited Google Scholar search."""
        self.logger.warning("Google Scholar adapter provides limited functionality")
        self._log_request(success=False)
        return []


class AdapterFactory:
    """Factory class for creating database adapters."""

    SUPPORTED_DATABASES = {
        "arxiv": ArxivAdapter,
        "ieee": IEEEAdapter,
        "elsevier": ElsevierAdapter,
        "springer": SpringerAdapter,
        "wiley": WileyAdapter,
        "acm": ACMAdapter,
        "google_scholar": GoogleScholarAdapter,
    }

    @staticmethod
    def create_adapter(database_name: str, **kwargs) -> Optional[DatabaseAdapter]:
        """
        Create a database adapter instance.

        Args:
                database_name: Name of the database
                **kwargs: Adapter-specific parameters (api_key, rate_limit, etc.)

        Returns:
                DatabaseAdapter instance or None if not supported
        """

        if database_name not in AdapterFactory.SUPPORTED_DATABASES:
            return None

        try:
            adapter_class = adapters[database_name]

            # Handle different parameter requirements
            if database_name == "arxiv":
                return adapter_class(rate_limit=kwargs.get("rate_limit", 3.0))
            elif database_name in ["ieee", "elsevier", "springer", "wiley"]:
                api_key = kwargs.get("api_key")
                if not api_key:
                    return None
                return adapter_class(
                    api_key=api_key, rate_limit=kwargs.get("rate_limit", 100.0)
                )
            elif database_name in ["acm", "google_scholar"]:
                return adapter_class(rate_limit=kwargs.get("rate_limit", 1.0))

        except Exception as e:
            logging.getLogger(__name__).error(
                f"Failed to create adapter for {database_name}: {e}"
            )
            return None

    @staticmethod
    def get_supported_databases() -> List[str]:
        """Get list of supported database names."""
        return list(AdapterFactory.SUPPORTED_DATABASES.keys())

    @staticmethod
    def get_adapter_requirements(database_name: str) -> Dict[str, Any]:
        """Get requirements for a specific adapter."""
        requirements = {
            "arxiv": {"api_key_required": False, "rate_limit": 3.0, "free": True},
            "ieee": {
                "api_key_required": True,
                "rate_limit": 100.0,
                "free": False,
                "api_signup_url": "https://developer.ieee.org/",
            },
            "elsevier": {
                "api_key_required": True,
                "rate_limit": 100.0,
                "free": False,
                "api_signup_url": "https://dev.elsevier.com/",
            },
            "springer": {
                "api_key_required": True,
                "rate_limit": 100.0,
                "free": True,
                "api_signup_url": "https://dev.springernature.com/",
            },
            "wiley": {
                "api_key_required": True,
                "rate_limit": 100.0,
                "free": False,
                "api_signup_url": "https://onlinelibrary.wiley.com/library-info/resources/text-and-datamining",
            },
            "acm": {
                "api_key_required": False,
                "rate_limit": 1.0,
                "free": True,
                "note": "No public API available, requires web scraping",
            },
            "google_scholar": {
                "api_key_required": False,
                "rate_limit": 1.0,
                "free": True,
                "note": "Blocks automated access, limited functionality",
            },
        }

        return requirements.get(database_name, {})


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)

    # Test ArXiv adapter
    arxiv = AdapterFactory.create_adapter("arxiv", rate_limit=3.0)
    if arxiv and arxiv.is_available():
        print(f"Testing {arxiv.get_database_name()} adapter...")
        papers = arxiv.search("machine learning", max_results=5)
        print(f"Found {len(papers)} papers")

        stats = arxiv.get_stats()
        print(f"Adapter stats: {stats}")

    # Show supported databases
    databases = AdapterFactory.get_supported_databases()
    print(f"\nSupported databases: {databases}")

    # Show requirements for each database
    print(f"\nDatabase requirements:")
    for db in databases:
        req = AdapterFactory.get_adapter_requirements(db)
        print(f"  {db}: {req}")
