#!/usr/bin/env python3
import time
import json
import urllib.request
import urllib.parse
import urllib.error
import xml.etree.ElementTree as ET
import re
import requests
import logging
from bs4 import BeautifulSoup
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
from datetime import datetime
from .paper import Paper, PaperFormatter


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

    def to_google_scholar_query(self, parsed_query: Dict) -> str:
        """Convert parsed query to Google Scholar format."""
        terms = []

        for group in parsed_query["groups"]:
            if group["type"] == "OR":
                group_terms = " OR ".join(
                    ['"' + term.replace('"', "") + '"' for term in group["terms"]]
                )
                terms.append(f"({group_terms})")
            elif group["type"] == "AND":
                group_terms = " ".join(
                    ['"' + term.replace('"', "") + '"' for term in group["terms"]]
                )
                terms.append(group_terms)
            else:
                clean_term = group["terms"][0].replace('"', "")
                terms.append(f'"{clean_term}"')

        final_query = " ".join(terms)
        self.logger.debug(f"Converted to Google Scholar query: {final_query}")
        return final_query


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
        self._availability_cache = None
        self._availability_check_time = 0
        self._availability_cache_duration = 300  # 5 minutes

        # Statistics
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_papers_found": 0,
            "last_request_time": None,
            "rate_limit_hits": 0,
            "availability_checks": 0,
            "last_availability_check": None,
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

    def _log_availability_check(self):
        """Log availability check statistics."""
        self.stats["availability_checks"] += 1
        self.stats["last_availability_check"] = datetime.now().isoformat()

    def _check_live_availability(self) -> bool:
        """Perform live availability check. Override in subclasses."""
        return True

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
            "availability_checks": 0,
            "last_availability_check": None,
        }

    def is_available(self) -> bool:
        """Check if the database adapter is available and configured with caching."""
        current_time = time.time()

        # Use cached result if still valid
        if (
            self._availability_cache is not None
            and current_time - self._availability_check_time
            < self._availability_cache_duration
        ):
            return self._availability_cache

        # Perform live check
        self._log_availability_check()
        try:
            self._availability_cache = self._check_live_availability()
        except Exception as e:
            self.logger.warning(f"Availability check failed: {e}")
            self._availability_cache = False

        self._availability_check_time = current_time
        return self._availability_cache

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

    def _check_live_availability(self) -> bool:
        """Check if arXiv API is accessible."""
        try:
            test_url = f"{self.base_url}?search_query=all:test&max_results=1"
            with urllib.request.urlopen(test_url, timeout=10) as response:
                return response.status == 200
        except Exception as e:
            self.logger.debug(f"ArXiv availability check failed: {e}")
            return False

    def search(
        self, query: str, max_results: int = 100, categories: List[str] = None, **kwargs
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


class IEEEAdapter(DatabaseAdapter):
    """Adapter for IEEE Xplore API with Boolean query support."""

    def __init__(self, api_key: str, rate_limit: float = 100.0):
        super().__init__(rate_limit=rate_limit, api_key=api_key)
        self.base_url = "http://ieeexploreapi.ieee.org/api/v1/search/articles"

    def get_database_name(self) -> str:
        return "ieee"

    def _check_live_availability(self) -> bool:
        """Check if IEEE API is accessible and API key is valid."""
        if not self.api_key:
            return False

        try:
            params = {
                "apikey": self.api_key,
                "querytext": "test",
                "max_records": 1,
                "start_record": 1,
            }
            url = f"{self.base_url}?{urllib.parse.urlencode(params)}"

            with urllib.request.urlopen(url, timeout=10) as response:
                data = json.loads(response.read().decode("utf-8"))
                # Check for valid response structure
                return "articles" in data or "total_records" in data
        except Exception as e:
            self.logger.debug(f"IEEE availability check failed: {e}")
            return False

    def _rate_limit_wait(self):
        """IEEE has hourly rate limits."""
        elapsed = time.time() - self.last_request
        if elapsed < (3600.0 / self.rate_limit):
            time.sleep((3600.0 / self.rate_limit) - elapsed)
        self.last_request = time.time()

    def search(
        self, query: str, max_results: int = 100, year_min: int = None, **kwargs
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

    def _check_live_availability(self) -> bool:
        """Check if Elsevier API is accessible and API key is valid."""
        if not self.api_key:
            return False

        try:
            headers = {"X-ELS-APIKey": self.api_key, "Accept": "application/json"}
            params = {"query": "test", "count": 1}
            url = f"{self.base_url}?{urllib.parse.urlencode(params)}"

            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode("utf-8"))
                return "search-results" in data
        except Exception as e:
            self.logger.debug(f"Elsevier availability check failed: {e}")
            return False

    def search(self, query: str, max_results: int = 100, **kwargs) -> List[Paper]:
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

    def _check_live_availability(self) -> bool:
        """Check if Springer API is accessible and API key is valid."""
        if not self.api_key:
            return False

        try:
            params = {
                "api_key": self.api_key,
                "q": "test",
                "s": 1,
                "p": 1,
            }
            url = f"{self.base_url}?{urllib.parse.urlencode(params)}"

            with urllib.request.urlopen(url, timeout=10) as response:
                data = json.loads(response.read().decode("utf-8"))
                return "records" in data or "result" in data
        except Exception as e:
            self.logger.debug(f"Springer availability check failed: {e}")
            return False

    def search(
        self, query: str, max_results: int = 100, year_min: int = None, **kwargs
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

    def _check_live_availability(self) -> bool:
        """Check if Wiley API is accessible and API key is valid."""
        if not self.api_key:
            return False

        try:
            headers = {
                "Wiley-TDM-Client-Token": self.api_key,
                "Accept": "application/json",
            }
            params = {"query": "test", "count": 1}
            url = f"{self.base_url}?{urllib.parse.urlencode(params)}"

            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode("utf-8"))
                return "items" in data or response.status == 200
        except Exception as e:
            self.logger.debug(f"Wiley availability check failed: {e}")
            return False

    def search(self, query: str, max_results: int = 100, **kwargs) -> List[Paper]:
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
    """
    Database adapter for the ACM Digital Library.

    NOTE: This adapter uses web scraping as ACM does not provide a public
    search API. It is fragile and may break if ACM changes its website layout.
    """

    BASE_URL = "https://dl.acm.org/"

    def __init__(self, rate_limit: float = 0.5):  # A gentler rate limit for scraping
        super().__init__(rate_limit=rate_limit)
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
        )

    def get_database_name(self) -> str:
        """Get the name of this database."""
        return "ACM Digital Library"

    def _check_live_availability(self) -> bool:
        """Check if the ACM Digital Library is reachable and functional."""
        try:
            # Test both the main site and search functionality
            response = self.session.head(self.BASE_URL, timeout=10)
            if response.status_code != 200:
                return False

            # Test a simple search to verify functionality
            test_search_url = urllib.parse.urljoin(
                self.BASE_URL, "action/doSearch?AllField=test"
            )
            response = self.session.get(test_search_url, timeout=10)

            # Check if we get a valid search results page
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                # Look for search results container or no results message
                return (
                    soup.select("div.issue-item__content")
                    or soup.select("div.search__no-results")
                    or "search" in response.text.lower()
                )

            return False
        except Exception as e:
            self.logger.debug(f"ACM availability check failed: {e}")
            return False

    def preprocess_query(self, query: str) -> str:
        """URL-encode the query for the search endpoint."""
        return urllib.parse.quote_plus(query.strip())

    def search(self, query: str, max_results: int = 20, **kwargs) -> List[Paper]:
        """
        Search the ACM Digital Library by scraping its search results page.

        Args:
                query: The search term.
                max_results: Maximum number of results to return (default is 20 per page).
                **kwargs: Not used in this implementation.

        Returns:
                A list of Paper objects.
        """
        if not self.is_available():
            self.logger.warning("ACM Digital Library is not available")
            return []

        if not self.validate_query(query):
            self.logger.warning("Search query is empty or invalid.")
            return []

        processed_query = self.preprocess_query(query)
        # ACM uses 'AllField' for a general search
        search_url = urllib.parse.urljoin(
            self.BASE_URL, f"action/doSearch?AllField={processed_query}"
        )

        self._rate_limit_wait()
        papers = []
        try:
            self.logger.info(f"Searching ACM for: '{query}'")
            response = self.session.get(search_url, timeout=15)
            response.raise_for_status()  # Raise an exception for bad status codes

            soup = BeautifulSoup(response.text, "html.parser")

            # Find all list items that contain search results
            results = soup.select("div.issue-item__content")

            if not results:
                self.logger.info("No search results found on the page.")

            for item in results[:max_results]:
                try:
                    title_tag = item.select_one("h5.issue-item__title a")
                    title = title_tag.get_text(strip=True) if title_tag else "N/A"
                    url = (
                        urllib.parse.urljoin(self.BASE_URL, title_tag["href"])
                        if title_tag and title_tag.has_attr("href")
                        else None
                    )

                    # Extract authors from the author list
                    author_tags = item.select("ul[aria-label='authors'] li a")
                    authors = [author.get_text(strip=True) for author in author_tags]

                    # Extract DOI
                    doi_tag = item.select_one("a.issue-item__doi")
                    doi = doi_tag.get_text(strip=True) if doi_tag else None

                    # Extract publication date
                    date_tag = item.select_one("span.epub-section__date")
                    pub_date = date_tag.get_text(strip=True) if date_tag else None

                    # Extract year from publication date
                    year = 0
                    if pub_date:
                        year_match = re.search(r"\b(19|20)\d{2}\b", pub_date)
                        if year_match:
                            year = int(year_match.group())

                    # Abstract/snippet is not reliably available on the search page
                    abstract = "Abstract not available on search results page."

                    paper = Paper(
                        title=title,
                        authors=authors,
                        doi=doi,
                        url=url,
                        abstract=abstract,
                        year=year,
                        database_source=self.get_database_name(),
                        publication_type="article",
                        venue="ACM Digital Library",
                        publisher="ACM",
                    )
                    papers.append(paper)
                except Exception as e:
                    self.logger.error(f"Error parsing a paper item: {e}", exc_info=True)

            self._log_request(success=True, papers_found=len(papers))
            self.logger.info(f"Found {len(papers)} papers on ACM for query '{query}'.")

        except requests.RequestException as e:
            self.logger.error(f"Failed to fetch search results from ACM: {e}")
            self._log_request(success=False)

        return papers


class GoogleScholarAdapter(DatabaseAdapter):
    """
    Enhanced Google Scholar adapter with actual functionality.

    Note: Google Scholar actively blocks automated requests and requires
    careful rate limiting and proper headers to work reliably.
    """

    BASE_URL = "https://scholar.google.com/"

    def __init__(self, rate_limit: float = 0.2):  # Very conservative rate limit
        super().__init__(rate_limit=rate_limit)
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive",
            }
        )

    def get_database_name(self) -> str:
        return "google_scholar"

    def _check_live_availability(self) -> bool:
        """Check if Google Scholar is accessible without triggering blocks."""
        try:
            # Test with a simple HEAD request to avoid triggering anti-bot measures
            response = self.session.head(self.BASE_URL, timeout=10)

            # Google Scholar returns 200 for the main page
            if response.status_code == 200:
                return True

            # Sometimes returns 302 redirects which may still indicate availability
            if response.status_code in [302, 301]:
                return True

            return False
        except Exception as e:
            self.logger.debug(f"Google Scholar availability check failed: {e}")
            return False

    def _is_blocked(self, response_text: str) -> bool:
        """Check if we've been blocked by Google Scholar."""
        blocked_indicators = [
            "automated queries",
            "unusual traffic",
            "captcha",
            "blocked",
            "robot",
            "bot detection",
        ]

        text_lower = response_text.lower()
        return any(indicator in text_lower for indicator in blocked_indicators)

    def _extract_citation_count(self, citation_text: str) -> int:
        """Extract citation count from citation text."""
        if not citation_text:
            return 0

        # Look for "Cited by X" pattern
        match = re.search(r"cited by (\d+)", citation_text.lower())
        if match:
            return int(match.group(1))

        return 0

    def _parse_scholar_result(self, result_div) -> Optional[Paper]:
        """Parse a single Google Scholar search result."""
        try:
            # Extract title
            title_elem = result_div.select_one("h3 a")
            title = title_elem.get_text(strip=True) if title_elem else ""
            url = title_elem.get("href", "") if title_elem else ""

            # Extract authors and publication info
            authors = []
            venue = ""
            year = 0

            # The publication info is usually in a div with class 'gs_a'
            pub_info = result_div.select_one(".gs_a")
            if pub_info:
                pub_text = pub_info.get_text(strip=True)

                # Try to extract year (usually at the end)
                year_match = re.search(r"\b(19|20)\d{2}\b", pub_text)
                if year_match:
                    year = int(year_match.group())

                # Authors are usually before the first dash
                if " - " in pub_text:
                    author_part = pub_text.split(" - ")[0]
                    # Split by commas and clean up
                    authors = [author.strip() for author in author_part.split(",")]

                    # Venue info is often after the first dash
                    venue_parts = pub_text.split(" - ")[1:]
                    venue = " - ".join(venue_parts)

            # Extract abstract/snippet
            abstract_elem = result_div.select_one(".gs_rs")
            abstract = abstract_elem.get_text(strip=True) if abstract_elem else ""

            # Extract citation count
            citations = 0
            citation_elem = result_div.select_one(".gs_fl a")
            if citation_elem and "cited by" in citation_elem.get_text().lower():
                citations = self._extract_citation_count(citation_elem.get_text())

            return Paper(
                title=title,
                authors=authors,
                abstract=abstract,
                year=year,
                venue=venue,
                url=url,
                citations=citations,
                database_source=self.get_database_name(),
                publication_type="article",
            )

        except Exception as e:
            self.logger.error(f"Error parsing Google Scholar result: {e}")
            return None

    def search(
        self, query: str, max_results: int = 20, year_min: int = None, **kwargs
    ) -> List[Paper]:
        """
        Search Google Scholar with careful rate limiting and error handling.

        Args:
                query: Search query
                max_results: Maximum number of results (limited to reduce blocking risk)
                year_min: Minimum publication year
                **kwargs: Additional parameters

        Returns:
                List[Paper]: Found papers (may be empty if blocked)
        """
        if not self.is_available():
            self.logger.warning("Google Scholar is not available")
            return []

        # Limit max results to reduce blocking risk
        max_results = min(max_results, 20)

        self._rate_limit_wait()

        try:
            # Parse and format query
            parsed_query = self.query_parser.parse_boolean_query(query)
            scholar_query = self.query_parser.to_google_scholar_query(parsed_query)

            # Add year filter if specified
            if year_min:
                scholar_query += f" after:{year_min}"

            # Build search URL
            params = {"q": scholar_query, "hl": "en", "num": max_results, "start": 0}

            search_url = f"{self.BASE_URL}scholar?" + urllib.parse.urlencode(params)
            self.logger.info(f"Searching Google Scholar: {search_url}")

            # Add random delay to appear more human-like
            import random

            time.sleep(random.uniform(1, 3))

            response = self.session.get(search_url, timeout=15)
            response.raise_for_status()

            # Check if we've been blocked
            if self._is_blocked(response.text):
                self.logger.warning("Google Scholar has blocked our request")
                self._log_request(success=False)
                return []

            soup = BeautifulSoup(response.text, "html.parser")

            # Find search results
            results = soup.select(
                "div.gs_r, div[data-lid]"
            )  # Different possible selectors

            papers = []
            for result in results[:max_results]:
                paper = self._parse_scholar_result(result)
                if paper:
                    papers.append(paper)

            self.logger.info(f"Google Scholar returned {len(papers)} papers")
            self._log_request(success=True, papers_found=len(papers))

            return papers

        except Exception as e:
            self.logger.error(f"Google Scholar search failed: {e}")
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
        adapters = AdapterFactory.SUPPORTED_DATABASES
        if database_name not in adapters:
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
            "arxiv": {
                "api_key_required": False,
                "rate_limit": 3.0,
                "free": True,
                "scraping_based": False,
                "reliability": "high",
                "database_url": "https://arxiv.org/",
                "api_docs_url": "https://info.arxiv.org/help/api/index.html",
                "note": "Official XML API with excellent reliability and comprehensive metadata.",
                "features": [
                    "title",
                    "authors",
                    "abstract",
                    "categories",
                    "arxiv_id",
                    "pdf_url",
                    "publication_date",
                ],
                "limitations": [
                    "preprints_only",
                    "no_citation_counts",
                    "academic_categories_only",
                ],
            },
            "ieee": {
                "api_key_required": True,
                "rate_limit": 100.0,
                "free": False,
                "scraping_based": False,
                "reliability": "high",
                "database_url": "https://ieeexplore.ieee.org/",
                "api_docs_url": "https://developer.ieee.org/",
                "api_signup_url": "https://developer.ieee.org/",
                "note": "Official REST API requiring subscription. High-quality engineering and computer science papers.",
                "features": [
                    "title",
                    "authors",
                    "abstract",
                    "doi",
                    "citation_counts",
                    "publication_info",
                    "pdf_access",
                ],
                "limitations": [
                    "requires_paid_subscription",
                    "engineering_focus",
                    "rate_limited_per_hour",
                ],
            },
            "elsevier": {
                "api_key_required": True,
                "rate_limit": 100.0,
                "free": False,
                "scraping_based": False,
                "reliability": "high",
                "database_url": "https://www.sciencedirect.com/",
                "api_docs_url": "https://dev.elsevier.com/",
                "api_signup_url": "https://dev.elsevier.com/",
                "note": "Official ScienceDirect API requiring subscription. Extensive collection across multiple disciplines.",
                "features": [
                    "title",
                    "authors",
                    "abstract",
                    "doi",
                    "full_text_access",
                    "publication_info",
                    "subject_areas",
                ],
                "limitations": [
                    "requires_paid_subscription",
                    "complex_authentication",
                    "usage_quotas",
                ],
            },
            "springer": {
                "api_key_required": True,
                "rate_limit": 100.0,
                "free": True,
                "scraping_based": False,
                "reliability": "high",
                "database_url": "https://link.springer.com/",
                "api_docs_url": "https://dev.springernature.com/",
                "api_signup_url": "https://dev.springernature.com/",
                "note": "Official API with free tier available. Good coverage of scientific literature.",
                "features": [
                    "title",
                    "authors",
                    "abstract",
                    "doi",
                    "publication_info",
                    "open_access_indicators",
                ],
                "limitations": [
                    "rate_limits_on_free_tier",
                    "some_content_requires_subscription",
                ],
            },
            "wiley": {
                "api_key_required": True,
                "rate_limit": 100.0,
                "free": False,
                "scraping_based": False,
                "reliability": "medium",
                "database_url": "https://onlinelibrary.wiley.com/",
                "api_docs_url": "https://onlinelibrary.wiley.com/library-info/resources/text-and-datamining",
                "api_signup_url": "https://onlinelibrary.wiley.com/library-info/resources/text-and-datamining",
                "note": "TDM (Text and Data Mining) API requiring institutional access. Focus on academic journals.",
                "features": [
                    "title",
                    "authors",
                    "abstract",
                    "doi",
                    "publication_info",
                    "full_text_mining",
                ],
                "limitations": [
                    "requires_institutional_access",
                    "complex_authentication",
                    "limited_free_access",
                ],
            },
            "acm": {
                "api_key_required": False,
                "rate_limit": 0.5,
                "free": True,
                "scraping_based": True,
                "reliability": "medium",
                "database_url": "https://dl.acm.org/",
                "api_docs_url": None,
                "note": "Uses web scraping with robust live availability checks. May break if ACM changes website structure.",
                "features": [
                    "title",
                    "authors",
                    "doi",
                    "publication_date",
                    "basic_metadata",
                ],
                "limitations": [
                    "no_abstracts_in_search_results",
                    "limited_to_20_results_per_page",
                    "web_scraping_fragility",
                ],
            },
            "google_scholar": {
                "api_key_required": False,
                "rate_limit": 0.2,
                "free": True,
                "scraping_based": True,
                "reliability": "low",
                "database_url": "https://scholar.google.com/",
                "api_docs_url": None,
                "note": "Functional but actively blocks automated access. Use very conservative rate limits and expect occasional failures.",
                "features": [
                    "title",
                    "authors",
                    "citations",
                    "abstracts",
                    "publication_info",
                    "year_filtering",
                ],
                "limitations": [
                    "frequent_blocking",
                    "captcha_challenges",
                    "rate_limiting_required",
                    "results_limited_to_20",
                ],
                "recommendations": [
                    "use_as_fallback_only",
                    "implement_captcha_handling",
                    "rotate_user_agents",
                ],
            },
        }

        return requirements.get(database_name, {})


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)

    # Test ArXiv adapter with live availability check
    arxiv = AdapterFactory.create_adapter("arxiv", rate_limit=3.0)
    if arxiv and arxiv.is_available():
        print(f"Testing {arxiv.get_database_name()} adapter...")
        papers = arxiv.search("machine learning", max_results=5)
        print(f"Found {len(papers)} papers")

        stats = arxiv.get_stats()
        print(f"Adapter stats: {stats}")
    else:
        print("ArXiv is not available")

    # Test ACM adapter
    print("\n" + "=" * 50)
    acm = AdapterFactory.create_adapter("acm", rate_limit=0.5)
    if acm:
        print(f"Testing {acm.get_database_name()} adapter...")
        if acm.is_available():
            print("✓ ACM is available, attempting search...")
            papers = acm.search("machine learning", max_results=3)
            print(f"Found {len(papers)} papers from ACM")
            if papers:
                print(f"Sample paper: {papers[0].title}")
        else:
            print(
                "✗ ACM Digital Library is not available (website may be down or blocking)"
            )
    else:
        print("✗ Could not create ACM adapter")

    # Test Google Scholar (with comprehensive warnings)
    print("\n" + "=" * 50)
    scholar = AdapterFactory.create_adapter("google_scholar", rate_limit=0.2)
    if scholar:
        print(f"Testing {scholar.get_database_name()} adapter...")
        print("⚠️  WARNING: Google Scholar aggressively blocks automated requests")
        print("⚠️  This test may fail or trigger CAPTCHA challenges")
        if scholar.is_available():
            print("✓ Google Scholar appears accessible, attempting search...")
            papers = scholar.search("deep learning", max_results=5)
            if papers:
                print(f"✓ Successfully found {len(papers)} papers from Google Scholar")
                print(f"Sample paper: {papers[0].title}")
            else:
                print("✗ No papers returned (likely blocked or no results)")
        else:
            print("✗ Google Scholar is not available or blocking requests")
    else:
        print("✗ Could not create Google Scholar adapter")

    # Show supported databases
    databases = AdapterFactory.get_supported_databases()
    print(f"\nSupported databases: {databases}")

    # Show requirements for each database
    print(f"\nDatabase requirements:")
    for db in databases:
        req = AdapterFactory.get_adapter_requirements(db)
        print(f"  {db}: {req}")

    # Test availability for all adapters
    print(f"\nLive availability check:")
    for db in databases:
        adapter = AdapterFactory.create_adapter(db)
        if adapter:
            available = adapter.is_available()
            print(f"  {db}: {'✓ Available' if available else '✗ Not available'}")
        else:
            print(f"  {db}: ✗ Could not create adapter (missing API key?)")
