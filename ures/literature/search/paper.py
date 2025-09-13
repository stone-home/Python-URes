#!/usr/bin/env python3
"""
Database Models and Paper Representation
Handles data models, database operations, and caching for literature search.
"""

import json
import sqlite3
import hashlib
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import re
import logging
from datetime import datetime


@dataclass
class Paper:
    """Represents a research paper with standardized metadata."""

    title: str
    authors: List[str]
    abstract: str = ""
    year: int = 0
    venue: str = ""
    doi: str = ""
    arxiv_id: str = ""
    url: str = ""
    citations: int = 0
    keywords: List[str] = None
    database_source: str = ""
    pdf_url: str = ""
    publication_type: str = ""  # article, conference, preprint, etc.
    issue: str = ""
    volume: str = ""
    pages: str = ""
    publisher: str = ""

    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []
        self._normalize_fields()

    def _normalize_fields(self):
        """Normalize and clean all fields to ensure consistency."""
        self.title = self._clean_text(self.title)
        self.authors = [
            self._clean_author_name(author) for author in self.authors if author.strip()
        ]
        self.abstract = self._clean_text(self.abstract)

        if not isinstance(self.year, int) or self.year < 1900 or self.year > 2030:
            self.year = 0

        self.venue = self._clean_text(self.venue)
        self.doi = self._normalize_doi(self.doi)
        self.publication_type = self._clean_text(self.publication_type).lower()
        self.issue = self._clean_text(self.issue)
        self.volume = self._clean_text(self.volume)
        self.pages = self._clean_text(self.pages)
        self.publisher = self._clean_text(self.publisher)

        try:
            self.citations = max(0, int(self.citations))
        except (ValueError, TypeError):
            self.citations = 0

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text fields."""
        if not text or not isinstance(text, str):
            return ""

        text = re.sub(r"\s+", " ", text.strip())
        text = re.sub(r"<[^>]+>", "", text)
        text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
        return text

    def _clean_author_name(self, author: str) -> str:
        """Clean and normalize author names."""
        if not author or not isinstance(author, str):
            return ""

        author = re.sub(r"\s+", " ", author.strip())
        author = re.sub(r"\([^)]*\)", "", author)
        author = re.sub(r"\S+@\S+", "", author)
        return author.strip()

    def _normalize_doi(self, doi: str) -> str:
        """Normalize DOI format."""
        if not doi or not isinstance(doi, str):
            return ""

        doi = re.sub(
            r"^(doi:|DOI:|https?://doi\.org/|https?://dx\.doi\.org/)", "", doi.strip()
        )
        if re.match(r"^10\.\d+/.+", doi):
            return doi
        return ""

    def get_canonical_id(self) -> str:
        """Generate a canonical identifier for deduplication."""
        if self.doi:
            return f"doi:{self.doi}"
        elif self.arxiv_id:
            return f"arxiv:{self.arxiv_id}"
        else:
            title_norm = re.sub(r"[^\w\s]", "", self.title.lower()).strip()
            first_author = self.authors[0] if self.authors else ""
            content = f"{title_norm}|{first_author}|{self.year}"
            return f"hash:{hashlib.md5(content.encode()).hexdigest()}"

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "Paper":
        """Create Paper instance from dictionary."""
        defaults = {
            "title": "",
            "authors": [],
            "abstract": "",
            "year": 0,
            "venue": "",
            "doi": "",
            "arxiv_id": "",
            "url": "",
            "citations": 0,
            "keywords": [],
            "database_source": "",
            "pdf_url": "",
            "publication_type": "",
            "issue": "",
            "volume": "",
            "pages": "",
            "publisher": "",
        }
        paper_data = {**defaults, **data}
        return cls(**paper_data)

    def similarity_score(self, other: "Paper") -> float:
        """Calculate similarity score with another paper (0-1)."""
        if not isinstance(other, Paper):
            return 0.0

        score = 0.0

        # Title similarity (highest weight)
        if self.title and other.title:
            title1 = re.sub(r"[^\w\s]", "", self.title.lower())
            title2 = re.sub(r"[^\w\s]", "", other.title.lower())
            if title1 == title2:
                score += 0.5
            elif title1 in title2 or title2 in title1:
                score += 0.3

        # Author overlap
        if self.authors and other.authors:
            common_authors = set(self.authors) & set(other.authors)
            if common_authors:
                score += 0.2 * (
                    len(common_authors) / max(len(self.authors), len(other.authors))
                )

        # DOI or arXiv ID match
        if self.doi and other.doi and self.doi == other.doi:
            score += 0.3
        elif self.arxiv_id and other.arxiv_id and self.arxiv_id == other.arxiv_id:
            score += 0.3

        # Year proximity
        if self.year and other.year:
            year_diff = abs(self.year - other.year)
            if year_diff == 0:
                score += 0.1
            elif year_diff <= 1:
                score += 0.05

        return min(score, 1.0)


class CacheManager:
    """Manages local caching of search results and paper metadata."""

    def __init__(self, cache_dir: str = "./lit_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.db_path = self.cache_dir / "papers.db"
        self.logger = logging.getLogger(__name__)
        self._init_database()

    def _init_database(self):
        """Initialize SQLite database for caching."""
        with sqlite3.connect(self.db_path) as conn:
            # Enable foreign keys
            conn.execute("PRAGMA foreign_keys = ON")

            # Drop existing tables to ensure clean schema
            conn.execute("DROP TABLE IF EXISTS search_results")
            conn.execute("DROP TABLE IF EXISTS searches")
            conn.execute("DROP TABLE IF EXISTS papers")

            # Create papers table
            conn.execute(
                """
                         CREATE TABLE papers
                         (
                             id               TEXT PRIMARY KEY,
                             title            TEXT NOT NULL,
                             authors          TEXT,
                             abstract         TEXT,
                             year             INTEGER,
                             venue            TEXT,
                             doi              TEXT,
                             arxiv_id         TEXT,
                             url              TEXT,
                             citations        INTEGER,
                             keywords         TEXT,
                             database_source  TEXT,
                             pdf_url          TEXT,
                             publication_type TEXT,
                             issue            TEXT,
                             volume           TEXT,
                             pages            TEXT,
                             publisher        TEXT,
                             cached_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                             updated_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                         )
			             """
            )

            # Create searches table
            conn.execute(
                """
                         CREATE TABLE searches
                         (
                             id            INTEGER PRIMARY KEY AUTOINCREMENT,
                             query_hash    TEXT UNIQUE,
                             query         TEXT NOT NULL,
                             databases     TEXT,
                             total_results INTEGER,
                             searched_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                         )
			             """
            )

            # Create search_results junction table
            conn.execute(
                """
                         CREATE TABLE search_results
                         (
                             search_id       INTEGER,
                             paper_id        TEXT,
                             relevance_score REAL DEFAULT 0.0,
                             FOREIGN KEY (search_id) REFERENCES searches (id) ON DELETE CASCADE,
                             FOREIGN KEY (paper_id) REFERENCES papers (id) ON DELETE CASCADE,
                             PRIMARY KEY (search_id, paper_id)
                         )
			             """
            )

            # Create indexes for better performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_papers_title ON papers(title)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_papers_year ON papers(year)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_papers_source ON papers(database_source)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_papers_doi ON papers(doi)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_papers_arxiv ON papers(arxiv_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_searches_hash ON searches(query_hash)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_searches_date ON searches(searched_at)"
            )

    def _generate_paper_id(self, paper: Paper) -> str:
        """Generate unique ID for a paper."""
        content = f"{paper.title}|{paper.authors}|{paper.year}|{paper.database_source}"
        return hashlib.md5(content.encode()).hexdigest()

    def _generate_query_hash(self, query: str, databases: List[str]) -> str:
        """Generate hash for search query."""
        content = f"{query}|{sorted(databases)}"
        return hashlib.md5(content.encode()).hexdigest()

    def cache_paper(self, paper: Paper) -> bool:
        """
        Cache a paper in the database.

        Args:
                paper: Paper object to cache

        Returns:
                bool: Success status
        """
        try:
            paper_id = self._generate_paper_id(paper)
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO papers
                    (id, title, authors, abstract, year, venue, doi, arxiv_id, url, citations,
                     keywords, database_source, pdf_url, publication_type, issue, volume, pages, publisher, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                    (
                        paper_id,
                        paper.title,
                        json.dumps(paper.authors),
                        paper.abstract,
                        paper.year,
                        paper.venue,
                        paper.doi,
                        paper.arxiv_id,
                        paper.url,
                        paper.citations,
                        json.dumps(paper.keywords),
                        paper.database_source,
                        paper.pdf_url,
                        paper.publication_type,
                        paper.issue,
                        paper.volume,
                        paper.pages,
                        paper.publisher,
                    ),
                )
            return True
        except Exception as e:
            self.logger.error(f"Failed to cache paper: {e}")
            return False

    def get_cached_papers(
        self,
        query: str = None,
        year_min: int = None,
        database_source: str = None,
        limit: int = None,
    ) -> List[Paper]:
        """
        Retrieve cached papers with optional filtering.

        Args:
                query: Search within title and abstract
                year_min: Minimum publication year
                database_source: Filter by database source
                limit: Maximum number of results

        Returns:
                List[Paper]: List of matching papers
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                sql = """
                      SELECT id, \
                             title, \
                             authors, \
                             abstract, year, venue, doi, arxiv_id, url, citations, keywords, database_source, pdf_url, publication_type, issue, volume, pages, publisher
                      FROM papers \
                      WHERE 1=1 \
				      """
                params = []

                if query:
                    sql += " AND (title LIKE ? OR abstract LIKE ?)"
                    params.extend([f"%{query}%", f"%{query}%"])

                if year_min:
                    sql += " AND year >= ?"
                    params.append(year_min)

                if database_source:
                    sql += " AND database_source = ?"
                    params.append(database_source)

                sql += " ORDER BY year DESC, citations DESC"

                if limit:
                    sql += " LIMIT ?"
                    params.append(limit)

                cursor = conn.execute(sql, params)
                papers = []
                for row in cursor.fetchall():
                    paper_data = {
                        "title": row[1] or "",
                        "authors": json.loads(row[2]) if row[2] else [],
                        "abstract": row[3] or "",
                        "year": row[4] or 0,
                        "venue": row[5] or "",
                        "doi": row[6] or "",
                        "arxiv_id": row[7] or "",
                        "url": row[8] or "",
                        "citations": row[9] or 0,
                        "keywords": json.loads(row[10]) if row[10] else [],
                        "database_source": row[11] or "",
                        "pdf_url": row[12] or "",
                        "publication_type": row[13] or "",
                        "issue": row[14] or "",
                        "volume": row[15] or "",
                        "pages": row[16] or "",
                        "publisher": row[17] or "",
                    }
                    papers.append(Paper.from_dict(paper_data))

                return papers
        except Exception as e:
            self.logger.error(f"Failed to retrieve cached papers: {e}")
            return []

    def cache_search_results(
        self, query: str, databases: List[str], results: List[Paper]
    ) -> bool:
        """
        Cache search results with metadata.

        Args:
                query: Search query
                databases: List of databases searched
                results: List of paper results

        Returns:
                bool: Success status
        """
        try:
            query_hash = self._generate_query_hash(query, databases)

            with sqlite3.connect(self.db_path) as conn:
                # First cache all papers
                for paper in results:
                    self.cache_paper(paper)

                # Insert or update search record
                cursor = conn.execute(
                    """
                    INSERT OR REPLACE INTO searches (query_hash, query, databases, total_results)
                    VALUES (?, ?, ?, ?)
                """,
                    (query_hash, query, json.dumps(databases), len(results)),
                )

                search_id = cursor.lastrowid

                # Clear existing search results
                conn.execute(
                    "DELETE FROM search_results WHERE search_id = ?", (search_id,)
                )

                # Insert search results
                for i, paper in enumerate(results):
                    paper_id = self._generate_paper_id(paper)
                    relevance_score = 1.0 - (
                        i / len(results)
                    )  # Simple relevance based on order

                    conn.execute(
                        """
                                 INSERT INTO search_results (search_id, paper_id, relevance_score)
                                 VALUES (?, ?, ?)
					             """,
                        (search_id, paper_id, relevance_score),
                    )

            return True
        except Exception as e:
            self.logger.error(f"Failed to cache search results: {e}")
            return False

    def get_cached_search(
        self, query: str, databases: List[str], max_age_hours: int = 24
    ) -> Optional[List[Paper]]:
        """
        Get cached search results if recent enough.

        Args:
                query: Search query
                databases: List of databases
                max_age_hours: Maximum age in hours

        Returns:
                Optional[List[Paper]]: Cached results or None
        """
        try:
            query_hash = self._generate_query_hash(query, databases)

            with sqlite3.connect(self.db_path) as conn:
                # Find recent search
                cursor = conn.execute(
                    """
                    SELECT id FROM searches
                    WHERE query_hash = ? AND searched_at > datetime('now', '-{} hours')
                    ORDER BY searched_at DESC LIMIT 1
                """.format(
                        max_age_hours
                    ),
                    (query_hash,),
                )

                row = cursor.fetchone()
                if not row:
                    return None

                search_id = row[0]

                # Get papers from search results
                cursor = conn.execute(
                    """
                                      SELECT p.id,
                                             p.title,
                                             p.authors,
                                             p.abstract,
                                             p.year,
                                             p.venue,
                                             p.doi,
                                             p.arxiv_id,
                                             p.url,
                                             p.citations,
                                             p.keywords,
                                             p.database_source,
                                             p.pdf_url,
                                             p.publication_type,
                                             p.issue,
                                             p.volume,
                                             p.pages,
                                             p.publisher
                                      FROM papers p
                                               JOIN search_results sr ON p.id = sr.paper_id
                                      WHERE sr.search_id = ?
                                      ORDER BY sr.relevance_score DESC
				                      """,
                    (search_id,),
                )

                papers = []
                for row in cursor.fetchall():
                    paper_data = {
                        "title": row[1] or "",
                        "authors": json.loads(row[2]) if row[2] else [],
                        "abstract": row[3] or "",
                        "year": row[4] or 0,
                        "venue": row[5] or "",
                        "doi": row[6] or "",
                        "arxiv_id": row[7] or "",
                        "url": row[8] or "",
                        "citations": row[9] or 0,
                        "keywords": json.loads(row[10]) if row[10] else [],
                        "database_source": row[11] or "",
                        "pdf_url": row[12] or "",
                        "publication_type": row[13] or "",
                        "issue": row[14] or "",
                        "volume": row[15] or "",
                        "pages": row[16] or "",
                        "publisher": row[17] or "",
                    }
                    papers.append(Paper.from_dict(paper_data))

                return papers
        except Exception as e:
            self.logger.error(f"Failed to get cached search: {e}")
            return None

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                stats = {}

                # Paper stats
                cursor = conn.execute("SELECT COUNT(*) FROM papers")
                stats["total_papers"] = cursor.fetchone()[0]

                cursor = conn.execute(
                    "SELECT COUNT(DISTINCT database_source) FROM papers"
                )
                stats["unique_sources"] = cursor.fetchone()[0]

                cursor = conn.execute(
                    "SELECT database_source, COUNT(*) FROM papers GROUP BY database_source"
                )
                stats["papers_by_source"] = dict(cursor.fetchall())

                # Search stats
                cursor = conn.execute("SELECT COUNT(*) FROM searches")
                stats["total_searches"] = cursor.fetchone()[0]

                cursor = conn.execute(
                    """
                                      SELECT COUNT(*)
                                      FROM searches
                                      WHERE searched_at > datetime('now', '-24 hours')
				                      """
                )
                stats["recent_searches"] = cursor.fetchone()[0]

                # Year distribution
                cursor = conn.execute(
                    """
                                      SELECT year, COUNT (*)
                                      FROM papers
                                      WHERE year > 0
                                      GROUP BY year
                                      ORDER BY year DESC
                                          LIMIT 10
				                      """
                )
                stats["recent_years"] = dict(cursor.fetchall())

                return stats
        except Exception as e:
            self.logger.error(f"Failed to get cache stats: {e}")
            return {}

    def cleanup_cache(self, days_old: int = 30) -> bool:
        """
        Clean up old cache entries.

        Args:
                days_old: Remove entries older than this many days

        Returns:
                bool: Success status
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Remove old searches (cascade will remove search_results)
                cursor = conn.execute(
                    """
                    DELETE FROM searches
                    WHERE searched_at < datetime('now', '-{} days')
                """.format(
                        days_old
                    )
                )

                searches_removed = cursor.rowcount

                # Remove orphaned papers (not referenced by any recent search)
                cursor = conn.execute(
                    """
                    DELETE FROM papers
                    WHERE id NOT IN (
                        SELECT DISTINCT sr.paper_id
                        FROM search_results sr
                        JOIN searches s ON sr.search_id = s.id
                        WHERE s.searched_at > datetime('now', '-{} days')
                    )
                    AND cached_at < datetime('now', '-{} days')
                """.format(
                        days_old, days_old
                    )
                )

                papers_removed = cursor.rowcount

                # Vacuum database to reclaim space
                conn.execute("VACUUM")

                self.logger.info(
                    f"Cache cleanup: removed {searches_removed} searches, {papers_removed} papers"
                )
                return True
        except Exception as e:
            self.logger.error(f"Failed to cleanup cache: {e}")
            return False

    def find_duplicates(self, similarity_threshold: float = 0.8) -> List[List[Paper]]:
        """
        Find potential duplicate papers in cache.

        Args:
                similarity_threshold: Minimum similarity score to consider duplicates

        Returns:
                List[List[Paper]]: Groups of similar papers
        """
        try:
            papers = self.get_cached_papers()
            duplicates = []
            processed = set()

            for i, paper1 in enumerate(papers):
                if i in processed:
                    continue

                similar_group = [paper1]
                for j, paper2 in enumerate(papers[i + 1 :], i + 1):
                    if j in processed:
                        continue

                    similarity = paper1.similarity_score(paper2)
                    if similarity >= similarity_threshold:
                        similar_group.append(paper2)
                        processed.add(j)

                if len(similar_group) > 1:
                    duplicates.append(similar_group)

                processed.add(i)

            return duplicates
        except Exception as e:
            self.logger.error(f"Failed to find duplicates: {e}")
            return []

    def export_papers(
        self, format: str = "json", filename: str = None
    ) -> Optional[str]:
        """
        Export all cached papers to file.

        Args:
                format: Export format ('json', 'csv')
                filename: Output filename

        Returns:
                Optional[str]: Filename if successful
        """
        try:
            papers = self.get_cached_papers()

            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"cached_papers_{timestamp}.{format}"

            if format == "json":
                data = [paper.to_dict() for paper in papers]
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

            elif format == "csv":
                import csv

                if papers:
                    with open(filename, "w", newline="", encoding="utf-8") as f:
                        fieldnames = [
                            "title",
                            "authors",
                            "year",
                            "venue",
                            "abstract",
                            "doi",
                            "url",
                            "citations",
                            "database_source",
                            "publication_type",
                        ]
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()

                        for paper in papers:
                            writer.writerow(
                                {
                                    "title": paper.title,
                                    "authors": "; ".join(paper.authors),
                                    "year": paper.year,
                                    "venue": paper.venue,
                                    "abstract": paper.abstract,
                                    "doi": paper.doi,
                                    "url": paper.url,
                                    "citations": paper.citations,
                                    "database_source": paper.database_source,
                                    "publication_type": paper.publication_type,
                                }
                            )
            else:
                raise ValueError(f"Unsupported format: {format}")

            return filename
        except Exception as e:
            self.logger.error(f"Failed to export papers: {e}")
            return None


class PaperFormatter:
    """Unified formatter for normalizing papers from different database sources."""

    @staticmethod
    def format_arxiv_paper(entry_data: Dict) -> Paper:
        """Format arXiv entry data into standardized Paper object."""
        title = entry_data.get("title", "").strip().replace("\n", " ")
        title = re.sub(r"\s+", " ", title)

        authors = []
        for author in entry_data.get("authors", []):
            if isinstance(author, dict):
                name = author.get("name", "")
            else:
                name = str(author)
            if name.strip():
                authors.append(name.strip())

        abstract = entry_data.get("summary", "").strip().replace("\n", " ")
        abstract = re.sub(r"\s+", " ", abstract)

        arxiv_id = entry_data.get("id", "").split("/")[-1]
        year = PaperFormatter._extract_arxiv_year(arxiv_id)

        keywords = []
        categories = entry_data.get("categories", [])
        if isinstance(categories, str):
            keywords = [categories]
        elif isinstance(categories, list):
            keywords = categories

        return Paper(
            title=title,
            authors=authors,
            abstract=abstract,
            year=year,
            venue="arXiv",
            arxiv_id=arxiv_id,
            url=entry_data.get("id", ""),
            keywords=keywords,
            database_source="arxiv",
            pdf_url=entry_data.get("pdf_url", ""),
            publication_type="preprint",
        )

    @staticmethod
    def format_ieee_paper(entry_data: Dict) -> Paper:
        """Format IEEE entry data into standardized Paper object."""
        title = entry_data.get("title", "")

        authors = []
        if "authors" in entry_data:
            if isinstance(entry_data["authors"], list):
                authors = [
                    author.get("full_name", "") for author in entry_data["authors"]
                ]
            elif isinstance(entry_data["authors"], dict):
                authors = [entry_data["authors"].get("full_name", "")]

        return Paper(
            title=title,
            authors=authors,
            abstract=entry_data.get("abstract", ""),
            year=int(entry_data.get("publication_year", 0)),
            venue=entry_data.get("publication_title", ""),
            doi=entry_data.get("doi", ""),
            url=entry_data.get("pdf_url", ""),
            citations=int(entry_data.get("citing_paper_count", 0)),
            database_source="ieee",
            pdf_url=entry_data.get("pdf_url", ""),
            publication_type="article",
            volume=entry_data.get("volume", ""),
            issue=entry_data.get("issue", ""),
            pages=PaperFormatter._format_pages(
                entry_data.get("start_page", ""), entry_data.get("end_page", "")
            ),
            publisher="IEEE",
        )

    @staticmethod
    def format_elsevier_paper(entry_data: Dict) -> Paper:
        """Format Elsevier/ScienceDirect entry data."""
        title = entry_data.get("dc:title", "")

        authors = []
        if "authors" in entry_data and "author" in entry_data["authors"]:
            author_list = entry_data["authors"]["author"]
            if isinstance(author_list, list):
                for author in author_list:
                    if isinstance(author, dict):
                        given = author.get("ce:given-name", "")
                        surname = author.get("ce:surname", "")
                        full_name = f"{given} {surname}".strip()
                        if full_name:
                            authors.append(full_name)

        return Paper(
            title=title,
            authors=authors,
            abstract=entry_data.get("dc:description", ""),
            year=(
                int(entry_data.get("prism:coverDate", "").split("-")[0])
                if entry_data.get("prism:coverDate")
                else 0
            ),
            venue=entry_data.get("prism:publicationName", ""),
            doi=entry_data.get("prism:doi", ""),
            url=(
                entry_data.get("link", [{}])[0].get("@href", "")
                if entry_data.get("link")
                else ""
            ),
            database_source="elsevier",
            publication_type="article",
            volume=entry_data.get("prism:volume", ""),
            issue=entry_data.get("prism:issueIdentifier", ""),
            pages=entry_data.get("prism:pageRange", ""),
            publisher="Elsevier",
        )

    @staticmethod
    def format_springer_paper(entry_data: Dict) -> Paper:
        """Format Springer entry data."""
        title = entry_data.get("title", "")

        authors = []
        if "creators" in entry_data:
            for creator in entry_data["creators"]:
                if isinstance(creator, dict):
                    name = creator.get("creator", "")
                else:
                    name = str(creator)
                if name:
                    authors.append(name)

        return Paper(
            title=title,
            authors=authors,
            abstract=entry_data.get("abstract", ""),
            year=(
                int(entry_data.get("publicationDate", "").split("-")[0])
                if entry_data.get("publicationDate")
                else 0
            ),
            venue=entry_data.get("publicationName", ""),
            doi=entry_data.get("doi", ""),
            url=(
                entry_data.get("url", [{}])[0].get("value", "")
                if entry_data.get("url")
                else ""
            ),
            database_source="springer",
            publication_type=entry_data.get("contentType", "article").lower(),
            publisher="Springer",
        )

    @staticmethod
    def format_wiley_paper(entry_data: Dict) -> Paper:
        """Format Wiley entry data."""
        return Paper(
            title=entry_data.get("title", ""),
            authors=entry_data.get("authors", []),
            abstract=entry_data.get("abstract", ""),
            year=int(entry_data.get("publicationYear", 0)),
            venue=entry_data.get("source", ""),
            doi=entry_data.get("doi", ""),
            url=entry_data.get("onlineLibraryUrl", ""),
            database_source="wiley",
            publication_type="article",
            publisher="Wiley",
        )

    @staticmethod
    def _extract_arxiv_year(arxiv_id: str) -> int:
        """Extract publication year from arXiv ID."""
        if not arxiv_id:
            return 0

        if "." in arxiv_id and len(arxiv_id.split(".")[0]) == 4:
            year_part = arxiv_id.split(".")[0][:2]
            year = 2000 + int(year_part)
            return year if year <= datetime.now().year else 1900 + int(year_part)

        if "/" in arxiv_id:
            year_part = arxiv_id.split("/")[1][:2]
            if year_part.isdigit():
                year = 2000 + int(year_part)
                return year if year <= datetime.now().year else 1900 + int(year_part)

        return 0

    @staticmethod
    def _format_pages(start_page: str, end_page: str) -> str:
        """Format page numbers."""
        if start_page and end_page:
            return f"{start_page}-{end_page}"
        elif start_page:
            return start_page
        return ""


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)

    # Test cache manager
    cache = CacheManager("./test_cache")

    # Create test paper
    test_paper = Paper(
        title="Test Paper on Machine Learning",
        authors=["John Doe", "Jane Smith"],
        abstract="This is a test abstract about machine learning algorithms.",
        year=2023,
        venue="Test Conference",
        doi="10.1000/test.doi",
        database_source="test",
    )

    # Cache the paper
    success = cache.cache_paper(test_paper)
    print(f"Paper cached: {success}")

    # Retrieve cached papers
    papers = cache.get_cached_papers(query="machine learning")
    print(f"Found {len(papers)} papers matching query")

    # Get cache stats
    stats = cache.get_cache_stats()
    print(f"Cache stats: {stats}")
