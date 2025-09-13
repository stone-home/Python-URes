#!/usr/bin/env python3
import json
import logging
import csv
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from ures.secrets import SecureKeyManager, StorageMethod
from .paper import Paper, CacheManager
from .adapters import DatabaseAdapter, AdapterFactory


class DatabaseConfig:
    """Enhanced configuration management with secure API key integration."""

    def __init__(
        self, config_dir: Optional[str] = None, app_name: str = "literature-search"
    ):
        config_dir = config_dir or Path.home() / ".ures_lit_search"
        self.config_dir = config_dir
        self.app_name = app_name
        self.logger = logging.getLogger(__name__)
        self.key_manager = SecureKeyManager(app_name, config_dir=self.config_dir)
        self.config = self._load_config()

    @property
    def config_path(self):
        return self.config_dir / "lit_search_config.json"

    def _load_config(self) -> Dict:
        """Load configuration from file or create default."""
        default_config = {
            "databases": {
                "arxiv": {
                    "enabled": True,
                    "rate_limit": 3,
                    "categories": [
                        "cs.LG",
                        "cs.NI",
                        "cs.SY",
                        "cs.PF",
                        "cs.DC",
                        "cs.AI",
                        "cs.SE",
                    ],
                },
                "ieee": {
                    "enabled": False,
                    "rate_limit": 100,
                },
                "springer": {
                    "enabled": False,
                    "rate_limit": 100,
                },
                "elsevier": {
                    "enabled": False,
                    "rate_limit": 100,
                },
                "wiley": {
                    "enabled": False,
                    "rate_limit": 100,
                },
                "acm": {
                    "enabled": False,
                    "rate_limit": 100,
                },
                "google_scholar": {
                    "enabled": False,  # Disabled by default due to blocking
                    "rate_limit": 1,
                },
            },
            "cache": {
                "directory": f"{self.config_dir}/cache",
                "expire_days": 30,
                "max_age_hours": 24,
            },
            "search": {
                "max_results": 100,
                "deduplication": True,
                "min_year": 2018,
                "similarity_threshold": 0.8,
            },
            "export": {
                "default_format": "json",
                "include_abstracts": True,
                "max_export_size": 10000,
            },
        }

        try:
            with open(self.config_path, "r") as f:
                config = json.load(f)
                # Initialize all API key information for each DB with dummy values when missing
                for db_name in default_config["databases"].keys():
                    if self.key_manager.get_key(db_name) is None:
                        self.key_manager.store_key(
                            db_name, "dummy", StorageMethod.ENCRYPTED
                        )
                # Merge with defaults to ensure all keys exist
                for section, values in default_config.items():
                    if section not in config:
                        config[section] = values
                    elif isinstance(values, dict):
                        for key, value in values.items():
                            if key not in config[section]:
                                config[section][key] = value
        except FileNotFoundError:
            config = default_config
            self._save_config(config)

        return config

    def _save_config(self, config: Dict):
        """Save configuration to file."""
        try:
            with open(self.config_path, "w") as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save config: {e}")

    def get_database_config(self, db_name: str) -> Dict:
        """Get configuration for a specific database."""
        return self.config.get("databases", {}).get(db_name, {})

    def is_database_enabled(self, db_name: str) -> bool:
        """Check if a database is enabled."""
        return self.get_database_config(db_name).get("enabled", False)

    def get_api_key(self, db_name: str) -> Optional[str]:
        """Get API key for a database using secure storage."""
        db_config = self.get_database_config(db_name)

        if "api_key" in db_config and db_config["api_key"]:
            return db_config["api_key"]

        # Use secure key manager
        return self.key_manager.get_key(db_name)

    def set_api_key(
        self,
        db_name: str,
        api_key: str,
        method: StorageMethod = StorageMethod.ENCRYPTED,
    ) -> bool:
        """
        Set API key for a database using secure storage.

        Args:
                        db_name: Database name
                        api_key: API key or reference (depends on method)
                        method: Storage method ('encrypted', 'env', '1password', 'keychain')

        Returns:
                        bool: Success status
        """
        # Store the key securely
        success = self.key_manager.store_key(db_name, api_key, method)

        if success:
            # Update config to reflect the storage method
            if "databases" not in self.config:
                self.config["databases"] = {}
            if db_name not in self.config["databases"]:
                self.config["databases"][db_name] = {}

            self.config["databases"][db_name]["api_key_method"] = method
            self.config["databases"][db_name]["enabled"] = True

            self._save_config(self.config)

        return success

    def list_api_keys(self) -> Dict[str, Dict]:
        """List all stored API keys (without revealing actual keys)."""
        return self.key_manager.list_keys()

    def delete_api_key(self, db_name: str) -> bool:
        """Delete API key for a database."""
        success = self.key_manager.delete_key(db_name)

        if success:
            # Update config
            if db_name in self.config.get("databases", {}):
                self.config["databases"][db_name]["enabled"] = False
                self._save_config(self.config)

        return success

    def update_config(self, section: str, key: str, value: Any) -> bool:
        """Update a configuration value."""
        try:
            if section not in self.config:
                self.config[section] = {}

            self.config[section][key] = value
            self._save_config(self.config)
            return True
        except Exception as e:
            self.logger.error(f"Failed to update config: {e}")
            return False

    def get_config_value(self, section: str, key: str, default=None):
        """Get a configuration value."""
        return self.config.get(section, {}).get(key, default)


class LiteratureSearchEngine:
    """Main search engine coordinating multiple database adapters with Boolean query support."""

    def __init__(
        self, config_dir: Optional[str] = None, app_name: str = "literature-search"
    ):
        """
        Initialize the Literature Search Engine.

        Args:
                        config_dir: Path to a directory used to store configuration file
                        app_name: Application name for key management
        """
        self.config = DatabaseConfig(config_dir, app_name)
        self.cache = CacheManager(self.config.config["cache"]["directory"])
        self.adapters: Dict[str, DatabaseAdapter] = {}
        self.logger = logging.getLogger(__name__)

        # Statistics
        self.stats = {
            "total_searches": 0,
            "cached_searches": 0,
            "total_papers_found": 0,
            "databases_used": set(),
            "last_search_time": None,
        }

        self._init_adapters()

    def _init_adapters(self):
        """Initialize database adapters based on configuration."""
        supported_databases = AdapterFactory.get_supported_databases()

        for db_name in supported_databases:
            if not self.config.is_database_enabled(db_name):
                continue

            try:
                db_config = self.config.get_database_config(db_name)
                api_key = self.config.get_api_key(db_name)
                rate_limit = db_config.get("rate_limit", 100)

                adapter = AdapterFactory.create_adapter(
                    db_name, api_key=api_key, rate_limit=rate_limit
                )

                if adapter and adapter.is_available():
                    self.adapters[db_name] = adapter
                    self.logger.info(f"Initialized {db_name} adapter")
                else:
                    if api_key:
                        self.logger.warning(f"Failed to initialize {db_name} adapter")
                    else:
                        self.logger.info(
                            f"{db_name} adapter not configured (missing API key)"
                        )

            except Exception as e:
                self.logger.error(f"Error initializing {db_name} adapter: {e}")

    def search(
        self,
        query: str,
        databases: List[str] = None,
        max_results: int = None,
        use_cache: bool = True,
        year_min: int = None,
        **kwargs,
    ) -> List[Paper]:
        """
        Search multiple databases with Boolean query support.

        Args:
                        query: Boolean search query
                        databases: List of database names to search
                        max_results: Maximum results per database
                        use_cache: Whether to use cached results
                        year_min: Minimum publication year
                        **kwargs: Additional search parameters

        Returns:
                        List of Paper objects, deduplicated and unified
        """
        if not query or not query.strip():
            return []

        if databases is None:
            databases = list(self.adapters.keys())

        if max_results is None:
            max_results = self.config.config["search"]["max_results"]

        if year_min is None:
            year_min = self.config.config["search"]["min_year"]

        self.logger.info(f"Searching with Boolean query: {query}")
        self.logger.info(f"Target databases: {databases}")

        # Update statistics
        self.stats["total_searches"] += 1
        self.stats["last_search_time"] = datetime.now().isoformat()

        # Check cache first
        if use_cache:
            max_age_hours = self.config.config["cache"]["max_age_hours"]
            cached_results = self.cache.get_cached_search(
                query, databases, max_age_hours
            )
            if cached_results:
                self.logger.info(f"Retrieved {len(cached_results)} papers from cache")
                self.stats["cached_searches"] += 1
                return self._filter_by_year(cached_results, year_min)

        # Perform searches across all databases
        all_papers = []
        for db_name in databases:
            if db_name not in self.adapters:
                self.logger.warning(f"Database {db_name} not available")
                continue

            self.logger.info(f"Searching {db_name}...")
            try:
                adapter = self.adapters[db_name]

                # Database-specific parameters
                search_kwargs = {}
                if db_name == "arxiv":
                    categories = self.config.get_database_config("arxiv").get(
                        "categories", []
                    )
                    search_kwargs["categories"] = categories
                if year_min:
                    search_kwargs["year_min"] = year_min

                papers = adapter.search(query, max_results, **search_kwargs)

                # Apply year filter
                if year_min:
                    papers = [p for p in papers if p.year >= year_min]

                all_papers.extend(papers)
                self.logger.info(f"Found {len(papers)} papers from {db_name}")
                self.stats["databases_used"].add(db_name)

                # Cache individual papers
                for paper in papers:
                    self.cache.cache_paper(paper)

            except Exception as e:
                self.logger.error(f"Error searching {db_name}: {e}")

        # Deduplicate results using canonical IDs
        if self.config.config["search"]["deduplication"]:
            all_papers = self._deduplicate_papers(all_papers)

        # Cache search results
        if use_cache and all_papers:
            self.cache.cache_search_results(query, databases, all_papers)

        # Sort by relevance (citations and year)
        all_papers.sort(key=lambda p: (-p.citations, -p.year))

        self.stats["total_papers_found"] += len(all_papers)
        self.logger.info(f"Total unique papers found: {len(all_papers)}")
        return all_papers

    def _filter_by_year(self, papers: List[Paper], year_min: int) -> List[Paper]:
        """Filter papers by minimum year."""
        if not year_min:
            return papers
        return [p for p in papers if p.year >= year_min]

    def _deduplicate_papers(self, papers: List[Paper]) -> List[Paper]:
        """Remove duplicate papers using canonical IDs and similarity."""
        if not papers:
            return papers

        unique_papers = []
        seen_ids = set()

        # Sort papers by priority: DOI > arXiv > citations > year
        papers_sorted = sorted(
            papers,
            key=lambda p: (
                0 if p.doi else 1 if p.arxiv_id else 2,
                -p.citations,
                -p.year,
            ),
        )

        similarity_threshold = self.config.config["search"]["similarity_threshold"]

        for paper in papers_sorted:
            canonical_id = paper.get_canonical_id()

            # Check for exact matches first
            if canonical_id in seen_ids:
                continue

            # Check for similar papers
            is_duplicate = False
            for existing_paper in unique_papers:
                if paper.similarity_score(existing_paper) >= similarity_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                seen_ids.add(canonical_id)
                unique_papers.append(paper)

        self.logger.info(f"Removed {len(papers) - len(unique_papers)} duplicates")
        return unique_papers

    def get_cached_papers(
        self,
        query: str = None,
        year_min: int = None,
        database_source: str = None,
        limit: int = None,
    ) -> List[Paper]:
        """Retrieve papers from local cache."""
        return self.cache.get_cached_papers(query, year_min, database_source, limit)

    def export_results(
        self,
        papers: List[Paper],
        format: str = None,
        filename: str = None,
        include_abstracts: bool = None,
    ) -> Optional[str]:
        """
        Export search results to various formats.

        Args:
                        papers: List of papers to export
                        format: Export format ('json', 'csv', 'bibtex')
                        filename: Output filename
                        include_abstracts: Whether to include abstracts

        Returns:
                        str: Path to exported file or None if failed
        """
        if not papers:
            self.logger.warning("No papers to export")
            return None

        if format is None:
            format = self.config.config["export"]["default_format"]

        if include_abstracts is None:
            include_abstracts = self.config.config["export"]["include_abstracts"]

        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"literature_search_{timestamp}.{format}"

        try:
            max_size = self.config.config["export"]["max_export_size"]
            if len(papers) > max_size:
                self.logger.warning(f"Limiting export to {max_size} papers")
                papers = papers[:max_size]

            if format == "json":
                return self._export_json(papers, filename, include_abstracts)
            elif format == "bibtex":
                return self._export_bibtex(papers, filename)
            elif format == "csv":
                return self._export_csv(papers, filename, include_abstracts)
            else:
                raise ValueError(f"Unsupported export format: {format}")

        except Exception as e:
            self.logger.error(f"Export failed: {e}")
            return None

    def _export_json(
        self, papers: List[Paper], filename: str, include_abstracts: bool
    ) -> str:
        """Export papers to JSON format."""
        data = []
        for paper in papers:
            paper_dict = paper.to_dict()
            if not include_abstracts:
                paper_dict.pop("abstract", None)
            data.append(paper_dict)

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return filename

    def _export_bibtex(self, papers: List[Paper], filename: str) -> str:
        """Export papers to BibTeX format."""
        with open(filename, "w", encoding="utf-8") as f:
            for i, paper in enumerate(papers):
                entry_type = "article" if paper.venue else "misc"
                entry_id = f"paper{i + 1}"

                f.write(f"@{entry_type}{{{entry_id},\n")
                f.write(f"  title={{{paper.title}}},\n")
                f.write(f"  author={{{' and '.join(paper.authors)}}},\n")
                if paper.year:
                    f.write(f"  year={{{paper.year}}},\n")
                if paper.venue:
                    f.write(f"  journal={{{paper.venue}}},\n")
                if paper.doi:
                    f.write(f"  doi={{{paper.doi}}},\n")
                if paper.url:
                    f.write(f"  url={{{paper.url}}},\n")
                f.write("}\n\n")
        return filename

    def _export_csv(
        self, papers: List[Paper], filename: str, include_abstracts: bool
    ) -> str:
        """Export papers to CSV format."""
        with open(filename, "w", newline="", encoding="utf-8") as f:
            if not papers:
                return filename

            fieldnames = [
                "title",
                "authors",
                "year",
                "venue",
                "doi",
                "url",
                "citations",
                "database_source",
                "publication_type",
            ]
            if include_abstracts:
                fieldnames.insert(4, "abstract")

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for paper in papers:
                row = {
                    "title": paper.title,
                    "authors": "; ".join(paper.authors),
                    "year": paper.year,
                    "venue": paper.venue,
                    "doi": paper.doi,
                    "url": paper.url,
                    "citations": paper.citations,
                    "database_source": paper.database_source,
                    "publication_type": paper.publication_type,
                }
                if include_abstracts:
                    row["abstract"] = paper.abstract

                writer.writerow(row)
        return filename

    def analyze_search_coverage(
        self, query: str, databases: List[str] = None
    ) -> Dict[str, Any]:
        """Analyze search coverage across databases."""
        if databases is None:
            databases = list(self.adapters.keys())

        coverage_report = {
            "query": query,
            "databases_searched": [],
            "results_per_database": {},
            "total_papers": 0,
            "unique_papers": 0,
            "overlap_analysis": {},
            "adapter_stats": {},
        }

        all_papers = []
        db_papers = {}

        for db_name in databases:
            if db_name not in self.adapters:
                continue

            try:
                adapter = self.adapters[db_name]
                papers = adapter.search(query, 50)  # Limited for analysis
                db_papers[db_name] = papers
                all_papers.extend(papers)

                coverage_report["databases_searched"].append(db_name)
                coverage_report["results_per_database"][db_name] = len(papers)
                coverage_report["adapter_stats"][db_name] = adapter.get_stats()

            except Exception as e:
                self.logger.error(f"Error in coverage analysis for {db_name}: {e}")

        coverage_report["total_papers"] = len(all_papers)

        # Calculate unique papers
        unique_papers = self._deduplicate_papers(all_papers)
        coverage_report["unique_papers"] = len(unique_papers)

        # Calculate overlap between databases
        for db1 in db_papers:
            for db2 in db_papers:
                if db1 < db2:  # Avoid duplicate pairs
                    overlap = self._calculate_overlap(db_papers[db1], db_papers[db2])
                    coverage_report["overlap_analysis"][f"{db1}_vs_{db2}"] = overlap

        return coverage_report

    def _calculate_overlap(self, papers1: List[Paper], papers2: List[Paper]) -> int:
        """Calculate overlap between two sets of papers."""
        ids1 = set(paper.get_canonical_id() for paper in papers1)
        ids2 = set(paper.get_canonical_id() for paper in papers2)
        return len(ids1.intersection(ids2))

    def get_engine_stats(self) -> Dict[str, Any]:
        """Get comprehensive engine statistics."""
        stats = self.stats.copy()
        stats["databases_used"] = list(stats["databases_used"])
        stats["available_adapters"] = list(self.adapters.keys())
        stats["cache_stats"] = self.cache.get_cache_stats()

        # Add adapter statistics
        adapter_stats = {}
        for name, adapter in self.adapters.items():
            adapter_stats[name] = adapter.get_stats()
        stats["adapter_stats"] = adapter_stats

        return stats

    def cleanup_cache(self, days_old: int = None) -> bool:
        """Clean up old cache entries."""
        if days_old is None:
            days_old = self.config.config["cache"]["expire_days"]

        return self.cache.cleanup_cache(days_old)

    def find_duplicates(self, similarity_threshold: float = None) -> List[List[Paper]]:
        """Find potential duplicate papers in cache."""
        if similarity_threshold is None:
            similarity_threshold = self.config.config["search"]["similarity_threshold"]

        return self.cache.find_duplicates(similarity_threshold)

    def reload_adapters(self):
        """Reload all database adapters (useful after config changes)."""
        self.adapters.clear()
        self._init_adapters()
        self.logger.info("Reloaded database adapters")


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Initialize search engine
    engine = LiteratureSearchEngine()
    print(f"Configuration File Location: {engine.config.config_path}")
    print(
        f"Initialized Literature Search Engine with adapters: {list(engine.adapters.keys())}"
    )

    # Example Boolean search query
    boolean_query = '("software" OR "system") AND ("performance prediction" OR "performance modeling") AND ("deep learning" OR "neural network")'

    print(f"🔍 Literature Search Engine Test")
    print(f"Query: {boolean_query}")
    print("-" * 80)

    # Check available adapters
    available_adapters = list(engine.adapters.keys())
    print(f"Available adapters: {available_adapters}")

    if available_adapters:
        # Perform search
        papers = engine.search(query=boolean_query, max_results=10, year_min=2020)

        print(f"\n📚 Found {len(papers)} papers")

        # Display first few results
        for i, paper in enumerate(papers[:3], 1):
            print(f"\n{i}. {paper.title}")
            print(f"   Authors: {', '.join(paper.authors[:2])}")
            print(f"   Year: {paper.year}, Venue: {paper.venue}")
            print(f"   Source: {paper.database_source}")
            if paper.doi:
                print(f"   DOI: {paper.doi}")

        # Export results
        if papers:
            json_file = engine.export_results(papers, format="json")
            bibtex_file = engine.export_results(papers, format="bibtex")
            print(f"\n💾 Exported to: {json_file}, {bibtex_file}")

        # Show statistics
        stats = engine.get_engine_stats()
        print(f"\n📊 Engine Statistics:")
        print(f"   Total searches: {stats['total_searches']}")
        print(f"   Cached searches: {stats['cached_searches']}")
        print(f"   Total papers found: {stats['total_papers_found']}")
        print(f"   Cache stats: {stats['cache_stats']}")

    else:
        print("No database adapters available. Please configure API keys.")
        print("Use the secure key manager to set up database access.")
