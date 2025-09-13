#!/usr/bin/env python3
"""
Literature Search Command Line Interface
Enhanced CLI with interactive setup, configuration management, and quick search capabilities.
"""

import argparse
import sys
import logging
import json
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

# Import your modules
from ures.secrets import SecureKeyManager, StorageMethod
from ures.literature.search import (
    LiteratureSearchEngine,
    DatabaseConfig,
    AdapterFactory,
)


class InteractiveSetup:
    """Interactive setup for API keys and configuration."""

    def __init__(self, key_manager: SecureKeyManager, config: DatabaseConfig):
        self.key_manager = key_manager
        self.config = config
        self.logger = logging.getLogger(__name__)

    def setup_secrets(self):
        """Interactive API key setup."""
        print("🔐 Literature Search - API Key Setup")
        print("=" * 60)
        print("This wizard will help you configure API keys for academic databases.")
        print("You can use different storage methods for enhanced security.\n")

        # Get all supported databases
        databases = AdapterFactory.get_supported_databases()

        # Group by API key requirement
        api_databases = []
        free_databases = []

        for db in databases:
            req = AdapterFactory.get_adapter_requirements(db)
            if req.get("api_key_required", False):
                api_databases.append((db, req))
            else:
                free_databases.append((db, req))

        # Show free databases first
        if free_databases:
            print("📚 Free Databases (No API Key Required):")
            for db, req in free_databases:
                status = "✅" if not req.get("scraping_based", False) else "⚠️"
                print(f"  {status} {db}: {req.get('note', 'Available')}")
            print()

        # Setup API databases
        if api_databases:
            print("🔑 Databases Requiring API Keys:")
            for db, req in api_databases:
                print(f"  💰 {db}: {req.get('note', '')}")
                if req.get("api_signup_url"):
                    print(f"      Signup: {req['api_signup_url']}")
            print()

            # Ask user which databases to configure
            print(
                "Which databases would you like to configure? (Enter numbers separated by spaces)"
            )
            for i, (db, req) in enumerate(api_databases, 1):
                free_indicator = (
                    " (Free tier available)" if req.get("free", False) else ""
                )
                print(f"  {i}. {db}{free_indicator}")
            print(f"  {len(api_databases) + 1}. Configure all")
            print(f"  {len(api_databases) + 2}. Skip API key setup")

            try:
                choice = input("\nEnter your choice: ").strip()
                if not choice or choice == str(len(api_databases) + 2):
                    print("Skipping API key setup.")
                    return

                # Parse choices
                if choice == str(len(api_databases) + 1):
                    selected_dbs = api_databases
                else:
                    indices = [int(x) - 1 for x in choice.split()]
                    selected_dbs = [
                        api_databases[i] for i in indices if 0 <= i < len(api_databases)
                    ]

                # Configure selected databases
                for db, req in selected_dbs:
                    self._setup_database_key(db, req)

            except (ValueError, IndexError):
                print("❌ Invalid choice. Skipping API key setup.")

        print("\n✅ API key setup completed!")

    def _setup_database_key(self, db_name: str, requirements: Dict):
        """Setup API key for a specific database."""
        print(f"\n📋 Configuring {db_name.upper()}")
        print("-" * 40)

        # Show database information
        if requirements.get("note"):
            print(f"Info: {requirements['note']}")

        if requirements.get("api_signup_url"):
            print(f"Get API key: {requirements['api_signup_url']}")

        if requirements.get("api_docs_url"):
            print(f"Documentation: {requirements['api_docs_url']}")

        # Check if key already exists
        existing_key = self.key_manager.get_key(db_name)
        if existing_key and existing_key != "dummy":
            print(f"✅ API key already configured for {db_name}")
            update = input("Do you want to update it? (y/N): ").strip().lower()
            if update != "y":
                return

        # Storage method selection
        print("\nChoose storage method:")
        print("1. Encrypted local storage (recommended)")
        print("2. Environment variable")
        print("3. 1Password (requires 1Password SDK)")
        print("4. macOS Keychain")
        print("5. Skip this database")

        method_choice = input("Select method (1-5): ").strip()

        if method_choice == "5":
            return

        storage_methods = {
            "1": StorageMethod.ENCRYPTED,
            "2": StorageMethod.ENV,
            "3": StorageMethod.ONEPASSWORD,
            "4": StorageMethod.KEYCHAIN,
        }

        method = storage_methods.get(method_choice, StorageMethod.ENCRYPTED)

        if method == StorageMethod.ENV:
            env_var = input(
                f"Environment variable name (default: {db_name.upper()}_API_KEY): "
            ).strip()
            if not env_var:
                env_var = f"{db_name.upper()}_API_KEY"

            print(
                f"Please set the environment variable: export {env_var}='your_api_key_here'"
            )
            input("Press Enter when you've set the environment variable...")

            success = self.key_manager.store_key(db_name, env_var, method)

        elif method == StorageMethod.ONEPASSWORD:
            if not SecureKeyManager.is_onepassword_available():
                print(
                    "❌ 1Password SDK not available. Install with: pip install onepassword"
                )
                print(SecureKeyManager.get_onepassword_setup_instructions())
                return

            op_reference = input(
                "1Password secret reference (e.g., op://vault/item/field): "
            ).strip()
            if not op_reference:
                print("❌ No 1Password reference provided")
                return

            success = self.key_manager.store_key(db_name, op_reference, method)

        elif method == StorageMethod.KEYCHAIN:
            if sys.platform != "darwin":
                print("❌ Keychain is only available on macOS")
                return

            service_name = input(
                f"Keychain service name (default: {db_name}_api_key): "
            ).strip()
            if not service_name:
                service_name = f"{db_name}_api_key"

            print(
                f"Please add your API key to Keychain with service name: {service_name}"
            )
            input("Press Enter when you've added the key to Keychain...")

            success = self.key_manager.store_key(db_name, service_name, method)

        else:  # Encrypted storage
            api_key = input(f"Enter {db_name} API key: ").strip()
            if not api_key:
                print("❌ No API key provided")
                return

            success = self.key_manager.store_key(db_name, api_key, method)

        if success:
            print(f"✅ {db_name} API key configured successfully")
            # Enable the database in config
            self.config.update_config("databases", f"{db_name}.enabled", True)
        else:
            print(f"❌ Failed to configure {db_name} API key")

    def setup_configuration(self):
        """Interactive configuration setup."""
        print("\n⚙️ Literature Search - Configuration Setup")
        print("=" * 60)

        # Search configuration
        print("📚 Search Configuration:")

        current_max = self.config.get_config_value("search", "max_results", 100)
        max_results = input(f"Maximum results per database [{current_max}]: ").strip()
        if max_results.isdigit():
            self.config.update_config("search", "max_results", int(max_results))

        current_min_year = self.config.get_config_value("search", "min_year", 2018)
        min_year = input(f"Minimum publication year [{current_min_year}]: ").strip()
        if min_year.isdigit():
            self.config.update_config("search", "min_year", int(min_year))

        current_dedup = self.config.get_config_value("search", "deduplication", True)
        dedup = (
            input(f"Enable deduplication? (y/n) [{'y' if current_dedup else 'n'}]: ")
            .strip()
            .lower()
        )
        if dedup in ["y", "n"]:
            self.config.update_config("search", "deduplication", dedup == "y")

        # Cache configuration
        print("\n💾 Cache Configuration:")

        current_cache_days = self.config.get_config_value("cache", "expire_days", 30)
        cache_days = input(f"Cache expiration days [{current_cache_days}]: ").strip()
        if cache_days.isdigit():
            self.config.update_config("cache", "expire_days", int(cache_days))

        current_max_age = self.config.get_config_value("cache", "max_age_hours", 24)
        max_age = input(f"Search cache max age hours [{current_max_age}]: ").strip()
        if max_age.isdigit():
            self.config.update_config("cache", "max_age_hours", int(max_age))

        # ArXiv categories
        print("\n📋 ArXiv Categories (for computer science research):")
        current_categories = self.config.get_config_value(
            "databases", "arxiv.categories", []
        )
        print(f"Current categories: {', '.join(current_categories)}")

        categories_input = input(
            "Enter categories (comma-separated) or press Enter to keep current: "
        ).strip()
        if categories_input:
            categories = [cat.strip() for cat in categories_input.split(",")]
            self.config.update_config("databases", "arxiv.categories", categories)

        print("\n✅ Configuration setup completed!")


class LiteratureSearchCLI:
    """Enhanced command-line interface for literature search."""

    def __init__(self):
        self.engine = None
        self.config = None
        self.logger = logging.getLogger(__name__)

    def setup_logging(self, level: str = "INFO"):
        """Setup logging configuration."""
        log_level = getattr(logging, level.upper(), logging.INFO)
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    def init_engine(self, config_dir: Optional[str] = None) -> bool:
        """Initialize the search engine."""
        try:
            self.config = DatabaseConfig(config_dir)
            self.engine = LiteratureSearchEngine(config_dir)
            return True
        except Exception as e:
            print(f"❌ Failed to initialize search engine: {e}")
            return False

    def cmd_init(self, args):
        """Initialize the literature search system interactively."""
        print("🚀 Welcome to Literature Search System!")
        print("This wizard will help you set up everything you need.")
        print("=" * 60)

        # Initialize config and key manager
        config_dir = args.config_dir
        if not config_dir:
            config_dir = Path.home() / ".ures_lit_search"

        # Create directories
        Path(config_dir).mkdir(parents=True, exist_ok=True)

        # Initialize components
        key_manager = SecureKeyManager("literature-search", config_dir=config_dir)
        config = DatabaseConfig(config_dir)

        setup = InteractiveSetup(key_manager, config)

        # Run setup steps
        setup.setup_secrets()
        setup.setup_configuration()

        print(f"\n🎉 Setup completed!")
        print(f"Configuration stored in: {config.config_path}")
        print(f"API keys stored securely in: {key_manager.config_dir}")
        print(f"\nYou can now use: python {sys.argv[0]} search 'your query here'")

    def cmd_status(self, args):
        """Show comprehensive system status."""
        print("📊 Literature Search System Status")
        print("=" * 60)

        # Initialize to get config
        if not self.init_engine(args.config_dir):
            return

        # API Keys Status
        print("🔐 API Keys Status:")
        print("-" * 30)

        key_info = self.config.list_api_keys()
        if not key_info:
            print("  No API keys configured")
        else:
            for service, info in key_info.items():
                status = "✅" if info.get("has_key", False) else "❌"
                method = info.get("method", "unknown")
                print(f"  {service:15} {status} ({method})")

        # Database Status
        print(f"\n📚 Database Status:")
        print("-" * 30)

        databases = AdapterFactory.get_supported_databases()
        for db in databases:
            req = AdapterFactory.get_adapter_requirements(db)
            enabled = self.config.is_database_enabled(db)

            # Check if adapter is actually available
            available = False
            if enabled and db in self.engine.adapters:
                adapter = self.engine.adapters[db]
                available = adapter.is_available()

            # Status indicators
            config_status = "✅" if enabled else "⚪"
            avail_status = "🟢" if available else "🔴" if enabled else "⚪"
            api_status = "🔑" if req.get("api_key_required", False) else "🆓"

            print(f"  {db:15} {config_status}{avail_status}{api_status}")

        print(f"\nLegend:")
        print(f"  Config: ✅ Enabled ⚪ Disabled")
        print(f"  Status: 🟢 Available 🔴 Error ⚪ Not configured")
        print(f"  Access: 🔑 API Key Required 🆓 Free")

        # Cache Status
        cache_stats = self.engine.get_engine_stats().get("cache_stats", {})
        print(f"\n💾 Cache Status:")
        print("-" * 30)
        print(f"  Total papers: {cache_stats.get('total_papers', 0)}")
        print(f"  Recent searches: {cache_stats.get('recent_searches', 0)}")
        print(f"  Papers by source: {cache_stats.get('papers_by_source', {})}")

        # System Info
        engine_stats = self.engine.get_engine_stats()
        print(f"\n🔍 Search Engine Status:")
        print("-" * 30)
        print(f"  Available adapters: {len(self.engine.adapters)}")
        print(f"  Total searches: {engine_stats.get('total_searches', 0)}")
        print(f"  Cached searches: {engine_stats.get('cached_searches', 0)}")
        print(f"  Papers found: {engine_stats.get('total_papers_found', 0)}")

    def cmd_search(self, args):
        """Perform literature search with enhanced output."""
        if not self.init_engine(args.config_dir):
            return

        query = args.query
        if not query:
            print("❌ No search query provided")
            return

        # Check if any adapters are available
        if not self.engine.adapters:
            print("❌ No database adapters available.")
            print("Run the initialization wizard first:")
            print(f"  python {sys.argv[0]} init")
            return

        print(f"🔍 Searching Literature")
        print("=" * 50)
        print(f"Query: {query}")

        # Show search parameters
        params_info = []
        if args.databases:
            params_info.append(f"Databases: {', '.join(args.databases)}")
        else:
            params_info.append(f"Databases: {', '.join(self.engine.adapters.keys())}")

        if args.max_results:
            params_info.append(f"Max results: {args.max_results}")
        if args.year_min:
            params_info.append(f"Min year: {args.year_min}")
        if args.no_cache:
            params_info.append("Cache: disabled")

        if params_info:
            print(f"Parameters: {' | '.join(params_info)}")

        print("-" * 50)

        # Perform search with timing
        start_time = datetime.now()

        try:
            papers = self.engine.search(
                query=query,
                databases=args.databases,
                max_results=args.max_results,
                year_min=args.year_min,
                use_cache=not args.no_cache,
            )

            search_time = (datetime.now() - start_time).total_seconds()

            if not papers:
                print("📭 No papers found matching your query.")
                self._show_suggestions(query)
                return

            # Display results summary
            print(f"📚 Found {len(papers)} papers in {search_time:.2f}s")

            # Group by database source
            by_source = {}
            for paper in papers:
                source = paper.database_source
                by_source.setdefault(source, 0)
                by_source[source] += 1

            source_summary = ", ".join(
                [f"{src}: {count}" for src, count in by_source.items()]
            )
            print(f"Sources: {source_summary}")
            print("=" * 50)

            # Display detailed results
            display_count = min(args.show, len(papers))
            for i, paper in enumerate(papers[:display_count], 1):
                self._display_paper(i, paper, args.show_abstracts, args.show_urls)

            if len(papers) > display_count:
                print(f"\n... and {len(papers) - display_count} more papers")
                print(f"Use --show {len(papers)} to see all results")

            # Export if requested
            if args.export:
                self._export_results(papers, args)

            # Show statistics if verbose
            if args.verbose:
                self._show_search_stats()

        except Exception as e:
            print(f"❌ Search failed: {e}")
            if args.verbose:
                import traceback

                traceback.print_exc()

    def _display_paper(
        self, index: int, paper, show_abstracts: bool = False, show_urls: bool = False
    ):
        """Display a single paper with rich formatting."""
        print(f"\n{index}. {paper.title}")

        # Authors
        if paper.authors:
            authors_display = ", ".join(paper.authors[:3])
            if len(paper.authors) > 3:
                authors_display += f" and {len(paper.authors) - 3} others"
            print(f"   👥 {authors_display}")

        # Publication info
        pub_info = []
        if paper.year:
            pub_info.append(str(paper.year))
        if paper.venue:
            pub_info.append(paper.venue)
        if paper.citations > 0:
            pub_info.append(f"{paper.citations} citations")
        if pub_info:
            print(f"   📖 {' | '.join(pub_info)}")

        # Source and type
        source_info = [paper.database_source]
        if paper.publication_type:
            source_info.append(paper.publication_type)
        print(f"   🔗 {' | '.join(source_info)}")

        # Identifiers
        ids = []
        if paper.doi:
            ids.append(f"DOI: {paper.doi}")
        if paper.arxiv_id:
            ids.append(f"arXiv: {paper.arxiv_id}")
        if ids:
            print(f"   🆔 {' | '.join(ids)}")

        # Keywords
        if paper.keywords:
            keywords_display = ", ".join(paper.keywords[:5])
            if len(paper.keywords) > 5:
                keywords_display += "..."
            print(f"   🏷️  {keywords_display}")

        # URLs
        if show_urls:
            if paper.url:
                print(f"   🌐 {paper.url}")
            if paper.pdf_url and paper.pdf_url != paper.url:
                print(f"   📄 {paper.pdf_url}")

        # Abstract
        if show_abstracts and paper.abstract:
            abstract = (
                paper.abstract[:400] + "..."
                if len(paper.abstract) > 400
                else paper.abstract
            )
            print(f"   📝 {abstract}")

    def _show_suggestions(self, query: str):
        """Show search suggestions when no results found."""
        print("\n💡 Suggestions:")
        print("   • Try broader search terms")
        print("   • Remove year restrictions with --year-min")
        print("   • Use Boolean operators: 'term1 OR term2', '(term1 AND term2)'")
        print("   • Check if databases are properly configured with 'status' command")

    def _export_results(self, papers, args):
        """Export search results."""
        try:
            filename = self.engine.export_results(
                papers,
                format=args.format,
                filename=args.output,
                include_abstracts=args.show_abstracts,
            )
            print(f"\n💾 Results exported to: {filename}")
        except Exception as e:
            print(f"❌ Export failed: {e}")

    def _show_search_stats(self):
        """Show detailed search statistics."""
        stats = self.engine.get_engine_stats()
        print(f"\n📊 Search Statistics:")
        print(f"   Total searches performed: {stats['total_searches']}")
        print(f"   Cached searches used: {stats['cached_searches']}")
        print(f"   Databases queried: {', '.join(stats['databases_used'])}")

        # Adapter-specific stats
        for name, adapter_stats in stats.get("adapter_stats", {}).items():
            success_rate = 0
            if adapter_stats["total_requests"] > 0:
                success_rate = (
                    adapter_stats["successful_requests"]
                    / adapter_stats["total_requests"]
                    * 100
                )
            print(
                f"   {name}: {success_rate:.1f}% success rate, {adapter_stats['total_papers_found']} papers"
            )

    def cmd_config(self, args):
        """Configuration management."""
        if not self.init_engine(args.config_dir):
            return

        if args.config_action == "show":
            print("⚙️ Current Configuration")
            print("=" * 40)
            print(json.dumps(self.config.config, indent=2))

        elif args.config_action == "edit":
            # Interactive configuration editing
            setup = InteractiveSetup(self.config.key_manager, self.config)
            setup.setup_configuration()

        elif args.config_action == "keys":
            # API key management
            setup = InteractiveSetup(self.config.key_manager, self.config)
            setup.setup_secrets()

        elif args.config_action == "test":
            # Test all configured API keys
            print("🧪 Testing API Keys")
            print("-" * 30)

            key_info = self.config.list_api_keys()
            for service, info in key_info.items():
                if info.get("has_key", False):
                    test_result = self.config.key_manager.test_key_access(service)
                    status = "✅" if test_result["accessible"] else "❌"
                    print(f"  {service:15} {status}")
                    if test_result.get("error"):
                        print(f"                  Error: {test_result['error']}")

    def cmd_cache(self, args):
        """Enhanced cache management."""
        if not self.init_engine(args.config_dir):
            return

        if args.cache_action == "stats":
            stats = self.engine.cache.get_cache_stats()
            print("📊 Cache Statistics")
            print("=" * 40)
            print(f"Total papers: {stats.get('total_papers', 0):,}")
            print(f"Unique sources: {stats.get('unique_sources', 0)}")
            print(f"Total searches: {stats.get('total_searches', 0)}")
            print(f"Recent searches (24h): {stats.get('recent_searches', 0)}")

            # Papers by source
            papers_by_source = stats.get("papers_by_source", {})
            if papers_by_source:
                print(f"\nPapers by database:")
                for source, count in sorted(
                    papers_by_source.items(), key=lambda x: x[1], reverse=True
                ):
                    print(f"  {source:15}: {count:,}")

            # Recent years distribution
            recent_years = stats.get("recent_years", {})
            if recent_years:
                print(f"\nPapers by year (top 5):")
                for year, count in list(recent_years.items())[:5]:
                    print(f"  {year}: {count:,}")

        elif args.cache_action == "clean":
            days = args.days or 30
            print(f"🧹 Cleaning cache entries older than {days} days...")
            success = self.engine.cleanup_cache(days)
            print(
                "✅ Cache cleanup completed" if success else "❌ Cache cleanup failed"
            )

        elif args.cache_action == "export":
            format_type = args.format or "json"
            filename = args.output
            print(f"💾 Exporting cached papers to {format_type.upper()}...")

            result = self.engine.cache.export_papers(format_type, filename)
            if result:
                print(f"✅ Exported to: {result}")
            else:
                print("❌ Export failed")

        elif args.cache_action == "duplicates":
            threshold = args.similarity or 0.8
            print(f"🔍 Finding duplicates (similarity ≥ {threshold})...")

            duplicates = self.engine.find_duplicates(threshold)
            if duplicates:
                print(f"Found {len(duplicates)} groups of similar papers:")
                for i, group in enumerate(duplicates[:10], 1):  # Limit display
                    print(f"\nGroup {i} ({len(group)} papers):")
                    for paper in group:
                        print(
                            f"  • {paper.title[:80]}{'...' if len(paper.title) > 80 else ''}"
                        )
                        print(f"    {paper.year} | {paper.database_source}")

                if len(duplicates) > 10:
                    print(f"\n... and {len(duplicates) - 10} more groups")
            else:
                print("No duplicates found")

    def run(self):
        """Main CLI entry point with comprehensive command structure."""
        parser = argparse.ArgumentParser(
            description="Literature Search System - Advanced academic database search with Boolean queries",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # First time setup
  python literature_search.py init

  # Quick searches
  python literature_search.py search "machine learning performance"
  python literature_search.py search '"deep learning" AND optimization' --export

  # Advanced searches
  python literature_search.py search 'software AND (performance OR efficiency)' \\
      --databases arxiv ieee --year-min 2020 --max-results 50

  # System management
  python literature_search.py status
  python literature_search.py config keys
  python literature_search.py cache stats

Boolean Query Examples:
  • "machine learning" AND performance
  • software OR system AND "neural network"
  • ("deep learning" OR "artificial intelligence") AND optimization
  • cloud computing AND NOT "edge computing"
            """,
        )

        # Global options
        parser.add_argument(
            "--verbose", "-v", action="store_true", help="Enable verbose output"
        )
        parser.add_argument("--config-dir", help="Custom configuration directory")
        parser.add_argument(
            "--log-level",
            default="INFO",
            choices=["DEBUG", "INFO", "WARNING", "ERROR"],
            help="Set logging level",
        )

        subparsers = parser.add_subparsers(dest="command", help="Available commands")

        # Init command - Interactive setup
        init_parser = subparsers.add_parser("init", help="Interactive system setup")

        # Search command - Enhanced with quick access
        search_parser = subparsers.add_parser(
            "search", help="Search academic literature"
        )
        search_parser.add_argument(
            "query", help="Search query (supports Boolean operators)"
        )
        search_parser.add_argument(
            "--databases", nargs="+", help="Specific databases to search"
        )
        search_parser.add_argument(
            "--max-results", type=int, help="Maximum results per database"
        )
        search_parser.add_argument(
            "--year-min", type=int, help="Minimum publication year"
        )
        search_parser.add_argument(
            "--show", type=int, default=10, help="Number of results to display"
        )
        search_parser.add_argument(
            "--show-abstracts", action="store_true", help="Include abstracts in output"
        )
        search_parser.add_argument(
            "--show-urls", action="store_true", help="Include URLs in output"
        )
        search_parser.add_argument(
            "--no-cache", action="store_true", help="Disable cache usage"
        )
        search_parser.add_argument(
            "--export", action="store_true", help="Export results to file"
        )
        search_parser.add_argument(
            "--format",
            choices=["json", "csv", "bibtex"],
            default="json",
            help="Export format",
        )
        search_parser.add_argument("--output", help="Output filename for export")

        # Status command - System overview
        status_parser = subparsers.add_parser("status", help="Show system status")

        # Config command - Configuration management
        config_parser = subparsers.add_parser("config", help="Configuration management")
        config_subparsers = config_parser.add_subparsers(
            dest="config_action", help="Config actions"
        )

        config_show = config_subparsers.add_parser(
            "show", help="Show current configuration"
        )
        config_edit = config_subparsers.add_parser(
            "edit", help="Interactive configuration editor"
        )
        config_keys = config_subparsers.add_parser("keys", help="Manage API keys")
        config_test = config_subparsers.add_parser(
            "test", help="Test API key accessibility"
        )

        # Cache command - Enhanced cache management
        cache_parser = subparsers.add_parser("cache", help="Cache management")
        cache_subparsers = cache_parser.add_subparsers(
            dest="cache_action", help="Cache actions"
        )

        cache_stats = cache_subparsers.add_parser("stats", help="Show cache statistics")

        cache_clean = cache_subparsers.add_parser(
            "clean", help="Clean old cache entries"
        )
        cache_clean.add_argument(
            "--days", type=int, help="Remove entries older than N days"
        )

        cache_export = cache_subparsers.add_parser(
            "export", help="Export cached papers"
        )
        cache_export.add_argument("--format", choices=["json", "csv"], default="json")
        cache_export.add_argument("--output", help="Output filename")

        cache_duplicates = cache_subparsers.add_parser(
            "duplicates", help="Find duplicate papers"
        )
        cache_duplicates.add_argument(
            "--similarity",
            type=float,
            default=0.8,
            help="Similarity threshold (0.0-1.0)",
        )

        # Coverage command - Search coverage analysis
        coverage_parser = subparsers.add_parser(
            "coverage", help="Analyze search coverage"
        )
        coverage_parser.add_argument("query", help="Query for coverage analysis")
        coverage_parser.add_argument(
            "--databases", nargs="+", help="Databases to analyze"
        )

        # Quick commands for common operations
        quick_parser = subparsers.add_parser("quick", help="Quick operations")
        quick_subparsers = quick_parser.add_subparsers(
            dest="quick_action", help="Quick actions"
        )

        quick_arxiv = quick_subparsers.add_parser("arxiv", help="Quick arXiv search")
        quick_arxiv.add_argument("query", help="Search query")
        quick_arxiv.add_argument("--max", type=int, default=20, help="Max results")

        quick_all = quick_subparsers.add_parser(
            "all", help="Search all available databases"
        )
        quick_all.add_argument("query", help="Search query")
        quick_all.add_argument(
            "--max", type=int, default=10, help="Max results per database"
        )

        # Parse arguments and handle commands
        if len(sys.argv) == 1:
            parser.print_help()
            return

        args = parser.parse_args()

        # Setup logging
        self.setup_logging(args.log_level)

        # Handle commands
        if args.command == "init":
            self.cmd_init(args)
        elif args.command == "search":
            self.cmd_search(args)
        elif args.command == "status":
            self.cmd_status(args)
        elif args.command == "config":
            self.cmd_config(args)
        elif args.command == "cache":
            self.cmd_cache(args)
        elif args.command == "coverage":
            self.cmd_coverage(args)
        elif args.command == "quick":
            self.cmd_quick(args)
        else:
            parser.print_help()

    def cmd_coverage(self, args):
        """Analyze search coverage across databases."""
        if not self.init_engine(args.config_dir):
            return

        query = args.query
        print(f"📈 Search Coverage Analysis")
        print("=" * 50)
        print(f"Query: {query}")

        if args.databases:
            print(f"Analyzing databases: {', '.join(args.databases)}")

        print("-" * 50)

        try:
            coverage = self.engine.analyze_search_coverage(query, args.databases)

            # Results summary
            print(f"📊 Coverage Summary:")
            print(f"   Total papers found: {coverage['total_papers']}")
            print(f"   Unique papers: {coverage['unique_papers']}")
            print(f"   Databases searched: {len(coverage['databases_searched'])}")

            # Results per database
            if coverage["results_per_database"]:
                print(f"\n📚 Results by Database:")
                for db, count in coverage["results_per_database"].items():
                    percentage = (
                        (count / coverage["total_papers"] * 100)
                        if coverage["total_papers"] > 0
                        else 0
                    )
                    print(f"   {db:15}: {count:3} papers ({percentage:5.1f}%)")

            # Overlap analysis
            if coverage["overlap_analysis"]:
                print(f"\n🔄 Database Overlaps:")
                for pair, overlap in coverage["overlap_analysis"].items():
                    db1, db2 = pair.split("_vs_")
                    total1 = coverage["results_per_database"].get(db1, 0)
                    total2 = coverage["results_per_database"].get(db2, 0)

                    if total1 > 0 and total2 > 0:
                        overlap_pct = overlap / min(total1, total2) * 100
                        print(
                            f"   {db1} ↔ {db2}: {overlap} papers ({overlap_pct:.1f}% overlap)"
                        )

            # Recommendations
            print(f"\n💡 Recommendations:")
            if coverage["unique_papers"] < coverage["total_papers"] * 0.8:
                print(
                    "   • High overlap detected - consider focusing on fewer databases"
                )

            no_result_dbs = [
                db
                for db, count in coverage["results_per_database"].items()
                if count == 0
            ]
            if no_result_dbs:
                print(
                    f"   • No results from: {', '.join(no_result_dbs)} - try different terms"
                )

            if coverage["unique_papers"] > 0:
                best_db = max(
                    coverage["results_per_database"].items(), key=lambda x: x[1]
                )
                print(f"   • Best single source: {best_db[0]} ({best_db[1]} papers)")

        except Exception as e:
            print(f"❌ Coverage analysis failed: {e}")
            if args.verbose:
                import traceback

                traceback.print_exc()

    def cmd_quick(self, args):
        """Quick search operations for common use cases."""
        if not self.init_engine(args.config_dir):
            return

        if args.quick_action == "arxiv":
            print(f"🚀 Quick arXiv Search")
            print("=" * 30)

            if "arxiv" not in self.engine.adapters:
                print("❌ arXiv adapter not available")
                return

            papers = self.engine.search(
                query=args.query,
                databases=["arxiv"],
                max_results=args.max,
                use_cache=True,
            )

            self._display_quick_results(papers, "arXiv")

        elif args.quick_action == "all":
            print(f"🚀 Quick Multi-Database Search")
            print("=" * 35)

            if not self.engine.adapters:
                print("❌ No database adapters available")
                return

            papers = self.engine.search(
                query=args.query, max_results=args.max, use_cache=True
            )

            self._display_quick_results(papers, "All Databases")

    def _display_quick_results(self, papers, source_name):
        """Display quick search results in compact format."""
        if not papers:
            print(f"📭 No papers found in {source_name}")
            return

        print(f"📚 Found {len(papers)} papers from {source_name}")
        print("-" * 50)

        for i, paper in enumerate(papers[:10], 1):
            # Compact display format
            authors = ", ".join(paper.authors[:2])
            if len(paper.authors) > 2:
                authors += " et al."

            year_venue = f"({paper.year})" if paper.year else ""
            if paper.venue:
                year_venue += f" {paper.venue}"

            print(f"{i:2}. {paper.title}")
            print(f"    {authors} {year_venue}")

            if paper.citations > 0:
                print(f"    📊 {paper.citations} citations")

            print()

        if len(papers) > 10:
            print(f"... and {len(papers) - 10} more papers")
            print(f"Use 'search' command with --show {len(papers)} to see all")


def main():
    """Main entry point."""
    try:
        cli = LiteratureSearchCLI()
        cli.run()
    except KeyboardInterrupt:
        print("\n\n👋 Search interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
