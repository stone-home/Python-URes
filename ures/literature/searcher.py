# API Key Management CLI
class APIKeyManager:
    """Command-line interface for managing API keys securely."""

    def __init__(self):
        self.config = DatabaseConfig()

    def setup_interactive(self):
        """Interactive setup for API keys."""
        print("🔐 Literature Search API Key Setup")
        print("=" * 50)

        databases = ["ieee", "springer", "elsevier", "wiley", "acm"]

        for db in databases:
            print(f"\n📚 Setting up {db.upper()} API Key")
            print("-" * 30)

            choice = input(f"Configure {db}? (y/n): ").lower().strip()
            if choice != "y":
                continue

            print("\nChoose storage method:")
            print("1. Encrypted locally (recommended)")
            print("2. Environment variable")
            print("3. 1Password CLI")
            print("4. macOS Keychain")

            method_choice = input("Enter choice (1-4): ").strip()

            if method_choice == "1":
                api_key = input(f"Enter {db} API key: ").strip()
                if api_key:
                    self.config.set_api_key(db, api_key, "encrypted")
                    print(f"✅ {db} API key stored securely (encrypted)")

            elif method_choice == "2":
                env_var = input(
                    f"Enter environment variable name (e.g., {db.upper()}_API_KEY): "
                ).strip()
                if env_var:
                    self.config.set_api_key(db, env_var, "env")
                    print(f"✅ {db} configured to use ${env_var}")
                    print(f"   Remember to set: export {env_var}=your_api_key")

            elif method_choice == "3":
                op_ref = input(
                    f"Enter 1Password reference (e.g., op://vault/{db}-api-key/credential): "
                ).strip()
                if op_ref:
                    self.config.set_api_key(db, op_ref, "1password")
                    print(f"✅ {db} configured to use 1Password")
                    print(f"   Reference: {op_ref}")

            elif method_choice == "4":
                service_name = input(
                    f"Enter Keychain service name (e.g., {db}-api-key): "
                ).strip()
                if service_name:
                    self.config.set_api_key(db, service_name, "keychain")
                    print(f"✅ {db} configured to use Keychain")
                    print(f"   Service: {service_name}")
                    print(
                        f"   Add to keychain: security add-generic-password -s {service_name} -a {db} -w your_api_key"
                    )

            else:
                print("Invalid choice, skipping...")

        print(f"\n🎉 Setup complete!")
        self.show_status()

    def show_status(self):
        """Show current API key configuration status."""
        print("\n📊 API Key Status")
        print("=" * 40)

        stored_keys = self.config.list_api_keys()
        databases = ["ieee", "springer", "elsevier", "wiley", "acm"]

        for db in databases:
            if db in stored_keys:
                info = stored_keys[db]
                status = "✅ Configured" if info["has_key"] else "❌ Key not accessible"
                method = info["method"]
                print(f"{db:12}: {status:15} ({method})")

                if method == "env" and "env_var" in info:
                    env_val = "SET" if os.getenv(info["env_var"]) else "NOT SET"
                    print(f"            Environment: ${info['env_var']} = {env_val}")
                elif method == "1password" and "op_reference" in info:
                    print(f"            1Password: {info['op_reference']}")
                elif method == "keychain" and "keychain_service" in info:
                    print(f"            Keychain: {info['keychain_service']}")
            else:
                print(f"{db:12}: ❌ Not configured")

        print  # !/usr/bin/env python3


"""
Advanced Literature Search Library
Supports complex Boolean queries across multiple academic databases.
Designed for comprehensive literature reviews with unified result formatting.
"""

import json
import sqlite3
import hashlib
import time
import urllib.request
import urllib.parse
import urllib.error
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import xml.etree.ElementTree as ET
import re
import logging
from datetime import datetime, timedelta
import os
import base64
import subprocess


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
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "Paper":
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

        parsed = {
            "quoted_phrases": quoted_phrases,
            "groups": groups,
            "original_query": query,
        }

        return parsed

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


class SecureKeyManager:
    """Secure API key management supporting multiple storage methods."""

    def __init__(self, config_dir: str = "./lit_cache"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self.key_file = self.config_dir / "encrypted_keys.json"
        self.logger = logging.getLogger(__name__)

        # Generate or load encryption key
        self.encryption_key = self._get_or_create_encryption_key()

    def _get_or_create_encryption_key(self) -> bytes:
        """Get or create local encryption key."""
        key_path = self.config_dir / ".encryption_key"

        if key_path.exists():
            with open(key_path, "rb") as f:
                return f.read()
        else:
            # Generate new key
            import secrets

            key = secrets.token_bytes(32)  # 256-bit key
            with open(key_path, "wb") as f:
                f.write(key)
            # Set restrictive permissions (Unix-like systems)
            try:
                os.chmod(key_path, 0o600)
            except:
                pass
            return key

    def _encrypt_value(self, value: str) -> str:
        """Encrypt a value using simple XOR (for basic protection)."""
        if not value:
            return ""

        # Simple XOR encryption (for basic obfuscation)
        key_bytes = self.encryption_key
        value_bytes = value.encode("utf-8")
        encrypted = bytearray()

        for i, byte in enumerate(value_bytes):
            encrypted.append(byte ^ key_bytes[i % len(key_bytes)])

        return base64.b64encode(encrypted).decode("utf-8")

    def _decrypt_value(self, encrypted_value: str) -> str:
        """Decrypt a value."""
        if not encrypted_value:
            return ""

        try:
            encrypted_bytes = base64.b64decode(encrypted_value.encode("utf-8"))
            key_bytes = self.encryption_key
            decrypted = bytearray()

            for i, byte in enumerate(encrypted_bytes):
                decrypted.append(byte ^ key_bytes[i % len(key_bytes)])

            return decrypted.decode("utf-8")
        except Exception as e:
            self.logger.error(f"Failed to decrypt value: {e}")
            return ""

    def store_api_key(self, service: str, api_key: str, method: str = "encrypted"):
        """
        Store API key using specified method.

        Args:
                service: Service name (e.g., 'ieee', 'springer')
                api_key: The API key to store
                method: Storage method ('encrypted', 'env', '1password', 'keychain')
        """
        if method == "encrypted":
            self._store_encrypted_key(service, api_key)
        elif method == "env":
            self._store_env_reference(service, api_key)
        elif method == "1password":
            self._store_1password_reference(service, api_key)
        elif method == "keychain":
            self._store_keychain_reference(service, api_key)
        else:
            raise ValueError(f"Unsupported storage method: {method}")

    def _store_encrypted_key(self, service: str, api_key: str):
        """Store encrypted API key locally."""
        encrypted_keys = {}
        if self.key_file.exists():
            with open(self.key_file, "r") as f:
                encrypted_keys = json.load(f)

        encrypted_keys[service] = {
            "method": "encrypted",
            "value": self._encrypt_value(api_key),
            "created_at": datetime.now().isoformat(),
        }

        with open(self.key_file, "w") as f:
            json.dump(encrypted_keys, f, indent=2)

        # Set restrictive permissions
        try:
            os.chmod(self.key_file, 0o600)
        except:
            pass

    def _store_env_reference(self, service: str, env_var_name: str):
        """Store reference to environment variable."""
        encrypted_keys = {}
        if self.key_file.exists():
            with open(self.key_file, "r") as f:
                encrypted_keys = json.load(f)

        encrypted_keys[service] = {
            "method": "env",
            "env_var": env_var_name,
            "created_at": datetime.now().isoformat(),
        }

        with open(self.key_file, "w") as f:
            json.dump(encrypted_keys, f, indent=2)

    def _store_1password_reference(self, service: str, op_reference: str):
        """Store 1Password CLI reference."""
        encrypted_keys = {}
        if self.key_file.exists():
            with open(self.key_file, "r") as f:
                encrypted_keys = json.load(f)

        encrypted_keys[service] = {
            "method": "1password",
            "op_reference": op_reference,  # e.g., "op://vault/item/field"
            "created_at": datetime.now().isoformat(),
        }

        with open(self.key_file, "w") as f:
            json.dump(encrypted_keys, f, indent=2)

    def _store_keychain_reference(self, service: str, keychain_service: str):
        """Store macOS Keychain reference."""
        encrypted_keys = {}
        if self.key_file.exists():
            with open(self.key_file, "r") as f:
                encrypted_keys = json.load(f)

        encrypted_keys[service] = {
            "method": "keychain",
            "keychain_service": keychain_service,
            "created_at": datetime.now().isoformat(),
        }

        with open(self.key_file, "w") as f:
            json.dump(encrypted_keys, f, indent=2)

    def get_api_key(self, service: str) -> str:
        """Retrieve API key using stored method."""
        if not self.key_file.exists():
            return ""

        try:
            with open(self.key_file, "r") as f:
                encrypted_keys = json.load(f)

            if service not in encrypted_keys:
                return ""

            key_info = encrypted_keys[service]
            method = key_info.get("method", "encrypted")

            if method == "encrypted":
                return self._decrypt_value(key_info.get("value", ""))
            elif method == "env":
                env_var = key_info.get("env_var", "")
                return os.getenv(env_var, "")
            elif method == "1password":
                return self._get_1password_key(key_info.get("op_reference", ""))
            elif method == "keychain":
                return self._get_keychain_key(key_info.get("keychain_service", ""))

        except Exception as e:
            self.logger.error(f"Failed to retrieve API key for {service}: {e}")

        return ""

    def _get_1password_key(self, op_reference: str) -> str:
        """Retrieve API key from 1Password CLI."""
        try:
            result = subprocess.run(
                ["op", "read", op_reference], capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                self.logger.error(f"1Password CLI error: {result.stderr}")
        except subprocess.TimeoutExpired:
            self.logger.error("1Password CLI timeout")
        except FileNotFoundError:
            self.logger.error(
                "1Password CLI not found. Install with: brew install 1password-cli"
            )
        except Exception as e:
            self.logger.error(f"1Password CLI error: {e}")

        return ""

    def _get_keychain_key(self, service_name: str) -> str:
        """Retrieve API key from macOS Keychain."""
        try:
            result = subprocess.run(
                [
                    "security",
                    "find-generic-password",
                    "-s",
                    service_name,
                    "-w",  # Output password only
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                return result.stdout.strip()
            else:
                self.logger.error(f"Keychain error: {result.stderr}")
        except subprocess.TimeoutExpired:
            self.logger.error("Keychain access timeout")
        except Exception as e:
            self.logger.error(f"Keychain error: {e}")

        return ""

    def list_stored_keys(self) -> Dict[str, Dict]:
        """List all stored API key references (without revealing actual keys)."""
        if not self.key_file.exists():
            return {}

        try:
            with open(self.key_file, "r") as f:
                encrypted_keys = json.load(f)

            # Remove sensitive values for display
            display_keys = {}
            for service, info in encrypted_keys.items():
                display_info = {
                    "method": info.get("method", "encrypted"),
                    "created_at": info.get("created_at", ""),
                    "has_key": bool(self.get_api_key(service)),
                }

                if info.get("method") == "env":
                    display_info["env_var"] = info.get("env_var", "")
                elif info.get("method") == "1password":
                    display_info["op_reference"] = info.get("op_reference", "")
                elif info.get("method") == "keychain":
                    display_info["keychain_service"] = info.get("keychain_service", "")

                display_keys[service] = display_info

            return display_keys

        except Exception as e:
            self.logger.error(f"Failed to list stored keys: {e}")
            return {}

    def delete_api_key(self, service: str):
        """Delete stored API key reference."""
        if not self.key_file.exists():
            return

        try:
            with open(self.key_file, "r") as f:
                encrypted_keys = json.load(f)

            if service in encrypted_keys:
                del encrypted_keys[service]

                with open(self.key_file, "w") as f:
                    json.dump(encrypted_keys, f, indent=2)

                self.logger.info(f"Deleted API key for {service}")

        except Exception as e:
            self.logger.error(f"Failed to delete API key for {service}: {e}")


class DatabaseConfig:
    """Enhanced configuration management with secure API key storage."""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "lit_search_config.json"
        self.config = self._load_config()
        self.key_manager = SecureKeyManager()

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
                    "api_key_method": "encrypted",  # or "env", "1password", "keychain"
                    "api_key_reference": "",  # depends on method
                    "rate_limit": 100,
                },
                "springer": {
                    "enabled": False,
                    "api_key_method": "encrypted",
                    "api_key_reference": "",
                    "rate_limit": 100,
                },
                "elsevier": {
                    "enabled": False,
                    "api_key_method": "encrypted",
                    "api_key_reference": "",
                    "rate_limit": 100,
                },
                "wiley": {
                    "enabled": False,
                    "api_key_method": "encrypted",
                    "api_key_reference": "",
                    "rate_limit": 100,
                },
                "acm": {
                    "enabled": False,
                    "api_key_method": "encrypted",
                    "api_key_reference": "",
                    "rate_limit": 100,
                },
                "google_scholar": {"enabled": True, "rate_limit": 1},
            },
            "cache": {"directory": "./lit_cache", "expire_days": 30},
            "search": {"max_results": 100, "deduplication": True, "min_year": 2018},
            "security": {
                "api_key_storage": "encrypted",  # default method
                "auto_decrypt": True,
            },
        }

        try:
            with open(self.config_path, "r") as f:
                config = json.load(f)
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
        with open(self.config_path, "w") as f:
            json.dump(config, f, indent=2)

    def get_database_config(self, db_name: str) -> Dict:
        """Get configuration for a specific database."""
        return self.config.get("databases", {}).get(db_name, {})

    def is_database_enabled(self, db_name: str) -> bool:
        """Check if a database is enabled."""
        return self.get_database_config(db_name).get("enabled", False)

    def get_api_key(self, db_name: str) -> str:
        """Get API key for a database using secure storage."""
        db_config = self.get_database_config(db_name)

        # Legacy support - check for direct api_key field
        if "api_key" in db_config and db_config["api_key"]:
            return db_config["api_key"]

        # Use secure key manager
        return self.key_manager.get_api_key(db_name)

    def set_api_key(self, db_name: str, api_key: str, method: str = "encrypted"):
        """
        Set API key for a database using secure storage.

        Args:
                db_name: Database name
                api_key: API key or reference (depends on method)
                method: Storage method ('encrypted', 'env', '1password', 'keychain')
        """
        # Store the key securely
        self.key_manager.store_api_key(db_name, api_key, method)

        # Update config to reflect the storage method
        if "databases" not in self.config:
            self.config["databases"] = {}
        if db_name not in self.config["databases"]:
            self.config["databases"][db_name] = {}

        self.config["databases"][db_name]["api_key_method"] = method
        self.config["databases"][db_name]["enabled"] = True

        # Store reference info based on method
        if method == "env":
            self.config["databases"][db_name][
                "api_key_reference"
            ] = api_key  # env var name
        elif method == "1password":
            self.config["databases"][db_name][
                "api_key_reference"
            ] = api_key  # op:// reference
        elif method == "keychain":
            self.config["databases"][db_name][
                "api_key_reference"
            ] = api_key  # service name
        else:
            self.config["databases"][db_name]["api_key_reference"] = "encrypted_locally"

        self._save_config(self.config)

    def list_api_keys(self) -> Dict[str, Dict]:
        """List all stored API keys (without revealing actual keys)."""
        return self.key_manager.list_stored_keys()

    def delete_api_key(self, db_name: str):
        """Delete API key for a database."""
        self.key_manager.delete_api_key(db_name)

        # Update config
        if db_name in self.config.get("databases", {}):
            self.config["databases"][db_name]["enabled"] = False
            self.config["databases"][db_name]["api_key_reference"] = ""
            self._save_config(self.config)


class CacheManager:
    """Manages local caching of search results and paper metadata."""

    def __init__(self, cache_dir: str = "./lit_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.db_path = self.cache_dir / "papers.db"
        self._init_database()
        self.logger = logging.getLogger(__name__)

    def _init_database(self):
        """Initialize SQLite database for caching."""
        with sqlite3.connect(self.db_path) as conn:
            # Drop existing tables to ensure clean schema
            conn.execute("DROP TABLE IF EXISTS papers")
            conn.execute("DROP TABLE IF EXISTS searches")

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
                             cached_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                         )
			             """
            )

            conn.execute(
                """
                         CREATE TABLE searches
                         (
                             query_hash  TEXT PRIMARY KEY,
                             query       TEXT,
                             results     TEXT,
                             searched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                         )
			             """
            )

            conn.execute("CREATE INDEX IF NOT EXISTS idx_title ON papers(title)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_year ON papers(year)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_source ON papers(database_source)"
            )

    def _generate_paper_id(self, paper: Paper) -> str:
        """Generate unique ID for a paper."""
        content = f"{paper.title}|{paper.authors}|{paper.year}"
        return hashlib.md5(content.encode()).hexdigest()

    def _generate_query_hash(self, query: str, databases: List[str]) -> str:
        """Generate hash for search query."""
        content = f"{query}|{sorted(databases)}"
        return hashlib.md5(content.encode()).hexdigest()

    def cache_paper(self, paper: Paper):
        """Cache a paper in the database."""
        paper_id = self._generate_paper_id(paper)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO papers
                (id, title, authors, abstract, year, venue, doi, arxiv_id, url, citations,
                 keywords, database_source, pdf_url, publication_type, issue, volume, pages, publisher)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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

    def get_cached_papers(self, query: str = None, year_min: int = None) -> List[Paper]:
        """Retrieve cached papers with optional filtering."""
        with sqlite3.connect(self.db_path) as conn:
            sql = """SELECT id, \
                            title, \
                            authors, \
                            abstract, year, venue, doi, arxiv_id, url, citations, keywords, database_source, pdf_url, publication_type, issue, volume, pages, publisher
                     FROM papers \
                     WHERE 1=1"""
            params = []

            if query:
                sql += " AND (title LIKE ? OR abstract LIKE ?)"
                params.extend([f"%{query}%", f"%{query}%"])

            if year_min:
                sql += " AND year >= ?"
                params.append(year_min)

            sql += " ORDER BY year DESC, citations DESC"

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

    def cache_search_results(
        self, query: str, databases: List[str], results: List[Paper]
    ):
        """Cache search results."""
        query_hash = self._generate_query_hash(query, databases)
        results_json = json.dumps([paper.to_dict() for paper in results])

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO searches VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            """,
                (query_hash, query, results_json),
            )

    def get_cached_search(
        self, query: str, databases: List[str], max_age_hours: int = 24
    ) -> Optional[List[Paper]]:
        """Get cached search results if recent enough."""
        query_hash = self._generate_query_hash(query, databases)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT results, searched_at FROM searches
                WHERE query_hash = ? AND searched_at > datetime('now', '-{} hours')
            """.format(
                    max_age_hours
                ),
                (query_hash,),
            )

            row = cursor.fetchone()
            if row:
                results_data = json.loads(row[0])
                return [Paper.from_dict(data) for data in results_data]

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


class ArxivAdapter:
    """Adapter for arXiv API with Boolean query support."""

    def __init__(self, rate_limit: float = 3.0):
        self.base_url = "http://export.arxiv.org/api/query"
        self.rate_limit = rate_limit
        self.last_request = 0
        self.logger = logging.getLogger(__name__)
        self.query_parser = QueryParser()

    def _rate_limit_wait(self):
        """Implement rate limiting."""
        elapsed = time.time() - self.last_request
        if elapsed < (1.0 / self.rate_limit):
            time.sleep((1.0 / self.rate_limit) - elapsed)
        self.last_request = time.time()

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
            return papers

        except Exception as e:
            self.logger.error(f"ArXiv search failed: {e}")
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


class IEEEAdapter:
    """Adapter for IEEE Xplore API with Boolean query support."""

    def __init__(self, api_key: str, rate_limit: float = 100.0):
        self.api_key = api_key
        self.base_url = "http://ieeexploreapi.ieee.org/api/v1/search/articles"
        self.rate_limit = rate_limit
        self.last_request = 0
        self.logger = logging.getLogger(__name__)
        self.query_parser = QueryParser()

    def _rate_limit_wait(self):
        """Implement rate limiting."""
        elapsed = time.time() - self.last_request
        if elapsed < (3600.0 / self.rate_limit):
            time.sleep((3600.0 / self.rate_limit) - elapsed)
        self.last_request = time.time()

    def search(
        self, query: str, max_results: int = 100, year_min: int = None
    ) -> List[Paper]:
        """Search IEEE Xplore with Boolean query support."""
        if not self.api_key:
            self.logger.warning("IEEE API key not provided")
            return []

        self._rate_limit_wait()

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

        try:
            url = f"{self.base_url}?{urllib.parse.urlencode(params)}"
            with urllib.request.urlopen(url, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))

            papers = []
            for article in data.get("articles", []):
                paper = PaperFormatter.format_ieee_paper(article)
                papers.append(paper)

            return papers

        except Exception as e:
            self.logger.error(f"IEEE search failed: {e}")
            return []


class ElsevierAdapter:
    """Adapter for Elsevier/ScienceDirect API."""

    def __init__(self, api_key: str, rate_limit: float = 100.0):
        self.api_key = api_key
        self.base_url = "https://api.elsevier.com/content/search/sciencedirect"
        self.rate_limit = rate_limit
        self.last_request = 0
        self.logger = logging.getLogger(__name__)
        self.query_parser = QueryParser()

    def search(self, query: str, max_results: int = 100) -> List[Paper]:
        """Search Elsevier ScienceDirect."""
        if not self.api_key:
            self.logger.warning("Elsevier API key not provided")
            return []

        parsed_query = self.query_parser.parse_boolean_query(query)
        simple_query = self.query_parser.to_simple_query(parsed_query)

        headers = {"X-ELS-APIKey": self.api_key, "Accept": "application/json"}

        params = {"query": simple_query, "count": min(max_results, 100)}

        try:
            url = f"{self.base_url}?{urllib.parse.urlencode(params)}"
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))

            papers = []
            for entry in data.get("search-results", {}).get("entry", []):
                paper = PaperFormatter.format_elsevier_paper(entry)
                papers.append(paper)

            return papers

        except Exception as e:
            self.logger.error(f"Elsevier search failed: {e}")
            return []


class SpringerAdapter:
    """Adapter for Springer Nature API."""

    def __init__(self, api_key: str, rate_limit: float = 100.0):
        self.api_key = api_key
        self.base_url = "http://api.springernature.com/meta/v1/json"
        self.rate_limit = rate_limit
        self.last_request = 0
        self.logger = logging.getLogger(__name__)
        self.query_parser = QueryParser()

    def search(
        self, query: str, max_results: int = 100, year_min: int = None
    ) -> List[Paper]:
        """Search Springer Nature."""
        if not self.api_key:
            self.logger.warning("Springer API key not provided")
            return []

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

        try:
            url = f"{self.base_url}?{urllib.parse.urlencode(params)}"
            with urllib.request.urlopen(url, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))

            papers = []
            for record in data.get("records", []):
                paper = PaperFormatter.format_springer_paper(record)
                papers.append(paper)

            return papers

        except Exception as e:
            self.logger.error(f"Springer search failed: {e}")
            return []


class WileyAdapter:
    """Adapter for Wiley Online Library API."""

    def __init__(self, api_key: str, rate_limit: float = 100.0):
        self.api_key = api_key
        self.base_url = "https://api.wiley.com/onlinelibrary/tdm/v1/articles"
        self.rate_limit = rate_limit
        self.last_request = 0
        self.logger = logging.getLogger(__name__)
        self.query_parser = QueryParser()

    def search(self, query: str, max_results: int = 100) -> List[Paper]:
        """Search Wiley Online Library."""
        if not self.api_key:
            self.logger.warning("Wiley API key not provided")
            return []

        parsed_query = self.query_parser.parse_boolean_query(query)
        simple_query = self.query_parser.to_simple_query(parsed_query)

        headers = {"Wiley-TDM-Client-Token": self.api_key, "Accept": "application/json"}

        params = {"query": simple_query, "count": min(max_results, 100)}

        try:
            url = f"{self.base_url}?{urllib.parse.urlencode(params)}"
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))

            papers = []
            for item in data.get("items", []):
                paper = PaperFormatter.format_wiley_paper(item)
                papers.append(paper)

            return papers

        except Exception as e:
            self.logger.error(f"Wiley search failed: {e}")
            return []


class ACMAdapter:
    """Adapter for ACM Digital Library."""

    def __init__(self, api_key: str = None, rate_limit: float = 100.0):
        self.api_key = api_key
        self.base_url = "https://dl.acm.org/action/doSearch"
        self.rate_limit = rate_limit
        self.last_request = 0
        self.logger = logging.getLogger(__name__)
        self.query_parser = QueryParser()

    def search(self, query: str, max_results: int = 100) -> List[Paper]:
        """Search ACM Digital Library (limited scraping approach)."""
        self.logger.warning(
            "ACM Digital Library requires web scraping - limited functionality"
        )
        # ACM doesn't have a public API, would require careful web scraping
        # For now, return empty list to avoid blocking issues
        return []


class GoogleScholarAdapter:
    """Limited Google Scholar adapter."""

    def __init__(self, rate_limit: float = 1.0):
        self.rate_limit = rate_limit
        self.last_request = 0
        self.logger = logging.getLogger(__name__)
        self.query_parser = QueryParser()

    def search(
        self, query: str, max_results: int = 20, year_min: int = None
    ) -> List[Paper]:
        """Limited Google Scholar search."""
        self.logger.warning("Google Scholar adapter provides limited functionality")
        return []


class LiteratureSearchEngine:
    """Main search engine coordinating multiple database adapters with Boolean query support."""

    def __init__(self, config_path: Optional[str] = None):
        self.config = DatabaseConfig(config_path)
        self.cache = CacheManager(self.config.config["cache"]["directory"])
        self.adapters = {}
        self.query_parser = QueryParser()
        self.logger = logging.getLogger(__name__)

        self._init_adapters()

    def _init_adapters(self):
        """Initialize database adapters based on configuration."""
        if self.config.is_database_enabled("arxiv"):
            config = self.config.get_database_config("arxiv")
            self.adapters["arxiv"] = ArxivAdapter(config.get("rate_limit", 3))

        if self.config.is_database_enabled("ieee"):
            api_key = self.config.get_api_key("ieee")
            if api_key:
                config = self.config.get_database_config("ieee")
                self.adapters["ieee"] = IEEEAdapter(
                    api_key, config.get("rate_limit", 100)
                )
            else:
                self.logger.warning("IEEE API key not configured")

        if self.config.is_database_enabled("springer"):
            api_key = self.config.get_api_key("springer")
            if api_key:
                config = self.config.get_database_config("springer")
                self.adapters["springer"] = SpringerAdapter(
                    api_key, config.get("rate_limit", 100)
                )
            else:
                self.logger.warning("Springer API key not configured")

        if self.config.is_database_enabled("elsevier"):
            api_key = self.config.get_api_key("elsevier")
            if api_key:
                config = self.config.get_database_config("elsevier")
                self.adapters["elsevier"] = ElsevierAdapter(
                    api_key, config.get("rate_limit", 100)
                )
            else:
                self.logger.warning("Elsevier API key not configured")

        if self.config.is_database_enabled("wiley"):
            api_key = self.config.get_api_key("wiley")
            if api_key:
                config = self.config.get_database_config("wiley")
                self.adapters["wiley"] = WileyAdapter(
                    api_key, config.get("rate_limit", 100)
                )
            else:
                self.logger.warning("Wiley API key not configured")

        if self.config.is_database_enabled("acm"):
            config = self.config.get_database_config("acm")
            self.adapters["acm"] = ACMAdapter(rate_limit=config.get("rate_limit", 100))

        if self.config.is_database_enabled("google_scholar"):
            config = self.config.get_database_config("google_scholar")
            self.adapters["google_scholar"] = GoogleScholarAdapter(
                config.get("rate_limit", 1)
            )

    def search(
        self,
        query: str,
        databases: List[str] = None,
        max_results: int = None,
        use_cache: bool = True,
        year_min: int = None,
    ) -> List[Paper]:
        """
        Search multiple databases with Boolean query support.

        Args:
                query: Boolean search query (e.g., '("software" OR "system") AND "configuration"')
                databases: List of database names to search
                max_results: Maximum results per database
                use_cache: Whether to use cached results
                year_min: Minimum publication year

        Returns:
                List of Paper objects, deduplicated and unified
        """
        if databases is None:
            databases = list(self.adapters.keys())

        if max_results is None:
            max_results = self.config.config["search"]["max_results"]

        if year_min is None:
            year_min = self.config.config["search"]["min_year"]

        self.logger.info(f"Searching with Boolean query: {query}")
        self.logger.info(f"Target databases: {databases}")

        # Check cache first
        if use_cache:
            cached_results = self.cache.get_cached_search(query, databases)
            if cached_results:
                self.logger.info(f"Retrieved {len(cached_results)} papers from cache")
                return self._filter_by_year(cached_results, year_min)

        # Perform searches across all databases
        all_papers = []
        for db_name in databases:
            if db_name not in self.adapters:
                self.logger.warning(f"Database {db_name} not available")
                continue

            self.logger.info(f"Searching {db_name}...")
            try:
                if db_name == "arxiv":
                    categories = self.config.get_database_config("arxiv").get(
                        "categories", []
                    )
                    papers = self.adapters[db_name].search(
                        query, max_results, categories
                    )
                else:
                    papers = self.adapters[db_name].search(query, max_results, year_min)

                # Apply year filter
                if year_min:
                    papers = [p for p in papers if p.year >= year_min]

                all_papers.extend(papers)
                self.logger.info(f"Found {len(papers)} papers from {db_name}")

                # Cache individual papers
                for paper in papers:
                    self.cache.cache_paper(paper)

            except Exception as e:
                self.logger.error(f"Error searching {db_name}: {e}")

        # Deduplicate results using canonical IDs
        if self.config.config["search"]["deduplication"]:
            all_papers = self._deduplicate_papers(all_papers)

        # Cache search results
        if use_cache:
            self.cache.cache_search_results(query, databases, all_papers)

        # Sort by relevance (citations and year)
        all_papers.sort(key=lambda p: (-p.citations, -p.year))

        self.logger.info(f"Total unique papers found: {len(all_papers)}")
        return all_papers

    def _filter_by_year(self, papers: List[Paper], year_min: int) -> List[Paper]:
        """Filter papers by minimum year."""
        if not year_min:
            return papers
        return [p for p in papers if p.year >= year_min]

    def _deduplicate_papers(self, papers: List[Paper]) -> List[Paper]:
        """Remove duplicate papers using canonical IDs and advanced matching."""
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

        for paper in papers_sorted:
            canonical_id = paper.get_canonical_id()

            if canonical_id not in seen_ids:
                seen_ids.add(canonical_id)
                unique_papers.append(paper)

        self.logger.info(f"Removed {len(papers) - len(unique_papers)} duplicates")
        return unique_papers

    def get_cached_papers(self, query: str = None, year_min: int = None) -> List[Paper]:
        """Retrieve papers from local cache."""
        return self.cache.get_cached_papers(query, year_min)

    def export_results(
        self, papers: List[Paper], format: str = "json", filename: str = None
    ) -> str:
        """Export search results to various formats."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"literature_search_{timestamp}.{format}"

        if format == "json":
            return self._export_json(papers, filename)
        elif format == "bibtex":
            return self._export_bibtex(papers, filename)
        elif format == "csv":
            return self._export_csv(papers, filename)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def _export_json(self, papers: List[Paper], filename: str) -> str:
        """Export papers to JSON format."""
        data = [paper.to_dict() for paper in papers]
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

    def _export_csv(self, papers: List[Paper], filename: str) -> str:
        """Export papers to CSV format."""
        import csv

        with open(filename, "w", newline="", encoding="utf-8") as f:
            if not papers:
                return filename

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
                    }
                )
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
        }

        all_papers = []
        db_papers = {}

        for db_name in databases:
            if db_name in self.adapters:
                try:
                    papers = self.adapters[db_name].search(
                        query, 50
                    )  # Limited for analysis
                    db_papers[db_name] = papers
                    all_papers.extend(papers)

                    coverage_report["databases_searched"].append(db_name)
                    coverage_report["results_per_database"][db_name] = len(papers)

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


# API Key Management CLI
class APIKeyManager:
    """Command-line interface for managing API keys securely."""

    def __init__(self):
        self.config = DatabaseConfig()

    def setup_interactive(self):
        """Interactive setup for API keys."""
        print("🔐 Literature Search API Key Setup")
        print("=" * 50)

        databases = ["ieee", "springer", "elsevier", "wiley", "acm"]

        for db in databases:
            print(f"\n📚 Setting up {db.upper()} API Key")
            print("-" * 30)

            choice = input(f"Configure {db}? (y/n): ").lower().strip()
            if choice != "y":
                continue

            print("\nChoose storage method:")
            print("1. Encrypted locally (recommended)")
            print("2. Environment variable")
            print("3. 1Password CLI")
            print("4. macOS Keychain")

            method_choice = input("Enter choice (1-4): ").strip()

            if method_choice == "1":
                api_key = input(f"Enter {db} API key: ").strip()
                if api_key:
                    self.config.set_api_key(db, api_key, "encrypted")
                    print(f"✅ {db} API key stored securely (encrypted)")

            elif method_choice == "2":
                env_var = input(
                    f"Enter environment variable name (e.g., {db.upper()}_API_KEY): "
                ).strip()
                if env_var:
                    self.config.set_api_key(db, env_var, "env")
                    print(f"✅ {db} configured to use ${env_var}")
                    print(f"   Remember to set: export {env_var}=your_api_key")

            elif method_choice == "3":
                op_ref = input(
                    f"Enter 1Password reference (e.g., op://vault/{db}-api-key/credential): "
                ).strip()
                if op_ref:
                    self.config.set_api_key(db, op_ref, "1password")
                    print(f"✅ {db} configured to use 1Password")
                    print(f"   Reference: {op_ref}")

            elif method_choice == "4":
                service_name = input(
                    f"Enter Keychain service name (e.g., {db}-api-key): "
                ).strip()
                if service_name:
                    self.config.set_api_key(db, service_name, "keychain")
                    print(f"✅ {db} configured to use Keychain")
                    print(f"   Service: {service_name}")
                    print(
                        f"   Add to keychain: security add-generic-password -s {service_name} -a {db} -w your_api_key"
                    )

            else:
                print("Invalid choice, skipping...")

        print(f"\n🎉 Setup complete!")
        self.show_status()

    def show_status(self):
        """Show current API key configuration status."""
        print("\n📊 API Key Status")
        print("=" * 40)

        stored_keys = self.config.list_api_keys()
        databases = ["ieee", "springer", "elsevier", "wiley", "acm"]

        for db in databases:
            if db in stored_keys:
                info = stored_keys[db]
                status = "✅ Configured" if info["has_key"] else "❌ Key not accessible"
                method = info["method"]
                print(f"{db:12}: {status:15} ({method})")

                if method == "env" and "env_var" in info:
                    env_val = "SET" if os.getenv(info["env_var"]) else "NOT SET"
                    print(f"            Environment: ${info['env_var']} = {env_val}")
                elif method == "1password" and "op_reference" in info:
                    print(f"            1Password: {info['op_reference']}")
                elif method == "keychain" and "keychain_service" in info:
                    print(f"            Keychain: {info['keychain_service']}")
            else:
                print(f"{db:12}: ❌ Not configured")

        print(
            f'\nℹ️  To reconfigure any database, run: python -c "from searcher import APIKeyManager; APIKeyManager().setup_interactive()"'
        )

    def add_key(self, database: str, api_key: str, method: str = "encrypted"):
        """Add API key programmatically."""
        try:
            self.config.set_api_key(database, api_key, method)
            print(f"✅ {database} API key configured using {method} method")
        except Exception as e:
            print(f"❌ Failed to configure {database}: {e}")

    def test_keys(self):
        """Test API key accessibility."""
        print("\n🧪 Testing API Key Access")
        print("=" * 40)

        databases = ["ieee", "springer", "elsevier", "wiley"]

        for db in databases:
            try:
                api_key = self.config.get_api_key(db)
                if api_key:
                    # Mask the key for display
                    masked_key = (
                        api_key[:8] + "..." + api_key[-4:]
                        if len(api_key) > 12
                        else "***"
                    )
                    print(f"{db:12}: ✅ Accessible ({masked_key})")
                else:
                    print(f"{db:12}: ❌ Not found")
            except Exception as e:
                print(f"{db:12}: ❌ Error - {e}")

    def delete_key(self, database: str):
        """Delete API key for a database."""
        try:
            self.config.delete_api_key(database)
            print(f"✅ {database} API key deleted")
        except Exception as e:
            print(f"❌ Failed to delete {database} key: {e}")


def setup_api_keys():
    """Convenience function for API key setup."""
    manager = APIKeyManager()
    manager.setup_interactive()


# Example usage and testing
def main():
    """Example usage of the Advanced Literature Search Library with secure API keys."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    print("🔍 Advanced Literature Search Library")
    print("=" * 60)

    # Check if API keys are configured
    manager = APIKeyManager()
    stored_keys = DatabaseConfig().list_api_keys()

    if not stored_keys:
        print("\n⚠️  No API keys configured yet.")
        choice = input("Would you like to set up API keys now? (y/n): ").lower().strip()
        if choice == "y":
            manager.setup_interactive()
            print("\n" + "=" * 60)
    else:
        print("\n📊 Current API Key Status:")
        manager.show_status()
        print("\n" + "=" * 60)

    # Initialize search engine
    engine = LiteratureSearchEngine()

    # Example Boolean search query for ML/Systems research
    boolean_query = '("software" OR "system") AND "configuration" AND ("performance prediction" OR "performance modeling" OR "performance learning" OR "configuration tuning") AND ("deep learning" OR "neural network")'

    print(f"\n🔍 Searching with Boolean query:")
    print(f"'{boolean_query}'")
    print("-" * 80)

    # Search across available databases
    available_databases = list(engine.adapters.keys())
    print(f"Available databases: {available_databases}")

    papers = engine.search(
        query=boolean_query,
        max_results=20,
        year_min=2020,
        databases=available_databases[:3],  # Limit to first 3 to avoid rate limits
    )

    print(f"\n📚 Found {len(papers)} papers")
    print("=" * 80)

    # Display results
    for i, paper in enumerate(papers[:5], 1):  # Show first 5 results
        print(f"\n{i}. {paper.title}")
        print(
            f"   Authors: {', '.join(paper.authors[:3])}{'...' if len(paper.authors) > 3 else ''}"
        )
        print(f"   Year: {paper.year}, Venue: {paper.venue}")
        print(f"   Source: {paper.database_source}")
        print(f"   Citations: {paper.citations}")
        if paper.doi:
            print(f"   DOI: {paper.doi}")
        if paper.abstract:
            print(f"   Abstract: {paper.abstract[:200]}...")
        if paper.keywords:
            print(f"   Keywords: {', '.join(paper.keywords[:5])}")

    # Export results
    if papers:
        try:
            json_file = engine.export_results(papers, format="json")
            bibtex_file = engine.export_results(papers, format="bibtex")
            csv_file = engine.export_results(papers, format="csv")

            print(f"\n💾 Results exported to:")
            print(f"  📄 JSON: {json_file}")
            print(f"  📖 BibTeX: {bibtex_file}")
            print(f"  📊 CSV: {csv_file}")
        except Exception as e:
            print(f"❌ Export failed: {e}")

    # Demonstrate search coverage analysis
    if papers:
        print(f"\n📈 Performing coverage analysis...")
        try:
            coverage = engine.analyze_search_coverage(
                boolean_query, available_databases[:2]
            )
            print(f"Coverage Report:")
            print(f"  🔍 Databases searched: {coverage['databases_searched']}")
            print(f"  📄 Total papers: {coverage['total_papers']}")
            print(f"  🎯 Unique papers: {coverage['unique_papers']}")
            if coverage["overlap_analysis"]:
                print(
                    f"  🔄 Database overlaps: {list(coverage['overlap_analysis'].keys())}"
                )
        except Exception as e:
            print(f"❌ Coverage analysis failed: {e}")

    # Show API key management options
    print(f"\n⚙️  API Key Management:")
    print(
        f'  • Setup keys: python -c "from searcher import setup_api_keys; setup_api_keys()"'
    )
    print(
        f'  • Check status: python -c "from searcher import APIKeyManager; APIKeyManager().show_status()"'
    )
    print(
        f'  • Test access: python -c "from searcher import APIKeyManager; APIKeyManager().test_keys()"'
    )

    # Show usage examples for different key storage methods
    print(f"\n🔐 Secure API Key Storage Examples:")
    print(f"")
    print(f"1. Local Encryption (Recommended):")
    print(f"   from searcher import DatabaseConfig")
    print(f"   config = DatabaseConfig()")
    print(f"   config.set_api_key('ieee', 'your-api-key', 'encrypted')")
    print(f"")
    print(f"2. Environment Variables:")
    print(f"   export IEEE_API_KEY=your-api-key")
    print(f"   config.set_api_key('ieee', 'IEEE_API_KEY', 'env')")
    print(f"")
    print(f"3. 1Password CLI:")
    print(f"   # Store in 1Password, then:")
    print(
        f"   config.set_api_key('ieee', 'op://vault/ieee-api/credential', '1password')"
    )
    print(f"")
    print(f"4. macOS Keychain:")
    print(f"   security add-generic-password -s ieee-api -a ieee -w your-api-key")
    print(f"   config.set_api_key('ieee', 'ieee-api', 'keychain')")


if __name__ == "__main__":
    main()
