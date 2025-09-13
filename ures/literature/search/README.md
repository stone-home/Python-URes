## 🔧 Detailed Usage

### API Key Management

```bash
# Setup keys interactively
python main_cli.py setup-keys

# Test key accessibility
python main_cli.py test-keys

# Show current configuration
python main_cli.py status
```

### Advanced Searches

```bash
# Complex Boolean query
python main_cli.py search '("performance prediction" OR "performance modeling") AND ("deep learning" OR "neural network")' \
  --databases arxiv ieee \
  --year-min 2020 \
  --max-results 50 \
  --show 20 \
  --show-abstracts \
  --export \
  --format bibtex

# Search specific databases only
python main_cli.py search "federated learning" --databases arxiv springer

# Disable caching for fresh results
python main_cli.py search "edge computing" --no-cache# Literature Search System

A comprehensive, modular literature search library for academic research with secure API key management and Boolean query support.

## 🚀 Features

- **Multi-Database Search**: Search across arXiv, IEEE Xplore, Springer, Elsevier, Wiley, and more
- **Boolean Query Support**: Complex search queries with AND, OR, parentheses, and quoted phrases
- **Secure Key Management**: Multiple storage methods (local encryption, 1Password, macOS Keychain, environment variables)
- **Intelligent Caching**: SQLite-based caching with deduplication and similarity detection
- **Multiple Export Formats**: JSON, BibTeX, CSV export for reference managers
- **Modular Architecture**: Clean separation of concerns with abstract base classes
- **Command-Line Interface**: Comprehensive CLI for all operations

## 📁 File Structure

```
literature_search/
├── secure_key_manager.py      # 🔐 Secure API key management SDK
├── database_models.py         # 📊 Paper models and database operations
├── database_adapters.py       # 🔌 Database adapter implementations
├── literature_search_engine.py # 🔍 Main search engine orchestrator
├── main_cli.py                # 💻 Command-line interface
└── README.md                  # 📖 This documentation
```

## 🏗️ Architecture

### 1. **Secure Key Manager** (`secure_key_manager.py`)
- Standalone SDK for secure API key storage
- Supports multiple storage methods:
  - **Local Encryption**: AES-encrypted local storage (recommended)
  - **Environment Variables**: `export API_KEY=value`
  - **1Password CLI**: `op://vault/item/field` references
  - **macOS Keychain**: System keychain integration

### 2. **Database Models** (`database_models.py`)
- `Paper`: Unified paper representation with normalization
- `CacheManager`: SQLite-based caching with advanced features
- `PaperFormatter`: Converts database-specific formats to unified format

### 3. **Database Adapters** (`database_adapters.py`)
- `DatabaseAdapter`: Abstract base class for all adapters
- Concrete implementations for each database:
  - `ArxivAdapter`: Free, no API key required
  - `IEEEAdapter`: Requires API key
  - `ElsevierAdapter`: Requires API key
  - `SpringerAdapter`: Requires API key
  - `WileyAdapter`: Requires API key
- `AdapterFactory`: Factory pattern for adapter creation

### 4. **Search Engine** (`literature_search_engine.py`)
- `LiteratureSearchEngine`: Main orchestrator
- `DatabaseConfig`: Configuration management
- Coordinates adapters, caching, and result processing

### 5. **CLI Interface** (`main_cli.py`)
- Comprehensive command-line interface
- Integrates all components seamlessly

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the files
# No external dependencies required (uses only Python standard library)
```

### 2. Setup API Keys

```bash
# Interactive setup for all databases
python main_cli.py setup-keys

# Check current status
python main_cli.py status
```

### 3. Basic Search

```bash
# Simple search
python main_cli.py search "machine learning performance"

# Boolean search with export
python main_cli.py search '("software" OR "system") AND "configuration"' --export --format bibtex
```

## 🔧 Detailed Usage

### API Key Management

```bash
# Setup keys interactively
python main_cli.py setup-keys

# Test key accessibility
python main_cli.py test-keys

# Show current configuration
python main_cli.py status
```

### Advanced Searches

```bash
# Complex Boolean query
python main_cli.py search '("performance prediction" OR "performance modeling") AND ("deep learning" OR "neural network")' \
  --databases arxiv ieee \
  --year-min 2020 \
  --max-results 50 \
  --show 20 \
  --show-abstracts \
  --export \
  --format bibtex

# Search specific databases only
python main_cli.py search "federated learning" --databases arxiv springer

# Disable caching for fresh results
python main_cli.py search "edge computing" --no-cache
```

### Cache Management

```bash
# Show cache statistics
python main_cli.py cache stats

# Clean old entries (default: 30 days)
python main_cli.py cache clean --days 60

# Export cached papers
python main_cli.py cache export --format csv --output my_papers.csv

# Find duplicate papers
python main_cli.py cache duplicates --similarity 0.9
```

### Coverage Analysis

```bash
# Analyze search coverage across databases
python main_cli.py coverage "machine learning optimization" --verbose

# Compare specific databases
python main_cli.py coverage "neural architecture search" --databases arxiv ieee springer
```

### Configuration

```bash
# Show current configuration
python main_cli.py config show

# Update configuration
python main_cli.py config set --section search --key max_results --value 200
```

## 🔐 API Key Storage Methods

### 1. Local Encryption (Recommended)
```python
from secure_key_manager import SecureKeyManager

manager = SecureKeyManager("literature-search")
manager.store_key("ieee", "your-api-key", "encrypted")
```

### 2. Environment Variables
```bash
export IEEE_API_KEY="your-api-key"
export SPRINGER_API_KEY="your-springer-key"

# Then configure references
python -c "
from literature_search_engine import DatabaseConfig
config = DatabaseConfig()
config.set_api_key('ieee', 'IEEE_API_KEY', 'env')
config.set_api_key('springer', 'SPRINGER_API_KEY', 'env')
"
```

### 3. 1Password SDK (Official)
```bash
# Install the official 1Password SDK
pip install onepassword

# Create a Service Account in 1Password
# 1. Go to your 1Password account settings
# 2. Create a new Service Account
# 3. Copy the service account token (starts with 'ops_')

# Set the environment variable
export OP_SERVICE_ACCOUNT_TOKEN="ops_your_token_here"

# Store your API keys in 1Password items, then reference them
python -c "
from literature_search_engine import DatabaseConfig
config = DatabaseConfig()
config.set_api_key('ieee', 'op://Private/IEEE-API/credential', '1password')
"
```

**1Password SDK Usage Example:**
```python
import asyncio
from secure_key_manager import SecureKeyManager

async def main():
    manager = SecureKeyManager("literature-search")

    # Store reference to 1Password item
    success = manager.store_key("ieee", "op://Private/IEEE-API/credential", "1password")

    # Retrieve key asynchronously (recommended for 1Password)
    api_key = await manager.get_key_async("ieee")

    # Or use synchronous wrapper (runs async internally)
    api_key_sync = manager.get_key("ieee")

asyncio.run(main())
```

### 4. macOS Keychain
```bash
# Add to keychain
security add-generic-password -s ieee-api -a literature-search -w your-api-key

# Configure reference
python -c "
from literature_search_engine import DatabaseConfig
config = DatabaseConfig()
config.set_api_key('ieee', 'ieee-api', 'keychain')
"
```

### 1Password Setup Instructions

The system now uses the official 1Password SDK for better security and reliability:

1. **Install 1Password SDK**:
   ```bash
   pip install onepassword
   ```

2. **Create Service Account**:
   - Log into your 1Password account
   - Go to Settings → Developer → Service Accounts
   - Create a new Service Account
   - Copy the token (starts with `ops_`)

3. **Set Environment Variable**:
   ```bash
   export OP_SERVICE_ACCOUNT_TOKEN="ops_your_service_account_token"
   ```

4. **Store API Keys in 1Password**:
   - Create secure notes or login items for each API key
   - Note the secret references (format: `op://vault/item/field`)

5. **Configure Key Manager**:
   ```python
   from secure_key_manager import SecureKeyManager

   manager = SecureKeyManager("literature-search")

   # Store references to 1Password items
   manager.store_key("ieee", "op://Private/IEEE-API/credential", "1password")
   manager.store_key("springer", "op://Private/Springer-API/password", "1password")
   ```

**1Password Benefits**:
- 🔒 Enterprise-grade security
- 👥 Team sharing capabilities
- 🔄 Automatic rotation support
- 📱 Cross-platform access
- 🛡️ Zero-trust architecture

## 🌐 Supported Databases

| Database | API Required | Free | Status | Setup URL |
|----------|-------------|------|--------|-----------|
| arXiv | ❌ No | ✅ Yes | ✅ Active | - |
| IEEE Xplore | ✅ Yes | ❌ Paid | ✅ Active | https://developer.ieee.org/ |
| Springer Nature | ✅ Yes | ✅ Yes | ✅ Active | https://dev.springernature.com/ |
| Elsevier/ScienceDirect | ✅ Yes | ❌ Paid | ✅ Active | https://dev.elsevier.com/ |
| Wiley Online Library | ✅ Yes | ❌ Paid | ✅ Active | https://onlinelibrary.wiley.com/library-info/resources/text-and-datamining |
| ACM Digital Library | ❌ No | ✅ Yes | 🚧 Limited | - |
| Google Scholar | ❌ No | ✅ Yes | 🚧 Limited | - |

## 📊 Boolean Query Examples

```python
# Basic operators
"machine learning AND performance"
"software OR system"
"neural network NOT optimization"

# Grouping with parentheses
'("deep learning" OR "neural network") AND ("performance" OR "optimization")'

# Complex nested queries
'("software" OR "system") AND "configuration" AND ("performance prediction" OR "performance modeling" OR "performance learning") AND ("deep learning" OR "neural network")'

# Quoted phrases for exact matches
'"federated learning" AND "edge computing"'
```

## 🔧 Programmatic Usage

### Basic Search
```python
from literature_search_engine import LiteratureSearchEngine

# Initialize engine
engine = LiteratureSearchEngine()

# Search with Boolean query
papers = engine.search(
    query='("machine learning" OR "deep learning") AND "performance"',
    max_results=100,
    year_min=2020
)

# Export results
engine.export_results(papers, format="bibtex", filename="my_papers.bib")
```

### Async Key Management (1Password)
```python
import asyncio
from secure_key_manager import SecureKeyManager

async def setup_keys():
    manager = SecureKeyManager("my-app")

    # Store 1Password references
    await manager.store_key("ieee", "op://Private/IEEE-API/credential", "1password")

    # Retrieve keys asynchronously
    ieee_key = await manager.get_key_async("ieee")

    return ieee_key

# Run async setup
api_key = asyncio.run(setup_keys())
```

### Sync Key Management (Other Methods)
```python
from secure_key_manager import SecureKeyManager

# Initialize key manager
manager = SecureKeyManager("my-app")

# Store encrypted API key
manager.store_key("ieee", "your-api-key", "encrypted")

# Store environment variable reference
manager.store_key("springer", "SPRINGER_API_KEY", "env")

# Retrieve API key (works synchronously for non-1Password methods)
api_key = manager.get_key("ieee")

# List stored keys
keys = manager.list_keys()
```

### Cache Operations
```python
from database_models import CacheManager

# Initialize cache
cache = CacheManager("./my_cache")

# Get cached papers
papers = cache.get_cached_papers(query="machine learning", year_min=2020)

# Get cache statistics
stats = cache.get_cache_stats()

# Find duplicates
duplicates = cache.find_duplicates(similarity_threshold=0.8)
```

## 🌐 Supported Databases

| Database | API Required | Free | Status   |
|----------|-------------|------|----------|
| arXiv | ❌ No | ✅ Yes | ✅ Active |
| IEEE Xplore | ✅ Yes | ❌ Paid | ✅ Active |
| Springer Nature | ✅ Yes | ✅ Yes | ✅ Active |
| Elsevier/ScienceDirect | ✅ Yes | ❌ Paid | ✅ Active |
| Wiley Online Library | ✅ Yes | ❌ Paid | ✅ Active |
| ACM Digital Library | ❌ No | ✅ Yes | ✅ Active |
| Google Scholar | ❌ No | ✅ Yes | ✅ Active |

### API Key Signup URLs:
- **IEEE Xplore**: https://developer.ieee.org/
- **Springer Nature**: https://dev.springernature.com/
- **Elsevier**: https://dev.elsevier.com/
- **Wiley**: https://onlinelibrary.wiley.com/library-info/resources/text-and-datamining

## 📊 Boolean Query Examples

```python
# Basic operators
"machine learning AND performance"
"software OR system"
"neural network NOT optimization"

# Grouping with parentheses
'("deep learning" OR "neural network") AND ("performance" OR "optimization")'

# Complex nested queries
'("software" OR "system") AND "configuration" AND ("performance prediction" OR "performance modeling" OR "performance learning") AND ("deep learning" OR "neural network")'

# Quoted phrases for exact matches
'"federated learning" AND "edge computing"'
```

## 🐛 Troubleshooting

### Common Issues

1. **No results found**
   ```bash
   # Check if databases are configured
   python main_cli.py status

   # Test API keys
   python main_cli.py test-keys
   ```

2. **1Password SDK issues**
   ```bash
   # Check if SDK is installed
   python -c "from secure_key_manager import SecureKeyManager; print(SecureKeyManager.is_onepassword_available())"

   # Check service account token
   echo $OP_SERVICE_ACCOUNT_TOKEN

   # Get setup instructions
   python secure_key_manager.py literature-search 1password-info
   ```

3. **API key errors**
   ```bash
   # Reconfigure keys
   python main_cli.py setup-keys

   # Check key accessibility
   python main_cli.py test-keys
   ```

4. **Cache issues**
   ```bash
   # Clear cache
   python main_cli.py cache clean --days 0

   # Check cache stats
   python main_cli.py cache stats
   ```

5. **Permission errors**
   ```bash
   # Check file permissions
   ls -la lit_search_config.json
   ls -la ./lit_cache/
   ```

### 1Password Troubleshooting

**Common 1Password Issues:**

1. **SDK not installed**:
   ```bash
   pip install onepassword
   ```

2. **Service Account Token not set**:
   ```bash
   export OP_SERVICE_ACCOUNT_TOKEN="ops_your_token_here"
   ```

3. **Invalid secret reference**:
   - Ensure the format is correct: `op://vault/item/field`
   - Check that the item exists in 1Password
   - Verify service account has access to the vault

4. **Async/Sync issues**:
   ```python
   # Use async version for 1Password
   import asyncio
   from secure_key_manager import SecureKeyManager

   async def get_key():
       manager = SecureKeyManager("app")
       return await manager.get_key_async("service")

   # Or use sync wrapper (less efficient but simpler)
   key = manager.get_key("service")  # Runs async internally
   ```

### Debug Mode
```bash
# Enable verbose logging
python main_cli.py search "query" --verbose --log-level DEBUG
```

## 🔄 Migration Guide

### From CLI to SDK
If you were using the 1Password CLI method, migrate to the official SDK:

1. **Install SDK**:
   ```bash
   pip install onepassword
   ```

2. **Create Service Account** (instead of personal auth)
3. **Update references** (same `op://` format works)
4. **Set environment variable**:
   ```bash
   export OP_SERVICE_ACCOUNT_TOKEN="ops_token"  # instead of op signin
   ```

### Benefits of SDK over CLI:
- ✅ No need for interactive sign-in
- ✅ Better security with service accounts
- ✅ Automatic token refresh
- ✅ Better error handling
- ✅ Async support for better performance

## 🤝 Contributing

The modular architecture makes it easy to:

1. **Add new databases**: Inherit from `DatabaseAdapter`
2. **Add new storage methods**: Extend `SecureKeyManager`
3. **Add new export formats**: Extend export methods in `LiteratureSearchEngine`
4. **Add new CLI commands**: Extend `LiteratureSearchCLI`

### Adding a New Database Adapter

```python
from database_adapters import DatabaseAdapter
from database_models import Paper

class NewDatabaseAdapter(DatabaseAdapter):
    def get_database_name(self) -> str:
        return "newdb"

    def search(self, query: str, max_results: int = 100) -> List[Paper]:
        # Implement database-specific search logic
        pass
```

### Adding a New Storage Method

```python
class SecureKeyManager:
    def _store_new_method_reference(self, service: str, reference: str) -> bool:
        # Implement new storage method
        pass

    def _get_new_method_key(self, reference: str) -> Optional[str]:
        # Implement key retrieval
        pass
```

## 🔧 Advanced Configuration

### Custom 1Password Integration

```python
import asyncio
from onepassword.client import Client as OnePasswordClient
from secure_key_manager import SecureKeyManager

class CustomKeyManager(SecureKeyManager):
    async def _get_onepassword_client(self):
        # Custom 1Password client configuration
        token = os.getenv("CUSTOM_OP_TOKEN")
        return await OnePasswordClient.authenticate(
            auth=token,
            integration_name="My Custom Integration",
            integration_version="v2.0.0"
        )
```

### Custom Database Adapter with Async

```python
import asyncio
from database_adapters import DatabaseAdapter

class AsyncDatabaseAdapter(DatabaseAdapter):
    async def search_async(self, query: str, max_results: int = 100) -> List[Paper]:
        # Implement async search
        pass

    def search(self, query: str, max_results: int = 100) -> List[Paper]:
        # Wrapper for sync compatibility
        return asyncio.run(self.search_async(query, max_results))
```

## 📦 Dependencies

The system is designed with minimal dependencies:

### Core Dependencies (Built-in Python)
- `json` - Configuration and data serialization
- `sqlite3` - Local caching database
- `pathlib` - Cross-platform path handling
- `hashlib` - Cryptographic operations
- `asyncio` - Asynchronous operations
- `subprocess` - External command execution
- `urllib` - HTTP requests for APIs

### Optional Dependencies
- `onepassword` - Official 1Password SDK (recommended for 1Password integration)
  ```bash
  pip install onepassword
  ```

### Installation
```bash
# Core functionality (no external deps)
# Just download the Python files

# With 1Password support
pip install onepassword

# Development dependencies (optional)
pip install pytest pytest-asyncio  # For testing
```

## 📄 License

This project is designed for academic and research use. Please respect the terms of service of individual database APIs and ensure appropriate attribution in academic work.

### API Terms of Service
- **arXiv**: Free for non-commercial use
- **IEEE**: Requires subscription and API agreement
- **Springer**: Free tier available, commercial use requires license
- **Elsevier**: Requires subscription and API agreement
- **Wiley**: Requires subscription and API agreement

### 1Password Integration
- Requires 1Password account and service account
- Subject to 1Password's terms of service
- Service accounts may have usage limits

## 🆘 Support

For issues and questions:

1. **Check the troubleshooting section**
2. **Review the API documentation** of specific databases
3. **Enable debug logging** for detailed error information:
   ```bash
   python main_cli.py search "query" --verbose --log-level DEBUG
   ```
4. **Check 1Password setup** if using 1Password integration:
   ```bash
   python secure_key_manager.py literature-search 1password-info
   ```

### Getting Help

- **Configuration issues**: Use `python main_cli.py status`
- **API key problems**: Use `python main_cli.py test-keys`
- **Cache problems**: Use `python main_cli.py cache stats`
- **1Password issues**: Check service account token and secret references
- **Search problems**: Try with `--no-cache --verbose` flags

## ⚙️ Configuration

The system uses a JSON configuration file (`lit_search_config.json`):

```json
{
  "databases": {
    "arxiv": {
      "enabled": true,
      "rate_limit": 3,
      "categories": ["cs.LG", "cs.AI", "cs.CV"]
    },
    "ieee": {
      "enabled": true,
      "api_key_method": "encrypted",
      "rate_limit": 100
    }
  },
  "cache": {
    "directory": "./lit_cache",
    "expire_days": 30,
    "max_age_hours": 24
  },
  "search": {
    "max_results": 100,
    "deduplication": true,
    "min_year": 2018,
    "similarity_threshold": 0.8
  }
}
```

## 🔍 Advanced Features

### Deduplication
The system automatically identifies and removes duplicate papers using:
- DOI matching
- arXiv ID matching
- Title similarity analysis
- Author overlap detection

### Caching Strategy
- **Search Results**: Cached by query hash and database combination
- **Individual Papers**: Cached by canonical ID
- **Automatic Cleanup**: Configurable expiration
- **Statistics Tracking**: Comprehensive cache analytics

### Rate Limiting
- **Per-Database**: Configurable rate limits
- **Automatic Throttling**: Prevents API quota exhaustion
- **Statistics Tracking**: Monitor request patterns

### Export Formats
- **JSON**: Complete metadata with all fields
- **BibTeX**: Ready for LaTeX and reference managers
- **CSV**: Spreadsheet-compatible format

## 🐛 Troubleshooting

### Common Issues

1. **No results found**
   ```bash
   # Check if databases are configured
   python main_cli.py status

   # Test API keys
   python main_cli.py test-keys
   ```

2. **API key errors**
   ```bash
   # Reconfigure keys
   python main_cli.py setup-keys

   # Check key accessibility
   python main_cli.py test-keys
   ```

3. **Cache issues**
   ```bash
   # Clear cache
   python main_cli.py cache clean --days 0

   # Check cache stats
   python main_cli.py cache stats
   ```

4. **Permission errors**
   ```bash
   # Check file permissions
   ls -la lit_search_config.json
   ls -la ./lit_cache/
   ```

### Debug Mode
```bash
# Enable verbose logging
python main_cli.py search "query" --verbose --log-level DEBUG
```

## 🤝 Contributing

The modular architecture makes it easy to:

1. **Add new databases**: Inherit from `DatabaseAdapter`
2. **Add new storage methods**: Extend `SecureKeyManager`
3. **Add new export formats**: Extend export methods in `LiteratureSearchEngine`
4. **Add new CLI commands**: Extend `LiteratureSearchCLI`

### Adding a New Database Adapter

```python
from database_adapters import DatabaseAdapter
from database_models import Paper

class NewDatabaseAdapter(DatabaseAdapter):
    def get_database_name(self) -> str:
        return "newdb"

    def search(self, query: str, max_results: int = 100) -> List[Paper]:
        # Implement database-specific search logic
        pass
```

## 📄 License

This project is designed for academic and research use. Please respect the terms of service of individual database APIs and ensure appropriate attribution in academic work.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the API documentation of specific databases
3. Enable debug logging for detailed error information
