#!/usr/bin/env python3
"""
Secure Key Manager SDK
A standalone library for secure API key management supporting multiple storage methods.
Can be used across different projects and libraries.
"""

import json
import os
import base64
import subprocess
import secrets
import asyncio
import enum
from typing import Dict, Optional, Union
from pathlib import Path
from datetime import datetime
import logging

try:
    from onepassword.client import Client as OnePasswordClient

    ONEPASSWORD_SDK_AVAILABLE = True
except ImportError:
    OnePasswordClient = None
    ONEPASSWORD_SDK_AVAILABLE = False


class StorageMethod(enum.Enum):
    ENCRYPTED = "encrypted"
    ENV = "env"
    ONEPASSWORD = "1password"
    KEYCHAIN = "keychain"


class SecureKeyManager:
    """
    Secure API key management supporting multiple storage methods.
    This is a standalone SDK that can be used by any application.
    """

    def __init__(
        self, app_name: str = "default", config_dir: Optional[Union[str, Path]] = None
    ):
        """
        Initialize the SecureKeyManager.

        Args:
            app_name: Application name for namespacing keys
            config_dir: Custom directory for storing encrypted keys
        """
        self.app_name = app_name

        if config_dir:
            self.config_dir = Path(config_dir)
        else:
            # Use platform-appropriate config directory
            if os.name == "nt":  # Windows
                self.config_dir = (
                    Path.home() / "AppData" / "Local" / f"SecureKeys-{app_name}"
                )
            else:  # Unix-like
                self.config_dir = Path.home() / f".ures-secure-keys-{app_name}"

        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.key_file = self.config_dir / "encrypted_keys.json"
        self.logger = logging.getLogger(__name__)

        # Generate or load encryption key
        self.encryption_key = self._get_or_create_encryption_key()

        # 1Password client cache
        self._onepassword_client = None

    def _get_or_create_encryption_key(self) -> bytes:
        """Get or create local encryption key."""
        key_path = self.config_dir / ".encryption_key"

        if key_path.exists():
            try:
                with open(key_path, "rb") as f:
                    return f.read()
            except Exception as e:
                self.logger.warning(
                    f"Failed to read existing key, generating new one: {e}"
                )

        # Generate new 256-bit key
        key = secrets.token_bytes(32)
        try:
            with open(key_path, "wb") as f:
                f.write(key)
            # Set restrictive permissions (Unix-like systems)
            if os.name != "nt":
                os.chmod(key_path, 0o600)
        except Exception as e:
            self.logger.error(f"Failed to save encryption key: {e}")

        return key

    def _encrypt_value(self, value: str) -> str:
        """Encrypt a value using XOR with random key."""
        if not value:
            return ""

        try:
            # Use XOR encryption with the key
            key_bytes = self.encryption_key
            value_bytes = value.encode("utf-8")
            encrypted = bytearray()

            for i, byte in enumerate(value_bytes):
                encrypted.append(byte ^ key_bytes[i % len(key_bytes)])

            return base64.b64encode(encrypted).decode("utf-8")
        except Exception as e:
            self.logger.error(f"Encryption failed: {e}")
            return ""

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
            self.logger.error(f"Decryption failed: {e}")
            return ""

    async def _get_onepassword_client(self) -> Optional[OnePasswordClient]:
        """Get or create 1Password client."""
        if not ONEPASSWORD_SDK_AVAILABLE:
            self.logger.error(
                "1Password SDK not available. Install with: pip install onepassword"
            )
            return None

        if self._onepassword_client is not None:
            return self._onepassword_client

        try:
            # Get service account token from environment
            token = os.getenv("OP_SERVICE_ACCOUNT_TOKEN")
            if not token:
                self.logger.error(
                    "OP_SERVICE_ACCOUNT_TOKEN environment variable not set"
                )
                return None

            # Create and authenticate client
            self._onepassword_client = await OnePasswordClient.authenticate(
                auth=token,
                integration_name=f"{self.app_name} Key Manager",
                integration_version="v1.0.0",
            )

            return self._onepassword_client

        except Exception as e:
            self.logger.error(f"Failed to authenticate with 1Password: {e}")
            return None

    def store_key(
        self,
        service: str,
        api_key: str,
        method: StorageMethod = StorageMethod.ENCRYPTED,
    ) -> bool:
        """
        Store API key using specified method.

        Args:
            service: Service name (e.g., 'ieee', 'springer')
            api_key: The API key to store or reference
            method: Storage method ('encrypted', 'env', '1password', 'keychain')

        Returns:
            bool: Success status
        """
        try:
            if method == StorageMethod.ENCRYPTED:
                return self._store_encrypted_key(service, api_key)
            elif method == StorageMethod.ENV:
                return self._store_env_reference(service, api_key)
            elif method == StorageMethod.ONEPASSWORD:
                return self._store_1password_reference(service, api_key)
            elif method == StorageMethod.KEYCHAIN:
                return self._store_keychain_reference(service, api_key)
            else:
                raise ValueError(f"Unsupported storage method: {method}")
        except Exception as e:
            self.logger.error(f"Failed to store key for {service}: {e}")
            return False

    def _store_encrypted_key(self, service: str, api_key: str) -> bool:
        """Store encrypted API key locally."""
        try:
            encrypted_keys = self._load_encrypted_keys()

            encrypted_keys[service] = {
                "method": StorageMethod.ENCRYPTED.value,
                "value": self._encrypt_value(api_key),
                "created_at": datetime.now().isoformat(),
                "app_name": self.app_name,
            }

            return self._save_encrypted_keys(encrypted_keys)
        except Exception as e:
            self.logger.error(f"Failed to store encrypted key for {service}: {e}")
            return False

    def _store_env_reference(self, service: str, env_var_name: str) -> bool:
        """Store reference to environment variable."""
        try:
            encrypted_keys = self._load_encrypted_keys()

            encrypted_keys[service] = {
                "method": StorageMethod.ENV.value,
                "env_var": env_var_name,
                "created_at": datetime.now().isoformat(),
                "app_name": self.app_name,
            }

            return self._save_encrypted_keys(encrypted_keys)
        except Exception as e:
            self.logger.error(f"Failed to store env reference for {service}: {e}")
            return False

    def _store_1password_reference(self, service: str, op_reference: str) -> bool:
        """Store 1Password SDK reference."""
        try:
            encrypted_keys = self._load_encrypted_keys()

            encrypted_keys[service] = {
                "method": StorageMethod.ONEPASSWORD.value,
                "op_reference": op_reference,
                "created_at": datetime.now().isoformat(),
                "app_name": self.app_name,
            }

            return self._save_encrypted_keys(encrypted_keys)
        except Exception as e:
            self.logger.error(f"Failed to store 1Password reference for {service}: {e}")
            return False

    def _store_keychain_reference(self, service: str, keychain_service: str) -> bool:
        """Store macOS Keychain reference."""
        try:
            encrypted_keys = self._load_encrypted_keys()

            encrypted_keys[service] = {
                "method": StorageMethod.KEYCHAIN.value,
                "keychain_service": keychain_service,
                "created_at": datetime.now().isoformat(),
                "app_name": self.app_name,
            }

            return self._save_encrypted_keys(encrypted_keys)
        except Exception as e:
            self.logger.error(f"Failed to store keychain reference for {service}: {e}")
            return False

    def get_key(self, service: str) -> Optional[str]:
        """
        Retrieve API key using stored method (synchronous wrapper).

        Args:
            service: Service name

        Returns:
            str or None: The API key if found and accessible
        """
        # For 1Password, we need to handle async
        encrypted_keys = self._load_encrypted_keys()

        if service not in encrypted_keys:
            return None

        key_info = encrypted_keys[service]
        method = key_info.get("method", StorageMethod.ENCRYPTED.value)

        if method == StorageMethod.ONEPASSWORD.value:
            # Run async method in event loop
            try:
                return asyncio.run(
                    self._get_1password_key_async(key_info.get("op_reference", ""))
                )
            except Exception as e:
                self.logger.error(f"Failed to get 1Password key synchronously: {e}")
                return None
        else:
            # Handle other methods synchronously
            return self._get_key_sync(service)

    def _get_key_sync(self, service: str) -> Optional[str]:
        """Get key using synchronous methods."""
        try:
            encrypted_keys = self._load_encrypted_keys()

            if service not in encrypted_keys:
                return None

            key_info = encrypted_keys[service]
            method = key_info.get("method", StorageMethod.ENCRYPTED.value)

            if method == StorageMethod.ENCRYPTED.value:
                return self._decrypt_value(key_info.get("value", ""))
            elif method == StorageMethod.ENV.value:
                env_var = key_info.get("env_var", "")
                return os.getenv(env_var)
            elif method == StorageMethod.KEYCHAIN.value:
                return self._get_keychain_key(key_info.get("keychain_service", ""))

        except Exception as e:
            self.logger.error(f"Failed to retrieve key for {service}: {e}")

        return None

    async def get_key_async(self, service: str) -> Optional[str]:
        """
        Retrieve API key using stored method (async version).

        Args:
            service: Service name

        Returns:
            str or None: The API key if found and accessible
        """
        try:
            encrypted_keys = self._load_encrypted_keys()

            if service not in encrypted_keys:
                return None

            key_info = encrypted_keys[service]
            method = key_info.get("method", StorageMethod.ENCRYPTED.value)

            if method == StorageMethod.ENCRYPTED.value:
                return self._decrypt_value(key_info.get("value", ""))
            elif method == StorageMethod.ENV.value:
                env_var = key_info.get("env_var", "")
                return os.getenv(env_var)
            elif method == StorageMethod.ONEPASSWORD.value:
                return await self._get_1password_key_async(
                    key_info.get("op_reference", "")
                )
            elif method == StorageMethod.KEYCHAIN.value:
                return self._get_keychain_key(key_info.get("keychain_service", ""))

        except Exception as e:
            self.logger.error(f"Failed to retrieve key for {service}: {e}")

        return None

    async def _get_1password_key_async(self, op_reference: str) -> Optional[str]:
        """Retrieve API key from 1Password using official SDK."""
        if not op_reference:
            return None

        if not ONEPASSWORD_SDK_AVAILABLE:
            self.logger.error("1Password SDK not available")
            return None

        try:
            client = await self._get_onepassword_client()
            if not client:
                return None

            # Use the official SDK to resolve the secret
            value = await client.secrets.resolve(op_reference)
            return value

        except Exception as e:
            self.logger.error(f"1Password SDK error: {e}")
            return None

    def _get_1password_key_cli_fallback(self, op_reference: str) -> Optional[str]:
        """Fallback to CLI method if SDK is not available."""
        if not op_reference:
            return None

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

        return None

    def _get_keychain_key(self, service_name: str) -> Optional[str]:
        """Retrieve API key from macOS Keychain."""
        if not service_name or os.name == "nt":  # Skip on Windows
            return None

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
                self.logger.debug(f"Keychain entry not found for {service_name}")
        except subprocess.TimeoutExpired:
            self.logger.error("Keychain access timeout")
        except FileNotFoundError:
            self.logger.debug("Security command not found (not on macOS)")
        except Exception as e:
            self.logger.error(f"Keychain error: {e}")

        return None

    def list_keys(self) -> Dict[str, Dict]:
        """
        List all stored API key references without revealing actual keys.

        Returns:
            dict: Service information including method and accessibility
        """
        try:
            encrypted_keys = self._load_encrypted_keys()

            display_keys = {}
            for service, info in encrypted_keys.items():
                # Only show keys for this app
                if info.get("app_name") != self.app_name:
                    continue

                display_info = {
                    "method": info.get("method", StorageMethod.ENCRYPTED.value),
                    "created_at": info.get("created_at", ""),
                    "has_key": bool(self.get_key(service)),
                }

                method = info.get("method", StorageMethod.ENCRYPTED.value)
                if method == StorageMethod.ENV.value:
                    display_info["env_var"] = info.get("env_var", "")
                elif method == StorageMethod.ONEPASSWORD.value:
                    display_info["op_reference"] = info.get("op_reference", "")
                elif method == StorageMethod.KEYCHAIN.value:
                    display_info["keychain_service"] = info.get("keychain_service", "")

                display_keys[service] = display_info

            return display_keys

        except Exception as e:
            self.logger.error(f"Failed to list keys: {e}")
            return {}

    def delete_key(self, service: str) -> bool:
        """
        Delete stored API key reference.

        Args:
            service: Service name

        Returns:
            bool: Success status
        """
        try:
            encrypted_keys = self._load_encrypted_keys()

            if service in encrypted_keys:
                del encrypted_keys[service]
                success = self._save_encrypted_keys(encrypted_keys)
                if success:
                    self.logger.info(f"Deleted key for {service}")
                return success

            return True  # Already deleted

        except Exception as e:
            self.logger.error(f"Failed to delete key for {service}: {e}")
            return False

    def test_key_access(self, service: str) -> Dict[str, any]:
        """
        Test if a key is accessible and return status info.

        Args:
            service: Service name

        Returns:
            dict: Status information
        """
        result = {
            "service": service,
            "exists": False,
            "accessible": False,
            "method": None,
            "error": None,
        }

        try:
            encrypted_keys = self._load_encrypted_keys()

            if service not in encrypted_keys:
                result["error"] = "Key not configured"
                return result

            result["exists"] = True
            result["method"] = encrypted_keys[service].get(
                "method", StorageMethod.ENCRYPTED.value
            )

            key = self.get_key(service)
            if key:
                result["accessible"] = True
                result["key_length"] = len(key)
            else:
                result["error"] = "Key exists but not accessible"

        except Exception as e:
            result["error"] = str(e)

        return result

    def _load_encrypted_keys(self) -> Dict:
        """Load encrypted keys from file."""
        if not self.key_file.exists():
            return {}

        try:
            with open(self.key_file, "r") as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load encrypted keys: {e}")
            return {}

    def _save_encrypted_keys(self, encrypted_keys: Dict) -> bool:
        """Save encrypted keys to file."""
        try:
            with open(self.key_file, "w") as f:
                json.dump(encrypted_keys, f, indent=2)

            # Set restrictive permissions
            if os.name != "nt":
                os.chmod(self.key_file, 0o600)

            return True
        except Exception as e:
            self.logger.error(f"Failed to save encrypted keys: {e}")
            return False

    def export_config(self, include_encrypted: bool = False) -> Dict:
        """
        Export configuration for backup or migration.

        Args:
            include_encrypted: Whether to include encrypted values (use with caution)

        Returns:
            dict: Configuration data
        """
        try:
            encrypted_keys = self._load_encrypted_keys()

            if not include_encrypted:
                # Remove encrypted values for safety
                safe_keys = {}
                for service, info in encrypted_keys.items():
                    if info.get("app_name") == self.app_name:
                        safe_info = {k: v for k, v in info.items() if k != "value"}
                        safe_keys[service] = safe_info
                return safe_keys

            return {
                k: v
                for k, v in encrypted_keys.items()
                if v.get("app_name") == self.app_name
            }

        except Exception as e:
            self.logger.error(f"Failed to export config: {e}")
            return {}

    def import_config(self, config_data: Dict, overwrite: bool = False) -> bool:
        """
        Import configuration from backup.

        Args:
            config_data: Configuration data
            overwrite: Whether to overwrite existing keys

        Returns:
            bool: Success status
        """
        try:
            encrypted_keys = self._load_encrypted_keys()

            for service, info in config_data.items():
                if not overwrite and service in encrypted_keys:
                    continue

                # Ensure app_name is set
                info["app_name"] = self.app_name
                encrypted_keys[service] = info

            return self._save_encrypted_keys(encrypted_keys)

        except Exception as e:
            self.logger.error(f"Failed to import config: {e}")
            return False

    @staticmethod
    def is_onepassword_available() -> bool:
        """Check if 1Password SDK is available."""
        return ONEPASSWORD_SDK_AVAILABLE

    @staticmethod
    def get_onepassword_setup_instructions() -> str:
        """Get setup instructions for 1Password integration."""
        return """
    1Password Setup Instructions:

    1. Install the 1Password SDK:
       pip install onepassword

    2. Create a Service Account:
       - Go to your 1Password account settings
       - Create a new Service Account
       - Note the token (starts with 'ops_')

    3. Set Environment Variable:
       export OP_SERVICE_ACCOUNT_TOKEN="ops_your_token_here"

    4. Store your API keys in 1Password:
       - Create items for each API key
       - Note the secret references (e.g., "op://vault/item/field")

    5. Configure the key manager:
       manager.store_key('ieee', 'op://Private/IEEE-API/credential', '1password')
        """


if __name__ == "__main__":
    sm = SecureKeyManager(app_name="testapp")
    sm.store_key(
        "example_service",
        "op://DevOps/Ansible-PW for Vault Secrets/password",
        method=StorageMethod.ONEPASSWORD,
    )
    print(sm.get_key("example_service"))
