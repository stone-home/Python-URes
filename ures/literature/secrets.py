import os
from ures.secrets import SecureKeyManager, StorageMethod


class KeyManagerCLI:
    """Command-line interface for the SecureKeyManager."""

    def __init__(self, app_name: str = "default"):
        self.manager = SecureKeyManager(app_name)
        self.app_name = app_name

    def interactive_setup(self, services: list = None):
        """Interactive setup for API keys."""
        if not services:
            services = ["ieee", "springer", "elsevier", "wiley", "acm"]

        print(f"🔐 Secure Key Manager Setup ({self.app_name})")
        print("=" * 60)

        for service in services:
            print(f"\n📚 Setting up {service.upper()} API Key")
            print("-" * 40)

            choice = input(f"Configure {service}? (y/n): ").lower().strip()
            if choice != "y":
                continue

            print("\nChoose storage method:")
            print("1. Encrypted locally (recommended)")
            print("2. Environment variable")
            print(
                "3. 1Password SDK"
                + (
                    " ✅"
                    if SecureKeyManager.is_onepassword_available()
                    else " ❌ Not installed"
                )
            )
            print("4. macOS Keychain")

            method_choice = input("Enter choice (1-4): ").strip()

            if method_choice == "1":
                api_key = input(f"Enter {service} API key: ").strip()
                if api_key:
                    if self.manager.store_key(
                        service, api_key, StorageMethod.ENCRYPTED
                    ):
                        print(f"✅ {service} API key stored securely (encrypted)")
                    else:
                        print(f"❌ Failed to store {service} API key")

            elif method_choice == "2":
                env_var = input(
                    f"Enter environment variable name (e.g., {service.upper()}_API_KEY): "
                ).strip()
                if env_var:
                    if self.manager.store_key(service, env_var, StorageMethod.ENV):
                        print(f"✅ {service} configured to use ${env_var}")
                        print(f"   Remember to set: export {env_var}=your_api_key")
                    else:
                        print(f"❌ Failed to configure {service}")

            elif method_choice == "3":
                if not SecureKeyManager.is_onepassword_available():
                    print("❌ 1Password SDK not installed.")
                    print("Install with: pip install onepassword")
                    print(SecureKeyManager.get_onepassword_setup_instructions())
                    continue

                if not os.getenv("OP_SERVICE_ACCOUNT_TOKEN"):
                    print("❌ OP_SERVICE_ACCOUNT_TOKEN environment variable not set.")
                    print(SecureKeyManager.get_onepassword_setup_instructions())
                    continue

                op_ref = input(
                    f"Enter 1Password reference (e.g., op://Private/{service.upper()}-API/credential): "
                ).strip()
                if op_ref:
                    if self.manager.store_key(
                        service, op_ref, StorageMethod.ONEPASSWORD
                    ):
                        print(f"✅ {service} configured to use 1Password SDK")
                        print(f"   Reference: {op_ref}")
                    else:
                        print(f"❌ Failed to configure {service}")

            elif method_choice == "4":
                service_name = input(
                    f"Enter Keychain service name (e.g., {service}-api-key): "
                ).strip()
                if service_name:
                    if self.manager.store_key(
                        service, service_name, StorageMethod.KEYCHAIN
                    ):
                        print(f"✅ {service} configured to use Keychain")
                        print(f"   Service: {service_name}")
                        print(
                            f"   Add to keychain: security add-generic-password -s {service_name} -a {service} -w your_api_key"
                        )
                    else:
                        print(f"❌ Failed to configure {service}")
            else:
                print("Invalid choice, skipping...")

        print(f"\n🎉 Setup complete!")
        self.show_status()

    def show_status(self):
        """Show current API key configuration status."""
        print(f"\n📊 API Key Status ({self.app_name})")
        print("=" * 50)

        stored_keys = self.manager.list_keys()

        if not stored_keys:
            print("No API keys configured.")
            return

        for service, info in stored_keys.items():
            status = "✅ Configured" if info["has_key"] else "❌ Key not accessible"
            method = info["method"]
            print(f"{service:15}: {status:20} ({method})")

            if method == "env" and "env_var" in info:
                env_val = "SET" if os.getenv(info["env_var"]) else "NOT SET"
                print(f"                 Environment: ${info['env_var']} = {env_val}")
            elif method == "1password" and "op_reference" in info:
                print(f"                 1Password: {info['op_reference']}")
            elif method == "keychain" and "keychain_service" in info:
                print(f"                 Keychain: {info['keychain_service']}")

    def test_keys(self):
        """Test API key accessibility."""
        print(f"\n🧪 Testing API Key Access ({self.app_name})")
        print("=" * 50)

        stored_keys = self.manager.list_keys()

        for service in stored_keys.keys():
            test_result = self.manager.test_key_access(service)

            if test_result["accessible"]:
                key_length = test_result.get("key_length", 0)
                masked_key = "*" * min(key_length, 8) + f"({key_length} chars)"
                print(f"{service:15}: ✅ Accessible {masked_key}")
            else:
                error = test_result.get("error", "Unknown error")
                print(f"{service:15}: ❌ {error}")


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) > 1:
        app_name = sys.argv[1]
    else:
        app_name = "literature-search"

    cli = KeyManagerCLI(app_name)

    if len(sys.argv) > 2:
        command = sys.argv[2]
        if command == "setup":
            cli.interactive_setup()
        elif command == "status":
            cli.show_status()
        elif command == "test":
            cli.test_keys()
        elif command == "1password-info":
            print(SecureKeyManager.get_onepassword_setup_instructions())
        else:
            print("Available commands: setup, status, test, 1password-info")
    else:
        cli.interactive_setup()
