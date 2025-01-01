import argparse
from pathlib import Path
from typing import Dict, List
from ures.markdown import MarkdownDocument
from ures.timedate import time_now


class ReleasePostMetadata:
    """Represents metadata for a release post."""

    def __init__(
        self,
        project_name: str,
        release_version: str,
        release_url: str,
        is_draft: bool = False,
        post_tags: List[str] = None,
        post_categories: List[str] = None,
        sidebar_weight: int = 10,
    ):
        """Initialize release post metadata.

        Args:
            project_name: Name of the project
            release_version: Version string for the release
            release_url: URL to the release
            is_draft: Whether this is a draft post
            post_tags: List of tags for the post
            post_categories: List of categories for the post
            sidebar_weight: Weight for sidebar ordering
        """
        self.project_name = project_name
        self.release_version = release_version
        self.release_url = release_url
        self.is_draft = is_draft
        self.post_tags = post_tags or ["project", "release"]
        self.post_categories = post_categories or ["repository"]
        self.sidebar_weight = sidebar_weight

    def create_sidebar_config(self) -> Dict:
        """Generate sidebar navigation configuration for the release post."""
        return {
            "sidebar": {
                "name": self.release_version,
                "identifier": ReleasePost.create_version_identifier(
                    self.project_name, self.release_version
                ),
                "parent": ReleasePost.create_project_identifier(self.project_name),
                "weight": self.sidebar_weight,
            }
        }

    def to_dict(self) -> Dict:
        """Convert metadata to dictionary format for document creation."""
        return {
            "date": time_now(iso8601=False, format="%Y-%m-%dT%H:%M:%SZ"),
            "draft": self.is_draft,
            "title": self.release_version,
            "description": f"Release notes for {self.project_name} version {self.release_version}",
            "menu": self.create_sidebar_config(),
            "tags": self.post_tags,
            "categories": self.post_categories,
            "link": self.release_url,
        }


class ReleasePost(MarkdownDocument):
    """Manages creation and storage of project release posts."""

    def __init__(self, metadata: ReleasePostMetadata, release_notes: str):
        """Initialize a release post.

        Args:
            metadata: Configuration and metadata for the post
            release_notes: Content of the release notes
        """
        self._metadata_obj = metadata  # Store the original metadata object
        # Convert metadata to dictionary before passing to parent class
        metadata_dict = metadata.to_dict()
        super().__init__(content=release_notes, metadata=metadata_dict)
        self.add_content(
            f"Read more at [{self._metadata_obj.project_name}]({self._metadata_obj.release_url})",
            append=True,
        )

    @staticmethod
    def create_project_identifier(project_name: str) -> str:
        """Create a unique identifier for the project section.

        Args:
            project_name: Name of the project

        Returns:
            Project section identifier
        """
        return f"project-{project_name}"

    @staticmethod
    def create_version_identifier(project_name: str, version: str) -> str:
        """Create a unique identifier for the release version.

        Args:
            project_name: Name of the project
            version: Version string

        Returns:
            Version-specific identifier
        """
        return f"{project_name}-{version}"

    @staticmethod
    def get_content_root(base_dir: str) -> Path:
        """Get the root directory for all release posts.

        Args:
            base_dir: Site content base directory

        Returns:
            Root directory path for release posts
        """
        return Path(base_dir).joinpath("content/posts/project")

    def generate_post_path(self, base_dir: str) -> Path:
        """Generate the full file path for the release post.

        Args:
            base_dir: Site content base directory

        Returns:
            Complete file path for the release post
        """
        content_root = self.get_content_root(base_dir)
        return content_root.joinpath(
            f"{self._metadata_obj.project_name}/{self._metadata_obj.release_version}/index.md"
        )

    def create_project_landing_page(self, base_dir: str) -> None:
        """Create or update the project's main landing page.

        Args:
            base_dir: Site content base directory
        """
        project_name = self._metadata_obj.project_name
        display_name = project_name[0].upper() + project_name[1:]

        landing_page_metadata = {
            "title": display_name,
            "menu": {
                "sidebar": {
                    "name": display_name,
                    "identifier": self.create_project_identifier(project_name),
                    "parent": "projects",
                    "weight": self._metadata_obj.sidebar_weight,
                }
            },
        }

        landing_page = MarkdownDocument(metadata=landing_page_metadata)
        landing_page_path = self.get_content_root(base_dir).joinpath(
            f"{project_name}/_index.md"
        )

        landing_page_path.parent.mkdir(parents=True, exist_ok=True)
        landing_page.save(landing_page_path)

    def save_to_disk(self, base_dir: str) -> None:
        """Save the release post and ensure all necessary directories exist.

        Args:
            base_dir: Site content base directory
        """
        post_path = self.generate_post_path(base_dir)

        if not post_path.parent.parent.exists():
            self.create_project_landing_page(base_dir)

        post_path.parent.mkdir(parents=True, exist_ok=True)
        super().save(post_path)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Generate a release post from release notes."
    )

    parser.add_argument("--project", "-p", required=True, help="Name of the project")

    parser.add_argument("--version", "-v", required=True, help="Release version")

    parser.add_argument("--url", "-u", required=True, help="URL to the release")

    parser.add_argument(
        "--notes",
        "-n",
        required=True,
        help="Note Content",
    )

    parser.add_argument(
        "--output-dir", "-o", required=True, help="Base directory for output files"
    )

    parser.add_argument(
        "--draft",
        "-d",
        action="store_true",
        help="Mark post as draft",
    )

    parser.add_argument("--tags", "-t", nargs="+", help="Additional tags for the post")

    parser.add_argument(
        "--categories", "-c", nargs="+", help="Additional categories for the post"
    )

    parser.add_argument(
        "--weight", "-w", type=int, default=10, help="Sidebar weight (default: 10)"
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point for the script."""
    args = parse_args()

    # Read release notes content
    release_notes = args.notes

    # Create metadata object
    metadata = ReleasePostMetadata(
        project_name=args.project,
        release_version=args.version,
        release_url=args.url,
        is_draft=args.draft,
        post_tags=args.tags,
        post_categories=args.categories,
        sidebar_weight=args.weight,
    )

    # Create and save the release post
    post = ReleasePost(metadata=metadata, release_notes=release_notes)
    post.save_to_disk(args.output_dir)
    print(
        f"Release post created successfully at: {post.generate_post_path(args.output_dir)}"
    )


if __name__ == "__main__":
    main()
