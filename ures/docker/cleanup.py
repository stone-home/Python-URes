import docker
from typing import Optional


class DockerCleanup:
    """Remove dangling images and leftover containers from the local Docker engine.

    Examples:
        >>> from ures.docker import DockerCleanup
        >>> cleanup = DockerCleanup()
        >>> cleanup.dangling_images()
        >>> cleanup.stopped_containers()
    """

    def __init__(self, client: Optional[docker.DockerClient] = None):
        """Attach to a Docker client.

        Args:
            client (docker.DockerClient | None): A Docker SDK client. Defaults to
                ``docker.from_env()``.
        """
        self.client = client or docker.from_env()

    def dangling_images(self):
        """Delete dangling (untagged) local images."""
        client = self.client

        # List dangling images (dangling=true filter)
        dangling_images = client.images.list(filters={"dangling": True})

        if not dangling_images:
            print("No dangling images to remove.")
            return

        for image in dangling_images:
            try:
                print(f"Removing image: {image.id}")
                client.images.remove(image.id)
            except Exception as e:
                print(f"Error removing image {image.id}: {e}")

        print("Dangling images prune complete!")

    def stopped_containers(self):
        """Delete containers in ``exited`` or ``created`` state."""
        client = self.client

        stopped_containers = client.containers.list(
            all=True, filters={"status": "exited"}
        )
        just_created_containers = client.containers.list(
            all=True, filters={"status": "created"}
        )

        plan2remove_containers = stopped_containers + just_created_containers

        if not plan2remove_containers:
            print("No stopped xmem_container to prune.")
            return

        for container in plan2remove_containers:
            try:
                print(f"Removing container: {container.name} ({container.short_id})")
                container.remove()
            except Exception as e:
                print(f"Error removing container {container.name}: {e}")

        print("Pruning complete!")
