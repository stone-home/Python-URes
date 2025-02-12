if __name__ == "__main__":
    import docker
    import os

    print(os.getlogin())
    client = docker.from_env()

    image = client.images.get("hello-worl")
