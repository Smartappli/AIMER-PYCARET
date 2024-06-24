import os

import anyio
from dotenv import load_dotenv
from hypercorn.config import Config
from hypercorn.asyncio import serve
from main import app

load_dotenv()


def load_config() -> Config:
    """
    Load the server configuration from environment variables.

    This function reads the 'BIND' environment variable to get the
    bind addresses for the server. If 'BIND' is not set, it defaults to '0.0.0.0:8000'.
    It returns a Config object with the bind addresses set.

    Returns:
        Config: A Hypercorn Config object with the bind addresses configured.
    """
    bind_addresses = os.getenv("BIND", "0.0.0.0:8000").split(",")
    config = Config()
    config.bind = bind_addresses
    return config


async def main():
    """
    Main entry point for the asynchronous server.

    This function loads the server configuration and starts the Hypercorn server
    using AnyIO's task group to manage the asynchronous tasks.
    """
    config = load_config()
    async with anyio.create_task_group() as task_group:
        await task_group.spawn(serve, app, config)


if __name__ == "__main__":
    anyio.run(main)
