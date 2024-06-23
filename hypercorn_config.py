import anyio
import os
from dotenv import load_dotenv
from hypercorn.config import Config
from hypercorn.asyncio import serve
from main import (
    app,
)

load_dotenv()


def load_config() -> Config:
    bind_addresses = os.getenv("BIND", "0.0.0.0:8000").split(",")
    config = Config()
    config.bind = bind_addresses
    return config


async def main():
    config = load_config()
    async with anyio.create_task_group() as task_group:
        await task_group.spawn(serve, app, config)


if __name__ == "__main__":
    anyio.run(main)
