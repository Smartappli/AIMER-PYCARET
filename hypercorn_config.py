import anyio
from hypercorn.config import Config
from hypercorn.asyncio import serve
from main import app  # Assurez-vous que le chemin est correct selon votre structure de projet

config = Config()
config.bind = ["0.0.0.0:8000"]

async def main():
    async with anyio.create_task_group() as task_group:
        await task_group.spawn(serve, app, config)

if __name__ == "__main__":
    anyio.run(main)
