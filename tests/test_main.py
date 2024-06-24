import os
import pytest
import anyio
from unittest.mock import patch, MagicMock
from hypercorn.config import Config
from main import load_config, main


def test_load_config_default():
    # Test with default environment variable
    config = load_config()
    assert config.bind == ["0.0.0.0:8000"]


@patch.dict(os.environ, {"BIND": "127.0.0.1:5000,0.0.0.0:5001"})
def test_load_config_custom():
    # Test with custom environment variable
    config = load_config()
    assert config.bind == ["127.0.0.1:5000", "0.0.0.0:5001"]


@pytest.mark.anyio
@patch("main.serve", new_callable=MagicMock)
@patch("main.load_config", return_value=Config())
async def test_main(mock_load_config, mock_serve):
    # Test the main function
    async def dummy_serve(app, config):
        pass

    mock_serve.side_effect = dummy_serve

    async with anyio.create_task_group() as task_group:
        await task_group.spawn(main)

    mock_load_config.assert_called_once()
    mock_serve.assert_called_once()
