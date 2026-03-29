import os
import requests
import logging

def get_logger(name: str):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

logger = get_logger("pulseforge_api")

def execute_ollama_request(model: str, system_prompt: str, user_prompt: str) -> requests.Response:
    """
    Executes a secure tunneling request to the local device's Ollama instance,
    bypassing LocalTunnel's anti-bot browser warning screens.
    """
    url = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
    payload = {
        "model": model,
        "system": system_prompt,
        "prompt": user_prompt,
        "stream": False
    }
    headers = {
        "Content-Type": "application/json",
        "Bypass-Tunnel-Reminder": "true",
        "ngrok-skip-browser-warning": "true"
    }

    try:
        logger.info(f"Dispatching inference to Edgenode ({model}) via {url}")
        return requests.post(url, json=payload, headers=headers, timeout=300)
    except requests.exceptions.RequestException as e:
        logger.error(f"Edgenode Inference Failed: {str(e)}")
        raise e
