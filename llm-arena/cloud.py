"""Ollama connection setup and actionable startup diagnostics."""

from __future__ import annotations

import os

import httpx
import ollama

CLOUD_HOST = "https://ollama.com"


class CloudError(RuntimeError):
    """An Ollama connection or model configuration needs attention."""


def make_client(host: str) -> ollama.AsyncClient:
    host = host.rstrip("/")
    key = os.environ.get("OLLAMA_API_KEY", "").strip()
    if host == CLOUD_HOST and not key:
        raise CloudError(
            "OLLAMA_API_KEY is not set. Run setup.ps1 to enter a key from "
            "https://ollama.com/settings/keys, then use run.ps1."
        )
    headers = {"Authorization": f"Bearer {key}"} if key else None
    return ollama.AsyncClient(host=host, headers=headers, timeout=180)


def connection_error(error: Exception) -> CloudError:
    if isinstance(error, ollama.ResponseError):
        status = error.status_code
        if status in (401, 403):
            return CloudError("Ollama rejected authentication. Check your key or run setup.ps1 -ReplaceApiKey.")
        if status == 429:
            return CloudError("Ollama usage limit reached. Check https://ollama.com/settings/usage or retry later.")
        if status == 404:
            return CloudError("Ollama model or endpoint was not found. Run --list-models and check --host.")
        return CloudError(f"Ollama request failed (HTTP {status}). Check model availability and retry.")
    return CloudError("Cannot reach Ollama. Check --host, network access, and whether a local server is running.")


async def list_models(client: ollama.AsyncClient) -> list[str]:
    try:
        return [model.model for model in (await client.list()).models if model.model]
    except (ollama.ResponseError, httpx.RequestError, ConnectionError) as error:
        raise connection_error(error) from error


async def check_models(client: ollama.AsyncClient, models: list[str]) -> None:
    available = await list_models(client)
    missing = [model for model in dict.fromkeys(models) if model not in available]
    if missing:
        raise CloudError(f"Models unavailable: {', '.join(missing)}. Choose names from --list-models.")


async def check_inference(client: ollama.AsyncClient, models: list[str]) -> None:
    """Check authenticated inference and both sides of the tool-calling protocol."""
    tool = {"type": "function", "function": {
        "name": "cloud_probe", "description": "Check that the arena connection is ready.",
        "parameters": {"type": "object", "properties": {}},
    }}
    for model in dict.fromkeys(models):
        messages = [{"role": "user", "content": "Call cloud_probe with no arguments to check the connection."}]
        try:
            response = await client.chat(model=model, messages=messages, tools=[tool], think=False)
            calls = response.message.tool_calls or []
            if not calls or any(call.function.name != "cloud_probe" for call in calls):
                raise CloudError(f"{model} did not call the test tool. Choose a model supporting tool calling.")
            messages.append(response.message)
            messages.extend({"role": "tool", "tool_name": "cloud_probe", "content": "ready"} for _ in calls)
            messages.append({"role": "user", "content": "The check is complete. Reply briefly without using tools."})
            reply = await client.chat(model=model, messages=messages, tools=[tool], think=False)
            if not reply.message.content or reply.message.tool_calls:
                raise CloudError(f"{model} did not complete the test tool round trip. Retry or choose another model.")
        except (ollama.ResponseError, httpx.RequestError, ConnectionError) as error:
            raise connection_error(error) from error
        print(f"[OK] {model}: authenticated inference and tool round trip", flush=True)
