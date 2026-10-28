"""Azure AI Foundry client: many models, structured JSON output, and a disk cache.

Models come from ``data/models.yaml`` (id -> deployment, endpoint, API style, price).
Two API styles are supported:

* ``responses`` - Responses API with a strict JSON schema (OpenAI models). The reply is
  guaranteed to match the schema.
* ``chat_json`` - Chat Completions with ``json_object`` (Grok, Mistral on Foundry). The
  schema is put in the prompt, the reply is validated with pydantic, a stray single
  wrapper key is unwrapped, and validation errors are fed back on retry.

Every successful call is cached under ``results/cache/llm`` keyed by a hash of
(deployment, instructions, input, schema, temperature), so reruns are free and
reproducible. Delete the folder to force fresh calls. With ``LLM_CACHE_ONLY=1``
(``standup-exp run --cache-only``) nothing is sent: a missing reply is a failed call.

Configuration (``.env``):
    FOUNDRY_ENDPOINT        base URL of the default resource (…/openai/v1)
    BENCH_OPENAI_ENDPOINT   extra endpoints referenced by models.yaml
    BENCH_MODELS_ENDPOINT
    FOUNDRY_GEN_MODEL       optional model id used for transcript generation
    FOUNDRY_API_KEY         key for FOUNDRY_ENDPOINT; BENCH_API_KEY for BENCH_* endpoints.
                            Entra ID (Azure CLI / DefaultAzureCredential) is used when absent.
    PRICE_INPUT_PER_M / PRICE_OUTPUT_PER_M   fallback prices for models with price: null
"""

from __future__ import annotations

import copy
import contextvars
import hashlib
import json
import time
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from typing import TypeVar

import yaml
from pydantic import BaseModel, ValidationError

from .paths import CACHE_DIR, DATA_DIR, env, flag

T = TypeVar("T", bound=BaseModel)
MODELS_PATH = DATA_DIR / "models.yaml"
MAX_OUTPUT_TOKENS = 16000


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    cached: bool = False

    def __iadd__(self, other: "Usage") -> "Usage":
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.seconds += other.seconds
        return self


# ------------------------------------------------------------------ catalogue
@dataclass(frozen=True)
class ModelSpec:
    id: str
    deployment: str
    endpoint: str  # env var name
    api: str = "responses"
    price: tuple[float, float] | None = None
    timeout: float = 240.0


@lru_cache(maxsize=1)
def _catalogue() -> tuple[str, dict[str, ModelSpec]]:
    raw = yaml.safe_load(MODELS_PATH.read_text(encoding="utf-8"))
    specs = {}
    for m in raw["models"]:
        price = tuple(m["price"]) if m.get("price") else None
        specs[m["id"]] = ModelSpec(m["id"], m["deployment"], m["endpoint"], m.get("api", "responses"), price,
                                   float(m.get("timeout", 240)))
    return raw["default"], specs


def model_ids() -> list[str]:
    return list(_catalogue()[1])


def default_model() -> str:
    return _catalogue()[0]


def get_model(model_id: str | None = None) -> ModelSpec:
    """Resolve a model id; unknown ids are treated as a deployment on FOUNDRY_ENDPOINT."""
    default, specs = _catalogue()
    model_id = model_id or default
    return specs.get(model_id) or ModelSpec(model_id, model_id, "FOUNDRY_ENDPOINT")


def cost_usd(input_tokens: int, output_tokens: int, model_id: str | None = None) -> float | None:
    """USD cost, or None when the model's price is unknown."""
    price = get_model(model_id).price
    if price is None:
        pin, pout = env("PRICE_INPUT_PER_M"), env("PRICE_OUTPUT_PER_M")
        if not (pin and pout):
            return None if (input_tokens or output_tokens) else 0.0
        price = (float(pin), float(pout))
    return input_tokens / 1e6 * price[0] + output_tokens / 1e6 * price[1]


# --------------------------------------------------------------------- client
@lru_cache(maxsize=None)
def _client(endpoint_var: str):
    from openai import OpenAI

    endpoint = env(endpoint_var)
    if not endpoint:
        raise RuntimeError(f"{endpoint_var} is not set. Copy .env.example to .env and fill it in.")
    # Key per resource by prefix: FOUNDRY_ENDPOINT -> FOUNDRY_API_KEY, BENCH_*_ENDPOINT -> BENCH_API_KEY.
    api_key = env(f"{endpoint_var.split('_')[0]}_API_KEY")
    if not api_key:
        from azure.identity import (AzureCliCredential, ChainedTokenCredential, DefaultAzureCredential,
                                    get_bearer_token_provider)

        # az CLI first with a generous timeout (the default 10 s fails when the CPU is busy).
        credential = ChainedTokenCredential(AzureCliCredential(process_timeout=60), DefaultAzureCredential())
        scope = ("https://cognitiveservices.azure.com/.default" if "openai.azure.com" in endpoint
                 else "https://ai.azure.com/.default")
        api_key = get_bearer_token_provider(credential, scope)
    return OpenAI(base_url=endpoint, api_key=api_key, max_retries=1, timeout=240)


def strict_schema(model: type[BaseModel]) -> dict:
    """Pydantic JSON schema adapted to OpenAI strict mode.

    Strict mode requires every property to be listed in ``required`` and
    ``additionalProperties: false`` on every object. Optional fields stay
    nullable through ``anyOf [..., null]``.
    """
    schema = copy.deepcopy(model.model_json_schema())

    def fix(node):
        if isinstance(node, dict):
            node.pop("default", None)
            node.pop("title", None)
            if node.get("format") == "date":
                node.pop("format")
                node["description"] = (node.get("description", "") + " ISO date YYYY-MM-DD.").strip()
            if node.get("type") == "object" and "properties" in node:
                node["required"] = list(node["properties"])
                node["additionalProperties"] = False
            for v in node.values():
                fix(v)
        elif isinstance(node, list):
            for v in node:
                fix(v)

    fix(schema)
    return schema


def _parse(output_model: type[T], text: str) -> T:
    """Validate JSON text; unwrap ``{"anything": {...}}`` if the model added a wrapper key."""
    try:
        return output_model.model_validate_json(text)
    except ValidationError:
        data = json.loads(text)
        if isinstance(data, dict) and len(data) == 1:
            inner = next(iter(data.values()))
            if isinstance(inner, dict):
                return output_model.model_validate(inner)
        raise


def _call_responses(spec: ModelSpec, name, instructions, user_input, schema, temperature, output_model):
    kwargs = dict(
        model=spec.deployment, instructions=instructions, input=user_input,
        text={"format": {"type": "json_schema", "name": name, "strict": True, "schema": schema}},
    )
    if temperature is not None:
        kwargs["temperature"] = temperature
    try:
        response = _client(spec.endpoint).with_options(timeout=spec.timeout, max_retries=0 if spec.timeout > 240 else 1).responses.create(**kwargs)
    except Exception as exc:  # some reasoning deployments reject temperature
        if "temperature" not in str(exc).lower() or "temperature" not in kwargs:
            raise
        kwargs.pop("temperature")
        response = _client(spec.endpoint).with_options(timeout=spec.timeout, max_retries=0 if spec.timeout > 240 else 1).responses.create(**kwargs)
    if not response.output_text:
        raise ValueError("Model returned no content.")
    u = response.usage
    return _parse(output_model, response.output_text), (u.input_tokens or 0, u.output_tokens or 0)


def _call_chat_json(spec: ModelSpec, name, instructions, user_input, schema, temperature, output_model, feedback):
    system = (
        f"{instructions}\n\nReply with ONE JSON object that matches this JSON Schema exactly. "
        f"Use the property names as given, include every property (null when unknown) and add no "
        f"wrapper keys or commentary.\nSchema ({name}):\n{json.dumps(schema)}"
    )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user_input}]
    if feedback:
        messages.append({"role": "user", "content": f"Your previous reply was invalid: {feedback}\nReply again with valid JSON."})
    kwargs = dict(model=spec.deployment, messages=messages, response_format={"type": "json_object"},
                  max_tokens=MAX_OUTPUT_TOKENS)
    if temperature is not None:
        kwargs["temperature"] = temperature
    completion = _client(spec.endpoint).with_options(timeout=spec.timeout, max_retries=0 if spec.timeout > 240 else 1).chat.completions.create(**kwargs)
    text = completion.choices[0].message.content if completion.choices else None
    if not text:
        raise ValueError("Model returned no content.")
    u = completion.usage
    return _parse(output_model, text), (u.prompt_tokens or 0, u.completion_tokens or 0)


_TRACE: contextvars.ContextVar = contextvars.ContextVar("llm_trace", default=None)


@contextmanager
def trace_calls():
    """Record every ``structured`` call made on this thread inside the block: prompt, schema, output, usage.

    Off unless a caller asks for it, so the experiment's runs are unchanged. Cached replies are recorded
    too (with their original latency), so a trace can be rebuilt without new model calls.
    """
    calls: list[dict] = []
    token = _TRACE.set(calls)
    try:
        yield calls
    finally:
        _TRACE.reset(token)


def _trace(name, spec: "ModelSpec", instructions, user_input, schema, parsed: BaseModel, usage: "Usage", key: str) -> None:
    calls = _TRACE.get()
    if calls is not None:
        calls.append({
            "name": name, "model": spec.id, "deployment": spec.deployment, "api": spec.api,
            "instructions": instructions, "input": user_input, "schema": schema,
            "output": parsed.model_dump(mode="json"),
            "input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
            "seconds": round(usage.seconds, 3), "cached": usage.cached,
            "cost_usd": cost_usd(usage.input_tokens, usage.output_tokens, spec.id), "cache_key": key[:16],
        })


def structured(
    name: str,
    instructions: str,
    user_input: str,
    output_model: type[T],
    *,
    model: str | None = None,
    temperature: float | None = 0.0,
    use_cache: bool = True,
    retries: int = 3,
) -> tuple[T, Usage]:
    """Call ``model`` (a models.yaml id; default from the catalogue) and parse into ``output_model``."""
    spec = get_model(model)
    schema = strict_schema(output_model)

    key = hashlib.sha256(
        json.dumps([spec.deployment, instructions, user_input, schema, temperature], sort_keys=True).encode()
    ).hexdigest()
    cache_file = CACHE_DIR / "llm" / f"{key}.json"
    if use_cache and cache_file.exists():
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        u = data["usage"]
        parsed = output_model.model_validate(data["output"])
        usage = Usage(u["input_tokens"], u["output_tokens"], u["seconds"], cached=True)
        _trace(name, spec, instructions, user_input, schema, parsed, usage, key)
        return parsed, usage
    if flag("LLM_CACHE_ONLY"):
        raise RuntimeError(f"{spec.id}: no recorded reply (cache-only mode; the live call never completed)")

    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            t0 = time.perf_counter()
            if spec.api == "chat_json":
                feedback = str(last_error)[:1500] if isinstance(last_error, (ValidationError, ValueError)) else None
                parsed, (tin, tout) = _call_chat_json(
                    spec, name, instructions, user_input, schema, temperature, output_model, feedback)
            else:
                parsed, (tin, tout) = _call_responses(
                    spec, name, instructions, user_input, schema, temperature, output_model)
            usage = Usage(tin, tout, time.perf_counter() - t0)
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(
                json.dumps({"model": spec.id, "output": parsed.model_dump(mode="json"), "usage": usage.__dict__},
                           indent=1),
                encoding="utf-8",
            )
            _trace(name, spec, instructions, user_input, schema, parsed, usage, key)
            return parsed, usage
        except Exception as exc:  # noqa: BLE001 - retry any transient or validation failure
            last_error = exc
            if type(exc).__name__ == "APITimeoutError":
                break  # already waited the full client timeout (+1 SDK retry); don't triple it
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"LLM call '{name}' on {spec.id} failed after {retries} attempts: {last_error}")


# ----------------------------------------------------------------- embeddings
EMBED_USAGE = {"tokens": 0, "calls": 0, "seconds": 0.0}  # process-wide, for cost reporting
_embed_lock = __import__("threading").Lock()


@lru_cache(maxsize=1)
def embedding_spec() -> dict:
    raw = yaml.safe_load(MODELS_PATH.read_text(encoding="utf-8"))
    return (raw.get("embeddings") or [{}])[0]


def embed_openai(texts: list[str]):
    """Embed with the Foundry OpenAI embedding model; every vector is cached on disk.

    Returns an (n, d) float32 numpy array of unit-normalised vectors.
    """
    import numpy as np

    spec = embedding_spec()
    cache_dir = CACHE_DIR / "embed" / spec["deployment"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    keys = [hashlib.sha256(t.encode()).hexdigest()[:24] for t in texts]
    out: list = [None] * len(texts)
    missing = []
    for i, k in enumerate(keys):
        f = cache_dir / f"{k}.npy"
        if f.exists():
            out[i] = np.load(f)
        else:
            missing.append(i)
    if missing and flag("LLM_CACHE_ONLY"):
        raise RuntimeError("embedding not cached (cache-only mode)")
    for start in range(0, len(missing), 256):
        batch = missing[start:start + 256]
        t0 = time.perf_counter()
        resp = _client(spec["endpoint"]).embeddings.create(model=spec["deployment"], input=[texts[i] for i in batch])
        with _embed_lock:
            EMBED_USAGE["tokens"] += resp.usage.total_tokens or 0
            EMBED_USAGE["calls"] += 1
            EMBED_USAGE["seconds"] += time.perf_counter() - t0
            ledger = cache_dir.parent / "usage.jsonl"  # persistent record of every paid embedding call
            with ledger.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"deployment": spec["deployment"], "texts": len(batch),
                                    "tokens": resp.usage.total_tokens or 0,
                                    "seconds": round(time.perf_counter() - t0, 3)}) + "\n")
        for i, d in zip(batch, resp.data):
            v = np.asarray(d.embedding, dtype=np.float32)
            v /= np.linalg.norm(v) or 1.0
            np.save(cache_dir / f"{keys[i]}.npy", v)
            out[i] = v
    return np.vstack(out) if out else np.zeros((0, int(spec.get("dimensions", 1))), dtype=np.float32)
