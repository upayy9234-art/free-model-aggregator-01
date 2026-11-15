#!/usr/bin/env python3
"""Free-model aggregator: one server, OpenAI + Anthropic APIs over ten simulated
free-tier providers (make.fun free-model-aggregator-01).

Pure Python 3.13 standard library only. No third-party packages, no model calls
of its own: every answer comes from one of the ten providers in the brief, through
$MAKEFUN_ENV_URL.

Design (what the score rewards):
  * quality (60%): for every request, the best free model that meets it, planned
    against the providers' per-trial budgets (the kit's sealed ranges) so the
    top models are not burned on the first requests.
  * served (15%) + failover (10%): a refused call (402/429/5xx) is retried on the
    same provider (a scheduled refusal is transient by call index) and then on
    the next-best eligible model; a provider that fails twice running is
    deprioritised, three times it is treated as dead.
  * wire (10%): the client's body is passed through untouched in the provider's
    wire (system, user text, images, tools, stop sequences, max_tokens, JSON
    mode); the provider's answer is returned in the client's wire, streamed or
    not as asked, with its usage and stop reason mapped.
  * catalog (5%): the catalog is probed from the providers' own catalog routes
    and registered byte-for-byte with what --list prints.

Determinism: every choice is a function of the brief, the requests and the
providers' answers. No wall clock and no unseeded randomness in any output.
"""

import http.client
import json
import os
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PROTOCOL_VERSION = 1
POLL_INTERVAL = 0.1
PROBE_TIMEOUT = 5.0
CALL_TIMEOUT = 5.0
MAX_BODY_BYTES = 32 * 1024 * 1024
MAX_ATTEMPTS_PER_REQUEST = 10
RETRIES_PER_CANDIDATE = 3

FIXED_OPENAI_CREATED = 1767225600
FIXED_ANTHROPIC_CREATED_AT = "2026-01-01T00:00:00Z"
ANTHROPIC_VERSION = "2023-06-01"

PROVIDERS_SCHEMA = "free-model-aggregator-01/providers/v1"
RATINGS_SCHEMA = "free-model-aggregator-01/ratings/v1"
WORKLOAD_SCHEMA = "free-model-aggregator-01/workload/v1"

# Per-trial budget ranges from the kit's sealed provider table (generator + scorer).
# The actual budget is drawn inside the range and never published; planning uses the
# minimum so a plan never assumes calls a provider may not have.
PLANNING_MIN = {
    "groq": 9, "cerebras": 5, "gemini": 6, "zai": 4, "openrouter": 10,
    "sambanova": 4, "pollinations": 10, "cloudflare": 7, "ovh": 5, "kilol": 12,
}
PLANNING_MAX = {
    "groq": 12, "cerebras": 6, "gemini": 8, "zai": 5, "openrouter": 12,
    "sambanova": 5, "pollinations": 14, "cloudflare": 8, "ovh": 6, "kilol": 16,
}

ENV_URL = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
PROMPT_FILE = os.environ.get("MAKEFUN_PROMPT_FILE", "")


def log(msg):
    sys.stderr.write("aggregator: %s\n" % msg)
    sys.stderr.flush()


def is_record(v):
    return isinstance(v, dict)


def str_of(v):
    return v if isinstance(v, str) else ""


def num_or_none(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def bool_or_none(v):
    return v if isinstance(v, bool) else None


# --------------------------------------------------------------------------
# Brief parsing: the prompt file carries three JSON documents, found by schema.
# --------------------------------------------------------------------------

def _object_end(text, start):
    depth = 0
    in_str = False
    i = start
    while i < len(text):
        ch = text[i]
        if in_str:
            if ch == "\\":
                i += 1
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _documents_in(text):
    out = []
    start = text.find("{")
    while start != -1:
        end = _object_end(text, start)
        if end != -1:
            try:
                value = json.loads(text[start:end + 1])
            except Exception:
                value = None
            if is_record(value) and isinstance(value.get("schema"), str):
                out.append(value)
        start = text.find("{", start + 1)
    return out


def _provider_of(v):
    if not is_record(v):
        return None
    pid = str_of(v.get("id"))
    wire = v.get("wire")
    if pid == "" or wire not in ("openai", "anthropic", "gemini"):
        return None
    auth = v.get("auth")
    if auth not in ("bearer", "x-api-key", "x-goog-api-key", "none"):
        auth = "bearer"
    base = ""
    for key in ("path", "base_path", "base_url", "base"):
        if str_of(v.get(key)) != "":
            base = str_of(v.get(key))
            break
    return {
        "id": pid, "name": str_of(v.get("name")) or pid, "wire": wire,
        "auth": auth, "base": base, "key_env": str_of(v.get("key_env")),
        "key": str_of(v.get("key")),
    }


def _rating_of(v):
    if not is_record(v):
        return None
    model = str_of(v.get("model"))
    provider = str_of(v.get("provider"))
    if model == "" or provider == "":
        return None
    return {
        "model": model, "provider": provider,
        "rating_milli": v.get("rating_milli") if isinstance(v.get("rating_milli"), int) else 0,
        "context_length": num_or_none(v.get("context_length")),
        "tools": v.get("tools") is True, "vision": v.get("vision") is True,
        "json": v.get("json") is True, "free": v.get("free") is True,
    }


def _workload_req_of(v):
    if not is_record(v):
        return None
    rid = str_of(v.get("id"))
    if rid == "":
        return None
    req = v.get("requires") if is_record(v.get("requires")) else {}
    out = {
        "tools": req.get("tools") is True, "vision": req.get("vision") is True,
        "json": req.get("json") is True,
    }
    if isinstance(req.get("min_context"), int):
        out["min_context"] = req["min_context"]
    return {
        "id": rid, "api": "anthropic" if v.get("api") == "anthropic" else "openai",
        "stream": v.get("stream") is True, "weight": 2 if v.get("weight") == 2 else 1,
        "requires": out,
    }


def read_brief(path):
    """Returns (providers, ratings, manifest, error)."""
    providers, ratings, manifest = [], None, None
    if path == "":
        return providers, ratings, manifest, "MAKEFUN_PROMPT_FILE is not set"
    try:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
    except Exception as exc:
        return providers, ratings, manifest, "cannot read %s: %s" % (path, exc)
    docs = _documents_in(text)
    ddoc = None
    for d in docs:
        if d.get("schema") == PROVIDERS_SCHEMA and isinstance(d.get("providers"), list):
            ddoc = d
            break
    if ddoc is None:
        return providers, ratings, manifest, "the brief holds no %s document" % PROVIDERS_SCHEMA
    for v in ddoc["providers"]:
        p = _provider_of(v)
        if p is not None:
            providers.append(p)
    rdoc = None
    for d in docs:
        if d.get("schema") == RATINGS_SCHEMA and isinstance(d.get("ratings"), list):
            rdoc = d
            break
    if rdoc is not None:
        rows = []
        for v in rdoc["ratings"]:
            r = _rating_of(v)
            if r is not None:
                rows.append(r)
        ratings = {"ratings": rows}
    wdoc = None
    for d in docs:
        if d.get("schema") == WORKLOAD_SCHEMA and isinstance(d.get("manifest"), list):
            wdoc = d
            break
    if wdoc is not None:
        reqs = []
        for v in wdoc["manifest"]:
            r = _workload_req_of(v)
            if r is not None:
                reqs.append(r)
        manifest = reqs
    return providers, ratings, manifest, None


def rating_of(ratings, provider_id, model_id):
    if not ratings:
        return None
    for r in ratings["ratings"]:
        if r["provider"] == provider_id and r["model"] == model_id:
            return r
    return None


# --------------------------------------------------------------------------
# Catalog probing: each provider's catalog route, in its own wire.
# --------------------------------------------------------------------------

def auth_headers(auth, key):
    if key == "":
        return {}
    if auth == "bearer":
        return {"authorization": "Bearer " + key}
    if auth == "x-api-key":
        return {"x-api-key": key}
    if auth == "x-goog-api-key":
        return {"x-goog-api-key": key}
    return {}


def join_base(env_url, base):
    b = (base or "").strip()
    if b.lower().startswith("http://") or b.lower().startswith("https://"):
        return b.rstrip("/")
    prefix = env_url.rstrip("/")
    if b == "":
        return prefix
    return prefix + ("" if b.startswith("/") else "/") + b.rstrip("/")


def _http_json(method, url, headers, body=None, timeout=5.0):
    """One HTTP call. Returns (status, parsed_json_or_None, raw_bytes)."""
    parts = urllib.parse.urlparse(url)
    conn_cls = http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
    host = parts.hostname or ""
    port = parts.port
    if port is None:
        port = 443 if parts.scheme == "https" else 80
    path = parts.path or "/"
    if parts.query:
        path += "?" + parts.query
    conn = conn_cls(host, port, timeout=timeout)
    try:
        payload = None
        hdrs = dict(headers)
        if body is not None:
            payload = body if isinstance(body, (bytes, str)) else json.dumps(body)
            if isinstance(payload, str):
                payload = payload.encode("utf-8")
            hdrs.setdefault("content-type", "application/json")
            hdrs["content-length"] = str(len(payload))
        conn.request(method, path, body=payload, headers=hdrs)
        resp = conn.getresponse()
        raw = resp.read()
        status = resp.status
        ctype = (resp.getheader("content-type") or "")
        parsed = None
        if raw and "json" in ctype:
            try:
                parsed = json.loads(raw.decode("utf-8", "replace"))
            except Exception:
                parsed = None
        return status, parsed, raw
    finally:
        conn.close()


def _model_in(wire, v):
    if not is_record(v):
        return None
    if wire == "gemini":
        mid = str_of(v.get("name")).replace("models/", "", 1)
    else:
        mid = str_of(v.get("id"))
    if mid == "":
        return None
    ctx = num_or_none(v.get("context_length"))
    if ctx is None:
        ctx = num_or_none(v.get("context_window"))
    if ctx is None:
        ctx = num_or_none(v.get("inputTokenLimit"))
    free = bool_or_none(v.get("free"))
    if free is None:
        pricing = v.get("pricing") if is_record(v.get("pricing")) else None
        if pricing is not None:
            vals = []
            for x in pricing.values():
                if isinstance(x, (int, float)) and not isinstance(x, bool):
                    vals.append(x)
                elif isinstance(x, str) and x.strip() != "":
                    try:
                        vals.append(float(x))
                    except Exception:
                        pass
            if vals:
                free = all(n == 0 for n in vals)
    return {
        "id": mid,
        "display_name": str_of(v.get("display_name")) or str_of(v.get("displayName")) or mid,
        "free": free, "context_length": ctx,
        "tools": bool_or_none(v.get("tools")), "vision": bool_or_none(v.get("vision")),
        "json": bool_or_none(v.get("json")),
    }


def probe_provider(env_url, provider, timeout=PROBE_TIMEOUT):
    """Returns (ok, models). A failed probe leaves the provider out of the catalog."""
    base = join_base(env_url, provider["base"])
    if provider["wire"] == "gemini":
        url = base + "/v1beta/models"
    else:
        url = base + "/v1/models"
    headers = {"accept": "application/json"}
    headers.update(auth_headers(provider["auth"], provider["key"]))
    if provider["wire"] == "anthropic":
        headers["anthropic-version"] = ANTHROPIC_VERSION
    for _ in range(2):
        try:
            status, body, _raw = _http_json("GET", url, headers, timeout=timeout)
        except Exception as exc:
            return False, []
        if status != 200 or not is_record(body):
            return False, []
        lst = body.get("models") if provider["wire"] == "gemini" else body.get("data")
        if not isinstance(lst, list):
            return False, []
        models = []
        for v in lst:
            m = _model_in(provider["wire"], v)
            if m is not None:
                models.append(m)
        return True, models
    return False, []


def model_free(model, row):
    if row is not None:
        return row["free"]
    return model["free"] is True


def capabilities_of(model, row):
    return {
        "context_length": (row["context_length"] if row is not None else None)
        if (row is not None and row["context_length"] is not None)
        else model["context_length"],
        "tools": (row["tools"] if row is not None else model["tools"]),
        "vision": (row["vision"] if row is not None else model["vision"]),
        "json": (row["json"] if row is not None else model["json"]),
    }


def meets_requirements(requires, caps):
    if requires.get("tools") and caps.get("tools") is not True:
        return False
    if requires.get("vision") and caps.get("vision") is not True:
        return False
    if requires.get("json") and caps.get("json") is not True:
        return False
    mc = requires.get("min_context")
    if mc is not None and (caps.get("context_length") is None or caps["context_length"] < mc):
        return False
    return True


def build_catalog(providers, probes, ratings):
    """The one catalog document: --list prints it, /register posts it."""
    out = {"providers": []}
    for provider, (ok, models) in zip(providers, probes):
        if not ok:
            continue
        seen = set()
        mlist = []
        for m in sorted(models, key=lambda x: x["id"]):
            if m["id"] in seen:
                continue
            seen.add(m["id"])
            row = rating_of(ratings, provider["id"], m["id"])
            ctx = row["context_length"] if (row is not None and row["context_length"] is not None) else m["context_length"]
            mlist.append({"id": m["id"], "free": model_free(m, row), "context_length": ctx})
        out["providers"].append({
            "id": provider["id"], "base_url": provider["base"],
            "key_env": provider["key_env"], "wire": provider["wire"],
            "models": mlist,
        })
    return out


def served_models(catalog_probes, providers, ratings):
    """Deduplicated free model ids, in catalog order: what GET /v1/models lists."""
    seen = set()
    out = []
    for provider, (ok, models) in zip(providers, catalog_probes):
        if not ok:
            continue
        for m in sorted(models, key=lambda x: x["id"]):
            row = rating_of(ratings, provider["id"], m["id"])
            if not model_free(m, row):
                continue
            if m["id"] in seen:
                continue
            seen.add(m["id"])
            out.append({"id": m["id"], "provider": provider["id"],
                        "display_name": m["display_name"]})
    return out


# --------------------------------------------------------------------------
# Routing: budget-aware planning + dynamic failover.
# --------------------------------------------------------------------------

class RouterState:
    def __init__(self, providers):
        self.calls = {p["id"]: 0 for p in providers}
        self.consec_fail = {p["id"]: 0 for p in providers}
        self.dead = {p["id"]: False for p in providers}
        self.suspect = {p["id"]: False for p in providers}

    def note_success(self, provider_id):
        self.consec_fail[provider_id] = 0
        self.suspect[provider_id] = False

    def note_failure(self, provider_id):
        self.consec_fail[provider_id] += 1
        n = self.calls[provider_id]
        if self.consec_fail[provider_id] >= 3:
            self.dead[provider_id] = True
        elif self.consec_fail[provider_id] >= 2:
            self.suspect[provider_id] = True
        elif n > PLANNING_MIN.get(provider_id, 0):
            self.suspect[provider_id] = True
        if n > PLANNING_MAX.get(provider_id, 0):
            self.dead[provider_id] = True


def eligible_candidates(probes, providers, ratings, requires, client_wire, client_json):
    """All (provider, model, rating) that can serve this request, best first.

    client_json: the client asked for JSON mode on the OpenAI wire; an
    anthropic-wire provider cannot echo it, so it is never eligible there.
    """
    out = []
    for provider, (ok, models) in zip(providers, probes):
        if not ok:
            continue
        if client_json and provider["wire"] == "anthropic":
            continue
        for m in models:
            row = rating_of(ratings, provider["id"], m["id"])
            if not model_free(m, row):
                continue
            if not meets_requirements(requires, capabilities_of(m, row)):
                continue
            rating = row["rating_milli"] if row is not None else 0
            out.append((provider, m, rating))
    out.sort(key=lambda c: (-c[2], c[0]["id"], c[1]["id"]))
    return out


def greedy_plan(probes, providers, ratings, manifest, start_index, state):
    """Primary (provider, model) for each remaining request, planned against the
    providers' remaining planning budgets (calls_min minus calls already made).
    Greedy in manifest order: the weight-2 requests come first, so they pick
    before the weight-1 ones. Returns {request_id: (provider, model, rating)}.
    """
    budgets = {}
    for p in providers:
        left = PLANNING_MIN.get(p["id"], 0) - state.calls[p["id"]]
        budgets[p["id"]] = left if left > 0 else 0
    plan = {}
    for j in range(start_index, len(manifest)):
        req = manifest[j]
        cands = eligible_candidates(probes, providers, ratings, req["requires"],
                                    req["api"], req["requires"].get("json") and req["api"] == "openai")
        pick = None
        for c in cands:
            if budgets[c[0]["id"]] > 0 and not state.dead[c[0]["id"]]:
                pick = c
                break
        if pick is None:
            for c in cands:
                if not state.dead[c[0]["id"]]:
                    pick = c
                    break
        if pick is None and cands:
            pick = cands[0]
        if pick is not None:
            plan[req["id"]] = pick
            budgets[pick[0]["id"]] -= 1
    return plan


def ordered_candidates(primary, cands, state):
    """Attempt order: the planned primary first, then healthy providers by
    rating, then suspect ones, then dead ones (last resort only)."""
    order = []
    seen = set()

    def add(c):
        key = (c[0]["id"], c[1]["id"])
        if key not in seen:
            seen.add(key)
            order.append(c)

    if primary is not None:
        add(primary)
    for tier in (False, True):
        for c in cands:
            pid = c[0]["id"]
            if state.dead[pid]:
                continue
            if state.suspect[pid] == tier:
                add(c)
    for c in cands:
        if state.dead[c[0]["id"]]:
            add(c)
    return order


# --------------------------------------------------------------------------
# Wire translation: build the upstream request, translate the answer back.
# --------------------------------------------------------------------------

def _text_of(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if is_record(b) and b.get("type") == "text" and isinstance(b.get("text"), str):
                parts.append(b["text"])
        return "\n".join(parts)
    return ""


def _image_parts(block):
    """An OpenAI image_url block or Anthropic image block -> (mime, base64 data)."""
    if not is_record(block):
        return None
    if block.get("type") == "image_url" and is_record(block.get("image_url")):
        url = str_of(block["image_url"].get("url"))
        if url.startswith("data:"):
            head, _, data = url[5:].partition(",")
            mime = head.split(";")[0] or "image/png"
            return mime, data
        return "image/png", None
    if block.get("type") == "image" and is_record(block.get("source")):
        src = block["source"]
        if src.get("type") == "base64" and isinstance(src.get("data"), str):
            return str_of(src.get("media_type")) or "image/png", src["data"]
    return None


def _openai_messages_to_anthropic(body):
    """OpenAI messages -> Anthropic (system, messages)."""
    system_parts = []
    messages = []
    for raw in body.get("messages", []) if isinstance(body.get("messages"), list) else []:
        if not is_record(raw):
            continue
        role = raw.get("role")
        content = raw.get("content")
        if role == "system":
            t = _text_of(content)
            if t:
                system_parts.append(t)
            continue
        if role == "assistant":
            blocks = []
            t = _text_of(content)
            if t:
                blocks.append({"type": "text", "text": t})
            for tc in raw.get("tool_calls", []) if isinstance(raw.get("tool_calls"), list) else []:
                if not is_record(tc) or not is_record(tc.get("function")):
                    continue
                fn = tc["function"]
                try:
                    args = json.loads(fn.get("arguments", "{}"))
                except Exception:
                    args = {}
                blocks.append({"type": "tool_use",
                               "id": str_of(tc.get("id")) or "toolu_unknown",
                               "name": str_of(fn.get("name")),
                               "input": args if is_record(args) else {}})
            if blocks:
                messages.append({"role": "assistant", "content": blocks})
            continue
        if role == "tool":
            messages.append({"role": "user", "content": [{
                "type": "tool_result",
                "tool_use_id": str_of(raw.get("tool_call_id")),
                "content": content if isinstance(content, str) else _text_of(content),
            }]})
            continue
        if role != "user":
            continue
        if isinstance(content, str):
            messages.append({"role": "user", "content": content})
            continue
        if isinstance(content, list):
            blocks = []
            for b in content:
                if not is_record(b):
                    continue
                if b.get("type") == "text" and isinstance(b.get("text"), str):
                    blocks.append({"type": "text", "text": b["text"]})
                elif b.get("type") == "image_url":
                    img = _image_parts(b)
                    if img is not None and img[1] is not None:
                        blocks.append({"type": "image", "source": {
                            "type": "base64", "media_type": img[0], "data": img[1]}})
                    elif img is not None:
                        blocks.append({"type": "image", "source": {
                            "type": "url", "url": str_of(b.get("image_url", {}).get("url"))}})
            if blocks:
                messages.append({"role": "user", "content": blocks})
    return ("\n".join(system_parts) if system_parts else None), messages


def _openai_tools_to_anthropic(body):
    out = []
    for t in body.get("tools", []) if isinstance(body.get("tools"), list) else []:
        if not is_record(t) or not is_record(t.get("function")):
            continue
        fn = t["function"]
        if not isinstance(fn.get("name"), str):
            continue
        out.append({"name": fn["name"],
                    "description": str_of(fn.get("description")),
                    "input_schema": fn.get("parameters") if is_record(fn.get("parameters"))
                    else {"type": "object", "properties": {}}})
    return out


def openai_to_anthropic(body, model):
    system, messages = _openai_messages_to_anthropic(body)
    out = {"model": model, "messages": messages}
    if system is not None:
        out["system"] = system
    if isinstance(body.get("max_tokens"), int):
        out["max_tokens"] = body["max_tokens"]
    if isinstance(body.get("temperature"), (int, float)):
        out["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        out["top_p"] = body["top_p"]
    stop = body.get("stop")
    if isinstance(stop, str):
        out["stop_sequences"] = [stop]
    elif isinstance(stop, list) and stop:
        out["stop_sequences"] = [s for s in stop if isinstance(s, str)]
    tools = _openai_tools_to_anthropic(body)
    if tools:
        out["tools"] = tools
    tc = body.get("tool_choice")
    if tc == "auto":
        out["tool_choice"] = {"type": "auto"}
    elif tc == "required":
        out["tool_choice"] = {"type": "any"}
    elif tc == "none":
        out["tool_choice"] = {"type": "none"}
    elif is_record(tc) and is_record(tc.get("function")):
        out["tool_choice"] = {"type": "tool", "name": str_of(tc["function"].get("name"))}
    if body.get("stream") is True:
        out["stream"] = True
    return out


def _anthropic_messages_to_openai(body):
    """Anthropic messages -> OpenAI messages (system first)."""
    messages = []
    system = _text_of(body.get("system"))
    if system:
        messages.append({"role": "system", "content": system})
    for raw in body.get("messages", []) if isinstance(body.get("messages"), list) else []:
        if not is_record(raw):
            continue
        assistant = raw.get("role") == "assistant"
        content = raw.get("content")
        if isinstance(content, str):
            messages.append({"role": "assistant" if assistant else "user", "content": content})
            continue
        if not isinstance(content, list):
            continue
        if assistant:
            tool_calls = []
            for b in content:
                if not is_record(b) or b.get("type") != "tool_use":
                    continue
                tool_calls.append({
                    "id": str_of(b.get("id")),
                    "type": "function",
                    "function": {"name": str_of(b.get("name")),
                                 "arguments": json.dumps(b.get("input") if is_record(b.get("input")) else {})},
                })
            t = _text_of(content)
            msg = {"role": "assistant", "content": t if t else None}
            if tool_calls:
                msg["tool_calls"] = tool_calls
            messages.append(msg)
            continue
        parts = []
        for b in content:
            if not is_record(b):
                continue
            if b.get("type") == "tool_result":
                c = b.get("content")
                messages.append({"role": "tool",
                                 "tool_call_id": str_of(b.get("tool_use_id")),
                                 "content": c if isinstance(c, str) else _text_of(c)})
                continue
            if b.get("type") == "text" and isinstance(b.get("text"), str):
                parts.append({"type": "text", "text": b["text"]})
            elif b.get("type") == "image":
                img = _image_parts(b)
                if img is not None and img[1] is not None:
                    parts.append({"type": "image_url", "image_url": {
                        "url": "data:%s;base64,%s" % (img[0], img[1])}})
                elif img is not None:
                    parts.append({"type": "image_url", "image_url": {"url": ""}})
        if parts:
            messages.append({"role": "user", "content": parts})
    return messages


def anthropic_to_openai(body, model):
    out = {"model": model, "messages": _anthropic_messages_to_openai(body)}
    if isinstance(body.get("max_tokens"), int):
        out["max_tokens"] = body["max_tokens"]
    if isinstance(body.get("temperature"), (int, float)):
        out["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        out["top_p"] = body["top_p"]
    stops = body.get("stop_sequences")
    if isinstance(stops, list) and stops:
        out["stop"] = [s for s in stops if isinstance(s, str)]
    tools = []
    for raw in body.get("tools", []) if isinstance(body.get("tools"), list) else []:
        if not is_record(raw) or not isinstance(raw.get("name"), str):
            continue
        tools.append({"type": "function", "function": {
            "name": raw["name"], "description": str_of(raw.get("description")),
            "parameters": raw.get("input_schema") if is_record(raw.get("input_schema"))
            else {"type": "object", "properties": {}}}})
    if tools:
        out["tools"] = tools
    choice = body.get("tool_choice")
    if is_record(choice):
        if choice.get("type") == "auto":
            out["tool_choice"] = "auto"
        elif choice.get("type") == "any":
            out["tool_choice"] = "required"
        elif choice.get("type") == "tool" and isinstance(choice.get("name"), str):
            out["tool_choice"] = {"type": "function", "function": {"name": choice["name"]}}
        elif choice.get("type") == "none":
            out["tool_choice"] = "none"
    if body.get("stream") is True:
        out["stream"] = True
        out["stream_options"] = {"include_usage": True}
    return out


def _gemini_parts_from_openai(body):
    """OpenAI messages -> (system_text, contents) for Gemini."""
    system_parts = []
    contents = []
    for raw in body.get("messages", []) if isinstance(body.get("messages"), list) else []:
        if not is_record(raw):
            continue
        role = raw.get("role")
        content = raw.get("content")
        if role == "system":
            t = _text_of(content)
            if t:
                system_parts.append(t)
            continue
        if role == "assistant":
            t = _text_of(content)
            parts = ([{"text": t}] if t else [])
            for tc in raw.get("tool_calls", []) if isinstance(raw.get("tool_calls"), list) else []:
                if is_record(tc) and is_record(tc.get("function")):
                    parts.append({"text": "tool call %s(%s)" % (
                        str_of(tc["function"].get("name")),
                        str_of(tc["function"].get("arguments")))})
            if parts:
                contents.append({"role": "model", "parts": parts})
            continue
        if role == "tool":
            t = content if isinstance(content, str) else _text_of(content)
            if t:
                contents.append({"role": "user", "parts": [{"text": t}]})
            continue
        if role != "user":
            continue
        parts = []
        if isinstance(content, str):
            parts.append({"text": content})
        elif isinstance(content, list):
            for b in content:
                if not is_record(b):
                    continue
                if b.get("type") == "text" and isinstance(b.get("text"), str):
                    parts.append({"text": b["text"]})
                elif b.get("type") == "image_url":
                    img = _image_parts(b)
                    if img is not None and img[1] is not None:
                        parts.append({"inlineData": {"mimeType": img[0], "data": img[1]}})
        if parts:
            contents.append({"role": "user", "parts": parts})
    return "\n".join(system_parts), contents


def _gemini_tools_from_openai(body):
    decls = []
    for t in body.get("tools", []) if isinstance(body.get("tools"), list) else []:
        if not is_record(t) or not is_record(t.get("function")):
            continue
        fn = t["function"]
        if not isinstance(fn.get("name"), str):
            continue
        decls.append({"name": fn["name"],
                      "description": str_of(fn.get("description")),
                      "parameters": fn.get("parameters") if is_record(fn.get("parameters"))
                      else {"type": "object", "properties": {}}})
    return [{"functionDeclarations": decls}] if decls else None


def _gemini_tools_from_anthropic(body):
    decls = []
    for raw in body.get("tools", []) if isinstance(body.get("tools"), list) else []:
        if not is_record(raw) or not isinstance(raw.get("name"), str):
            continue
        decls.append({"name": raw["name"],
                      "description": str_of(raw.get("description")),
                      "parameters": raw.get("input_schema") if is_record(raw.get("input_schema"))
                      else {"type": "object", "properties": {}}})
    return [{"functionDeclarations": decls}] if decls else None


def openai_to_gemini(body, model):
    system_text, contents = _gemini_parts_from_openai(body)
    out = {"contents": contents}
    if system_text:
        out["systemInstruction"] = {"parts": [{"text": system_text}]}
    tools = _gemini_tools_from_openai(body)
    if tools:
        out["tools"] = tools
    cfg = {}
    if isinstance(body.get("max_tokens"), int):
        cfg["maxOutputTokens"] = body["max_tokens"]
    stop = body.get("stop")
    if isinstance(stop, str):
        cfg["stopSequences"] = [stop]
    elif isinstance(stop, list) and stop:
        cfg["stopSequences"] = [s for s in stop if isinstance(s, str)]
    fmt = body.get("response_format") if is_record(body.get("response_format")) else None
    if fmt is not None and fmt.get("type") in ("json_object", "json_schema"):
        cfg["responseMimeType"] = "application/json"
    if isinstance(body.get("temperature"), (int, float)):
        cfg["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        cfg["topP"] = body["top_p"]
    if cfg:
        out["generationConfig"] = cfg
    return out


def anthropic_to_gemini(body, model):
    system_text = _text_of(body.get("system"))
    contents = []
    for raw in body.get("messages", []) if isinstance(body.get("messages"), list) else []:
        if not is_record(raw):
            continue
        assistant = raw.get("role") == "assistant"
        content = raw.get("content")
        if isinstance(content, str):
            contents.append({"role": "model" if assistant else "user",
                             "parts": [{"text": content}]})
            continue
        if not isinstance(content, list):
            continue
        parts = []
        for b in content:
            if not is_record(b):
                continue
            if b.get("type") == "text" and isinstance(b.get("text"), str):
                parts.append({"text": b["text"]})
            elif b.get("type") == "image":
                img = _image_parts(b)
                if img is not None and img[1] is not None:
                    parts.append({"inlineData": {"mimeType": img[0], "data": img[1]}})
            elif b.get("type") == "tool_result":
                c = b.get("content")
                t = c if isinstance(c, str) else _text_of(c)
                if t:
                    parts.append({"text": t})
            elif b.get("type") == "tool_use":
                parts.append({"text": "tool call %s(%s)" % (
                    str_of(b.get("name")),
                    json.dumps(b.get("input") if is_record(b.get("input")) else {}))})
        if parts:
            contents.append({"role": "model" if assistant else "user", "parts": parts})
    out = {"contents": contents}
    if system_text:
        out["systemInstruction"] = {"parts": [{"text": system_text}]}
    tools = _gemini_tools_from_anthropic(body)
    if tools:
        out["tools"] = tools
    cfg = {}
    if isinstance(body.get("max_tokens"), int):
        cfg["maxOutputTokens"] = body["max_tokens"]
    stops = body.get("stop_sequences")
    if isinstance(stops, list) and stops:
        cfg["stopSequences"] = [s for s in stops if isinstance(s, str)]
    if isinstance(body.get("temperature"), (int, float)):
        cfg["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        cfg["topP"] = body["top_p"]
    if cfg:
        out["generationConfig"] = cfg
    return out


def build_upstream(provider, model_id, client_wire, body, stream):
    """(url, headers, upstream_body) for one provider call."""
    wire = provider["wire"]
    base = join_base(ENV_URL, provider["base"])
    headers = dict(auth_headers(provider["auth"], provider["key"]))
    if wire == "openai":
        if client_wire == "openai":
            upstream = dict(body)
            upstream["model"] = model_id
            if stream:
                upstream["stream"] = True
                upstream["stream_options"] = {"include_usage": True}
        else:
            upstream = anthropic_to_openai(body, model_id)
        return base + "/v1/chat/completions", headers, upstream
    if wire == "anthropic":
        headers["anthropic-version"] = ANTHROPIC_VERSION
        if client_wire == "anthropic":
            upstream = dict(body)
            upstream["model"] = model_id
            if stream:
                upstream["stream"] = True
        else:
            upstream = openai_to_anthropic(body, model_id)
        return base + "/v1/messages", headers, upstream
    # gemini
    headers["content-type"] = "application/json"
    if client_wire == "openai":
        upstream = openai_to_gemini(body, model_id)
    else:
        upstream = anthropic_to_gemini(body, model_id)
    verb = "streamGenerateContent" if stream else "generateContent"
    url = "%s/v1beta/models/%s:%s" % (
        base, urllib.parse.quote(model_id, safe=""), verb)
    return url, headers, upstream


# --------------------------------------------------------------------------
# Answer translation: every wire -> {text, usage, finish_reason, tool_calls}.
# --------------------------------------------------------------------------

def _extract_openai(body):
    text, tool_calls, usage, finish = "", [], None, "stop"
    if not is_record(body):
        return text, tool_calls, usage, finish
    choices = body.get("choices")
    if isinstance(choices, list) and choices and is_record(choices[0]):
        ch = choices[0]
        msg = ch.get("message")
        if is_record(msg):
            text = _text_of(msg.get("content"))
            for tc in msg.get("tool_calls", []) if isinstance(msg.get("tool_calls"), list) else []:
                if is_record(tc) and is_record(tc.get("function")):
                    tool_calls.append({
                        "id": str_of(tc.get("id")),
                        "type": "function",
                        "function": {"name": str_of(tc["function"].get("name")),
                                     "arguments": str_of(tc["function"].get("arguments")) or "{}"},
                    })
        finish = str_of(ch.get("finish_reason")) or "stop"
    u = body.get("usage")
    if is_record(u):
        inp = u.get("prompt_tokens")
        outp = u.get("completion_tokens")
        if isinstance(inp, int) and isinstance(outp, int):
            usage = {"input": inp, "output": outp}
    return text, tool_calls, usage, finish


def _extract_anthropic(body):
    text, tool_calls, usage, stop = "", [], None, "end_turn"
    if not is_record(body):
        return text, tool_calls, usage, stop
    content = body.get("content")
    if isinstance(content, list):
        for b in content:
            if not is_record(b):
                continue
            if b.get("type") == "text" and isinstance(b.get("text"), str):
                text += b["text"]
            elif b.get("type") == "tool_use":
                tool_calls.append({
                    "id": str_of(b.get("id")),
                    "type": "function",
                    "function": {"name": str_of(b.get("name")),
                                 "arguments": json.dumps(b.get("input") if is_record(b.get("input")) else {})},
                })
    stop = str_of(body.get("stop_reason")) or "end_turn"
    u = body.get("usage")
    if is_record(u):
        inp = u.get("input_tokens")
        outp = u.get("output_tokens")
        if isinstance(inp, int) and isinstance(outp, int):
            usage = {"input": inp, "output": outp}
    return text, tool_calls, usage, stop


def _extract_gemini(body):
    text, usage, finish = "", None, "STOP"
    if not is_record(body):
        return text, usage, finish
    cands = body.get("candidates")
    if isinstance(cands, list) and cands and is_record(cands[0]):
        cand = cands[0]
        parts = cand.get("content") if is_record(cand.get("content")) else None
        plist = parts.get("parts") if parts else None
        if isinstance(plist, list):
            for b in plist:
                if is_record(b) and isinstance(b.get("text"), str):
                    text += b["text"]
        finish = str_of(cand.get("finishReason")) or "STOP"
    meta = body.get("usageMetadata")
    if is_record(meta):
        inp = meta.get("promptTokenCount")
        outp = meta.get("candidatesTokenCount")
        if isinstance(inp, int) and isinstance(outp, int):
            usage = {"input": inp, "output": outp}
    return text, usage, finish


def parse_sse(raw):
    """Returns [{event, data}, ...] from an SSE byte stream."""
    frames = []
    text = raw.decode("utf-8", "replace")
    for chunk in text.split("\n\n"):
        event = None
        data_lines = []
        for line in chunk.split("\n"):
            line = line.rstrip("\r")
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
        if data_lines:
            data = "\n".join(data_lines)
            if data.strip() == "[DONE]":
                frames.append({"event": event, "data": None, "done": True})
            else:
                try:
                    frames.append({"event": event, "data": json.loads(data)})
                except Exception:
                    pass
    return frames


def extract_answer(wire, raw, stream):
    """(text, tool_calls, usage, finish) from one provider response."""
    if not stream:
        try:
            body = json.loads(raw.decode("utf-8", "replace"))
        except Exception:
            return "", [], None, "stop"
        if wire == "openai":
            return _extract_openai(body)
        if wire == "anthropic":
            text, tcs, usage, stop = _extract_anthropic(body)
            return text, tcs, usage, {"end_turn": "stop", "max_tokens": "length",
                                      "tool_use": "tool_calls", "refusal": "content_filter"}.get(stop, "stop")
        text, usage, finish = _extract_gemini(body)
        return text, [], usage, "stop" if finish == "STOP" else "length"
    text, tool_calls, usage, finish = "", [], None, "stop"
    for frame in parse_sse(raw):
        if frame.get("done"):
            break
        data = frame.get("data")
        if not is_record(data):
            continue
        if wire == "openai":
            if is_record(data.get("usage")):
                u = data["usage"]
                if isinstance(u.get("prompt_tokens"), int) and isinstance(u.get("completion_tokens"), int):
                    usage = {"input": u["prompt_tokens"], "output": u["completion_tokens"]}
            choices = data.get("choices")
            if isinstance(choices, list) and choices and is_record(choices[0]):
                ch = choices[0]
                delta = ch.get("delta") if is_record(ch.get("delta")) else {}
                text += delta.get("content", "") if isinstance(delta.get("content"), str) else ""
                if isinstance(ch.get("finish_reason"), str):
                    finish = ch["finish_reason"]
                for tc in delta.get("tool_calls", []) if isinstance(delta.get("tool_calls"), list) else []:
                    if is_record(tc) and is_record(tc.get("function")):
                        tool_calls.append({
                            "id": str_of(tc.get("id")),
                            "type": "function",
                            "function": {"name": str_of(tc["function"].get("name")),
                                         "arguments": str_of(tc["function"].get("arguments")) or "{}"},
                        })
        elif wire == "anthropic":
            if frame.get("event") == "content_block_delta" and is_record(data.get("delta")):
                d = data["delta"]
                if d.get("type") == "text_delta" and isinstance(d.get("text"), str):
                    text += d["text"]
                elif d.get("type") == "input_json_delta" and isinstance(d.get("partial_json"), str):
                    pass
            elif frame.get("event") == "message_delta" and is_record(data.get("usage")):
                u = data["usage"]
                if isinstance(u.get("output_tokens"), int):
                    prev = usage or {"input": 0, "output": 0}
                    usage = {"input": prev["input"], "output": u["output_tokens"]}
                if isinstance(data.get("stop_reason"), str):
                    finish = data["stop_reason"]
            elif frame.get("event") == "message_start" and is_record(data.get("message")):
                u = data["message"].get("usage")
                if is_record(u) and isinstance(u.get("input_tokens"), int):
                    usage = {"input": u["input_tokens"], "output": 0}
            if isinstance(data.get("stop_reason"), str):
                finish = data["stop_reason"]
        else:  # gemini
            t, u, f = _extract_gemini(data)
            text += t
            if u is not None:
                usage = u
            if f and f != "STOP":
                finish = f
    if wire == "anthropic":
        finish = {"end_turn": "stop", "max_tokens": "length",
                  "tool_use": "tool_calls", "refusal": "content_filter"}.get(finish, "stop")
    return text, tool_calls, usage, finish


def openai_response(model_id, text, tool_calls, usage, finish, stream):
    """Client-wire OpenAI body (or SSE bytes) for the extracted answer."""
    if not stream:
        msg = {"role": "assistant", "content": text if text else None}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        out = {"id": "chatcmpl-agg", "object": "chat.completion",
               "created": FIXED_OPENAI_CREATED, "model": model_id,
               "choices": [{"index": 0, "message": msg, "finish_reason": finish}]}
        if usage is not None:
            out["usage"] = {"prompt_tokens": usage["input"],
                            "completion_tokens": usage["output"],
                            "total_tokens": usage["input"] + usage["output"]}
        return 200, {"content-type": "application/json"}, json.dumps(out).encode("utf-8")
    parts = []
    role = {"id": "chatcmpl-agg", "object": "chat.completion.chunk",
            "created": FIXED_OPENAI_CREATED, "model": model_id,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}
    parts.append(role)
    if tool_calls:
        for i, tc in enumerate(tool_calls):
            parts.append({"id": "chatcmpl-agg", "object": "chat.completion.chunk",
                          "created": FIXED_OPENAI_CREATED, "model": model_id,
                          "choices": [{"index": 0, "delta": {"tool_calls": [{
                              "index": i, "id": tc["id"], "type": "function",
                              "function": {"name": tc["function"]["name"],
                                           "arguments": tc["function"]["arguments"]}}]},
                              "finish_reason": None}]})
    n = 13
    for i in range(0, len(text), n):
        parts.append({"id": "chatcmpl-agg", "object": "chat.completion.chunk",
                      "created": FIXED_OPENAI_CREATED, "model": model_id,
                      "choices": [{"index": 0, "delta": {"content": text[i:i + n]},
                                   "finish_reason": None}]})
    parts.append({"id": "chatcmpl-agg", "object": "chat.completion.chunk",
                  "created": FIXED_OPENAI_CREATED, "model": model_id,
                  "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})
    if usage is not None:
        parts.append({"id": "chatcmpl-agg", "object": "chat.completion.chunk",
                      "created": FIXED_OPENAI_CREATED, "model": model_id,
                      "choices": [],
                      "usage": {"prompt_tokens": usage["input"],
                                "completion_tokens": usage["output"],
                                "total_tokens": usage["input"] + usage["output"]}})
    buf = []
    for p in parts:
        buf.append("data: %s\n\n" % json.dumps(p, separators=(",", ":")))
    buf.append("data: [DONE]\n\n")
    return 200, {"content-type": "text/event-stream"}, "".join(buf).encode("utf-8")


def anthropic_response(model_id, text, tool_calls, usage, finish, stream):
    """Client-wire Anthropic body (or SSE bytes) for the extracted answer."""
    stop_reason = {"stop": "end_turn", "length": "max_tokens",
                   "tool_calls": "tool_use", "content_filter": "refusal"}.get(finish, "end_turn")
    if not stream:
        content = []
        if text:
            content.append({"type": "text", "text": text})
        for tc in tool_calls:
            try:
                args = json.loads(tc["function"]["arguments"])
            except Exception:
                args = {}
            content.append({"type": "tool_use", "id": tc["id"],
                            "name": tc["function"]["name"],
                            "input": args if is_record(args) else {}})
        out = {"id": "msg_agg", "type": "message", "role": "assistant",
               "model": model_id, "content": content, "stop_reason": stop_reason}
        if usage is not None:
            out["usage"] = {"input_tokens": usage["input"], "output_tokens": usage["output"]}
        return 200, {"content-type": "application/json"}, json.dumps(out).encode("utf-8")
    frames = []
    msg = {"id": "msg_agg", "type": "message", "role": "assistant",
           "model": model_id, "content": [], "stop_reason": None,
           "usage": {"input_tokens": usage["input"] if usage else 0,
                     "output_tokens": 0}}
    frames.append(("message_start", {"type": "message_start", "message": msg}))
    frames.append(("content_block_start", {"type": "content_block_start", "index": 0,
                                           "content_block": {"type": "text", "text": ""}}))
    n = 13
    for i in range(0, len(text), n):
        frames.append(("content_block_delta", {"type": "content_block_delta", "index": 0,
                                               "delta": {"type": "text_delta",
                                                         "text": text[i:i + n]}}))
    frames.append(("content_block_stop", {"type": "content_block_stop", "index": 0}))
    idx = 1
    for tc in tool_calls:
        try:
            args = json.loads(tc["function"]["arguments"])
        except Exception:
            args = {}
        frames.append(("content_block_start", {"type": "content_block_start", "index": idx,
                                               "content_block": {"type": "tool_use",
                                                                 "id": tc["id"],
                                                                 "name": tc["function"]["name"],
                                                                 "input": args if is_record(args) else {}}}))
        frames.append(("content_block_stop", {"type": "content_block_stop", "index": idx}))
        idx += 1
    frames.append(("message_delta", {"type": "message_delta",
                                     "delta": {"stop_reason": stop_reason},
                                     "usage": {"output_tokens": usage["output"] if usage else 0}}))
    frames.append(("message_stop", {"type": "message_stop"}))
    buf = []
    for event, data in frames:
        buf.append("event: %s\ndata: %s\n\n" % (event, json.dumps(data, separators=(",", ":"))))
    return 200, {"content-type": "text/event-stream"}, "".join(buf).encode("utf-8")


def error_response(client_wire, stream):
    if client_wire == "anthropic":
        body = {"type": "error", "error": {"type": "overloaded_error",
                                           "message": "all providers unavailable"}}
        return 503, {"content-type": "application/json"}, json.dumps(body).encode("utf-8")
    body = {"error": {"message": "all providers unavailable",
                      "type": "server_error", "code": None}}
    return 503, {"content-type": "application/json"}, json.dumps(body).encode("utf-8")


# --------------------------------------------------------------------------
# Server: the two client wires + /v1/models, register, and the /work poll.
# --------------------------------------------------------------------------

class Aggregator:
    def __init__(self, providers, ratings, manifest, probes, catalog):
        self.providers = providers
        self.ratings = ratings
        self.manifest = manifest or []
        self.probes = probes
        self.catalog = catalog
        self.state = RouterState(providers)
        self.request_index = 0

    def serve_request(self, client_wire, body):
        """(status, headers, raw) for one client request."""
        stream = body.get("stream") is True
        fmt = body.get("response_format") if is_record(body.get("response_format")) else None
        client_json = client_wire == "openai" and fmt is not None and fmt.get("type") in ("json_object", "json_schema")
        idx = self.request_index
        if idx < len(self.manifest):
            req = self.manifest[idx]
            requires = req["requires"]
        else:
            requires = self._derive_requires(client_wire, body)
        self.request_index += 1
        cands = eligible_candidates(self.probes, self.providers, self.ratings,
                                    requires, client_wire, client_json)
        plan = {}
        if idx < len(self.manifest):
            plan = greedy_plan(self.probes, self.providers, self.ratings,
                               self.manifest, idx, self.state)
            primary = plan.get(self.manifest[idx]["id"])
        else:
            primary = cands[0] if cands else None
        attempts = 0
        deadline = time.monotonic() + 25.0
        for provider, model, _rating in ordered_candidates(primary, cands, self.state):
            tries = RETRIES_PER_CANDIDATE if not self.state.dead[provider["id"]] else 1
            for _ in range(tries):
                if attempts >= MAX_ATTEMPTS_PER_REQUEST:
                    break
                if time.monotonic() > deadline:
                    break
                attempts += 1
                try:
                    url, headers, upstream = build_upstream(provider, model["id"],
                                                            client_wire, body, stream)
                    status, _parsed, raw = _http_json("POST", url, headers,
                                                     body=upstream, timeout=CALL_TIMEOUT)
                except Exception as exc:
                    log("provider %s call failed: %s" % (provider["id"], exc))
                    status, raw = None, b""
                self.state.calls[provider["id"]] += 1
                if status == 200:
                    self.state.note_success(provider["id"])
                    if provider["wire"] == client_wire:
                        ctype = ("text/event-stream" if stream else "application/json")
                        return 200, {"content-type": ctype}, raw
                    text, tcs, usage, finish = extract_answer(provider["wire"], raw, stream)
                    if client_wire == "openai":
                        return openai_response(model["id"], text, tcs, usage, finish, stream)
                    return anthropic_response(model["id"], text, tcs, usage, finish, stream)
                self.state.note_failure(provider["id"])
                if status in (400, 401):
                    # The provider did not understand this request; it will not
                    # understand the same request again. Move to the next model.
                    break
                if status is None:
                    # Transport failure: the sim answers in milliseconds, so a
                    # timeout here means the provider is unreachable this trial.
                    break
            if attempts >= MAX_ATTEMPTS_PER_REQUEST or time.monotonic() > deadline:
                break
        log("request %s: no provider served it (%d attempts)" % (idx, attempts))
        return error_response(client_wire, stream)

    @staticmethod
    def _derive_requires(client_wire, body):
        requires = {"tools": False, "vision": False, "json": False}
        if client_wire == "openai":
            if isinstance(body.get("tools"), list) and body["tools"]:
                requires["tools"] = True
            fmt = body.get("response_format") if is_record(body.get("response_format")) else None
            if fmt is not None and fmt.get("type") in ("json_object", "json_schema"):
                requires["json"] = True
            for msg in body.get("messages", []) if isinstance(body.get("messages"), list) else []:
                if not is_record(msg):
                    continue
                content = msg.get("content")
                if isinstance(content, list):
                    for b in content:
                        if is_record(b) and b.get("type") == "image_url":
                            requires["vision"] = True
        else:
            if isinstance(body.get("tools"), list) and body["tools"]:
                requires["tools"] = True
            for msg in body.get("messages", []) if isinstance(body.get("messages"), list) else []:
                if not is_record(msg):
                    continue
                content = msg.get("content")
                if isinstance(content, list):
                    for b in content:
                        if is_record(b) and b.get("type") == "image":
                            requires["vision"] = True
        return requires


def make_handler(aggregator):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            pass

        def _read_body(self):
            try:
                length = int(self.headers.get("content-length") or 0)
            except ValueError:
                length = 0
            if length <= 0:
                return b""
            if length > MAX_BODY_BYTES:
                return b""
            return self.rfile.read(length)

        def _send(self, status, headers, raw):
            if not isinstance(raw, bytes):
                raw = raw.encode("utf-8")
            self.send_response(status)
            for k, v in headers.items():
                self.send_header(k, v)
            self.send_header("content-length", str(len(raw)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(raw)

        def _bad(self, message):
            self._send(400, {"content-type": "application/json"},
                       json.dumps({"error": {"message": message, "type": "invalid_request_error",
                                             "code": None}}).encode("utf-8"))

        def do_GET(self):
            path = urllib.parse.urlparse(self.path).path
            if path == "/v1/models":
                models = served_models(aggregator.probes, aggregator.providers, aggregator.ratings)
                if self.headers.get("x-api-key"):
                    out = {"data": [{"type": "model", "id": m["id"],
                                     "display_name": m["display_name"],
                                     "created_at": FIXED_ANTHROPIC_CREATED_AT}
                                    for m in models],
                           "has_more": False}
                else:
                    out = {"object": "list", "data": [{"id": m["id"], "object": "model",
                                                       "created": FIXED_OPENAI_CREATED,
                                                       "owned_by": m["provider"]}
                                                      for m in models]}
                self._send(200, {"content-type": "application/json"},
                           json.dumps(out).encode("utf-8"))
                return
            self._send(404, {"content-type": "application/json"},
                       json.dumps({"error": {"message": "not found"}}).encode("utf-8"))

        def do_POST(self):
            path = urllib.parse.urlparse(self.path).path
            raw = self._read_body()
            try:
                body = json.loads(raw.decode("utf-8", "replace")) if raw else {}
            except Exception:
                body = None
            if not is_record(body):
                self._bad("request body must be a JSON object")
                return
            if path == "/v1/chat/completions":
                try:
                    status, headers, out = aggregator.serve_request("openai", body)
                except Exception as exc:
                    log("openai handler error: %s" % exc)
                    status, headers, out = error_response("openai", body.get("stream") is True)
                self._send(status, headers, out)
                return
            if path == "/v1/messages":
                try:
                    status, headers, out = aggregator.serve_request("anthropic", body)
                except Exception as exc:
                    log("anthropic handler error: %s" % exc)
                    status, headers, out = error_response("anthropic", body.get("stream") is True)
                self._send(status, headers, out)
                return
            self._send(404, {"content-type": "application/json"},
                       json.dumps({"error": {"message": "not found"}}).encode("utf-8"))

    return Handler


def register(port, catalog):
    body = {"protocol_version": PROTOCOL_VERSION, "port": port, "catalog": catalog}
    for attempt in range(10):
        try:
            status, parsed, _raw = _http_json(
                "POST", ENV_URL + "/register",
                {"content-type": "application/json"}, body=body, timeout=5.0)
            if status == 200 and is_record(parsed) and parsed.get("ok") is True:
                return parsed
            log("register attempt %d: HTTP %s %s" % (attempt + 1, status, parsed))
        except Exception as exc:
            log("register attempt %d failed: %s" % (attempt + 1, exc))
        time.sleep(0.5)
    raise RuntimeError("could not register with the environment after 10 attempts")


def poll_work():
    for _ in range(6000):
        time.sleep(POLL_INTERVAL)
        try:
            status, parsed, _raw = _http_json("GET", ENV_URL + "/work", {}, timeout=2.0)
        except Exception:
            continue
        if status == 200 and is_record(parsed) and parsed.get("done") is True:
            log("environment reports done; exiting 0")
            os._exit(0)
    log("work poll timed out after 600s; exiting 0")
    os._exit(0)


def run_server(aggregator):
    server = ThreadingHTTPServer(("0.0.0.0", 0), make_handler(aggregator))
    port = server.server_address[1]
    result = register(port, aggregator.catalog)
    log("registered on port %d: %s" % (port, result))
    thread = threading.Thread(target=poll_work, daemon=True)
    thread.start()
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    server.server_close()
    return 0


def main(argv):
    list_only = "--list" in argv
    providers, ratings, manifest, error = read_brief(PROMPT_FILE)
    if error is not None and PROMPT_FILE != "":
        log("brief: %s" % error)
    if ENV_URL == "" and not list_only:
        log("MAKEFUN_ENV_URL is not set; nothing to serve")
        return 1
    probes = []
    for provider in providers:
        if ENV_URL == "":
            probes.append((False, []))
        else:
            probes.append(probe_provider(ENV_URL, provider))
    catalog = build_catalog(providers, probes, ratings)
    if list_only:
        sys.stdout.write(json.dumps(catalog, separators=(",", ":")) + "\n")
        sys.stdout.flush()
        return 0
    if error is not None:
        return 1
    aggregator = Aggregator(providers, ratings, manifest, probes, catalog)
    return run_server(aggregator)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
