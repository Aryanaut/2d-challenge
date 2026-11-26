"""Generic call to the Lab 6 scene-understanding gateway through the course's own client.py.

    python client.py --server <url> --group <groupNN> --image <jpg> --prompt-file <txt>

Returns the raw model text plus, if the model answered in JSON, the parsed object.
"""

import json
import re
import subprocess
import time

GROUP_RE = re.compile(r"^group(0[1-9]|1[0-6])$")


def first_json_object(text: str):
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def parse_model_text(result):
    """The model's answer as a dict if it is JSON (also inside ```json fences), else None."""
    if isinstance(result, dict):
        return result
    return first_json_object(re.sub(r"```(?:json)?", "", str(result)))


def parse_client_output(text: str) -> dict:
    payload = first_json_object(text)
    if payload is None:
        return {"ok": False, "error": f"no JSON in client output: {text.strip()[:200]}"}
    if not payload.get("success", False):
        err = payload.get("error") or {}
        if isinstance(err, dict):
            return {"ok": False, "error": f"{err.get('code', 'unknown')}: {err.get('message', '')}"}
        return {"ok": False, "error": str(err)}
    result = payload.get("result", "")
    return {"ok": True, "result": result if isinstance(result, str) else json.dumps(result),
            "parsed": parse_model_text(result),
            "server_time_s": payload.get("processing_time_s")}


def query(client_python, client_script, server, group, image_path, prompt_path, timeout_s=20.0) -> dict:
    t0 = time.monotonic()
    cmd = [client_python, client_script, "--server", server, "--group", group,
           "--image", image_path, "--prompt-file", prompt_path]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        out = parse_client_output(proc.stdout + "\n" + proc.stderr)
    except subprocess.TimeoutExpired:
        out = {"ok": False, "error": f"timeout after {timeout_s:.0f} s"}
    except OSError as e:
        out = {"ok": False, "error": f"could not run client: {e}"}
    out["latency_s"] = round(time.monotonic() - t0, 2)
    return out


def config_problems(group, client_script) -> list:
    import os
    problems = []
    if not GROUP_RE.match(group or ""):
        problems.append(f"group '{group}' is not group01..group16 (server answers invalid_group)")
    if not os.path.isfile(client_script):
        problems.append(f"client_script not found: {client_script}")
    return problems
