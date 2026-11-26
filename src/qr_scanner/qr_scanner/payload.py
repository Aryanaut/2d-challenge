"""Turn the raw QR text into its meaning ("decrypt").

The arena codes seen so far contain single letters ("I", "J"). What they stand for comes from
the mission brief, so the meaning is configurable instead of hard-coded:

  decode_mode: raw     -> text as-is
               base64  -> base64-decode
               caesar  -> shift letters back by caesar_shift (A..Z, a..z)
               json    -> parse JSON, return it compactly
               auto    -> try json, then base64 (only if it decodes to printable text), else raw
  key_map:     ["I:storeroom", "J:kitchen"]  applied after decoding; whole-text match first,
               then per character (so "IJ" -> "storeroom kitchen")
"""

import base64
import binascii
import json
import string


def parse_key_map(entries) -> dict:
    mapping = {}
    for e in entries or []:
        if ":" in e:
            k, v = e.split(":", 1)
            mapping[k.strip()] = v.strip()
    return mapping


def caesar(text: str, shift: int) -> str:
    out = []
    for ch in text:
        if ch in string.ascii_uppercase:
            out.append(chr((ord(ch) - 65 - shift) % 26 + 65))
        elif ch in string.ascii_lowercase:
            out.append(chr((ord(ch) - 97 - shift) % 26 + 97))
        else:
            out.append(ch)
    return "".join(out)


def _try_base64(text: str):
    s = text.strip()
    if len(s) < 4 or len(s) % 4:
        return None
    try:
        raw = base64.b64decode(s, validate=True).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError, ValueError):
        return None
    return raw if raw and all(c.isprintable() or c.isspace() for c in raw) else None


def decode_payload(text: str, mode: str = "raw", caesar_shift: int = 0, key_map=None) -> str:
    mode = (mode or "raw").lower()
    out = text
    if mode == "base64":
        out = _try_base64(text) or text
    elif mode == "caesar":
        out = caesar(text, caesar_shift)
    elif mode == "json":
        try:
            out = json.dumps(json.loads(text), separators=(",", ":"))
        except json.JSONDecodeError:
            out = text
    elif mode == "auto":
        try:
            out = json.dumps(json.loads(text), separators=(",", ":"))
        except json.JSONDecodeError:
            out = _try_base64(text) or text

    if key_map:
        if out.strip() in key_map:
            return key_map[out.strip()]
        if all(ch in key_map or ch.isspace() for ch in out) and any(ch in key_map for ch in out):
            return " ".join(key_map[ch] for ch in out if ch in key_map)
    return out
