#!/usr/bin/env python3
"""Password-protect the report pages (StatiCrypt-style, no server needed).

Each *.html in the report dir is replaced -- just before git commit -- by a small
wrapper page holding the AES-256-GCM ciphertext of the original. The viewer types
the password once; the page derives the key with PBKDF2-SHA256 (600k iterations,
WebCrypto, no external JS) and decrypts in the browser. With "remember me" the
DERIVED KEYS (never passwords) are kept in localStorage; on load a page tries every
remembered key, so each password group unlocks together on that browser.

Passwords (.pagepass, gitignored) -- one password per page is supported:
    jupiter                       # single line          -> one password for all
or a mapping:
    default: alphago              # every page not listed below
    spread.html: jupiter          # per-page overrides
    plot.html: jupiter

.pagesalt (gitignored) holds the 16-byte site salt (hex), created on first run.

Idempotent + rotation-aware: an encrypted page embeds pwid = SHA256(salt|password)
[:8]. If the page's assigned password still matches its pwid it is skipped (no git
churn); if not (password changed / template upgrade), it is decrypted with any of
the configured passwords and re-encrypted with the assigned one.
"""
from __future__ import annotations

import base64
import glob
import hashlib
import os
import re
import secrets

from Crypto.Cipher import AES
from Crypto.Protocol.KDF import PBKDF2
from Crypto.Hash import SHA256

MARKER_PREFIX = "<!--pageenc-v1"          # any version of our wrapper
ITERATIONS = 600_000

TEMPLATE = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Protected report</title><style>
body{{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:0;
     background:radial-gradient(1200px 600px at 20% -10%,#1e293b,#0f172a);color:#e2e8f0;
     min-height:100vh;display:flex;align-items:center;justify-content:center}}
.box{{background:#1e293b;border:1px solid #334155;border-radius:16px;padding:36px 40px;
     box-shadow:0 10px 30px rgba(0,0,0,.35);max-width:360px;width:90%}}
h1{{font-size:18px;margin:0 0 6px}}p{{color:#94a3b8;font-size:13px;margin:0 0 18px}}
input[type=password]{{width:100%;padding:10px 12px;border-radius:8px;border:1px solid #334155;
     background:#0f172a;color:#e2e8f0;font-size:14px;outline:none}}
input[type=password]:focus{{border-color:#38bdf8}}
label{{display:flex;gap:8px;align-items:center;color:#94a3b8;font-size:12.5px;margin:12px 0}}
button{{width:100%;padding:10px;border:0;border-radius:8px;background:#38bdf8;color:#0f172a;
     font-weight:700;font-size:14px;cursor:pointer}}button:hover{{background:#7dd3fc}}
.err{{color:#f87171;font-size:12.5px;min-height:16px;margin-top:10px}}
.spin{{color:#94a3b8;font-size:12.5px;min-height:16px;margin-top:10px}}
</style></head><body>
<div class="box"><h1>&#128274; Protected report</h1>
<p>Enter the password to view this page.</p>
<form id="f"><input type="password" id="pw" placeholder="Password" autofocus>
<label><input type="checkbox" id="rm" checked> remember me on this browser</label>
<button type="submit">Unlock</button><div class="err" id="err"></div>
<div class="spin" id="spin"></div></form></div>
<script>
const SALT_HEX="{salt_hex}", ITER={iterations}, PAYLOAD_B64="{payload_b64}";
const KEYSTORE="pageenc_keys_v1", LEGACY="pageenc_key_v1";
function hex2buf(h){{const b=new Uint8Array(h.length/2);
  for(let i=0;i<b.length;i++)b[i]=parseInt(h.substr(2*i,2),16);return b;}}
function b642buf(s){{const bin=atob(s);const b=new Uint8Array(bin.length);
  for(let i=0;i<bin.length;i++)b[i]=bin.charCodeAt(i);return b;}}
function buf2hex(b){{return Array.from(new Uint8Array(b)).map(x=>x.toString(16).padStart(2,"0")).join("");}}
function storedKeys(){{
  let ks=[];try{{ks=JSON.parse(localStorage.getItem(KEYSTORE)||"[]");}}catch(e){{}}
  const old=localStorage.getItem(LEGACY);if(old&&!ks.includes(old))ks.push(old);
  return ks;}}
function rememberKey(hex){{
  const ks=storedKeys();if(!ks.includes(hex))ks.push(hex);
  localStorage.setItem(KEYSTORE,JSON.stringify(ks.slice(-8)));}}
async function deriveKey(pw){{
  const mat=await crypto.subtle.importKey("raw",new TextEncoder().encode(pw),"PBKDF2",false,["deriveBits"]);
  return await crypto.subtle.deriveBits({{name:"PBKDF2",hash:"SHA-256",salt:hex2buf(SALT_HEX),iterations:ITER}},mat,256);
}}
async function tryDecrypt(rawKeyBits){{
  const data=b642buf(PAYLOAD_B64);
  const iv=data.slice(0,12), ct=data.slice(12);
  const key=await crypto.subtle.importKey("raw",rawKeyBits,"AES-GCM",false,["decrypt"]);
  const pt=await crypto.subtle.decrypt({{name:"AES-GCM",iv:iv}},key,ct);
  const html=new TextDecoder().decode(pt);
  document.open();document.write(html);document.close();
}}
window.addEventListener("load",async()=>{{
  for(const k of storedKeys()){{
    try{{await tryDecrypt(hex2buf(k));return;}}catch(e){{}}
  }}
}});
document.getElementById("f").addEventListener("submit",async ev=>{{
  ev.preventDefault();
  const err=document.getElementById("err"),spin=document.getElementById("spin");
  err.textContent="";spin.textContent="deriving key\\u2026";
  try{{
    const bits=await deriveKey(document.getElementById("pw").value);
    spin.textContent="decrypting\\u2026";
    if(document.getElementById("rm").checked)rememberKey(buf2hex(bits));
    await tryDecrypt(bits);
  }}catch(e){{spin.textContent="";err.textContent="Wrong password.";}}
}});
</script></body></html>"""


def _pwid(salt: bytes, password: str) -> str:
    return hashlib.sha256(salt + password.encode()).hexdigest()[:8]


def load_passwords(dir_: str) -> tuple[str, dict[str, str]]:
    """(default_password, {filename: password}) from .pagepass."""
    path = os.path.join(dir_, ".pagepass")
    if not os.path.exists(path):
        raise SystemExit(f"[pageenc] missing {path} -- create it with the page password")
    default, per_page = None, {}
    lines = [ln.strip() for ln in open(path) if ln.strip() and not ln.strip().startswith("#")]
    for ln in lines:
        if ":" in ln:
            name, pw = ln.split(":", 1)
            name, pw = name.strip(), pw.strip()
            if name == "default":
                default = pw
            else:
                per_page[name] = pw
        elif len(lines) == 1:
            default = ln                      # single bare line -> one password for all
        else:
            raise SystemExit(f"[pageenc] bad .pagepass line (want 'name: password'): {ln!r}")
    if default is None:
        raise SystemExit("[pageenc] .pagepass needs a 'default: <password>' line")
    return default, per_page


def _load_salt(dir_: str) -> bytes:
    salt_path = os.path.join(dir_, ".pagesalt")
    if os.path.exists(salt_path):
        return bytes.fromhex(open(salt_path).read().strip())
    salt = secrets.token_bytes(16)
    with open(salt_path, "w") as f:
        f.write(salt.hex())
    print(f"[pageenc] generated site salt -> {salt_path}")
    return salt


def _derive(password: str, salt: bytes, cache: dict) -> bytes:
    if password not in cache:
        cache[password] = PBKDF2(password, salt, dkLen=32, count=ITERATIONS,
                                 hmac_hash_module=SHA256)
    return cache[password]


def decrypt_wrapper(page: str, candidates: list[bytes]):
    """Recover the plaintext of an encrypted wrapper page, or None."""
    m = re.search(r'PAYLOAD_B64="([^"]+)"', page)
    if not m:
        return None
    payload = base64.b64decode(m.group(1))
    nonce, ct, tag = payload[:12], payload[12:-16], payload[-16:]
    for key in candidates:
        try:
            return AES.new(key, AES.MODE_GCM, nonce=nonce).decrypt_and_verify(ct, tag)
        except ValueError:
            continue
    return None


def encrypt_dir(dir_: str) -> None:
    default_pw, per_page = load_passwords(dir_)
    salt = _load_salt(dir_)
    cache: dict[str, bytes] = {}
    all_pws = [default_pw] + list(per_page.values())
    done, skipped, failed = [], 0, []
    for path in sorted(glob.glob(os.path.join(dir_, "*.html"))):
        name = os.path.basename(path)
        pw = per_page.get(name, default_pw)
        key = _derive(pw, salt, cache)
        want_id = _pwid(salt, pw)
        raw = open(path, "rb").read()
        head = raw[:200].decode("utf-8", "ignore")
        if head.startswith(MARKER_PREFIX):
            if f"pwid={want_id}" in head:
                skipped += 1                  # already encrypted with the right password
                continue
            # password changed or old template -> decrypt with any known password
            plain = decrypt_wrapper(raw.decode("utf-8", "ignore"),
                                    [_derive(p, salt, cache) for p in all_pws])
            if plain is None:
                failed.append(name)
                continue
            raw = plain
        nonce = secrets.token_bytes(12)
        cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
        ct, tag = cipher.encrypt_and_digest(raw)
        page = (f"{MARKER_PREFIX} pwid={want_id}-->"
                + TEMPLATE.format(salt_hex=salt.hex(), iterations=ITERATIONS,
                                  payload_b64=base64.b64encode(nonce + ct + tag).decode()))
        with open(path, "w") as f:
            f.write(page)
        done.append(name)
    print(f"[pageenc] encrypted {len(done)} page(s) ({skipped} already current): "
          + ", ".join(done))
    if failed:
        print("[pageenc] WARNING could not re-key (no configured password decrypts): "
              + ", ".join(failed))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()
    encrypt_dir(args.dir)
