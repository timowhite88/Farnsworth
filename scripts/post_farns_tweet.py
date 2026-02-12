"""Post FARNS announcement thread to X/Twitter."""
import os
import json
import time
import hmac
import hashlib
import base64
import urllib.parse
import uuid
import httpx


def oauth1_header(method, url, params, consumer_key, consumer_secret, token, token_secret):
    """Generate OAuth 1.0a Authorization header."""
    oauth_params = {
        "oauth_consumer_key": consumer_key,
        "oauth_nonce": uuid.uuid4().hex,
        "oauth_signature_method": "HMAC-SHA1",
        "oauth_timestamp": str(int(time.time())),
        "oauth_token": token,
        "oauth_version": "1.0",
    }

    all_params = {**oauth_params, **params}
    sorted_params = sorted(all_params.items())
    param_string = "&".join(f"{urllib.parse.quote(k, safe='')}={urllib.parse.quote(str(v), safe='')}" for k, v in sorted_params)

    base_string = f"{method.upper()}&{urllib.parse.quote(url, safe='')}&{urllib.parse.quote(param_string, safe='')}"
    signing_key = f"{urllib.parse.quote(consumer_secret, safe='')}&{urllib.parse.quote(token_secret, safe='')}"

    signature = base64.b64encode(
        hmac.new(signing_key.encode(), base_string.encode(), hashlib.sha1).digest()
    ).decode()

    oauth_params["oauth_signature"] = signature
    auth_header = "OAuth " + ", ".join(
        f'{urllib.parse.quote(k, safe="")}="{urllib.parse.quote(v, safe="")}"'
        for k, v in sorted(oauth_params.items())
    )
    return auth_header


def post_tweet(text, reply_to=None):
    """Post a tweet using OAuth 1.0a."""
    url = "https://api.twitter.com/2/tweets"

    consumer_key = os.environ["X_API_KEY"]
    consumer_secret = os.environ["X_API_SECRET"]
    token = os.environ["X_OAUTH1_ACCESS_TOKEN"]
    token_secret = os.environ["X_OAUTH1_ACCESS_SECRET"]

    payload = {"text": text}
    if reply_to:
        payload["reply"] = {"in_reply_to_tweet_id": str(reply_to)}

    auth = oauth1_header("POST", url, {}, consumer_key, consumer_secret, token, token_secret)

    resp = httpx.post(
        url,
        headers={
            "Authorization": auth,
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=30.0,
    )

    if resp.status_code in (200, 201):
        data = resp.json()
        tweet_id = data["data"]["id"]
        print(f"  Posted tweet {tweet_id}: {text[:60]}...")
        return tweet_id
    else:
        print(f"  FAILED ({resp.status_code}): {resp.text[:200]}")
        return None


def main():
    # Thread tweets — no code, just the announcement
    tweets = [
        # Tweet 1 - Hook
        "We just built something that hasn't been done before:\n\nA multi-node AI mesh where GPUs ARE the identity.\n\nNo private keys. No certificates. No PKI.\n\nJust GPU silicon fingerprints + BLAKE3.\n\nWe call it FARNS.\n\n🧵",

        # Tweet 2 - The insight
        "The key insight: different GPU silicon produces different floating-point rounding on identical matrix multiplies.\n\nAn A40 and A6000 running the same torch.mm(A,B) with the same seed produce DIFFERENT results at the bit level.\n\nWe hash that into the node identity. Your GPU IS your credential.",

        # Tweet 3 - 5-layer auth
        "Proof-of-Swarm: 5 layers, zero keys\n\n1. Swarm Seed — shared 256-bit secret\n2. GPU Fingerprint — deterministic CUDA compute\n3. Temporal Hash Mesh — interleaved chains detect divergence\n4. Swarm Quorum — 2+ nodes must verify\n5. Rolling BLAKE3 Seals — per-packet, replay-proof\n\n~200 lines of Python.",

        # Tweet 4 - What it does
        "What does FARNS actually do?\n\nConnect to ONE node, see ALL models across the mesh. Route requests transparently.\n\nWe have 7 bots across 2 GPU servers:\n- Server 1 (A40): 6 models\n- Server 2 (A6000): qwen3-coder-next (80B)\n\nAny client sees all 7. Routing is automatic.",

        # Tweet 5 - Live results
        "Live results from today:\n\nLocal phi4 query through FARNS: 3.9 seconds\n\nRemote 80B qwen3-coder-next (A6000, via SSH tunnel): 57 seconds\n\nBoth through the same mesh connection, same auth, same protocol.\n\nRaw TCP + msgpack + TCP_NODELAY. No HTTP overhead.",

        # Tweet 6 - PRO expansion
        "FARNS also supports dynamic node addition.\n\nPRO users can install FARNS locally:\n- Min specs: RTX 4090+, <30ms latency\n- GPU benchmark + manual approval\n- They contribute compute AND get mesh access\n\nTheir GPU fingerprint becomes their identity. No keys to manage.",

        # Tweet 7 - CTA
        "10 files. ~1200 lines. All async Python.\n\nGPU-as-Identity. Temporal Hash Mesh. Rolling BLAKE3 Seals. Zero key management.\n\nFull technical breakdown on the Colosseum forum.\n\nhttps://ai.farnsworth.cloud\n\n$FARNS",
    ]

    print(f"Posting {len(tweets)}-tweet thread about FARNS...")
    print()

    root_id = None
    last_id = None

    for i, text in enumerate(tweets):
        print(f"Tweet {i+1}/{len(tweets)}:")
        tweet_id = post_tweet(text, reply_to=last_id)

        if tweet_id:
            if root_id is None:
                root_id = tweet_id
            last_id = tweet_id
        else:
            print(f"  Thread broken at tweet {i+1}, stopping.")
            break

        if i < len(tweets) - 1:
            time.sleep(2)  # Rate limit safety

    print()
    if root_id:
        print(f"Thread root: https://x.com/FarnsworthAI/status/{root_id}")
        print(f"Total tweets posted: {tweets.index(text) + 1 if last_id else 0}")
    else:
        print("No tweets posted.")


if __name__ == "__main__":
    main()
