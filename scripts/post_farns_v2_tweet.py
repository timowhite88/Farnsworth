"""Post FARNS v2.0 announcement thread to X/Twitter."""
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
    tweets = [
        # Tweet 1 - Hook
        "FARNS v2.0 just dropped.\n\n24 hours after v1.0, we added 4 new systems to our GPU-as-Identity mesh:\n\n- Proof-of-Inference\n- Latent Space Routing\n- GPU-Signed Model Attestation\n- Swarm Memory Crystallization\n\nAll tested end-to-end. All live now.\n\nThread:",

        # Tweet 2 - Proof-of-Inference
        "1/ Proof-of-Inference Consensus\n\nProof-of-Work wastes compute. Proof-of-Stake wastes capital.\n\nProof-of-Inference does USEFUL WORK.\n\nMultiple nodes run the same prompt, each creates a hardware-bound attestation:\n\nBLAKE3(gpu_fingerprint + output_hash + timestamp)\n\n2/3+ must agree = VERIFIED output.",

        # Tweet 3 - PoI live result
        "We tested it live:\n\nPhi4 on NVIDIA A40 answered \"4\" to \"What is 2+2?\"\n\nAttestation seal: 7fbc0fc952b2fb8e...\nOutput hash: e67a9c4536256f1e\nInference: 411ms\n\nThat seal is unforgeable without the physical GPU silicon.\n\nCryptographically verifiable AI computation.",

        # Tweet 4 - Latent Space Routing
        "2/ Latent Space Routing\n\nThe mesh now UNDERSTANDS what it routes.\n\nCode question -> qwen3-coder (80B)\nMath -> deepseek-r1\nCreative writing -> llama3\nFactual -> gemma2\nMultilingual -> qwen2.5\n\n6/6 correct. One packet, no model selection.\n\nThe first protocol-level semantic router for AI meshes.",

        # Tweet 5 - Attestation
        "3/ GPU-Signed Model Attestation\n\nEvery model that loads gets a hardware-signed certificate:\n\nBLAKE3(gpu_fp + model_weights + timestamp + identity + chain_prev)\n\nChained seals form an unbreakable provenance chain.\n\nProves WHAT model runs on WHICH GPU at WHAT time.\n\nSwap a model = chain breaks.",

        # Tweet 6 - Swarm Memory
        "4/ Swarm Memory Crystallization\n\nRAG is retrieval. Vector DBs are storage.\n\nSwarm Memory is CONSENSUS-VERIFIED knowledge.\n\nMultiple models independently verify a fact. 2/3+ agree = crystallized. Hallucinations get rejected.\n\nThe memory is self-healing. No single node controls it.",

        # Tweet 7 - CTA
        "FARNS v2.0:\n\n14 files. ~2000 lines. All async Python.\n\n- Proof-of-Inference (verifiable AI compute)\n- Latent Routing (semantic mesh routing)\n- Model Attestation (hardware provenance)\n- Swarm Memory (consensus knowledge)\n\nBuilt on GPU-as-Identity. Zero key management.\n\nhttps://ai.farnsworth.cloud\n\n$FARNS",
    ]

    print(f"Posting {len(tweets)}-tweet thread about FARNS v2.0...")
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
            time.sleep(2)

    print()
    if root_id:
        print(f"Thread root: https://x.com/FarnsworthAI/status/{root_id}")
        print(f"Total tweets posted: {i + 1 if last_id else 0}")
    else:
        print("No tweets posted.")


if __name__ == "__main__":
    main()
