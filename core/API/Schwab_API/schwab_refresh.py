# schwab_refresh.py

import base64
import json
import time
import requests

CLIENT_ID = "DodBaa8AR1twJGwT5srKqFUdl6SepUnCwck38IGFlxc8QVt9"
CLIENT_SECRET = "GAq7IMKHL5AiIJrZunI13WP7z7nGgAy0USgUxB0zZA3hTUP260kAJidKxjsAETBj"

TOKEN_URL = "https://api.schwabapi.com/v1/oauth/token"
TOKENS_FILE = "schwab_tokens.json"


def load_tokens():
    with open(TOKENS_FILE, "r") as f:
        return json.load(f)


def save_tokens(tokens: dict):
    # add a timestamp so you know when they were saved
    tokens["saved_at"] = int(time.time())
    with open(TOKENS_FILE, "w") as f:
        json.dump(tokens, f, indent=2)


def refresh_tokens():
    tokens = load_tokens()
    refresh_token = tokens.get("refresh_token")
    if not refresh_token:
        raise RuntimeError("No refresh_token found in schwab_tokens.json")

    credentials = f"{CLIENT_ID}:{CLIENT_SECRET}"
    basic_auth = base64.b64encode(credentials.encode("utf-8")).decode("utf-8")

    headers = {
        "Authorization": f"Basic {basic_auth}",
        "Content-Type": "application/x-www-form-urlencoded",
    }

    data = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
    }

    resp = requests.post(TOKEN_URL, headers=headers, data=data)

    if resp.status_code != 200:
        print("❌ Error refreshing tokens:")
        print("Status:", resp.status_code)
        print("Body:", resp.text)
        raise SystemExit(1)

    new_tokens = resp.json()
    save_tokens(new_tokens)
    print("✅ Tokens refreshed and saved.")
    return new_tokens


if __name__ == "__main__":
    refresh_tokens()
