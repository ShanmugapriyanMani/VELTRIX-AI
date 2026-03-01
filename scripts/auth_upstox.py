"""
Upstox Authentication Helper.

Fully automatic OAuth2 flow:
  python scripts/auth_upstox.py
  → Opens browser, starts local server, captures code, exchanges for token.

Manual code exchange:
  python scripts/auth_upstox.py --code YOUR_AUTH_CODE
"""

import os
import sys
import threading
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

# Project root
_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, _root)

# Load .env + .env.{stage} via env_loader (auto-loads on import)
from src.config.env_loader import get_config
get_config()  # Trigger env loading

import yaml
from src.data.fetcher import build_auth_from_config


class _CallbackHandler(BaseHTTPRequestHandler):
    """HTTP handler to capture the OAuth2 redirect callback."""
    auth_code = None

    def do_GET(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)

        if "code" in params:
            _CallbackHandler.auth_code = params["code"][0]
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(
                b"<html><body style='font-family:sans-serif;text-align:center;padding:50px'>"
                b"<h1>Authentication Successful!</h1>"
                b"<p>You can close this tab and return to the terminal.</p>"
                b"</body></html>"
            )
        else:
            self.send_response(400)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            error = params.get("error", ["unknown"])[0]
            self.wfile.write(
                f"<html><body><h1>Error: {error}</h1></body></html>".encode()
            )

    def log_message(self, format, *args):
        pass  # Suppress server logs


def _auto_auth(auth):
    """
    Automatic OAuth2 flow:
    1. Start a local HTTP server on the redirect URI port
    2. Open the login URL in the browser
    3. Wait for the callback with the auth code
    4. Exchange the code for an access token
    """
    from urllib.parse import urlparse as _urlparse
    parsed = _urlparse(auth.redirect_uri)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 5000

    server = HTTPServer((host, port), _CallbackHandler)
    server.timeout = 120  # 2 minute timeout

    login_url = auth.get_login_url()
    print(f"\nOpening browser for Upstox login...")
    print(f"URL: {login_url}\n")
    print(f"Waiting for callback on {auth.redirect_uri} ...")
    print("(If browser doesn't open, copy the URL above manually)\n")

    # Open browser in a separate thread so server starts immediately
    threading.Timer(0.5, webbrowser.open, args=[login_url]).start()

    # Wait for the callback (handle_request blocks until one request comes in)
    while _CallbackHandler.auth_code is None:
        server.handle_request()

    server.server_close()
    code = _CallbackHandler.auth_code
    print(f"Authorization code received!")
    print(f"Exchanging code for access token...\n")

    token = auth.exchange_code_for_token(code)
    print(f"Access token saved successfully!")
    print(f"Token (first 20 chars): {token[:20]}...")
    print(f"Saved to: {auth.access_token_path}")
    print(f"\nLIVE token valid until end of day. Re-run this script daily.")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Upstox Authentication Helper")
    parser.add_argument("--code", help="Authorization code from redirect URL (manual flow)")
    args = parser.parse_args()

    with open(os.path.join(_root, "config/config.yaml")) as f:
        config = yaml.safe_load(f)

    auth = build_auth_from_config(config)
    print(f"\nTrading Mode: LIVE")
    print(f"API Key: {auth.api_key[:8]}..." if auth.api_key else "API Key: (empty!)")

    # ── Option 1: Manual code exchange ──
    if args.code:
        token = auth.exchange_code_for_token(args.code)
        print(f"\nAccess token saved successfully!")
        print(f"Token (first 20 chars): {token[:20]}...")
        print(f"Saved to: {auth.access_token_path}")
        return

    # ── No args: Auto OAuth2 flow ──
    _auto_auth(auth)


if __name__ == "__main__":
    main()
