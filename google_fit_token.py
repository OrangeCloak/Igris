import os
import pickle
import base64
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ["https://www.googleapis.com/auth/fitness.activity.read"]
TOKEN_PATH = "google_fit_token.pkl"
CLIENT_SECRET_FILE = "client_secrets.json"  # Only needed locally

def ensure_token_file():
    if os.path.exists(TOKEN_PATH):
        return

    # If not running locally, try to decode from env var
    token_b64 = os.getenv("GOOGLE_FIT_TOKEN_B64")
    if token_b64:
        print("[🔓] Decoding token from environment...")
        with open(TOKEN_PATH, "wb") as f:
            f.write(base64.b64decode(token_b64))
    else:
        raise FileNotFoundError(
            "❌ google_fit_token.pkl not found and no base64 backup in environment.\n"
            "Run this script locally to generate a new token."
        )

def get_google_fit_credentials():
    ensure_token_file()

    creds = None
    if os.path.exists(TOKEN_PATH):
        with open(TOKEN_PATH, "rb") as token_file:
            creds = pickle.load(token_file)

    # Refresh or start auth flow
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print("[🔁] Refreshing expired token...")
            creds.refresh(Request())

            # Save the refreshed token
            try:
                with open(TOKEN_PATH, "wb") as token_file:
                    pickle.dump(creds, token_file)
                print("[✅] Token refreshed and updated on disk (temporary on Render).")
            except Exception as e:
                print(f"[⚠️] Failed to write refreshed token: {e}")
        else:
            if not os.path.exists(CLIENT_SECRET_FILE):
                raise FileNotFoundError("Missing 'client_secrets.json'. Required for first-time auth.")
            print("[🔐] Launching browser for local authentication...")
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, SCOPES)
            creds = flow.run_local_server(port=8080)

            with open(TOKEN_PATH, "wb") as token_file:
                pickle.dump(creds, token_file)
            print("[✅] New token generated and saved.")

    return creds

def get_access_token() -> str:
    creds = get_google_fit_credentials()
    return creds.token

if __name__ == "__main__":
    token = get_access_token()
    print(f"[✅] Access Token: {token}")
