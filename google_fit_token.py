import os
import pickle
import requests
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow

# Scopes required for reading Google Fit step data
SCOPES = ["https://www.googleapis.com/auth/fitness.activity.read"]
TOKEN_PATH = "google_fit_token.pkl"
CLIENT_SECRET_FILE = "client_secrets.json"


# Step 1: Write the client secret from env to file
if not os.path.exists(CLIENT_SECRET_FILE):
    client_secret_json = os.environ.get("GOOGLE_CLIENT_SECRET_JSON")
    if client_secret_json:
        with open(CLIENT_SECRET_FILE, "w") as f:
            f.write(client_secret_json)
    else:
        raise Exception("Missing GOOGLE_CLIENT_SECRET_JSON in Replit secrets.")

def get_google_fit_credentials():
    """
    Returns valid credentials (refreshes if expired).
    Saves the refresh token for long-term reuse.
    """
    creds = None

    if os.path.exists(TOKEN_PATH):
        with open(TOKEN_PATH, "rb") as token_file:
            creds = pickle.load(token_file)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print("[🔁] Refreshing expired token...")
            creds.refresh(Request())
        else:
            print("[🔐] Launching browser to authenticate with Google...")
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, SCOPES)
            creds = flow.run_local_server(port=8080)

        # Save credentials for next run
        with open(TOKEN_PATH, "wb") as token_file:
            pickle.dump(creds, token_file)

    return creds


def get_access_token() -> str:
    creds = get_google_fit_credentials()
    return creds.token


if __name__ == "__main__":
    token = get_access_token()
    print(f"[✅] Access Token: {token}")
