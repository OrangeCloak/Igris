import os
import pickle
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ["https://www.googleapis.com/auth/fitness.activity.read"]
TOKEN_PATH = "google_fit_token.pkl"
CLIENT_SECRET_FILE = "client_secrets.json"

if not os.path.exists(CLIENT_SECRET_FILE):
    raise FileNotFoundError("Missing 'client_secrets.json'. Place it in your project folder.")

def get_google_fit_credentials():
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

        with open(TOKEN_PATH, "wb") as token_file:
            pickle.dump(creds, token_file)

    return creds

def get_access_token() -> str:
    creds = get_google_fit_credentials()
    return creds.token

if __name__ == "__main__":
    token = get_access_token()
    print(f"[✅] Access Token: {token}")
