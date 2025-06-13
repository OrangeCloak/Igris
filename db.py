import os
import datetime
import pymongo
from dotenv import load_dotenv

load_dotenv("api.env")

MONGO_URI = os.environ.get("MONGO_URI")
DB_NAME = os.environ.get("MONGO_DB_NAME", "igris_db")
COLLECTION_NAME = os.environ.get("MONGO_COLLECTION_NAME", "exp_logs")

client = pymongo.MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]



def save_entry(entry: dict):
    collection.insert_one(entry)
    print(f"[DB] Saved entry for user: {entry.get('user')}")

def load_data(limit: int = 50):
    data = list(collection.find().sort("timestamp", -1).limit(limit))
    return data[::-1]  # reverse to get chronological order

def get_unsynced_entries():
    return list(collection.find({"status": "unsync"}))

def mark_entry_as_synced(entry_id: str):
    collection.update_one({"id": entry_id}, {"$set": {"status": "sync"}})
