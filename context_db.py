from pymongo import MongoClient
from datetime import datetime, timezone
import os

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "igris"
COLLECTION_NAME = "context_memory"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
context_collection = db[COLLECTION_NAME]

# Ensure TTL index is set (run only once, safe to re-run)
context_collection.create_index("timestamp", expireAfterSeconds=7200)

def save_context(username: str, message: str):
    context_collection.update_one(
        {"username": username},
        {
            "$set": {
                "last_message": message,
                "timestamp": datetime.now(timezone.utc)
            }
        },
        upsert=True
    )

def get_context(username: str) -> str | None:
    doc = context_collection.find_one({"username": username})
    return doc["last_message"] if doc else None
