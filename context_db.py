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

def get_context_by_message_id(message_id: int) -> str:
    result = context_collection.find_one({"telegram_message_id": message_id})
    return result["content"] if result else ""


def save_context(user_id, content, telegram_message_id=None):
    context_entry = {
        "user_id": user_id,
        "content": content,
        "timestamp": datetime.now().isoformat()
    }
    if telegram_message_id:
        context_entry["telegram_message_id"] = telegram_message_id
    context_collection.insert_one(context_entry)


def get_context(username: str) -> str | None:
    doc = context_collection.find_one({"username": username})
    return doc["last_message"] if doc else None
