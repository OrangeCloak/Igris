import os
import datetime
import pymongo
from dotenv import load_dotenv

load_dotenv("api.env")

# Unified DB and collections
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "igris"  # 👈 unified DB

client = pymongo.MongoClient(MONGO_URI)
db = client[DB_NAME]

# Collections
exp_logs = db["exp_logs"]
context_memory = db["context_memory"]


# ------------------------- EXP LOG HANDLING -------------------------

def save_entry(entry: dict):
    exp_logs.insert_one(entry)
    print(f"[DB] Saved entry for user: {entry.get('user')}")


def load_data(limit: int = 50):
    data = list(exp_logs.find().sort("timestamp", -1).limit(limit))
    return data[::-1]  # reverse to get chronological order


def get_unsynced_entries():
    return list(exp_logs.find({"status": "unsync"}))


def mark_entry_as_synced(entry_ids: list):
    result = exp_logs.update_many(
        {"id": {"$in": entry_ids}},
        {"$set": {"status": "sync"}}
    )
    print(f"[🔄] Synced {result.modified_count}/{len(entry_ids)} tasks.")


def auto_cleanup_if_doc_count_exceeds(limit: int = 5000, keep_latest: int = 1000):
    """
    Deletes oldest entries if total document count exceeds `limit`.
    Keeps only the most recent `keep_latest` entries.
    """
    try:
        total_docs = exp_logs.count_documents({})

        if total_docs > limit:
            to_delete = total_docs - keep_latest
            old_docs = exp_logs.find().sort("timestamp", 1).limit(to_delete)
            ids_to_delete = [doc["_id"] for doc in old_docs]

            result = exp_logs.delete_many({"_id": {"$in": ids_to_delete}})
            print(f"[🧹] Deleted {result.deleted_count} old entries (doc count > {limit})")
        else:
            print(f"[✅] Document count within limit: {total_docs}")
    except Exception as e:
        print(f"[ERROR] Cleanup failed: {e}")


# ------------------------- CONTEXT MEMORY -------------------------

def get_context_for_user(user: str):
    doc = context_memory.find_one({"user": user})
    return doc.get("context", []) if doc else []


def save_context_for_user(user: str, context: list):
    context_memory.update_one(
        {"user": user},
        {"$set": {
            "context": context,
            "last_updated": datetime.datetime.utcnow()
        }},
        upsert=True
    )
    print(f"[💬] Context updated for user: {user}")


def delete_context_for_user(user: str):
    result = context_memory.delete_one({"user": user})
    if result.deleted_count:
        print(f"[🗑️] Deleted context for user: {user}")
