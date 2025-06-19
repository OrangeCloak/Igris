from flask import Flask
import threading
import os
import igris_mongodb

app = Flask(__name__)

@app.route('/')
def home():
    return "🧠 Igris is alive."

# ✅ Run background threads (e.g. Notion syncing)
threading.Thread(target=igris_mongodb.start_background_threads_only, daemon=True).start()

# ✅ Run Telegram bot in a thread too (Render requires Flask to run in main thread)
threading.Thread(target=igris_mongodb.run_telegram_polling, daemon=True).start()

# ✅ Flask must be in the MAIN thread so Render can detect the port
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
