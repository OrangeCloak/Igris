from flask import Flask
import threading
import os
import igris_mongodb

app = Flask(__name__)

@app.route('/')
def home():
    return "🧠 Igris is alive."

# ✅ Run Flask server in a thread
def run_flask():
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

# ✅ Run Flask in background (Render will still detect it)
threading.Thread(target=run_flask, daemon=True).start()

# ✅ Start background threads for Notion, EXP, etc
threading.Thread(target=igris_mongodb.start_background_threads_only, daemon=True).start()

# ✅ Run Telegram polling in main thread (required by PTB)
if __name__ == '__main__':
    igris_mongodb.run_telegram_polling()
