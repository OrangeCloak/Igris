from flask import Flask
import threading
import os
import igris_mongodb

app = Flask(__name__)

@app.route('/')
def home():
    return "🧠 Igris is alive."

# ✅ Start background threads in another thread
def start_background_threads():
    igris_mongodb.start_background_threads_only()

# ✅ Start background threads in separate thread
threading.Thread(target=start_background_threads, daemon=True).start()

# ✅ Run bot in the MAIN thread (no threading)
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    # Start Telegram bot polling — must be main thread
    igris_mongodb.run_telegram_polling()
    # Start Flask server (optional, since Render needs this to expose HTTP)
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
