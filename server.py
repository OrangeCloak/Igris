import sys
sys.stdout.reconfigure(line_buffering=True)  # ✅ Enable print() flushing

from flask import Flask
import threading
import os
import igris_mongodb

app = Flask(__name__)

@app.route('/')
def home():
    return "🧠 Igris is alive."

def run_flask():
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)

if __name__ == "__main__":
    threading.Thread(target=run_flask, daemon=True).start()
    threading.Thread(target=igris_mongodb.start_background_threads_only, daemon=True).start()
    igris_mongodb.run_telegram_polling()
