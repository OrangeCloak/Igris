from flask import Flask
import threading
import igris_mongodb
import os

app = Flask(__name__)

@app.route('/')
def home():
    return "🧠 Igris is alive."

# Auto-start Igris in background
def run_bot():
    igris_mongodb.run_igris()

# Start once when the Flask server starts
threading.Thread(target=run_bot, daemon=True).start()

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
