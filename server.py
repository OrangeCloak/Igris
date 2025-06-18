from flask import Flask
import threading
import igris_mongodb

app = Flask(__name__)

@app.route('/')
def home():
    return "🧠 Igris is alive."

@app.route('/start')
def start_igris():
    # Start Telegram polling + background tasks in a thread
    def run_bot():
        igris_mongodb.run_igris()

    thread = threading.Thread(target=run_bot)
    thread.daemon = True
    thread.start()

    return "✅ Igris started."

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10000)
