from flask import Flask
from main import run_igris  # Assume your main entrypoint is here

app = Flask(__name__)

@app.route('/')
def home():
    return "Igris is running."

@app.route('/start')
def start():
    run_igris()
    return "Igris Started"

if __name__ == '__main__':
    app.run()
