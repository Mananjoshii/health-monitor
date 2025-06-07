# receiver.py (Flask server to receive ESP32 data)

from flask import Flask, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allows Streamlit to access this

buffer = []

@app.route('/postdata', methods=['POST'])
def post_data():
    content = request.get_json()
    if isinstance(content, list):  # batch
        for sample in content:
            buffer.append((sample['ir'], sample['red']))
    elif isinstance(content, dict):  # single
        buffer.append((content['ir'], content['red']))
    while len(buffer) > 1000:
        buffer.pop(0)
    return "OK", 200

@app.route('/reset', methods=['POST'])
def reset_buffer():
    global buffer
    buffer = []
    return {"status": "buffer cleared"}, 200


@app.route('/latest', methods=['GET'])
def get_latest():
    return {'data': buffer[-600:] if len(buffer) >= 600 else buffer}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
