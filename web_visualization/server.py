import os
import logging
from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")

# State to store the most recent data
current_state = {
    'init': None,
    'last_event': None
}

# Reduce werkzeug logging to prevent terminal clutter
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/init', methods=['POST'])
def init_api():
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    current_state['init'] = data
    # Broadcast to all connected clients
    socketio.emit('init_data', data)
    return jsonify({"status": "success", "message": "Initialization data received"}), 200

@app.route('/api/ping', methods=['POST'])
def ping_api():
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    current_state['last_event'] = data
    # Broadcast to all connected clients
    socketio.emit('ping_data', data)
    return jsonify({"status": "success", "message": "Ping data received"}), 200

@socketio.on('connect')
def handle_connect():
    # Send current state to newly connected client
    if current_state['init']:
        socketio.emit('init_data', current_state['init'])
    if current_state['last_event']:
        socketio.emit('ping_data', current_state['last_event'])

def start_server(host='0.0.0.0', port=5000):
    print(f"[Web Visualization] Starting server on http://{host}:{port}")
    socketio.run(app, host=host, port=port, allow_unsafe_werkzeug=True)

if __name__ == '__main__':
    start_server()
