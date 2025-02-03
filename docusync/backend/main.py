from flask import Flask, request, jsonify
import datetime

app = Flask(__name__)

# In‑memory store for version history. Replace with a persistent database in production.
version_history = {}

@app.route('/commit', methods=['POST'])
def commit_version():
    data = request.get_json()
    document_id = data.get('documentId')
    content = data.get('content')
    timestamp = datetime.datetime.utcnow().isoformat() + "Z"
    
    if not document_id or content is None:
        return jsonify({'error': 'Invalid payload'}), 400

    commit_entry = {
        'timestamp': timestamp,
        'content': content
    }
    if document_id not in version_history:
        version_history[document_id] = []
    version_history[document_id].append(commit_entry)
    
    return jsonify({'message': 'Commit successful', 'commit': commit_entry}), 200

@app.route('/history/<document_id>', methods=['GET'])
def get_history(document_id):
    history = version_history.get(document_id, [])
    return jsonify({'documentId': document_id, 'history': history}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
