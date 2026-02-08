import os
import json
import zipfile
import io
from flask import Flask, send_file, request, jsonify, render_template

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
METADATA_FILE = os.path.join(BASE_DIR, "metadata.json")
SESSION_AUDIO_MAP_FILE = os.path.join(BASE_DIR, "session_audio", "session_audio_map.json")
EXPORT_DIR = os.path.join(BASE_DIR, "exports")
EXPORT_MANIFEST = os.path.join(EXPORT_DIR, "manifest.json")

# New Output Location
SELECTED_DATA_DIR = os.path.join(BASE_DIR, "selected_data")
LABELS_FILE = os.path.join(SELECTED_DATA_DIR, "labels.json")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/metadata')
def get_metadata():
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'r') as f:
            return jsonify(json.load(f))
    return jsonify([])

@app.route('/api/session_audio_map')
def get_session_audio_map():
    if os.path.exists(SESSION_AUDIO_MAP_FILE):
        with open(SESSION_AUDIO_MAP_FILE, 'r') as f:
            return jsonify(json.load(f))
    return jsonify({})

@app.route('/api/labels')
def get_labels():
    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE, 'r') as f:
            return jsonify(json.load(f))
    return jsonify({})

@app.route('/api/save', methods=['POST'])
def save_labels():
    data = request.json
    
    # Ensure folder exists
    if not os.path.exists(SELECTED_DATA_DIR):
        os.makedirs(SELECTED_DATA_DIR)
        
    with open(LABELS_FILE, 'w') as f:
        json.dump(data, f, indent=2)
    return jsonify({"status": "ok", "path": LABELS_FILE})

@app.route('/api/export', methods=['POST'])
def run_export():
    """Run the export process."""
    try:
        from export_labels import process_exports
        manifest = process_exports()
        return jsonify({"status": "ok", "manifest": manifest})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/export/manifest')
def get_export_manifest():
    """Get the export manifest."""
    if os.path.exists(EXPORT_MANIFEST):
        with open(EXPORT_MANIFEST, 'r') as f:
            return jsonify(json.load(f))
    return jsonify({"video": [], "audio": [], "stats": {}})

@app.route('/api/export/download/<path:subpath>')
def download_export_file(subpath):
    """Download a single exported file."""
    filepath = os.path.join(EXPORT_DIR, subpath)
    if not os.path.exists(filepath):
        return "File not found", 404
    return send_file(filepath, as_attachment=True)

@app.route('/api/export/download-all')
def download_all_exports():
    """Download all exports as a ZIP file."""
    if not os.path.exists(EXPORT_DIR):
        return "No exports found", 404
    
    # Create ZIP in memory
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(EXPORT_DIR):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, EXPORT_DIR)
                zf.write(file_path, arcname)
    
    memory_file.seek(0)
    return send_file(
        memory_file,
        mimetype='application/zip',
        as_attachment=True,
        download_name='labeled_exports.zip'
    )

@app.route('/file')
def serve_file():
    filepath = request.args.get('path')
    if not filepath: return "No path provided", 400
    if not os.path.exists(filepath): return "File not found", 404
    return send_file(filepath)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)