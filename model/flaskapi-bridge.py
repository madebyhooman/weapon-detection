from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import uuid
import time
import json
import asyncio
import websockets
import threading
import logging
from werkzeug.utils import secure_filename
from werkzeug.middleware.shared_data import SharedDataMiddleware
import subprocess
import threading
import shutil
import base64
import cv2
import numpy as np
from PIL import Image
import io


# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('flask_api_bridge')


app = Flask(__name__)
CORS(app)


# Configuration
UPLOAD_FOLDER = 'uploads'
DETECTIONS_FOLDER = 'detections'  # New folder for saving detection images
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
WS_SERVER_URL = 'ws://localhost:8000/ws/flutter-bridge'
FLASK_PUBLIC_URL = 'https://yourflaskapp.loca.lt'  # Replace with your Localtunnel URL


# Create necessary folders
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DETECTIONS_FOLDER, exist_ok=True)


app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DETECTIONS_FOLDER'] = DETECTIONS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload


# In-memory database for simplicity
media_items = {}
detections_by_id = {}


# Add middleware to serve files directly
app.add_url_rule('/uploads/<filename>', 'uploaded_file', build_only=True)
app.add_url_rule('/detections/<path:filename>', 'detection_file', build_only=True)
app.wsgi_app = SharedDataMiddleware(app.wsgi_app, {
    '/uploads': app.config['UPLOAD_FOLDER'],
    '/detections': app.config['DETECTIONS_FOLDER']
})


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


async def forward_to_detector(media_id, username, url, source_name, original_filename):
    try:
        # Ensure URL has username parameter
        import urllib.parse
        parsed_url = urllib.parse.urlparse(url)
        query_params = urllib.parse.parse_qs(parsed_url.query)
        query_params["username"] = [username]
        new_query = urllib.parse.urlencode(query_params, doseq=True)
        updated_url = parsed_url._replace(query=new_query)
        url_with_params = urllib.parse.urlunparse(updated_url)
       
        logger.info(f"Forwarding to detector with URL: {url_with_params}")
       
        async with websockets.connect(WS_SERVER_URL) as websocket:
            source = {
                "url": url_with_params,
                "name": source_name,
                "source_type": "img",
                "media_id": media_id,
                "original_filename": original_filename  # Pass original filename to help with saving
            }
            logger.info(f"Forwarding media to detector: {source}")
            await websocket.send(json.dumps({"source": [source]}))


            max_retries = 3
            for attempt in range(max_retries):
                try:
                    msg = await asyncio.wait_for(websocket.recv(), timeout=30)
                    data = json.loads(msg)
                   
                    if "detections" in data:
                        logger.info(f"Received detections for {media_id}")
                        message_media_id = data.get("media_id", None)
                        if message_media_id and message_media_id != media_id:
                            logger.warning(f"Received detections for wrong media ID: expected {media_id}, got {message_media_id}")
                            continue
                       
                        detections_by_id[media_id] = data["detections"]
                       
                        # Save detection image if included
                        if "detection_image" in data:
                            save_detection_image(data["detection_image"], username, media_id, original_filename)
                           
                            # Update the media item with detection image URL
                            if username in media_items and media_id in media_items[username]:
                                detection_filename = f"{os.path.splitext(original_filename)[0]}_{media_id}_detection.jpg"
                                detection_path = f"/detections/{username}/{detection_filename}"
                                detection_url = f"{FLASK_PUBLIC_URL.rstrip('/')}{detection_path}"
                               
                                media_items[username][media_id]["detections"] = data["detections"]
                                media_items[username][media_id]["detection_image_url"] = detection_url
                        else:
                            if username in media_items and media_id in media_items[username]:
                                media_items[username][media_id]["detections"] = data["detections"]
                       
                        break
                    elif "ping" in data:
                        continue
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout waiting for detections for {media_id}, attempt {attempt+1}/{max_retries}")
                    if attempt == max_retries - 1:
                        logger.error(f"Max retries reached for {media_id}, giving up")
                        if media_id not in detections_by_id:
                            detections_by_id[media_id] = []
                            if username in media_items and media_id in media_items[username]:
                                media_items[username][media_id]["detections"] = []
    except Exception as e:
        logger.error(f"Error in WebSocket communication: {e}")
        detections_by_id[media_id] = []
        if username in media_items and media_id in media_items[username]:
            media_items[username][media_id]["detections"] = []


def save_detection_image(detection_image_b64, username, media_id, original_filename):
    """Save detection image with bounding boxes to user-specific folder"""
    try:
        # Create user-specific directory if it doesn't exist
        user_detection_dir = os.path.join(app.config['DETECTIONS_FOLDER'], username)
        os.makedirs(user_detection_dir, exist_ok=True)
       
        # Process the base64 image
        if detection_image_b64.startswith('data:image'):
            # Strip the data URI prefix if present
            detection_image_b64 = detection_image_b64.split(',')[1]
           
        image_data = base64.b64decode(detection_image_b64)
       
        # Generate a filename based on the original
        base_name = os.path.splitext(original_filename)[0]
        detection_filename = f"{base_name}_{media_id}_detection.jpg"
        file_path = os.path.join(user_detection_dir, detection_filename)
       
        # Save the image
        with open(file_path, 'wb') as f:
            f.write(image_data)
           
        logger.info(f"Saved detection image to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving detection image: {e}")
        return False


def start_websocket_task(media_id, username, url, source_name, original_filename):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(forward_to_detector(media_id, username, url, source_name, original_filename))
    loop.close()


@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/detections/<username>/<path:filename>')
def detection_file(username, filename):
    user_detection_dir = os.path.join(app.config['DETECTIONS_FOLDER'], username)
    return send_from_directory(user_detection_dir, filename)


@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400


    file = request.files['file']
    username = request.form.get('username')
    metadata = json.loads(request.form.get('metadata', '{}'))


    if not username:
        return jsonify({'error': 'Username is required'}), 400


    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400


    if file and allowed_file(file.filename):
        media_id = str(uuid.uuid4())
        original_filename = secure_filename(file.filename)
        base, ext = os.path.splitext(original_filename)
        unique_filename = f"{base}_{media_id}{ext}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        logger.info(f"Saved file to {file_path}")
       
        # Log the actual file on disk for debugging
        logger.debug(f"File exists on disk: {os.path.exists(file_path)}")
        logger.debug(f"File size: {os.path.getsize(file_path) if os.path.exists(file_path) else 'N/A'}")


        # Create API media URL (with username parameter)
        api_media_url = f"/api/media/{unique_filename}?username={username}"
       
        # Also create a direct URL to the uploads folder
        direct_url = f"/uploads/{unique_filename}"
       
        # Use FLASK_PUBLIC_URL instead of request.host_url for absolute URLs
        absolute_api_url = f"{FLASK_PUBLIC_URL.rstrip('/')}{api_media_url}"
        absolute_direct_url = f"{FLASK_PUBLIC_URL.rstrip('/')}{direct_url}"
       
        logger.info(f"Created API media URL: {absolute_api_url}")
        logger.info(f"Created direct URL: {absolute_direct_url}")


        timestamp = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        new_item = {
            "id": media_id,
            "url": absolute_api_url,
            "direct_url": absolute_direct_url,
            "filename": unique_filename,
            "original_filename": original_filename,
            "timestamp": timestamp,
            "metadata": metadata,
            "detections": [],
            "detection_image_url": None  # Will be populated when detection image is saved
        }


        if username not in media_items:
            media_items[username] = {}
        media_items[username][media_id] = new_item


        source_name = f"{username}_{media_id}"
       
        # Use the direct URL for detector forwarding
        thread = threading.Thread(
            target=start_websocket_task,
            args=(media_id, username, absolute_direct_url, source_name, original_filename)
        )
        thread.daemon = True
        thread.start()


        return jsonify({
            'id': media_id,
            'url': absolute_api_url,
            'direct_url': absolute_direct_url,
            'timestamp': timestamp,
            'message': 'File uploaded successfully'
        }), 201


    return jsonify({'error': 'File type not allowed'}), 400


@app.route('/api/media/<path:filename>')
def serve_media(filename):
    username = request.args.get('username')
    logger.info(f"Media request received for: {filename} with username: {username}")
   
    clean_filename = filename.split('?')[0]
    logger.debug(f"Cleaned filename: {clean_filename}")
   
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], clean_filename)
    logger.debug(f"Full file path: {file_path}")
   
    if os.path.exists(file_path):
        logger.info(f"File found by direct path: {file_path}")
        try:
            with open(file_path, 'rb') as f:
                _ = f.read(100)
            logger.info(f"File is readable, serving via send_from_directory")
            return send_from_directory(os.path.abspath(app.config['UPLOAD_FOLDER']), clean_filename)
        except IOError as e:
            logger.error(f"File exists but cannot be read: {e}")
            return jsonify({'error': f'File exists but cannot be read: {e}'}), 500
   
    logger.warning(f"File not found directly: {file_path}")
   
    if username:
        logger.debug(f"Searching in media_items for username: {username}")
        user_media = media_items.get(username, {})
       
        if user_media:
            user_filenames = [item.get('filename') for item in user_media.values()]
            logger.debug(f"User has these files in media_items: {user_filenames}")
       
        for media_id, item in user_media.items():
            item_filename = item.get('filename')
            if item_filename == clean_filename:
                logger.info(f"Found matching filename in media_items: {item_filename}")
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], item_filename)
                if os.path.exists(file_path):
                    logger.info(f"Serving file from media_items match: {file_path}")
                    try:
                        return send_from_directory(os.path.abspath(app.config['UPLOAD_FOLDER']), item_filename)
                    except Exception as e:
                        logger.error(f"Error serving file from media_items match: {e}")
                        return jsonify({'error': f'Error serving file: {e}'}), 500
                else:
                    logger.warning(f"File not found on disk despite media_items entry: {file_path}")
   
    logger.debug("Searching all media_items for matching filename")
    for user, user_media in media_items.items():
        for media_id, item in user_media.items():
            if item.get('filename') == clean_filename:
                logger.info(f"Found match in user {user}'s media_items")
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], clean_filename)
                if os.path.exists(file_path):
                    logger.info(f"Serving file from cross-user match: {file_path}")
                    try:
                        return send_from_directory(os.path.abspath(app.config['UPLOAD_FOLDER']), clean_filename)
                    except Exception as e:
                        logger.error(f"Error serving file from cross-user match: {e}")
                        return jsonify({'error': f'Error serving file: {e}'}), 500
   
    logger.debug("Last resort: looking for similar files in uploads directory")
    if os.path.exists(app.config['UPLOAD_FOLDER']):
        all_files = os.listdir(app.config['UPLOAD_FOLDER'])
        logger.debug(f"All files in upload directory: {all_files}")
       
        import re
        uuid_pattern = re.compile(r'([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})')
        uuid_match = uuid_pattern.search(clean_filename)
       
        if uuid_match:
            media_id = uuid_match.group(1)
            logger.debug(f"Extracted media_id from filename: {media_id}")
           
            for file in all_files:
                if media_id in file:
                    logger.info(f"Found file with matching media_id: {file}")
                    try:
                        return send_from_directory(os.path.abspath(app.config['UPLOAD_FOLDER']), file)
                    except Exception as e:
                        logger.error(f"Error serving file with matching media_id: {e}")
                        return jsonify({'error': f'Error serving file: {e}'}), 500
   
    logger.error(f"Media item not found after exhaustive search: {filename}")
   
    return jsonify({
        'error': 'Media item not found',
        'requested_filename': filename,
        'cleaned_filename': clean_filename,
        'username': username,
        'uploads_dir_exists': os.path.exists(app.config['UPLOAD_FOLDER']),
        'files_in_uploads': os.listdir(app.config['UPLOAD_FOLDER']) if os.path.exists(app.config['UPLOAD_FOLDER']) else [],
        'media_items_keys': list(media_items.keys()) if media_items else [],
        'user_media_count': len(media_items.get(username, {})) if username else 0
    }), 404


@app.route('/api/debug', methods=['GET'])
def debug_system():
    upload_folder = app.config['UPLOAD_FOLDER']
    detection_folder = app.config['DETECTIONS_FOLDER']
    upload_folder_abs = os.path.abspath(upload_folder)
    detection_folder_abs = os.path.abspath(detection_folder)
   
    folder_exists = os.path.exists(upload_folder)
    detection_folder_exists = os.path.exists(detection_folder)
    folder_is_dir = os.path.isdir(upload_folder) if folder_exists else False
    detection_is_dir = os.path.isdir(detection_folder) if detection_folder_exists else False
   
    folder_permissions = None
    if folder_exists:
        try:
            import stat
            folder_stat = os.stat(upload_folder)
            folder_permissions = stat.filemode(folder_stat.st_mode)
        except Exception as e:
            folder_permissions = f"Error getting permissions: {str(e)}"
   
    files_in_folder = []
    if folder_exists and folder_is_dir:
        try:
            files_in_folder = os.listdir(upload_folder)
            file_details = []
            for file in files_in_folder:
                file_path = os.path.join(upload_folder, file)
                try:
                    file_stat = os.stat(file_path)
                    file_details.append({
                        'name': file,
                        'size': file_stat.st_size,
                        'permissions': stat.filemode(file_stat.st_mode),
                        'readable': os.access(file_path, os.R_OK),
                        'writable': os.access(file_path, os.W_OK),
                        'executable': os.access(file_path, os.X_OK)
                    })
                except Exception as e:
                    file_details.append({
                        'name': file,
                        'error': str(e)
                    })
        except Exception as e:
            files_in_folder = [f"Error listing files: {str(e)}"]
   
    # Get detection folder structure
    detection_folders = []
    if detection_folder_exists and detection_is_dir:
        try:
            user_folders = os.listdir(detection_folder)
            for user_folder in user_folders:
                user_path = os.path.join(detection_folder, user_folder)
                if os.path.isdir(user_path):
                    detection_files = os.listdir(user_path)
                    detection_folders.append({
                        'username': user_folder,
                        'file_count': len(detection_files),
                        'files': detection_files[:10]  # List first 10 files only
                    })
        except Exception as e:
            detection_folders = [f"Error listing detection folders: {str(e)}"]
   
    media_items_summary = {}
    for username, user_media in media_items.items():
        media_items_summary[username] = {
            'count': len(user_media),
            'media_ids': list(user_media.keys()),
            'filenames': [item.get('filename') for item in user_media.values()],
            'detection_image_urls': [item.get('detection_image_url') for item in user_media.values()]
        }
   
    import platform
    import sys
    import flask
   
    routes = []
    for rule in app.url_map.iter_rules():
        routes.append({
            'endpoint': rule.endpoint,
            'methods': list(rule.methods),
            'path': str(rule)
        })
   
    test_send_from_directory = None
    if folder_exists and folder_is_dir and files_in_folder:
        test_file = files_in_folder[0]
        try:
            send_from_directory(upload_folder, test_file)
            test_send_from_directory = "Function appears to work correctly"
        except Exception as e:
            test_send_from_directory = f"Error: {str(e)}"
   
    debug_info = {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'app_config': {
            'upload_folder': upload_folder,
            'upload_folder_absolute': upload_folder_abs,
            'detections_folder': detection_folder,
            'detections_folder_absolute': detection_folder_abs,
            'max_content_length': app.config['MAX_CONTENT_LENGTH'],
            'allowed_extensions': ALLOWED_EXTENSIONS
        },
        'folder_info': {
            'uploads_exists': folder_exists,
            'uploads_is_directory': folder_is_dir,
            'uploads_permissions': folder_permissions,
            'detections_exists': detection_folder_exists,
            'detections_is_directory': detection_is_dir
        },
        'files': file_details if 'file_details' in locals() else files_in_folder,
        'detection_folders': detection_folders,
        'media_items': media_items_summary,
        'environment': {
            'python_version': platform.python_version(),
            'platform': platform.platform(),
            'cwd': os.getcwd(),
            'flask_version': flask.__version__
        },
        'routes': routes,
        'send_from_directory_test': test_send_from_directory,
        'websocket_url': WS_SERVER_URL,
    }
   
    return jsonify(debug_info)


@app.route('/api/media', methods=['GET'])
def get_media_items():
    username = request.args.get('username')
    if not username:
        return jsonify({'error': 'Username is required'}), 400
   
    logger.debug(f"Get media items request for user: {username}")
    logger.debug(f"Current media_items: {json.dumps(media_items.get(username, {}))}")
   
    user_media = media_items.get(username, {})
    return jsonify(list(user_media.values())), 200


@app.route('/api/media/<media_id>', methods=['GET'])
def get_media_item(media_id):
    username = request.args.get('username')
    if not username:
        return jsonify({'error': 'Username is required'}), 400
    user_media = media_items.get(username, {})
    item = user_media.get(media_id)
    if not item:
        return jsonify({'error': 'Media item not found'}), 404
    return jsonify(item), 200


@app.route('/api/media/<media_id>', methods=['DELETE'])
def delete_media_item(media_id):
    username = request.args.get('username')
    if not username:
        return jsonify({'error': 'Username is required'}), 400
    user_media = media_items.get(username, {})
    item = user_media.get(media_id)
    if not item:
        return jsonify({'error': 'Media item not found'}), 404
    try:
        # Delete original upload file
        os.remove(os.path.join(app.config['UPLOAD_FOLDER'], item['filename']))
       
        # Delete detection image if it exists
        if item.get('detection_image_url'):
            detection_filename = os.path.basename(item['detection_image_url'].split('/')[-1])
            detection_path = os.path.join(app.config['DETECTIONS_FOLDER'], username, detection_filename)
            if os.path.exists(detection_path):
                os.remove(detection_path)
                logger.info(f"Deleted detection image: {detection_path}")
    except OSError as e:
        logger.error(f"Error removing files: {e}")
   
    del media_items[username][media_id]
    if media_id in detections_by_id:
        del detections_by_id[media_id]
    return jsonify({'message': 'Media item deleted successfully'}), 200


@app.route('/api/detections/<media_id>', methods=['GET'])
def get_detections(media_id):
    username = request.args.get('username')
    detections = detections_by_id.get(media_id, [])
   
    # If username is provided, include detection image URL if available
    if username and username in media_items and media_id in media_items[username]:
        item = media_items[username][media_id]
        response = {
            'detections': detections,
            'detection_image_url': item.get('detection_image_url')
        }
        return jsonify(response), 200
   
    return jsonify(detections), 200


@app.route('/api/test', methods=['GET'])
def test_server():
    return jsonify({
        'status': 'ok',
        'message': 'Server is running',
        'upload_folder': app.config['UPLOAD_FOLDER'],
        'detections_folder': app.config['DETECTIONS_FOLDER'],
        'files_in_upload': os.listdir(app.config['UPLOAD_FOLDER']) if os.path.exists(app.config['UPLOAD_FOLDER']) else [],
        'detection_folders': os.listdir(app.config['DETECTIONS_FOLDER']) if os.path.exists(app.config['DETECTIONS_FOLDER']) else []
    })


def start_localtunnel():
    try:
        npx_path = shutil.which('npx')
        if not npx_path:
            logger.error("npx not found. Please ensure Node.js is installed and npx is in your PATH.")
            return
        command = [npx_path, 'localtunnel', '--port', '5000', '--subdomain', 'yourflaskapp']
        logger.info(f"Starting Localtunnel with command: {' '.join(command)}")
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=True
        )
        logger.info("Localtunnel started. Check output for the URL and password.")
        for line in process.stdout:
            logger.info(f"Localtunnel stdout: {line.strip()}")
        for line in process.stderr:
            logger.error(f"Localtunnel stderr: {line.strip()}")
    except Exception as e:
        logger.error(f"Failed to start Localtunnel: {e}")


if __name__ == '__main__':
    tunnel_thread = threading.Thread(target=start_localtunnel, daemon=True)
    tunnel_thread.start()
    app.run(host='0.0.0.0', port=5000, debug=True)