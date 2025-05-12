import cv2
import torch
import subprocess
import time
import asyncio
import json
import zlib
import platform
import base64
import os
import logging
import urllib.parse
from collections import defaultdict
from datetime import datetime
from io import BytesIO
from PIL import Image
from typing import Dict, Set
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
from ultralytics import YOLO
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Play alert sound function
def play_alert_sound():
    if platform.system() == "Windows":
        import winsound
        duration = 1000  # Reduced from 5000 to 1000ms to be less intrusive
        frequency = 1000
        winsound.Beep(frequency, duration)
    else:
        print('\a')  # Console bell

# Global FastAPI app
app = FastAPI()
clients: Dict[str, WebSocket] = {}
active_connections: Set[WebSocket] = set()

# Add a root route to provide basic information
@app.get("/")
async def root():
    """Provide basic information about the weapon detection service"""
    return {
        "service": "Weapon Detection Server",
        "version": "1.0.0",
        "status": "Operational",
        "websocket_endpoint": "/ws/{client_name}",
        "description": "WebSocket-based weapon detection service using YOLO model"
    }

# Add a health check route
@app.get("/health")
async def health_check():
    """Provide a health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device),
        "active_clients": len(clients)
    }

# Detection model configuration
MODEL_PATH = r"weapon-detection.pt"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = None  # Will be loaded in startup event

# Detection constants
WEAPON_CLASSES = ["Baton", "Knife", "Pistol", "Rifle", "Switchblades"]
ALERT_INTERVAL = 5.0  # seconds between alerts
last_alert_time = 0

# Detection service variables
active_sources = {}
processing_tasks = {}
detector_websocket = None  # Internal websocket connection for the detector client

def save_json(data):
    """Save detection data to log file"""
    with open("detections_log.json", "a") as file:
        file.write(json.dumps(data) + "\n")
    logger.info(f"💾 Saved detection to log file: {json.dumps(data)[:100]}...")

def encode_image_to_base64(image):
    """Convert OpenCV image to base64 string"""
    try:
        # Convert the image from BGR (OpenCV format) to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
       
        # Use PIL to convert to JPEG format
        pil_img = Image.fromarray(image_rgb)
       
        # Create a BytesIO object to store the image data
        buffer = BytesIO()
        pil_img.save(buffer, format="JPEG", quality=80)  # Lower quality for smaller payload
       
        # Get the byte data and encode to base64
        img_bytes = buffer.getvalue()
        base64_image = base64.b64encode(img_bytes).decode('utf-8')
       
        return f"data:image/jpeg;base64,{base64_image}"
    except Exception as e:
        logger.error(f"Error encoding image to base64: {e}")
        return None

async def process_url_with_username(src):
    """Process URL to include username from source name if needed"""
    if "url" in src and "name" in src:
        # Extract username from source name (format: username_media_id)
        if "_" in src["name"]:
            username = src["name"].split("_")[0]
           
            # Parse the URL and add username parameter
            parsed_url = urllib.parse.urlparse(src["url"])
            query_params = urllib.parse.parse_qs(parsed_url.query)
           
            # Only add if not already present
            if "username" not in query_params:
                query_params["username"] = [username]
                new_query = urllib.parse.urlencode(query_params, doseq=True)
               
                # Reconstruct URL with username
                updated_url = parsed_url._replace(query=new_query)
                src["url"] = urllib.parse.urlunparse(updated_url)
               
                logger.info(f"🔄 Updated URL for {src['name']}: {src['url']}")
            else:
                logger.info(f"ℹ️ Username already in URL for {src['name']}")
    return src

# ============ DETECTOR CLIENT FUNCTIONS ============

async def process_stream(source_url, source_name):
    """Process an RTSP or other streaming source"""
    global last_alert_time
   
    logger.info(f"🔄 Starting processing for stream: {source_name}")
   
    cap = cv2.VideoCapture(source_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        logger.error(f"❌ Failed to open stream: {source_url}")
        return
       
    fps = cap.get(cv2.CAP_PROP_FPS)
    logger.info(f"FPS for {source_name}: {fps}")
   
    frame_skip = int(fps * 0.5) if fps > 0 else 15  # Reduce frame rate for performance
    frame_count = 0

    try:
        while active_sources.get(source_name, {}).get("active", False):
            success, frame = cap.read()
            if not success:
                logger.warning(f"❌ Failed to read frame from {source_name}")
                # Try to reconnect if stream disconnects
                cap.release()
                await asyncio.sleep(2)
                cap = cv2.VideoCapture(source_url, cv2.CAP_FFMPEG)
                if not cap.isOpened():
                    logger.error(f"❌ Failed to reconnect to stream: {source_url}")
                    break
                continue

            frame_count += 1
            if frame_count % frame_skip != 0:
                continue

            frame = cv2.resize(frame, (416, 256))  
           
            results = model(frame, device=device, conf=0.1, iou=0.1)
            detections = []
            detection_frame = frame.copy()

            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    detected_object = model.names[class_id]
                    confidence = box.conf[0].item()

                    if detected_object in WEAPON_CLASSES:
                        current_time = time.time()
                        logger.info(f"⏳ Time since last alert: {current_time - last_alert_time:.2f} seconds")
                        if current_time - last_alert_time > ALERT_INTERVAL:
                            play_alert_sound()
                            last_alert_time = current_time
                            logger.info(f"🚨 Alert: {detected_object} detected!")

                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(detection_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"{model.names[class_id]} {confidence:.2f}"
                        cv2.putText(detection_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        # Format detection for Firebase compatibility
                        detections.append({
                            "label": detected_object,
                            "confidence": round(confidence, 2),
                            "box": [x1, y1, x2, y2]
                        })

            # Only try to display if we're not in headless mode
            try:
                resize_frame = cv2.resize(detection_frame, (640, 480))
                cv2.imshow(f"Smart Surveillance System Process - {source_name}", resize_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
            except:
                pass

            if detections:
                # Create a base64 encoded image of the frame with bounding boxes
                detection_image_b64 = encode_image_to_base64(detection_frame)
               
                # Format for Firebase
                firebase_formatted_detections = []
                for detection in detections:
                    firebase_formatted_detections.append({
                        "objectDetected": detection["label"],
                        "confidenceLevel": detection["confidence"],
                        "cameraName": source_name,
                        "timestamp": datetime.now().isoformat(),
                        "imageUrl": None  # Will be set by the server
                    })
               
                message = {
                    "source_name": source_name,
                    "detections": detections,
                    "firebase_detections": firebase_formatted_detections,
                    "detection_image": detection_image_b64  # New field with the image
                }
               
                # Forward detection results to all connected clients
                await forward_detections(message)

            # Short pause to prevent CPU overload
            await asyncio.sleep(0.01)

    except Exception as e:
        logger.error(f"❌ Error in process_stream for {source_name}: {e}")
    finally:
        cap.release()
        try:
            cv2.destroyWindow(f"Smart Surveillance System Process - {source_name}")
        except:
            pass
        logger.info(f"Stream processing ended for {source_name}")

async def process_image(source):
    """Process a single image for weapon detection"""
    image_path = source["url"]
    source_name = source["name"]
    media_id = source.get("media_id", "unknown")
    original_filename = source.get("original_filename", "image.jpg")
    username = source_name.split('_')[0] if '_' in source_name else "unknown"
   
    logger.info(f"🖼️ Processing image: {image_path}")
    logger.info(f"Original filename: {original_filename}, Username: {username}")

    try:
        # Add username parameter to URL if not present
        parsed_url = urllib.parse.urlparse(image_path)
        query_params = urllib.parse.parse_qs(parsed_url.query)
       
        # If no username in URL and we have a source name with username format
        if 'username' not in query_params and '_' in source_name:
            username = source_name.split('_')[0]
            # Reconstruct URL with username
            query_params['username'] = [username]
            new_query = urllib.parse.urlencode(query_params, doseq=True)
           
            # Create new URL with username parameter
            parsed_url = parsed_url._replace(query=new_query)
            image_path = urllib.parse.urlunparse(parsed_url)
            logger.info(f"🔄 Updated URL with username: {image_path}")
       
        # Download the image using requests
        import requests
        logger.info(f"📥 Attempting to download image from: {image_path}")
        response = requests.get(image_path)
       
        if response.status_code != 200:
            logger.error(f"❌ Failed to download image: HTTP {response.status_code}")
            logger.info(f"Response content: {response.text[:200]}...")  # Print first 200 chars of response
           
            # Send empty detections to prevent hanging
            message = {
                "source_name": source_name,
                "media_id": media_id,
                "detections": []
            }
            await forward_detections(message)
            logger.info(f"🔄 Sent empty detections due to image download failure")
            return
           
        # Convert to OpenCV format
        nparr = np.frombuffer(response.content, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
       
        if frame is None:
            logger.error("❌ Failed to decode image.")
            # Send empty detections
            message = {
                "source_name": source_name,
                "media_id": media_id,
                "detections": []
            }
            await forward_detections(message)
            logger.info(f"🔄 Sent empty detections due to image decode failure")
            return
    except Exception as e:
        logger.error(f"❌ Error processing image: {e}")
        # Send empty detections
        message = {
            "source_name": source_name,
            "media_id": media_id,
            "detections": []
        }
        await forward_detections(message)
        logger.info(f"🔄 Sent empty detections due to exception: {e}")
        return

    # Make a copy for detection visualization
    detection_frame = frame.copy()
   
    # Continue with detection as before
    results = model(frame)
    detections = []
    firebase_formatted_detections = []

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            detected_object = model.names[class_id]
            confidence = box.conf[0].item()

            if detected_object in WEAPON_CLASSES:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(detection_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{detected_object} {confidence:.2f}"
                cv2.putText(detection_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                detections.append({
                    "label": detected_object,
                    "confidence": round(confidence, 2), 
                    "box": [x1, y1, x2, y2]
                })
               
                # Create Firebase-formatted detection
                firebase_formatted_detections.append({
                    "objectDetected": detected_object,
                    "confidenceLevel": round(confidence, 2),
                    "cameraName": source_name,
                    "timestamp": datetime.now().isoformat(),
                    "imageUrl": None  # Will be set by the server
                })

    # Convert the detection frame to base64
    detection_image_b64 = encode_image_to_base64(detection_frame)
   
    message = {
        "source_name": source_name,
        "media_id": media_id,
        "original_filename": original_filename,
        "detections": detections,
        "firebase_detections": firebase_formatted_detections,
        "detection_image": detection_image_b64  # Send the annotated image
    }

    # Forward detection results to all connected clients
    await forward_detections(message)

    # Display the frame if not in headless mode
    try:
        cv2.imshow(f"Image Detection - {source_name}", detection_frame)
        cv2.waitKey(1000)  # Display for 1 second
        cv2.destroyWindow(f"Image Detection - {source_name}")
    except:
        logger.info("Running in headless mode, skipping display")

async def process_video(source):
    """Process a video file for weapon detection"""
    video_path = source["url"]
    source_name = source["name"]
    username = source_name.split('_')[0] if '_' in source_name else "unknown"
   
    logger.info(f"🎞️ Processing video: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("❌ Failed to open video.")
        return

    while cap.isOpened() and active_sources.get(source_name, {}).get("active", False):
        success, frame = cap.read()
        if not success:
            break

        # Make a copy for detection visualization
        detection_frame = frame.copy()
       
        results = model(frame)
        detections = []
        firebase_formatted_detections = []

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                detected_object = model.names[class_id]
                confidence = box.conf[0].item()

                if detected_object in WEAPON_CLASSES:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(detection_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{detected_object} {confidence:.2f}"
                    cv2.putText(detection_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    detections.append({
                        "label": detected_object,
                        "confidence": round(confidence, 2),
                        "box": [x1, y1, x2, y2]
                    })
                   
                    # Create Firebase-formatted detection
                    firebase_formatted_detections.append({
                        "objectDetected": detected_object,
                        "confidenceLevel": round(confidence, 2),
                        "cameraName": source_name,
                        "timestamp": datetime.now().isoformat(),
                        "imageUrl": None
                    })

        if detections:
            # Convert the detection frame to base64
            detection_image_b64 = encode_image_to_base64(detection_frame)
           
            message = {
                "source_name": source_name,
                "detections": detections,
                "firebase_detections": firebase_formatted_detections,
                "detection_image": detection_image_b64  # Send the annotated image
            }

            # Forward detections to all clients
            await forward_detections(message)

        try:
            cv2.imshow(f"Video Detection - {source_name}", detection_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        except:
            logger.info("Running in headless mode, skipping display")
           
        # Short pause to prevent CPU overload
        await asyncio.sleep(0.01)

    cap.release()
    try:
        cv2.destroyWindow(f"Video Detection - {source_name}")
    except:
        pass

async def start_source_processing(source):
    """Handle processing for a specific source"""
    source_name = source["name"]
    source_type = source.get("type", source.get("source_type", "rtsp"))
   
    # Mark this source as active
    active_sources[source_name] = {
        "source": source,
        "active": True,
        "type": source_type
    }
   
    logger.info(f"🔄 Starting processing for {source_name} of type {source_type}")
   
    try:
        if source_type == "img":
            await process_image(source)
        elif source_type == "video":
            await process_video(source)
        else:  # rtsp or other streaming types
            await process_stream(source["url"], source_name)
    except Exception as e:
        logger.error(f"❌ Error in source processing for {source_name}: {e}")
    finally:
        # Mark source as inactive when done
        if source_name in active_sources:
            active_sources[source_name]["active"] = False
       
        logger.info(f"✅ Finished processing for {source_name}")

async def handle_source(source):
    """Start or update a source processing task"""
    source_name = source["name"]
   
    # If we already have a task for this source, stop it
    if source_name in processing_tasks and not processing_tasks[source_name].done():
        # Mark the source as inactive to stop the processing loop
        if source_name in active_sources:
            active_sources[source_name]["active"] = False
       
        # Wait a bit for the task to clean up
        await asyncio.sleep(1)
   
    # Create a new task for this source
    task = asyncio.create_task(start_source_processing(source))
    processing_tasks[source_name] = task

async def forward_detections(message):
    """Forward detection results to all connected clients"""
    compressed_data = zlib.compress(json.dumps(message).encode('utf-8'))
    
    # Get source info for logging
    source_name = message.get("source_name", "unknown")
    detection_count = len(message.get("detections", []))
    
    # Save to log file
    save_json(message)
    
    # Forward to all clients
    for name, client_ws in list(clients.items()):
        if client_ws in active_connections:
            try:
                await client_ws.send_bytes(compressed_data)
                logger.info(f"📤 Forwarded {detection_count} detections from {source_name} to {name}")
            except Exception as e:
                logger.error(f"Failed to forward detection to {name}: {e}")
                # Remove failed connection
                if client_ws in active_connections:
                    active_connections.remove(client_ws)
                    logger.info(f"🧹 Removed {name} from active connections")

# ========== WEB SOCKET SERVER ROUTES ==========

@app.websocket("/ws/{client_name}")
async def websocket_endpoint(websocket: WebSocket, client_name: str):
    """Handle WebSocket connections from external clients"""
    global last_alert_time
    
    await websocket.accept()
    active_connections.add(websocket)
   
    # Handle duplicate connection names
    if client_name in clients:
        old_ws = clients[client_name]
        if old_ws in active_connections:
            logger.info(f"♻️ [{client_name}] Already connected. Closing old connection.")
            try:
                active_connections.remove(old_ws)
                await old_ws.close(code=1000)
                logger.info(f"✅ [{client_name}] Old connection closed successfully")
            except Exception as e:
                logger.warning(f"⚠️ [{client_name}] Error closing previous connection: {e}")
        else:
            logger.warning(f"⚠️ [{client_name}] Previous connection already closed")
   
    # Register new connection
    clients[client_name] = websocket
    logger.info(f"✅ [{client_name}] WebSocket Client Connected.")
    logger.info(f"📊 Current active clients: {list(clients.keys())}")

    # For detector-server registration - simplified in combined version
    is_detector = (client_name == "detector-server")

    # Maintain heartbeat
    heartbeat_task = None
   
    try:
        # Start a heartbeat task for this connection
        heartbeat_task = asyncio.create_task(send_heartbeat(websocket, client_name))
       
        while True:
            try:
                raw_data = await websocket.receive()
               
                # Handle disconnect signals
                if raw_data.get("type") == "websocket.disconnect":
                    logger.info(f"📢 [{client_name}] Received disconnect signal")
                    break

                json_data = None
               
                # Handle binary message with zlib decompression
                if "bytes" in raw_data and raw_data["bytes"] is not None:
                    try:
                        data_size = len(raw_data["bytes"])
                        logger.info(f"📦 [{client_name}] Received compressed binary data: {data_size} bytes")
                        decompressed = zlib.decompress(raw_data["bytes"]).decode("utf-8")
                        json_data = json.loads(decompressed)
                    except Exception as e:
                        logger.error(f"🚨 [{client_name}] Failed to decompress or parse binary message: {e}")
                        continue

                # Handle text-based JSON messages
                elif "text" in raw_data and raw_data["text"] is not None:
                    try:
                        logger.info(f"📝 [{client_name}] Received text message")
                        json_data = json.loads(raw_data["text"])
                    except Exception as e:
                        logger.error(f"🚨 [{client_name}] Failed to parse text message: {e}")
                        continue
                else:
                    logger.warning(f"⚠️ [{client_name}] Empty or unknown message format")
                    continue

                # Handle ping-pong for keep-alive
                if "ping" in json_data:
                    logger.debug(f"🏓 [{client_name}] Received ping, sending pong")
                    await websocket.send_text(json.dumps({"pong": "keep-alive"}))
                    continue
               
                # If client sends source(s)
                if "source" in json_data:
                    # Process all sources - add username parameter to URLs
                    processed_sources = []
                    logger.info(f"🎥 [{client_name}] Received sources request")
                   
                    for src in json_data["source"]:
                        processed_src = await process_url_with_username(src)
                        processed_sources.append(processed_src)
                        logger.debug(f"🎥 [{client_name}] Processed source: {processed_src}")
                   
                    # In combined version, we directly handle sources here
                    for source in processed_sources:
                        await handle_source(source)
                       
                    logger.info(f"🎥 Started processing {len(processed_sources)} sources")

                # If detections come from detector-server or other clients (in combined version, usually not needed)
                if "detections" in json_data:
                    source = json_data.get("source_name", "unknown")
                    media_id = json_data.get("media_id", None)
                   
                    # Log detection details
                    detection_count = len(json_data["detections"])
                    logger.info(f"🔍 [{client_name}] Received {detection_count} detections from {source}")
                   
                    # Save detections
                    save_json(json_data)
                   
                    # Forward to other clients
                    for name, client_ws in list(clients.items()):
                        if name != client_name and client_ws in active_connections:
                            try:
                                await client_ws.send_text(json.dumps(json_data))
                                logger.info(f"📤 [{client_name}] Detections forwarded to {name}")
                            except Exception as e:
                                logger.warning(f"⚠️ [{client_name}] Failed to forward to {name}: {e}")
                                if client_ws in active_connections:
                                    active_connections.remove(client_ws)

            except WebSocketDisconnect:
                logger.warning(f"⚠️ [{client_name}] Client Disconnected during receive.")
                break
           
            except asyncio.CancelledError:
                logger.info(f"📢 [{client_name}] Task cancelled")
                break

            except Exception as e:
                logger.error(f"🚨 [{client_name}] Error receiving message: {e}")
                continue

    except WebSocketDisconnect:
        logger.warning(f"⚠️ [{client_name}] Client Disconnected")
   
    except Exception as e:
        logger.error(f"🚨 [{client_name}] Unexpected WebSocket Error: {e}")

    # Clean up client after disconnect
    finally:
        # Cancel the heartbeat task if it exists
        if heartbeat_task and not heartbeat_task.done():
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass
           
        # Only remove client if it's still this websocket instance
        if clients.get(client_name) == websocket:
            clients.pop(client_name, None)
            logger.info(f"✅ [{client_name}] Client disconnected and cleaned up.")
       
        # Remove from active connections set
        if websocket in active_connections:
            active_connections.remove(websocket)
           
        logger.info(f"📊 Current active clients after cleanup: {list(clients.keys())}")

@app.websocket("/ws/detector-server")
async def detector_server_websocket(websocket: WebSocket):
    """Special WebSocket route specifically for the detector server - not needed in combined version
    but maintained for compatibility with external clients"""
    await websocket_endpoint(websocket, "detector-server")

async def send_heartbeat(websocket: WebSocket, client_name: str):
    """Send periodic heartbeats to keep connection alive"""
    try:
        while True:
            await asyncio.sleep(25)  # Send heartbeat every 25 seconds
            try:
                if websocket in active_connections:
                    await websocket.send_text(json.dumps({"heartbeat": "ping"}))
                    logger.debug(f"💓 Sent heartbeat to {client_name}")
                else:
                    logger.debug(f"⚠️ [{client_name}] Connection no longer active, stopping heartbeat")
                    break
            except Exception as e:
                logger.warning(f"⚠️ Failed to send heartbeat to {client_name}: {e}")
                break
    except asyncio.CancelledError:
        # Task was cancelled, exit gracefully
        logger.debug(f"💓 [{client_name}] Heartbeat task cancelled")
        pass

# ============= STARTUP & SHUTDOWN =============

@app.on_event("startup")
async def startup_event():
    """Load model and initialize everything when the app starts"""
    global model, last_alert_time
    
    # Initialize the alert time
    last_alert_time = time.time()
    
    # Load the YOLO model
    try:
        logger.info(f"Loading weapon detection model from: {MODEL_PATH}")
        model = YOLO(MODEL_PATH)
        logger.info(f"Model loaded successfully on {'GPU' if torch.cuda.is_available() else 'CPU'}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        # We'll continue anyway, just log the error

    logger.info("Application startup complete")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources when the app shuts down"""
    # Stop all active sources
    for source_name in list(active_sources.keys()):
        active_sources[source_name]["active"] = False
    
    # Close all OpenCV windows
    try:
        cv2.destroyAllWindows()
    except:
        pass
    
    logger.info("Application shutdown complete")

# Entry point when running this script directly
if __name__ == "__main__":
    logger.info("🚀 Starting Weapon Detction Server on port 8000")
    # Get port from environment variable for compatibility with hosting platforms like Render
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)