import cv2
import torch
import subprocess
import time
import winsound
import asyncio
import websockets
import json
from ultralytics import YOLO
import numpy as np
import sys
import zlib
import platform
import base64
from collections import defaultdict
from datetime import datetime
from io import BytesIO
from PIL import Image


MODEL_PATH = r"myenv\model\weapon-detection.pt"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = YOLO(MODEL_PATH)

WEAPON_CLASSES = ["Baton", "Knife", "Pistol", "Rifle", "Switchblades"]

ALERT_INTERVAL = 5
last_alert_time = 0


COLORS_10 = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]


# Global WebSocket connection
main_websocket = None
active_sources = {}
processing_tasks = {}


def play_alert_sound():
    if platform.system() == "Windows":
        duration = 1000  # Reduced from 5000 to 1000ms to be less intrusive
        frequency = 1000
        winsound.Beep(frequency, duration)
    else:
        print('\a')


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
        print(f"Error encoding image to base64: {e}")
        return None


async def process_stream(source_url, source_name):
    global last_alert_time, main_websocket
   
    print(f"🔄 Starting processing for stream: {source_name}")
   
    cap = cv2.VideoCapture(source_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print(f"❌ Failed to open stream: {source_url}")
        return
       
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"FPS for {source_name}: {fps}")
   
    frame_skip = int(fps * 0.5) if fps > 0 else 15  # Reduce frame rate for performance
    frame_count = 0


    try:
        while active_sources.get(source_name, {}).get("active", False):
            success, frame = cap.read()
            if not success:
                print(f"❌ Failed to read frame from {source_name}")
                # Try to reconnect if stream disconnects
                cap.release()
                await asyncio.sleep(2)
                cap = cv2.VideoCapture(source_url, cv2.CAP_FFMPEG)
                if not cap.isOpened():
                    print(f"❌ Failed to reconnect to stream: {source_url}")
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
                        print(f"⏳ Time since last alert: {current_time - last_alert_time:.2f} seconds")
                        if current_time - last_alert_time > ALERT_INTERVAL:
                            play_alert_sound()
                            last_alert_time = current_time
                            print(f"🚨 Alert: {detected_object} detected!")


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


            if detections and main_websocket:
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
               
                try:
                    compressed_data = zlib.compress(json.dumps(message).encode('utf-8'))
                    await main_websocket.send(compressed_data)
                    print(f"🔁 Sent detections with image to server: {len(detections)} detections")
                except websockets.exceptions.ConnectionClosed:
                    print("❌ WebSocket connection closed while sending detections")
                    break


            # Send periodic keep-alive message to maintain connection
            if frame_count % (frame_skip * 20) == 0 and main_websocket:
                try:
                    await main_websocket.send(json.dumps({"ping": "keep-alive"}).encode('utf-8'))
                except:
                    print("❌ Failed to send keep-alive")
                    break


            # Short pause to prevent CPU overload
            await asyncio.sleep(0.01)


    except Exception as e:
        print(f"❌ Error in process_stream for {source_name}: {e}")
    finally:
        cap.release()
        try:
            cv2.destroyWindow(f"Smart Surveillance System Process - {source_name}")
        except:
            pass
        print(f"Stream processing ended for {source_name}")


async def process_image(source):
    global main_websocket
   
    image_path = source["url"]
    source_name = source["name"]
    media_id = source.get("media_id", "unknown")
    original_filename = source.get("original_filename", "image.jpg")
    username = source_name.split('_')[0] if '_' in source_name else "unknown"
   
    print(f"🖼️ Processing image: {image_path}")
    print(f"Original filename: {original_filename}, Username: {username}")


    try:
        # Add username parameter to URL if not present
        import urllib.parse
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
            print(f"🔄 Updated URL with username: {image_path}")
       
        # Download the image using requests
        import requests
        print(f"📥 Attempting to download image from: {image_path}")
        response = requests.get(image_path)
       
        if response.status_code != 200:
            print(f"❌ Failed to download image: HTTP {response.status_code}")
            print(f"Response content: {response.text[:200]}...")  # Print first 200 chars of response
           
            # Send empty detections to prevent hanging
            if main_websocket:
                message = {
                    "source_name": source_name,
                    "media_id": media_id,
                    "detections": []
                }
                await main_websocket.send(zlib.compress(json.dumps(message).encode('utf-8')))
                print(f"🔄 Sent empty detections due to image download failure")
            return
           
        # Convert to OpenCV format
        nparr = np.frombuffer(response.content, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
       
        if frame is None:
            print("❌ Failed to decode image.")
            # Send empty detections
            if main_websocket:
                message = {
                    "source_name": source_name,
                    "media_id": media_id,
                    "detections": []
                }
                await main_websocket.send(zlib.compress(json.dumps(message).encode('utf-8')))
                print(f"🔄 Sent empty detections due to image decode failure")
            return
    except Exception as e:
        print(f"❌ Error processing image: {e}")
        # Send empty detections
        if main_websocket:
            message = {
                "source_name": source_name,
                "media_id": media_id,
                "detections": []
            }
            await main_websocket.send(zlib.compress(json.dumps(message).encode('utf-8')))
            print(f"🔄 Sent empty detections due to exception: {e}")
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


    if main_websocket:
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


        try:
            compressed_data = zlib.compress(json.dumps(message).encode('utf-8'))
            await main_websocket.send(compressed_data)
            print(f"🔁 Sent detections with image to server for {original_filename}: {len(detections)} detections")
        except Exception as e:
            print(f"❌ Error sending detections: {e}")


    # Display the frame
    try:
        cv2.imshow(f"Image Detection - {source_name}", detection_frame)
        cv2.waitKey(1000)  # Display for 1 second
        cv2.destroyWindow(f"Image Detection - {source_name}")
    except:
        print("Running in headless mode, skipping display")


async def process_video(source):
    global main_websocket
   
    video_path = source["url"]
    source_name = source["name"]
    username = source_name.split('_')[0] if '_' in source_name else "unknown"
   
    print(f"🎞️ Processing video: {video_path}")


    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Failed to open video.")
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


        if detections and main_websocket:
            # Convert the detection frame to base64
            detection_image_b64 = encode_image_to_base64(detection_frame)
           
            message = {
                "source_name": source_name,
                "detections": detections,
                "firebase_detections": firebase_formatted_detections,
                "detection_image": detection_image_b64  # Send the annotated image
            }


            try:
                compressed_data = zlib.compress(json.dumps(message).encode('utf-8'))
                await main_websocket.send(compressed_data)
                print(f"🔁 Sent detections with image to server: {len(detections)} detections")
            except websockets.exceptions.ConnectionClosed:
                print("❌ WebSocket connection closed while sending detections")
                break


        try:
            cv2.imshow(f"Video Detection - {source_name}", detection_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        except:
            print("Running in headless mode, skipping display")
           
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
   
    print(f"🔄 Starting processing for {source_name} of type {source_type}")
   
    try:
        if source_type == "img":
            await process_image(source)
        elif source_type == "video":
            await process_video(source)
        else:  # rtsp or other streaming types
            await process_stream(source["url"], source_name)
    except Exception as e:
        print(f"❌ Error in source processing for {source_name}: {e}")
    finally:
        # Mark source as inactive when done
        if source_name in active_sources:
            active_sources[source_name]["active"] = False
       
        print(f"✅ Finished processing for {source_name}")


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


async def connect_websocket():
    """Main WebSocket connection that manages all sources"""
    global main_websocket
   
    uri = "ws://localhost:8000/ws/detector-server"
    print("🚀 WebSocket Client Starting")
   
    # Keep reconnecting if the connection drops
    while True:
        try:
            async with websockets.connect(uri, max_size=None, ping_interval=20, ping_timeout=60) as websocket:
                main_websocket = websocket
                print("✅ Connected to WebSocket server")
               
                # Periodic heartbeat sender task
                heartbeat_task = asyncio.create_task(send_periodic_heartbeat(websocket))
               
                try:
                    while True:
                        try:
                            msg = await websocket.recv()
                           
                            # Handle binary (compressed) data
                            if isinstance(msg, bytes):
                                try:
                                    msg = zlib.decompress(msg).decode('utf-8')
                                except Exception as e:
                                    print(f"❌ Failed to decompress message: {e}")
                                    continue
                           
                            data = json.loads(msg)


                            if "source" in data:
                                sources = data["source"]
                                print(f"🎥 Received sources: {sources}")


                                # Handle each source
                                for source in sources:
                                    await handle_source(source)
                           
                            # Handle heartbeat messages
                            elif "heartbeat" in data:
                                await websocket.send(json.dumps({"pong": data["heartbeat"]}))
                               
                            # Handle ping-pong for keep-alive
                            elif "ping" in data:
                                await websocket.send(json.dumps({"pong": "keep-alive"}))
                       
                        except json.JSONDecodeError as e:
                            print(f"❌ Failed to parse JSON: {e}")
                            continue
                           
                        except asyncio.TimeoutError:
                            # Send keep-alive if timeout occurs
                            await websocket.send(json.dumps({"ping": "keep-alive"}))
                           
                        except websockets.exceptions.ConnectionClosed:
                            print("🔄 WebSocket connection closed. Reconnecting...")
                            break
               
                except Exception as e:
                    print(f"❌ Error in WebSocket connection: {e}")
               
                finally:
                    # Clean up heartbeat task
                    if not heartbeat_task.done():
                        heartbeat_task.cancel()
                        try:
                            await heartbeat_task
                        except asyncio.CancelledError:
                            pass
                   
                    # Clear the main websocket reference
                    main_websocket = None
       
        except Exception as e:
            print(f"🔄 Failed to connect to WebSocket: {e}, retrying in 3 seconds...")
            await asyncio.sleep(3)


async def send_periodic_heartbeat(websocket):
    """Send periodic heartbeats to keep the connection alive"""
    try:
        while True:
            await asyncio.sleep(20)  # Send heartbeat every 20 seconds
            try:
                await websocket.send(json.dumps({"ping": "keep-alive"}))
                print("💓 Sent heartbeat ping")
            except Exception as e:
                print(f"❌ Failed to send heartbeat: {e}")
                break
    except asyncio.CancelledError:
        # Task cancelled, exit gracefully
        pass


if __name__ == "__main__":
    # Allow time for other processes to start
    time.sleep(1)
   
    # Run the main websocket connection
    asyncio.run(connect_websocket())
