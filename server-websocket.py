
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
import json
import asyncio
import zlib
import logging
from typing import Dict, Set
import urllib.parse

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


app = FastAPI()
clients: Dict[str, WebSocket] = {}
active_connections: Set[WebSocket] = set()


def save_json(data):
    with open("detections_log.json", "a") as file:
        file.write(json.dumps(data) + "\n")
    logger.info(f"💾 Saved detection to log file: {json.dumps(data)[:100]}...")


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


@app.websocket("/ws/{client_name}")
async def websocket_endpoint(websocket: WebSocket, client_name: str):
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


    # Maintain heartbeat
    heartbeat_task = None
   
    try:
        # Start a heartbeat task for this connection
        heartbeat_task = asyncio.create_task(send_heartbeat(websocket, client_name))
       
        while True:
            try:
                raw_data = await websocket.receive()
               
                # Handle disconnect signals (should be caught by the outer try-except)
                if raw_data.get("type") == "websocket.disconnect":
                    logger.info(f"📢 [{client_name}] Received disconnect signal")
                    break


                json_data = None
               
                # Debug raw data type
                logger.info(f"📥 [{client_name}] Received raw data type: {type(raw_data)}")
                if "type" in raw_data:
                    logger.info(f"📥 [{client_name}] Raw data message type: {raw_data['type']}")


                # Handle binary message with zlib decompression
                if "bytes" in raw_data and raw_data["bytes"] is not None:
                    try:
                        data_size = len(raw_data["bytes"])
                        logger.info(f"📦 [{client_name}] Received compressed binary data: {data_size} bytes")
                        decompressed = zlib.decompress(raw_data["bytes"]).decode("utf-8")
                        logger.info(f"📦 [{client_name}] Decompressed size: {len(decompressed)} bytes")
                        json_data = json.loads(decompressed)
                        logger.info(f"📦 [{client_name}] Binary data keys: {list(json_data.keys())}")
                    except Exception as e:
                        logger.error(f"🚨 [{client_name}] Failed to decompress or parse binary message: {e}", exc_info=True)
                        continue


                # Handle text-based JSON messages
                elif "text" in raw_data and raw_data["text"] is not None:
                    try:
                        logger.info(f"📝 [{client_name}] Received text message: {raw_data['text'][:200]}...")
                        json_data = json.loads(raw_data["text"])
                        logger.info(f"📝 [{client_name}] Text data keys: {list(json_data.keys())}")
                    except Exception as e:
                        logger.error(f"🚨 [{client_name}] Failed to parse text message: {e}", exc_info=True)
                        continue


                else:
                    logger.warning(f"⚠️ [{client_name}] Empty or unknown message format: {raw_data}")
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
                    logger.info(f"🎥 [{client_name}] Received sources request: {json.dumps(json_data['source'])}")
                   
                    for src in json_data["source"]:
                        processed_src = await process_url_with_username(src)
                        processed_sources.append(processed_src)
                        logger.info(f"🎥 [{client_name}] Processed source: {processed_src}")
                   
                    # Now forward the modified sources to detector
                    if "detector-server" in clients and clients["detector-server"] in active_connections:
                        detector_ws = clients["detector-server"]
                        try:
                            forward_data = {
                                "source": processed_sources
                            }
                            await detector_ws.send_text(json.dumps(forward_data))
                            logger.info(f"📤 [{client_name}] Forwarded sources to detector-server: {json.dumps(forward_data)}")
                        except Exception as e:
                            logger.error(f"🚨 [{client_name}] Error sending to detector-server: {e}")
                            # Clean up invalid connection
                            if detector_ws in active_connections:
                                active_connections.remove(detector_ws)
                                logger.info(f"🧹 Removed detector-server from active connections")
                    else:
                        logger.warning(f"⚠️ [{client_name}] detector-server is not connected. Cannot forward sources.")
                        logger.info(f"📊 Current active clients: {list(clients.keys())}")


                # If detections come from detector-server or other clients
                if "detections" in json_data:
                    source = json_data.get("source_name", "unknown")
                    media_id = json_data.get("media_id", None)
                   
                    # Log detection details
                    detection_count = len(json_data["detections"])
                    logger.info(f"🔍 [{client_name}] Received {detection_count} detections from {source}")
                   
                    # Log each detection with its confidence
                    for i, detection in enumerate(json_data["detections"]):
                        logger.info(f"🔍 [{client_name}] Detection {i+1}: {detection.get('label', 'unknown')} - {detection.get('confidence', 0)}")
                   
                    # Add media_id logging
                    if media_id:
                        logger.info(f"🔍 [{client_name}] Media ID: {media_id}")
                   
                    # Firebase detections if present
                    if "firebase_detections" in json_data:
                        firebase_count = len(json_data["firebase_detections"])
                        logger.info(f"🔥 [{client_name}] Firebase detections: {firebase_count}")
                   
                    # Save to log file
                    save_json(json_data)
                   
                    # Forward detections to all clients except sender
                    for name, client_ws in list(clients.items()):
                        if name != client_name and client_ws in active_connections:
                            try:
                                await client_ws.send_text(json.dumps(json_data))
                                logger.info(f"📤 [{client_name}] Detections forwarded to {name}")
                            except Exception as e:
                                logger.warning(f"⚠️ [{client_name}] Failed to forward to {name}: {e}")
                                # Mark connection as potentially dead
                                if client_ws in active_connections:
                                    active_connections.remove(client_ws)
                                    logger.info(f"🧹 Removed {name} from active connections")


            except WebSocketDisconnect:
                logger.warning(f"⚠️ [{client_name}] Client Disconnected during receive.")
                break
           
            except asyncio.CancelledError:
                logger.info(f"📢 [{client_name}] Task cancelled")
                break


            except Exception as e:
                logger.error(f"🚨 [{client_name}] Error receiving message: {e}", exc_info=True)
                break


    except WebSocketDisconnect:
        logger.warning(f"⚠️ [{client_name}] Client Disconnected")
   
    except Exception as e:
        logger.error(f"🚨 [{client_name}] Unexpected WebSocket Error: {e}", exc_info=True)


    # Always clean up client after disconnect
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
        else:
            logger.info(f"♻️ [{client_name}] Disconnect signal from an old connection ignored.")
       
        # Remove from active connections set
        if websocket in active_connections:
            active_connections.remove(websocket)
           
        logger.info(f"📊 Current active clients after cleanup: {list(clients.keys())}")


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


if __name__ == "__main__":
    logger.info("🚀 Starting WebSocket Server on port 8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)