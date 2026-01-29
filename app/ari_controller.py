import json
import logging
import os
import threading
import time
from urllib.parse import urlencode

import requests
from websocket import WebSocketApp


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ari_controller")


ARI_BASE_URL = os.getenv("ARI_BASE_URL", "http://host.docker.internal:8088/ari")
ARI_USER = os.getenv("ARI_USER", "bot")
ARI_PASS = os.getenv("ARI_PASS", "botpass")
ARI_APP = os.getenv("ARI_APP", "phonebotMikh")

EXTERNAL_MEDIA_HOST = os.getenv("EXTERNAL_MEDIA_HOST", "host.docker.internal:10000")
EXTERNAL_MEDIA_FORMAT = os.getenv("EXTERNAL_MEDIA_FORMAT", "ulaw")

HTTP_TIMEOUT = float(os.getenv("ARI_HTTP_TIMEOUT", "6.0"))
RECONNECT_SEC = float(os.getenv("ARI_RECONNECT_SEC", "2.0"))


def ws_url_from_base(base_url: str) -> str:
    if base_url.startswith("https://"):
        return "wss://" + base_url[len("https://"):]
    if base_url.startswith("http://"):
        return "ws://" + base_url[len("http://"):]
    return base_url


class ARI:
    def __init__(self, base_url: str, user: str, password: str):
        self.base_url = base_url.rstrip("/")
        self.auth = (user, password)
        self.session = requests.Session()
        self.session.auth = self.auth

    def post(self, path: str, params=None, json_body=None):
        url = self.base_url + path
        r = self.session.post(url, params=params, json=json_body, timeout=HTTP_TIMEOUT)
        if r.status_code >= 400:
            raise RuntimeError(f"POST {path} {r.status_code} {r.text}")
        if not r.text:
            return None
        return r.json()

    def delete(self, path: str, params=None):
        url = self.base_url + path
        r = self.session.delete(url, params=params, timeout=HTTP_TIMEOUT)
        if r.status_code >= 400:
            raise RuntimeError(f"DELETE {path} {r.status_code} {r.text}")
        if not r.text:
            return None
        return r.json()


class SessionStore:
    def __init__(self):
        self.lock = threading.Lock()
        self.by_caller = {}

    def set(self, caller_channel_id: str, bridge_id: str, ext_channel_id: str):
        with self.lock:
            self.by_caller[caller_channel_id] = (bridge_id, ext_channel_id)

    def get(self, caller_channel_id: str):
        with self.lock:
            return self.by_caller.get(caller_channel_id)

    def pop(self, caller_channel_id: str):
        with self.lock:
            return self.by_caller.pop(caller_channel_id, None)

    def find_by_ext(self, ext_channel_id: str):
        with self.lock:
            for k, v in self.by_caller.items():
                if v[1] == ext_channel_id:
                    return k, v
        return None, None


ari = ARI(ARI_BASE_URL, ARI_USER, ARI_PASS)
store = SessionStore()


def safe_id(prefix: str, channel_id: str) -> str:
    s = channel_id.replace("-", "")
    if len(s) > 20:
        s = s[-20:]
    return f"{prefix}{s}"


def ensure_bridge(bridge_id: str):
    ari.post("/bridges", params={"type": "mixing", "bridgeId": bridge_id})


def add_to_bridge(bridge_id: str, channel_id: str):
    ari.post(f"/bridges/{bridge_id}/addChannel", params={"channel": channel_id})


def create_external_media(ext_channel_id: str):
    params = {
        "app": ARI_APP,
        "external_host": EXTERNAL_MEDIA_HOST,
        "format": EXTERNAL_MEDIA_FORMAT,
        "encapsulation": "rtp",
        "transport": "udp",
        "connection_type": "client",
        "direction": "both",
        "channelId": ext_channel_id
    }
    resp = ari.post("/channels/externalMedia", params=params)
    if resp and "id" in resp:
        return resp["id"]
    return ext_channel_id


def answer_channel(channel_id: str):
    ari.post(f"/channels/{channel_id}/answer")


def hangup_channel(channel_id: str):
    try:
        ari.delete(f"/channels/{channel_id}")
    except Exception:
        pass


def destroy_bridge(bridge_id: str):
    try:
        ari.delete(f"/bridges/{bridge_id}")
    except Exception:
        pass


def on_stasis_start(event):
    ch = event.get("channel") or {}
    channel_id = ch.get("id")
    if not channel_id:
        return

    if channel_id.startswith("ext"):
        return

    name = (ch.get("name") or "").lower()
    if name.startswith("unicastrtp/") or name.startswith("external/"):
        return

    bridge_id = safe_id("b", channel_id)
    ext_id = safe_id("ext", channel_id)

    try:
        answer_channel(channel_id)
    except Exception:
        pass

    try:
        ensure_bridge(bridge_id)
        add_to_bridge(bridge_id, channel_id)
        ext_channel_id = create_external_media(ext_id)
        add_to_bridge(bridge_id, ext_channel_id)
        store.set(channel_id, bridge_id, ext_channel_id)
        logger.info("Attached %s to bridge %s with external media %s", channel_id, bridge_id, ext_channel_id)
    except Exception as e:
        logger.error("StasisStart handling failed for %s: %s", channel_id, e)
        try:
            destroy_bridge(bridge_id)
        except Exception:
            pass


def on_stasis_end(event):
    ch = event.get("channel") or {}
    channel_id = ch.get("id")
    if not channel_id:
        return
    sess = store.pop(channel_id)
    if not sess:
        return
    bridge_id, ext_channel_id = sess
    hangup_channel(ext_channel_id)
    destroy_bridge(bridge_id)
    logger.info("Cleaned session for %s", channel_id)


def on_channel_destroyed(event):
    ch = event.get("channel") or {}
    channel_id = ch.get("id")
    if not channel_id:
        return

    if channel_id.startswith("ext"):
        caller_id, sess = store.find_by_ext(channel_id)
        if caller_id and sess:
            store.pop(caller_id)
            bridge_id, ext_channel_id = sess
            destroy_bridge(bridge_id)
        return

    sess = store.pop(channel_id)
    if not sess:
        return
    bridge_id, ext_channel_id = sess
    hangup_channel(ext_channel_id)
    destroy_bridge(bridge_id)


def handle_event(msg: str):
    try:
        event = json.loads(msg)
    except Exception:
        return
    etype = event.get("type")
    if etype == "StasisStart":
        on_stasis_start(event)
    elif etype == "StasisEnd":
        on_stasis_end(event)
    elif etype == "ChannelDestroyed":
        on_channel_destroyed(event)


def run_ws():
    ws_base = ws_url_from_base(ARI_BASE_URL)
    qs = urlencode({"app": ARI_APP, "api_key": f"{ARI_USER}:{ARI_PASS}"})
    url = f"{ws_base}/events?{qs}"

    def on_message(ws, message):
        if message:
            handle_event(message)

    def on_open(ws):
        logger.info("ARI WS connected")

    def on_close(ws, status, msg):
        logger.info("ARI WS closed")

    def on_error(ws, err):
        logger.error("ARI WS error: %s", err)

    while True:
        ws = WebSocketApp(url, on_message=on_message, on_open=on_open, on_close=on_close, on_error=on_error)
        ws.run_forever(ping_interval=30, ping_timeout=10)
        time.sleep(RECONNECT_SEC)


if __name__ == "__main__":
    logger.info("ARI controller starting base=%s app=%s ext=%s fmt=%s", ARI_BASE_URL, ARI_APP, EXTERNAL_MEDIA_HOST, EXTERNAL_MEDIA_FORMAT)
    run_ws()
