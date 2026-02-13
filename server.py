"""
server.py — PrimeAssist Flask server
==================================
Serves the chat API and the static frontend.
Chat history is saved to the  chat_history/  folder as JSON files.
Usage:  python server.py
"""

import json, os, re, uuid
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from app.rnn_matcher import RNNMatcher
from app.conversation import handle

app = Flask(__name__, static_folder="static", static_url_path="")

# Lazy-load model: start server quickly, load model on first request
matcher = None
def get_matcher():
    global matcher
    if matcher is None:
        print("Loading RNN model (this may take a few seconds)...")
        import time
        t0 = time.time()
        matcher = RNNMatcher("model")
        print(f"RNN model loaded in {time.time() - t0:.2f}s")
    return matcher

# ── Chat history ──────────────────────────────────────────────────────────────
HISTORY_DIR = os.path.join(os.path.dirname(__file__), "chat_history")
os.makedirs(HISTORY_DIR, exist_ok=True)


def save_turn(session_id: str, user_msg: str, reply: str,
              intent: str, confidence: float):
    """Append one conversation turn to the session's JSON log file."""
    path = os.path.join(HISTORY_DIR, f"{session_id}.json")

    # Load existing history or start fresh
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            history = json.load(f)
    else:
        history = {
            "session_id": session_id,
            "created_at": datetime.now().isoformat(),
            "turns": [],
        }

    history["turns"].append({
        "timestamp":  datetime.now().isoformat(),
        "user":       user_msg,
        "bot":        reply,
        "intent":     intent,
        "confidence": round(confidence, 4),
    })
    history["updated_at"] = datetime.now().isoformat()

    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    user_msg   = data.get("message", "").strip()
    session_id = data.get("session_id", str(uuid.uuid4()))

    if not user_msg:
        return jsonify({"reply": "Please type a message.", "intent": "", "confidence": 0})

    # Get RNN prediction (model will be loaded on first call)
    m = get_matcher()
    intent, rnn_response, confidence = m.predict(user_msg)

    # ── Low-confidence keyword fallback ──────────────────────────────────
    # The training data doesn't cover all common phrasings (e.g. "where is
    # my order").  When the model is uncertain, use keyword heuristics.
    if confidence < 0.75:
        msg_lower = user_msg.lower()
        keyword_rules = [
            (r"\b(where|track|status|eta)\b.*\b(order|package|shipment|delivery)\b", "track_order"),
            (r"\b(order|package|shipment|delivery)\b.*\b(where|track|status|eta)\b", "track_order"),
            (r"\b(where|status).*\brefund\b",  "track_refund"),
            (r"\brefund\b.*\b(where|status)\b", "track_refund"),
            (r"\b(cancel)\b.*\border\b",       "cancel_order"),
            (r"\border\b.*\b(cancel)\b",       "cancel_order"),
            (r"\b(recover|forgot|reset)\b.*\bpassword\b",   "recover_password"),
            (r"\bpassword\b.*\b(recover|forgot|reset)\b",   "recover_password"),
        ]
        for pattern, corrected_intent in keyword_rules:
            if re.search(pattern, msg_lower):
                intent = corrected_intent
                rnn_response = m.intent_responses.get(intent, rnn_response)
                confidence = 0.85  # mark as keyword-assisted
                break

    # Run through conversation engine (may ask follow-up or do DB lookup)
    reply = handle(session_id, user_msg, intent, rnn_response, confidence)

    # Save to chat history
    save_turn(session_id, user_msg, reply, intent, confidence)

    return jsonify({
        "reply": reply,
        "intent": intent,
        "confidence": round(confidence, 2),
        "session_id": session_id,
    })


@app.route("/api/history", methods=["GET"])
def list_history():
    """List all saved chat sessions."""
    files = sorted(os.listdir(HISTORY_DIR), reverse=True)
    sessions = []
    for f in files:
        if not f.endswith(".json"):
            continue
        path = os.path.join(HISTORY_DIR, f)
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        sessions.append({
            "session_id": data["session_id"],
            "created_at": data.get("created_at"),
            "turns":      len(data.get("turns", [])),
        })
    return jsonify(sessions)


@app.route("/api/history/<session_id>", methods=["GET"])
def get_history(session_id):
    """Get full chat history for a specific session."""
    path = os.path.join(HISTORY_DIR, f"{session_id}.json")
    if not os.path.exists(path):
        return jsonify({"error": "Session not found"}), 404
    with open(path, "r", encoding="utf-8") as f:
        return jsonify(json.load(f))


if __name__ == "__main__":
    print("\n  PrimeAssist is running at http://localhost:5000\n")
    print(f"  Chat history saved to: {os.path.abspath(HISTORY_DIR)}\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
