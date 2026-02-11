"""
conversation.py — Multi-turn conversation engine for PrimeAssist
=============================================================
Manages session state so the bot can ask follow-up questions
(e.g. "please provide your order number") and look up answers
in the SQLite database.
"""

import sqlite3, pathlib, re
from typing import Optional

DB_PATH = pathlib.Path(__file__).resolve().parent.parent / "db" / "primeassist.db"

# Intents that need a follow-up lookup
NEEDS_ORDER_ID   = {"track_order", "cancel_order", "get_refund", "track_refund",
                    "check_cancellation_fee"}
NEEDS_EMAIL      = {"recover_password", "change_shipping_address", "switch_account",
                    "registration_problems", "delete_account", "edit_account"}


def _get_db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


# ── DB lookup helpers ─────────────────────────────────────────────────────────

def lookup_order(order_id: str) -> Optional[dict]:
    con = _get_db()
    row = con.execute("SELECT * FROM orders WHERE order_id = ? COLLATE NOCASE",
                      (order_id.strip(),)).fetchone()
    con.close()
    return dict(row) if row else None


def lookup_refund_by_order(order_id: str) -> Optional[dict]:
    con = _get_db()
    row = con.execute("SELECT * FROM refunds WHERE order_id = ? COLLATE NOCASE",
                      (order_id.strip(),)).fetchone()
    con.close()
    return dict(row) if row else None


def lookup_account(email: str) -> Optional[dict]:
    con = _get_db()
    row = con.execute("SELECT * FROM accounts WHERE email = ? COLLATE NOCASE",
                      (email.strip(),)).fetchone()
    con.close()
    return dict(row) if row else None


def lookup_orders_by_email(email: str) -> list[dict]:
    con = _get_db()
    rows = con.execute("SELECT * FROM orders WHERE email = ? COLLATE NOCASE",
                       (email.strip(),)).fetchall()
    con.close()
    return [dict(r) for r in rows]


# ── Format helpers ────────────────────────────────────────────────────────────

def _fmt_order(o: dict) -> str:
    lines = [
        f"**Order {o['order_id']}**",
        f"Product: {o['product']}",
        f"Status: {o['status']}",
    ]
    if o.get("tracking"):
        lines.append(f"Tracking: {o['tracking']}")
    lines.append(f"Order date: {o['order_date']}")
    if o.get("est_delivery"):
        lines.append(f"Estimated delivery: {o['est_delivery']}")
    return "\n".join(lines)


def _fmt_refund(r: dict) -> str:
    lines = [
        f"**Refund {r['refund_id']}** (for order {r['order_id']})",
        f"Amount: ${r['amount']:.2f}",
        f"Status: {r['status']}",
        f"Requested: {r['requested_date']}",
    ]
    if r.get("processed_date"):
        lines.append(f"Processed: {r['processed_date']}")
    else:
        lines.append("Processing date: Pending")
    return "\n".join(lines)


def _fmt_account(a: dict) -> str:
    return "\n".join([
        f"**Account for {a['name']}**",
        f"Email: {a['email']}",
        f"Phone: {a['phone'] or 'N/A'}",
        f"Shipping address: {a['shipping_address'] or 'Not set'}",
        f"Member since: {a['member_since']}",
    ])


# ── Extract IDs from user text ────────────────────────────────────────────────

def _extract_order_id(text: str) -> Optional[str]:
    """Try to find an order ID like ORD-1234 or just a number."""
    m = re.search(r"(ORD[- ]?\d{3,})", text, re.IGNORECASE)
    if m:
        # Normalise to ORD-XXXX
        raw = m.group(1).upper().replace(" ", "-")
        if not raw.startswith("ORD-"):
            raw = "ORD-" + raw[3:]
        return raw
    # Bare number
    m = re.search(r"\b(\d{4,})\b", text)
    if m:
        return f"ORD-{m.group(1)}"
    return None


def _extract_email(text: str) -> Optional[str]:
    m = re.search(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", text)
    return m.group(0).lower() if m else None


# ── Session state ─────────────────────────────────────────────────────────────

class Session:
    """Tracks per-user conversation state."""
    def __init__(self):
        self.waiting_for: Optional[str] = None   # "order_id" | "email"
        self.pending_intent: Optional[str] = None
        self.rnn_response: Optional[str] = None


# ── Main conversation handler ─────────────────────────────────────────────────

_sessions: dict[str, Session] = {}


def get_session(session_id: str) -> Session:
    if session_id not in _sessions:
        _sessions[session_id] = Session()
    return _sessions[session_id]


def handle(session_id: str, user_msg: str, intent: str,
           rnn_response: str, confidence: float) -> str:
    """
    Given the RNN's prediction and the user message, decide what
    to reply — possibly asking for more info or doing a DB lookup.
    """
    sess = get_session(session_id)

    # ── If we're waiting for a follow-up answer ──────────────────────────────
    if sess.waiting_for == "order_id":
        oid = _extract_order_id(user_msg)
        if not oid:
            return ("I couldn't find an order number in your message. "
                    "Could you please provide it in the format **ORD-XXXX**? "
                    "(e.g. ORD-1003)")
        return _handle_order_lookup(sess, oid)

    if sess.waiting_for == "email":
        email = _extract_email(user_msg)
        if not email:
            return ("I couldn't find an email address in your message. "
                    "Could you please provide your registered email?")
        return _handle_email_lookup(sess, email)

    # ── Fresh intent from the RNN ────────────────────────────────────────────

    # Check if user already included an order ID in their message
    if intent in NEEDS_ORDER_ID:
        oid = _extract_order_id(user_msg)
        if oid:
            sess.pending_intent = intent
            return _handle_order_lookup(sess, oid)
        # Otherwise ask for it
        sess.waiting_for = "order_id"
        sess.pending_intent = intent
        sess.rnn_response = rnn_response
        prompts = {
            "track_order":            "I'd be happy to help you track your order! Could you please provide your **order number**? (e.g. ORD-1003)",
            "cancel_order":           "I can help you with the cancellation. Could you please provide your **order number**?",
            "get_refund":             "I'll look into a refund for you. Could you please provide your **order number**?",
            "track_refund":           "Let me check the refund status. Could you please provide your **order number**?",
            "check_cancellation_fee": "I can check the cancellation fee. Could you please provide your **order number**?",
        }
        return prompts.get(intent, "Could you please provide your **order number**?")

    if intent in NEEDS_EMAIL:
        email = _extract_email(user_msg)
        if email:
            sess.pending_intent = intent
            return _handle_email_lookup(sess, email)
        sess.waiting_for = "email"
        sess.pending_intent = intent
        sess.rnn_response = rnn_response
        prompts = {
            "recover_password":       "I can help you recover your password! Could you please provide your **registered email address**?",
            "change_shipping_address":"Sure! Could you please provide your **email address** so I can look up your account?",
            "switch_account":         "I can help you switch accounts. Could you provide your **email address**?",
            "delete_account":         "I can assist with account deletion. Could you please provide your **email address**?",
            "edit_account":           "To edit your account, could you please provide your **email address**?",
            "registration_problems":  "Let me look into that. Could you provide your **email address**?",
        }
        return prompts.get(intent, "Could you please provide your **email address**?")

    # ── Intents that don't need DB lookups — return the RNN response ─────────
    return rnn_response


def _handle_order_lookup(sess: Session, order_id: str) -> str:
    intent = sess.pending_intent
    sess.waiting_for = None
    sess.pending_intent = None

    order = lookup_order(order_id)
    if not order:
        return (f"I couldn't find order **{order_id}** in our system. "
                "Please double-check the order number and try again.")

    if intent == "track_order":
        return f"Here are the details for your order:\n\n{_fmt_order(order)}"

    if intent == "cancel_order":
        if order["status"] in ("Delivered", "Cancelled"):
            return (f"Order **{order_id}** is already **{order['status']}** "
                    "and cannot be cancelled.")
        return (f"Order **{order_id}** ({order['product']}) is currently "
                f"**{order['status']}**.\n\n"
                "To confirm cancellation, please contact our support team "
                "or reply with **CONFIRM** to proceed.")

    if intent in ("get_refund", "track_refund", "check_cancellation_fee"):
        refund = lookup_refund_by_order(order_id)
        if refund:
            return (f"Here is the refund information for order "
                    f"**{order_id}**:\n\n{_fmt_refund(refund)}")
        return (f"There is no refund on record for order **{order_id}** "
                f"({order['product']}, status: {order['status']}). "
                "Would you like to request one?")

    return f"Here are the details:\n\n{_fmt_order(order)}"


def _handle_email_lookup(sess: Session, email: str) -> str:
    intent = sess.pending_intent
    sess.waiting_for = None
    sess.pending_intent = None

    account = lookup_account(email)
    if not account:
        return (f"I couldn't find an account with email **{email}**. "
                "Please make sure you're using your registered email address.")

    if intent == "recover_password":
        return (f"I found your account, **{account['name']}**! "
                f"A password reset link has been sent to **{email}**. "
                "Please check your inbox (and spam folder).")

    if intent == "change_shipping_address":
        return (f"Your current shipping address is:\n"
                f"**{account['shipping_address'] or 'Not set'}**\n\n"
                "To update it, please provide your new address.")

    if intent in ("switch_account", "edit_account", "delete_account"):
        return (f"Here is your account information:\n\n"
                f"{_fmt_account(account)}\n\n"
                "How would you like to proceed?")

    return f"Here is your account:\n\n{_fmt_account(account)}"
