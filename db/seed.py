"""
seed.py — Populate the PrimeAssist SQLite database with sample data.
Usage:  python -m db.seed
"""

import sqlite3, os, pathlib

DB_PATH = pathlib.Path(__file__).resolve().parent / "primeassist.db"
SCHEMA  = pathlib.Path(__file__).resolve().parent / "schema.sql"


def seed():
    if DB_PATH.exists():
        DB_PATH.unlink()

    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    # Create tables
    cur.executescript(SCHEMA.read_text())

    # ── Accounts ──────────────────────────────────────────────────────────────
    accounts = [
        ("alice@example.com",   "Alice Johnson",   "+1-555-0101", "123 Oak St, New York, NY 10001",      "2024-03-15"),
        ("bob@example.com",     "Bob Smith",        "+1-555-0102", "456 Maple Ave, Los Angeles, CA 90001", "2024-06-20"),
        ("charlie@example.com", "Charlie Davis",    "+1-555-0103", "789 Pine Rd, Chicago, IL 60601",       "2024-09-10"),
        ("diana@example.com",   "Diana Lee",        "+1-555-0104", "321 Elm Blvd, Houston, TX 77001",      "2025-01-05"),
        ("eve@example.com",     "Eve Martinez",     "+1-555-0105", "654 Cedar Ln, Phoenix, AZ 85001",      "2025-04-22"),
    ]
    cur.executemany("INSERT INTO accounts VALUES (?,?,?,?,?)", accounts)

    # ── Orders ────────────────────────────────────────────────────────────────
    orders = [
        ("ORD-1001", "Alice Johnson",   "alice@example.com",   "Wireless Headphones",      "Delivered",   "TRK-50001", "2026-01-20", "2026-01-27"),
        ("ORD-1002", "Alice Johnson",   "alice@example.com",   "Phone Case (Black)",       "Shipped",     "TRK-50002", "2026-02-05", "2026-02-14"),
        ("ORD-1003", "Bob Smith",       "bob@example.com",     "Mechanical Keyboard",      "Processing",  None,        "2026-02-09", "2026-02-18"),
        ("ORD-1004", "Bob Smith",       "bob@example.com",     "USB-C Hub",                "Delivered",   "TRK-50004", "2026-01-10", "2026-01-16"),
        ("ORD-1005", "Charlie Davis",   "charlie@example.com", "Running Shoes (Size 10)",  "Shipped",     "TRK-50005", "2026-02-07", "2026-02-15"),
        ("ORD-1006", "Charlie Davis",   "charlie@example.com", "Yoga Mat",                 "Cancelled",   None,        "2026-02-01", None),
        ("ORD-1007", "Diana Lee",       "diana@example.com",   "Smartwatch Pro",           "Delivered",   "TRK-50007", "2026-01-25", "2026-02-01"),
        ("ORD-1008", "Diana Lee",       "diana@example.com",   "Laptop Stand",             "Processing",  None,        "2026-02-10", "2026-02-19"),
        ("ORD-1009", "Eve Martinez",    "eve@example.com",     "Bluetooth Speaker",        "Shipped",     "TRK-50009", "2026-02-06", "2026-02-13"),
        ("ORD-1010", "Eve Martinez",    "eve@example.com",     "Desk Lamp LED",            "Delivered",   "TRK-50010", "2026-01-15", "2026-01-22"),
    ]
    cur.executemany("INSERT INTO orders VALUES (?,?,?,?,?,?,?,?)", orders)

    # ── Refunds ───────────────────────────────────────────────────────────────
    refunds = [
        ("REF-2001", "ORD-1001", 59.99, "Completed",  "2026-01-30", "2026-02-04"),
        ("REF-2002", "ORD-1006", 29.99, "Completed",  "2026-02-02", "2026-02-06"),
        ("REF-2003", "ORD-1004", 34.99, "Processing", "2026-02-08", None),
        ("REF-2004", "ORD-1007", 199.99,"Pending",    "2026-02-10", None),
    ]
    cur.executemany("INSERT INTO refunds VALUES (?,?,?,?,?,?)", refunds)

    con.commit()
    con.close()
    print(f"[OK] Database seeded at {DB_PATH}")
    print(f"     {len(accounts)} accounts, {len(orders)} orders, {len(refunds)} refunds")


if __name__ == "__main__":
    seed()
