CREATE TABLE IF NOT EXISTS orders (
    order_id    TEXT PRIMARY KEY,
    customer    TEXT NOT NULL,
    email       TEXT NOT NULL,
    product     TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'Processing',
    tracking    TEXT,
    order_date  TEXT NOT NULL,
    est_delivery TEXT
);

CREATE TABLE IF NOT EXISTS refunds (
    refund_id       TEXT PRIMARY KEY,
    order_id        TEXT NOT NULL,
    amount          REAL NOT NULL,
    status          TEXT NOT NULL DEFAULT 'Pending',
    requested_date  TEXT NOT NULL,
    processed_date  TEXT,
    FOREIGN KEY (order_id) REFERENCES orders(order_id)
);

CREATE TABLE IF NOT EXISTS accounts (
    email           TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    phone           TEXT,
    shipping_address TEXT,
    member_since    TEXT
);
