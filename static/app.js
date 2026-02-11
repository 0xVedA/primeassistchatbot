/* ═══════════════════════════════════════════════════════════
   PrimeAssist — Frontend Logic
   ═══════════════════════════════════════════════════════════ */

const chatMessages = document.getElementById("chatMessages");
const userInput = document.getElementById("userInput");
const sendBtn = document.getElementById("sendBtn");
const clearBtn = document.getElementById("clearChat");
const intentBadge = document.getElementById("intentBadge");
const intentValue = document.getElementById("intentValue");
const confValue = document.getElementById("confValue");
const menuToggle = document.getElementById("menuToggle");
const sidebar = document.querySelector(".sidebar");

let sessionId = crypto.randomUUID ? crypto.randomUUID() : Date.now().toString();

// ── Helpers ──────────────────────────────────────────────

function scrollToBottom() {
    requestAnimationFrame(() => {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    });
}

function getTime() {
    return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

/** Minimal markdown-like rendering (bold, line breaks) */
function renderText(text) {
    return text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
        .replace(/\n/g, "<br>");
}

// ── Add Messages ─────────────────────────────────────────

function addUserMessage(text) {
    const div = document.createElement("div");
    div.className = "message user-message fade-in";
    div.innerHTML = `
        <div class="message-avatar"></div>
        <div class="message-content">
            <div class="message-bubble">${renderText(text)}</div>
            <div class="message-time">${getTime()}</div>
        </div>`;
    chatMessages.appendChild(div);
    scrollToBottom();
}

function addBotMessage(text) {
    const div = document.createElement("div");
    div.className = "message bot-message fade-in";
    div.innerHTML = `
        <div class="message-avatar">
            <svg width="20" height="20" viewBox="0 0 28 28" fill="none">
                <circle cx="14" cy="14" r="12" stroke="url(#grad2)" stroke-width="2"/>
                <circle cx="14" cy="14" r="6" fill="url(#grad2)"/>
            </svg>
        </div>
        <div class="message-content">
            <div class="message-bubble">${renderText(text)}</div>
            <div class="message-time">PrimeAssist · ${getTime()}</div>
        </div>`;
    chatMessages.appendChild(div);
    scrollToBottom();
}

function showTyping() {
    const div = document.createElement("div");
    div.className = "message bot-message fade-in";
    div.id = "typingMsg";
    div.innerHTML = `
        <div class="message-avatar">
            <svg width="20" height="20" viewBox="0 0 28 28" fill="none">
                <circle cx="14" cy="14" r="12" stroke="url(#grad2)" stroke-width="2"/>
                <circle cx="14" cy="14" r="6" fill="url(#grad2)"/>
            </svg>
        </div>
        <div class="message-content">
            <div class="typing-indicator">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        </div>`;
    chatMessages.appendChild(div);
    scrollToBottom();
}

function hideTyping() {
    const el = document.getElementById("typingMsg");
    if (el) el.remove();
}

function updateBadge(intent, conf) {
    intentBadge.style.display = "flex";
    intentValue.textContent = intent.replace(/_/g, " ");
    confValue.textContent = `${Math.round(conf * 100)}%`;
}

// ── API Call ──────────────────────────────────────────────

async function sendMessage(text) {
    addUserMessage(text);
    userInput.value = "";
    autoResize();
    sendBtn.disabled = true;
    showTyping();

    try {
        const res = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: text, session_id: sessionId }),
        });
        const data = await res.json();
        hideTyping();
        addBotMessage(data.reply);
        if (data.intent) updateBadge(data.intent, data.confidence);
    } catch (err) {
        hideTyping();
        addBotMessage("Sorry, something went wrong. Please try again.");
        console.error(err);
    }

    sendBtn.disabled = false;
    userInput.focus();
}

// ── Event Listeners ──────────────────────────────────────

sendBtn.addEventListener("click", () => {
    const text = userInput.value.trim();
    if (text) sendMessage(text);
});

userInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        const text = userInput.value.trim();
        if (text) sendMessage(text);
    }
});

// Auto-resize textarea
function autoResize() {
    userInput.style.height = "auto";
    userInput.style.height = Math.min(userInput.scrollHeight, 120) + "px";
}
userInput.addEventListener("input", autoResize);

// Clear chat
clearBtn.addEventListener("click", () => {
    sessionId = crypto.randomUUID ? crypto.randomUUID() : Date.now().toString();
    // Remove all messages except the welcome
    const msgs = chatMessages.querySelectorAll(".message");
    msgs.forEach((m, i) => { if (i > 0) m.remove(); });
    intentBadge.style.display = "none";
});

// Sidebar quick actions
document.querySelectorAll(".nav-btn[data-msg]").forEach((btn) => {
    btn.addEventListener("click", () => {
        const msg = btn.dataset.msg;
        if (msg) sendMessage(msg);
        // Close mobile sidebar
        sidebar.classList.remove("open");
        const overlay = document.querySelector(".sidebar-overlay");
        if (overlay) overlay.classList.remove("show");
    });
});

// Mobile menu
menuToggle.addEventListener("click", () => {
    sidebar.classList.toggle("open");
    let overlay = document.querySelector(".sidebar-overlay");
    if (!overlay) {
        overlay = document.createElement("div");
        overlay.className = "sidebar-overlay";
        document.body.appendChild(overlay);
        overlay.addEventListener("click", () => {
            sidebar.classList.remove("open");
            overlay.classList.remove("show");
        });
    }
    overlay.classList.toggle("show");
});
