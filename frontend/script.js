// Configuration: HF Spaces backend URL
// Update this when deploying to Vercel
const API_BASE_URL = window.location.hostname === 'localhost' 
    ? 'http://localhost:5000'
    : 'https://YOUR-HF-SPACE-NAME.hf.space';

const API = API_BASE_URL;
const chat = document.getElementById("chat");
const input = document.getElementById("question");
const sendBtn = document.getElementById("send-btn");
const status = document.getElementById("status");
const fileInput = document.getElementById("file-input");
const fileNames = document.getElementById("file-names");
const uploadBtn = document.getElementById("upload-btn");

// ── Helpers ──────────────────────────────────────
function addMsg(text, cls) {
    const div = document.createElement("div");
    div.className = "msg " + cls;
    div.textContent = text;
    chat.appendChild(div);
    chat.scrollTop = chat.scrollHeight;
    return div;
}

function showTyping() {
    const div = document.createElement("div");
    div.className = "typing";
    div.id = "typing";
    div.textContent = "Thinking...";
    chat.appendChild(div);
    chat.scrollTop = chat.scrollHeight;
}

function hideTyping() {
    const el = document.getElementById("typing");
    if (el) el.remove();
}

// ── Health check ─────────────────────────────────
async function checkHealth() {
    try {
        const res = await fetch(API + "/health");
        const data = await res.json();
        if (data.chain_ready) {
            status.textContent = "ready";
            status.className = "ready";
        } else {
            status.textContent = "no docs loaded";
            status.className = "loading";
        }
    } catch {
        status.textContent = "backend offline";
        status.className = "error";
    }
}

// ── Send question ────────────────────────────────
async function send() {
    const q = input.value.trim();
    if (!q) return;
    addMsg(q, "user");
    input.value = "";
    sendBtn.disabled = true;
    showTyping();

    try {
        const res = await fetch(API + "/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question: q }),
        });
        hideTyping();
        if (!res.ok) {
            const err = await res.json();
            addMsg("Error: " + (err.detail || res.statusText), "system");
        } else {
            const data = await res.json();
            addMsg(data.answer, "bot");
        }
    } catch (e) {
        hideTyping();
        addMsg("Could not reach the backend.", "system");
    }
    sendBtn.disabled = false;
    input.focus();
}

// ── Upload PDFs ──────────────────────────────────
fileInput.addEventListener("change", () => {
    const names = Array.from(fileInput.files).map(f => f.name).join(", ");
    fileNames.textContent = names || "No files selected";
    uploadBtn.disabled = fileInput.files.length === 0;
});

uploadBtn.addEventListener("click", async () => {
    if (!fileInput.files.length) return;
    uploadBtn.disabled = true;
    uploadBtn.textContent = "Uploading...";
    const form = new FormData();
    for (const f of fileInput.files) form.append("files", f);

    try {
        const res = await fetch(API + "/upload", { method: "POST", body: form });
        const data = await res.json();
        if (res.ok) {
            addMsg("\u2713 " + data.message, "system");
            checkHealth();
        } else {
            addMsg("Upload error: " + (data.detail || "unknown"), "system");
        }
    } catch {
        addMsg("Could not reach the backend.", "system");
    }
    uploadBtn.textContent = "Upload";
    uploadBtn.disabled = false;
    fileInput.value = "";
    fileNames.textContent = "No files selected";
});

// ── Events ───────────────────────────────────────
sendBtn.addEventListener("click", send);
input.addEventListener("keydown", (e) => { if (e.key === "Enter") send(); });

checkHealth();
setInterval(checkHealth, 60000);
