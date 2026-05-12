from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from flask import Flask, render_template_string, request, jsonify, Response
import json
import uuid
from datetime import datetime

app = Flask(__name__)
client = OpenAI(base_url="https://api.gapgpt.app/v1")

# In-memory storage for chat history
chats = {}


def chat_stream(messages):
    """Stream response from API"""
    with client.chat.completions.create(
        model="gpt-5-nano",
        max_tokens=1200,
        messages=messages,
        stream=True,
    ) as stream:
        for chunk in stream:
            if chunk.choices and len(chunk.choices) > 0 and chunk.choices[0].delta and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


@app.route("/")
def index():
    return render_template_string(
        """
        <!doctype html>
        <html lang="en">
        <head>
            <meta charset="utf-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1.0" />
            <title>Claude AI - Professional Chat</title>
            <style>
                @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
                
                :root {
                    --primary: #0f172a;
                    --secondary: #1e293b;
                    --tertiary: #334155;
                    --accent: #3b82f6;
                    --accent-light: #60a5fa;
                    --success: #10b981;
                    --warning: #f59e0b;
                    --danger: #ef4444;
                    --text-primary: #f1f5f9;
                    --text-secondary: #cbd5e1;
                    --text-tertiary: #94a3b8;
                    --border: #334155;
                    --bg-hover: rgba(51, 65, 85, 0.5);
                    --shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.3);
                    --shadow-sm: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
                }
                
                * {
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }
                
                html, body {
                    height: 100%;
                    width: 100%;
                    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                    background: linear-gradient(135deg, #0f172a 0%, #1a1f3a 50%, #0f172a 100%);
                    color: var(--text-primary);
                    overflow: hidden;
                }
                
                .container {
                    display: flex;
                    height: 100vh;
                    width: 100%;
                }
                
                /* SIDEBAR */
                .sidebar {
                    width: 320px;
                    background: rgba(15, 23, 42, 0.85);
                    border-right: 1px solid var(--border);
                    display: flex;
                    flex-direction: column;
                    backdrop-filter: blur(20px);
                    box-shadow: inset 0 0 20px rgba(0, 0, 0, 0.2);
                }
                
                .sidebar-header {
                    padding: 28px 24px;
                    border-bottom: 1px solid var(--border);
                }
                
                .logo {
                    display: flex;
                    align-items: center;
                    gap: 12px;
                    font-size: 21px;
                    font-weight: 700;
                    letter-spacing: -0.5px;
                    color: var(--accent);
                    margin-bottom: 20px;
                }
                
                .logo::before {
                    content: "✨";
                    font-size: 26px;
                    display: flex;
                    align-items: center;
                }
                
                .new-chat-btn {
                    width: 100%;
                    padding: 14px 16px;
                    background: linear-gradient(135deg, var(--accent) 0%, var(--accent-light) 100%);
                    color: white;
                    border: none;
                    border-radius: 12px;
                    font-weight: 600;
                    cursor: pointer;
                    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                    font-size: 14px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    gap: 8px;
                    box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
                }
                
                .new-chat-btn:hover {
                    transform: translateY(-2px);
                    box-shadow: 0 8px 25px rgba(59, 130, 246, 0.4);
                }
                
                .new-chat-btn:active {
                    transform: translateY(0);
                }
                
                .chat-list-header {
                    padding: 16px 24px;
                    font-size: 11px;
                    font-weight: 700;
                    letter-spacing: 1px;
                    color: var(--text-tertiary);
                    text-transform: uppercase;
                    border-bottom: 1px solid var(--border);
                }
                
                .chat-history {
                    flex: 1;
                    overflow-y: auto;
                    padding: 12px;
                }
                
                .chat-item {
                    padding: 12px 16px;
                    margin-bottom: 8px;
                    border-radius: 10px;
                    cursor: pointer;
                    font-size: 13px;
                    color: var(--text-secondary);
                    transition: all 0.2s ease;
                    border: 1px solid transparent;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    position: relative;
                    group: "chat-item";
                }
                
                .chat-item:hover {
                    background: var(--bg-hover);
                    color: var(--text-primary);
                    border-color: rgba(59, 130, 246, 0.2);
                }
                
                .chat-item.active {
                    background: rgba(59, 130, 246, 0.15);
                    color: var(--accent);
                    border-color: var(--accent);
                    font-weight: 600;
                }
                
                .chat-item-title {
                    flex: 1;
                    white-space: nowrap;
                    overflow: hidden;
                    text-overflow: ellipsis;
                }
                
                .chat-item-actions {
                    display: none;
                    gap: 4px;
                    margin-left: 8px;
                }
                
                .chat-item:hover .chat-item-actions {
                    display: flex;
                }
                
                .chat-action-btn {
                    background: none;
                    border: none;
                    color: var(--text-tertiary);
                    cursor: pointer;
                    padding: 4px;
                    border-radius: 6px;
                    transition: all 0.2s ease;
                    font-size: 14px;
                    display: flex;
                    align-items: center;
                }
                
                .chat-action-btn:hover {
                    background: rgba(59, 130, 246, 0.2);
                    color: var(--accent);
                }
                
                .chat-action-btn.delete:hover {
                    background: rgba(239, 68, 68, 0.2);
                    color: var(--danger);
                }
                
                .sidebar-footer {
                    padding: 16px 24px;
                    border-top: 1px solid var(--border);
                    display: flex;
                    gap: 12px;
                    font-size: 12px;
                    color: var(--text-tertiary);
                }
                
                /* MAIN CONTENT */
                .main-content {
                    flex: 1;
                    display: flex;
                    flex-direction: column;
                    background: linear-gradient(180deg, rgba(15, 23, 42, 0.8) 0%, rgba(30, 41, 59, 0.4) 100%);
                }
                
                .header {
                    padding: 24px 32px;
                    border-bottom: 1px solid var(--border);
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    backdrop-filter: blur(10px);
                }
                
                .header-title {
                    font-size: 26px;
                    font-weight: 700;
                    letter-spacing: -0.5px;
                    background: linear-gradient(135deg, var(--accent-light), var(--accent));
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    background-clip: text;
                }
                
                .header-info {
                    display: flex;
                    align-items: center;
                    gap: 12px;
                    color: var(--text-secondary);
                    font-size: 13px;
                }
                
                .status-indicator {
                    width: 10px;
                    height: 10px;
                    border-radius: 50%;
                    background: var(--success);
                    animation: pulse 2s infinite;
                    box-shadow: 0 0 10px rgba(16, 185, 129, 0.5);
                }
                
                @keyframes pulse {
                    0%, 100% { opacity: 1; }
                    50% { opacity: 0.5; }
                }
                
                /* MESSAGES */
                .messages-container {
                    flex: 1;
                    overflow-y: auto;
                    padding: 32px;
                    display: flex;
                    flex-direction: column;
                    gap: 24px;
                }
                
                .empty-state {
                    flex: 1;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    text-align: center;
                    gap: 20px;
                    color: var(--text-tertiary);
                }
                
                .empty-state-icon {
                    font-size: 56px;
                    opacity: 0.4;
                }
                
                .empty-state h2 {
                    font-size: 24px;
                    font-weight: 700;
                    color: var(--text-secondary);
                }
                
                .empty-state p {
                    font-size: 14px;
                    max-width: 400px;
                }
                
                .message {
                    display: flex;
                    gap: 16px;
                    max-width: 80%;
                    animation: slideIn 0.4s cubic-bezier(0.4, 0, 0.2, 1);
                }
                
                @keyframes slideIn {
                    from {
                        opacity: 0;
                        transform: translateY(12px);
                    }
                    to {
                        opacity: 1;
                        transform: translateY(0);
                    }
                }
                
                .message.user {
                    margin-left: auto;
                    flex-direction: row-reverse;
                }
                
                .message-avatar {
                    width: 36px;
                    height: 36px;
                    border-radius: 10px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 18px;
                    flex-shrink: 0;
                    font-weight: 700;
                }
                
                .message.user .message-avatar {
                    background: linear-gradient(135deg, var(--accent), var(--accent-light));
                    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
                }
                
                .message.assistant .message-avatar {
                    background: rgba(59, 130, 246, 0.15);
                    border: 2px solid var(--accent);
                }
                
                .message-content {
                    background: rgba(51, 65, 85, 0.35);
                    border: 1px solid rgba(148, 163, 184, 0.12);
                    border-radius: 16px;
                    padding: 16px 20px;
                    line-height: 1.7;
                    word-wrap: break-word;
                    white-space: pre-wrap;
                    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
                    backdrop-filter: blur(5px);
                }
                
                .message.user .message-content {
                    background: linear-gradient(135deg, rgba(59, 130, 246, 0.25), rgba(96, 165, 250, 0.15));
                    border-color: rgba(59, 130, 246, 0.4);
                }
                
                .message.assistant .message-content {
                    background: rgba(51, 65, 85, 0.4);
                    border-color: rgba(148, 163, 184, 0.2);
                }
                
                /* TYPING INDICATOR */
                .typing-indicator {
                    display: flex;
                    gap: 6px;
                    padding: 12px 0;
                }
                
                .typing-dot {
                    width: 10px;
                    height: 10px;
                    border-radius: 50%;
                    background: var(--accent);
                    animation: typingAnimation 1.4s infinite;
                    box-shadow: 0 0 6px rgba(59, 130, 246, 0.5);
                }
                
                .typing-dot:nth-child(2) {
                    animation-delay: 0.2s;
                }
                
                .typing-dot:nth-child(3) {
                    animation-delay: 0.4s;
                }
                
                @keyframes typingAnimation {
                    0%, 60%, 100% { opacity: 0.5; transform: translateY(0); }
                    30% { opacity: 1; transform: translateY(-12px); }
                }
                
                /* INPUT AREA */
                .input-area {
                    padding: 28px 32px;
                    border-top: 1px solid var(--border);
                    background: rgba(15, 23, 42, 0.95);
                    backdrop-filter: blur(10px);
                }
                
                .input-wrapper {
                    display: flex;
                    gap: 12px;
                    align-items: flex-end;
                }
                
                textarea {
                    flex: 1;
                    padding: 14px 18px;
                    border: 1px solid var(--border);
                    border-radius: 14px;
                    background: rgba(30, 41, 59, 0.6);
                    color: var(--text-primary);
                    font-family: 'Inter', sans-serif;
                    font-size: 14px;
                    outline: none;
                    resize: none;
                    max-height: 140px;
                    transition: all 0.3s ease;
                }
                
                textarea::placeholder {
                    color: var(--text-tertiary);
                }
                
                textarea:focus {
                    border-color: var(--accent);
                    background: rgba(30, 41, 59, 0.8);
                    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
                }
                
                .send-button {
                    padding: 14px 24px;
                    background: linear-gradient(135deg, var(--accent), var(--accent-light));
                    color: white;
                    border: none;
                    border-radius: 12px;
                    font-weight: 600;
                    cursor: pointer;
                    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    white-space: nowrap;
                    font-size: 14px;
                    box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
                }
                
                .send-button:hover:not(:disabled) {
                    transform: translateY(-2px);
                    box-shadow: 0 8px 25px rgba(59, 130, 246, 0.4);
                }
                
                .send-button:active:not(:disabled) {
                    transform: translateY(0);
                }
                
                .send-button:disabled {
                    opacity: 0.6;
                    cursor: not-allowed;
                }
                
                /* MODAL */
                .modal {
                    display: none;
                    position: fixed;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    background: rgba(0, 0, 0, 0.7);
                    backdrop-filter: blur(5px);
                    z-index: 1000;
                    align-items: center;
                    justify-content: center;
                }
                
                .modal.active {
                    display: flex;
                }
                
                .modal-content {
                    background: var(--secondary);
                    border: 1px solid var(--border);
                    border-radius: 16px;
                    padding: 32px;
                    width: 90%;
                    max-width: 400px;
                    box-shadow: var(--shadow);
                }
                
                .modal-title {
                    font-size: 20px;
                    font-weight: 700;
                    margin-bottom: 16px;
                    color: var(--text-primary);
                }
                
                .modal-input {
                    width: 100%;
                    padding: 12px 16px;
                    border: 1px solid var(--border);
                    border-radius: 10px;
                    background: var(--primary);
                    color: var(--text-primary);
                    font-family: 'Inter', sans-serif;
                    font-size: 14px;
                    margin-bottom: 20px;
                    outline: none;
                }
                
                .modal-input:focus {
                    border-color: var(--accent);
                    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
                }
                
                .modal-actions {
                    display: flex;
                    gap: 12px;
                }
                
                .modal-btn {
                    flex: 1;
                    padding: 12px 16px;
                    border: none;
                    border-radius: 10px;
                    font-weight: 600;
                    cursor: pointer;
                    transition: all 0.2s ease;
                    font-size: 14px;
                }
                
                .modal-btn-primary {
                    background: var(--accent);
                    color: white;
                }
                
                .modal-btn-primary:hover {
                    background: var(--accent-light);
                }
                
                .modal-btn-secondary {
                    background: var(--tertiary);
                    color: var(--text-primary);
                }
                
                .modal-btn-secondary:hover {
                    background: var(--border);
                }
                
                /* SCROLLBAR */
                ::-webkit-scrollbar {
                    width: 8px;
                    height: 8px;
                }
                
                ::-webkit-scrollbar-track {
                    background: transparent;
                }
                
                ::-webkit-scrollbar-thumb {
                    background: rgba(148, 163, 184, 0.25);
                    border-radius: 4px;
                }
                
                ::-webkit-scrollbar-thumb:hover {
                    background: rgba(148, 163, 184, 0.4);
                }
            </style>
        </head>
        <body>
            <div class="container">
                <!-- SIDEBAR -->
                <div class="sidebar">
                    <div class="sidebar-header">
                        <div class="logo">Claude AI</div>
                        <button class="new-chat-btn" id="new-chat-btn">+ New Chat</button>
                    </div>
                    
                    <div class="chat-list-header">Recent Chats</div>
                    <div class="chat-history" id="chat-history"></div>
                    
                    <div class="sidebar-footer">
                        <span>✨ AI-Powered Conversations</span>
                    </div>
                </div>
                
                <!-- MAIN CONTENT -->
                <div class="main-content">
                    <div class="header">
                        <div class="header-title" id="header-title">Let's Chat</div>
                        <div class="header-info">
                            <div class="status-indicator"></div>
                            <span>Ready</span>
                        </div>
                    </div>
                    
                    <div class="messages-container" id="messages">
                        <div class="empty-state">
                            <div class="empty-state-icon">💬</div>
                            <h2>Start Chatting</h2>
                            <p>Create a new conversation or select one from your history to continue where you left off</p>
                        </div>
                    </div>
                    
                    <div class="input-area">
                        <form id="chat-form" style="display: flex; flex-direction: column; gap: 12px;">
                            <div class="input-wrapper">
                                <textarea id="user-input" placeholder="Type your message here... (Shift+Enter for new line)" autocomplete="off" rows="1"></textarea>
                                <button type="submit" class="send-button" id="send-button">
                                    <span>Send</span>
                                    <span>↑</span>
                                </button>
                            </div>
                        </form>
                    </div>
                </div>
            </div>
            
            <!-- RENAME MODAL -->
            <div class="modal" id="rename-modal">
                <div class="modal-content">
                    <div class="modal-title">Rename Chat</div>
                    <input type="text" class="modal-input" id="rename-input" placeholder="Enter new chat name">
                    <div class="modal-actions">
                        <button class="modal-btn modal-btn-secondary" id="rename-cancel">Cancel</button>
                        <button class="modal-btn modal-btn-primary" id="rename-confirm">Save</button>
                    </div>
                </div>
            </div>
            
            <script>
                const messagesEl = document.getElementById("messages");
                const form = document.getElementById("chat-form");
                const input = document.getElementById("user-input");
                const sendButton = document.getElementById("send-button");
                const newChatBtn = document.getElementById("new-chat-btn");
                const chatHistory = document.getElementById("chat-history");
                const headerTitle = document.getElementById("header-title");
                const renameModal = document.getElementById("rename-modal");
                const renameInput = document.getElementById("rename-input");
                const renameConfirm = document.getElementById("rename-confirm");
                const renameCancel = document.getElementById("rename-cancel");
                
                let chats = JSON.parse(localStorage.getItem("chats")) || {};
                let currentChatId = null;
                let currentChatMessages = [];
                let renamingChatId = null;
                
                function generateChatId() {
                    return Date.now().toString();
                }
                
                function saveChatToStorage() {
                    localStorage.setItem("chats", JSON.stringify(chats));
                }
                
                function createNewChat() {
                    const chatId = generateChatId();
                    chats[chatId] = {
                        id: chatId,
                        name: "New Conversation",
                        messages: [],
                        pinned: false,
                        createdAt: new Date().toISOString()
                    };
                    saveChatToStorage();
                    loadChat(chatId);
                    renderChatList();
                }
                
                function loadChat(chatId) {
                    currentChatId = chatId;
                    const chat = chats[chatId];
                    if (!chat) return;
                    
                    currentChatMessages = chat.messages || [];
                    headerTitle.textContent = chat.name;
                    renderMessages();
                    renderChatList();
                }
                
                function renderChatList() {
                    chatHistory.innerHTML = "";
                    
                    const pinnedChats = Object.values(chats).filter(c => c.pinned).sort((a, b) => b.createdAt - a.createdAt);
                    const unpinnedChats = Object.values(chats).filter(c => !c.pinned).sort((a, b) => b.createdAt - a.createdAt);
                    
                    [...pinnedChats, ...unpinnedChats].forEach(chat => {
                        const item = document.createElement("div");
                        item.className = `chat-item ${chat.id === currentChatId ? "active" : ""}`;
                        item.innerHTML = `
                            <div class="chat-item-title">${chat.pinned ? "📌 " : ""}${chat.name}</div>
                            <div class="chat-item-actions">
                                <button class="chat-action-btn pin-btn" title="Pin/Unpin">
                                    ${chat.pinned ? "📍" : "📌"}
                                </button>
                                <button class="chat-action-btn rename-btn" title="Rename">✏️</button>
                                <button class="chat-action-btn delete delete" title="Delete">🗑️</button>
                            </div>
                        `;
                        
                        item.addEventListener("click", () => loadChat(chat.id));
                        
                        item.querySelector(".pin-btn").addEventListener("click", (e) => {
                            e.stopPropagation();
                            chats[chat.id].pinned = !chats[chat.id].pinned;
                            saveChatToStorage();
                            renderChatList();
                        });
                        
                        item.querySelector(".rename-btn").addEventListener("click", (e) => {
                            e.stopPropagation();
                            renamingChatId = chat.id;
                            renameInput.value = chat.name;
                            renameModal.classList.add("active");
                            renameInput.focus();
                        });
                        
                        item.querySelector(".delete").addEventListener("click", (e) => {
                            e.stopPropagation();
                            delete chats[chat.id];
                            saveChatToStorage();
                            if (chat.id === currentChatId) {
                                currentChatId = null;
                                messagesEl.innerHTML = `
                                    <div class="empty-state">
                                        <div class="empty-state-icon">💬</div>
                                        <h2>Start Chatting</h2>
                                        <p>Create a new conversation or select one from your history</p>
                                    </div>
                                `;
                            }
                            renderChatList();
                        });
                        
                        chatHistory.appendChild(item);
                    });
                }
                
                function renderMessages() {
                    if (currentChatMessages.length === 0) {
                        messagesEl.innerHTML = `
                            <div class="empty-state">
                                <div class="empty-state-icon">💬</div>
                                <h2>Start Chatting</h2>
                                <p>Create a new conversation or select one from your history</p>
                            </div>
                        `;
                        return;
                    }
                    
                    messagesEl.innerHTML = "";
                    currentChatMessages.forEach(msg => {
                        appendMessageToDOM(msg.role, msg.content, false);
                    });
                }
                
                function appendMessageToDOM(role, text, scroll = true) {
                    const messageEl = document.createElement("div");
                    messageEl.className = `message ${role}`;
                    
                    const avatar = role === "user" ? "👤" : "🤖";
                    messageEl.innerHTML = `
                        <div class="message-avatar">${avatar}</div>
                        <div class="message-content">${text}</div>
                    `;
                    messagesEl.appendChild(messageEl);
                    if (scroll) messagesEl.scrollTop = messagesEl.scrollHeight;
                    return messageEl;
                }
                
                function createTypingIndicator() {
                    const messageEl = document.createElement("div");
                    messageEl.className = "message assistant";
                    messageEl.id = "typing-message";
                    messageEl.innerHTML = `
                        <div class="message-avatar">🤖</div>
                        <div class="message-content">
                            <div class="typing-indicator">
                                <div class="typing-dot"></div>
                                <div class="typing-dot"></div>
                                <div class="typing-dot"></div>
                            </div>
                        </div>
                    `;
                    messagesEl.appendChild(messageEl);
                    messagesEl.scrollTop = messagesEl.scrollHeight;
                    return messageEl;
                }
                
                async function sendMessage() {
                    const text = input.value.trim();
                    if (!text) return;
                    
                    if (!currentChatId) {
                        createNewChat();
                    }
                    
                    appendMessageToDOM("user", text);
                    currentChatMessages.push({ role: "user", content: text });
                    
                    if (!chats[currentChatId].name.includes("Conversation")) {
                        chats[currentChatId].name = text.substring(0, 40) + (text.length > 40 ? "..." : "");
                    }
                    
                    input.value = "";
                    input.style.height = "auto";
                    sendButton.disabled = true;
                    saveChatToStorage();
                    
                    const typingMsg = createTypingIndicator();
                    renderChatList();

                    try {
                        const response = await fetch("/api/chat-stream", {
                            method: "POST",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify({ messages: currentChatMessages })
                        });
                        
                        const reader = response.body.getReader();
                        const decoder = new TextDecoder();
                        let fullResponse = "";
                        
                        typingMsg.remove();
                        const assistantMsg = appendMessageToDOM("assistant", "");
                        const contentDiv = assistantMsg.querySelector(".message-content");
                        
                        while (true) {
                            const { done, value } = await reader.read();
                            if (done) break;
                            
                            const chunk = decoder.decode(value);
                            fullResponse += chunk;
                            contentDiv.textContent = fullResponse;
                            messagesEl.scrollTop = messagesEl.scrollHeight;
                        }
                        
                        currentChatMessages.push({ role: "assistant", content: fullResponse });
                        chats[currentChatId].messages = currentChatMessages;
                        saveChatToStorage();
                    } catch (err) {
                        console.error(err);
                        typingMsg.remove();
                        appendMessageToDOM("assistant", "Sorry, something went wrong. Please try again.");
                    } finally {
                        sendButton.disabled = false;
                    }
                }
                
                form.addEventListener("submit", (event) => {
                    event.preventDefault();
                    sendMessage();
                });

                input.addEventListener("keydown", (event) => {
                    if (event.key === "Enter" && !event.shiftKey) {
                        event.preventDefault();
                        form.requestSubmit();
                    }
                });
                
                input.addEventListener("input", () => {
                    input.style.height = "auto";
                    input.style.height = Math.min(input.scrollHeight, 140) + "px";
                });
                
                newChatBtn.addEventListener("click", createNewChat);
                
                renameConfirm.addEventListener("click", () => {
                    const newName = renameInput.value.trim();
                    if (newName && renamingChatId) {
                        chats[renamingChatId].name = newName;
                        if (renamingChatId === currentChatId) {
                            headerTitle.textContent = newName;
                        }
                        saveChatToStorage();
                        renderChatList();
                    }
                    renameModal.classList.remove("active");
                });
                
                renameCancel.addEventListener("click", () => {
                    renameModal.classList.remove("active");
                });
                
                renameInput.addEventListener("keydown", (e) => {
                    if (e.key === "Enter") {
                        renameConfirm.click();
                    }
                    if (e.key === "Escape") {
                        renameCancel.click();
                    }
                });
                
                // Initialize
                renderChatList();
            </script>
        </body>
        </html>
        """
    )


@app.route("/api/chat-stream", methods=["POST"])
def api_chat_stream():
    body = request.get_json(force=True, silent=True) or {}
    messages = body.get("messages", [])
    
    def generate():
        try:
            for chunk in chat_stream(messages):
                yield chunk
        except Exception as e:
            yield f"Error: {str(e)}"
    
    return Response(generate(), mimetype="text/plain")


if __name__ == "__main__":
    app.run(debug=True, port=5000)