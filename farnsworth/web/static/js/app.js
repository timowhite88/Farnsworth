/**
 * Farnsworth AI - Neural Interface v3.0
 * Full-stack chat interface with all local features
 */

// ============================================
// STATE MANAGEMENT
// ============================================

const state = {
    wallet: null,
    verified: false,
    voiceEnabled: true,
    sidebarOpen: {
        left: window.innerWidth > 992,
        right: window.innerWidth > 992
    },
    currentProfile: 'default',
    focusTimer: {
        active: false,
        duration: 25 * 60,
        remaining: 25 * 60,
        task: '',
        interval: null
    },
    ws: null,
    wsConnected: false,
    chatHistory: [],
    features: {}
};

// ============================================
// INITIALIZATION
// ============================================

document.addEventListener('DOMContentLoaded', () => {
    initApp();
});

async function initApp() {
    // Check server status
    await checkServerStatus();

    // Initialize WebSocket
    initWebSocket();

    // Set up event listeners
    setupEventListeners();

    // Initialize Neural Canvas
    initNeuralCanvas();

    // Load initial data
    loadNotes();
    loadSnippets();
    loadHealthSummary();

    // Check for existing wallet connection
    checkExistingWallet();
}

// ============================================
// SERVER STATUS
// ============================================

async function checkServerStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();

        state.features = data.features || {};

        // Update status indicators
        updateStatusIndicator('memory', data.features?.memory);
        updateStatusIndicator('notes', data.features?.notes);
        updateStatusIndicator('health', data.features?.health);
        updateStatusIndicator('tools', data.features?.tools);
        updateStatusIndicator('thinking', data.features?.thinking);

        if (data.demo_mode) {
            const badge = document.getElementById('feature-badge');
            if (badge) {
                badge.textContent = '🧪 Demo Mode Active';
                badge.style.color = 'var(--warning)';
            }
        }

        return data;
    } catch (error) {
        console.error('Server status check failed:', error);
        showToast('Could not connect to server', 'error');
    }
}

function updateStatusIndicator(name, available) {
    const el = document.getElementById(`status-${name}`);
    if (el) {
        el.textContent = available ? '✅' : '❌';
        el.classList.toggle('online', available);
        el.classList.toggle('offline', !available);
    }
}

// ============================================
// WEBSOCKET
// ============================================

function initWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/live`;

    try {
        state.ws = new WebSocket(wsUrl);

        state.ws.onopen = () => {
            state.wsConnected = true;
            updateConnectionStatus('connected');
            console.log('WebSocket connected');
        };

        state.ws.onclose = () => {
            state.wsConnected = false;
            updateConnectionStatus('disconnected');
            // Reconnect after 3 seconds
            setTimeout(initWebSocket, 3000);
        };

        state.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            updateConnectionStatus('disconnected');
        };

        state.ws.onmessage = (event) => {
            handleWebSocketMessage(JSON.parse(event.data));
        };
    } catch (error) {
        console.error('WebSocket init failed:', error);
        updateConnectionStatus('disconnected');
    }
}

function updateConnectionStatus(status) {
    const dot = document.getElementById('connection-dot');
    const text = document.getElementById('connection-status-text');

    if (dot && text) {
        dot.className = 'status-dot ' + status;
        text.textContent = status === 'connected' ? 'Neural Link Active' :
                           status === 'connecting' ? 'Connecting...' : 'Disconnected';
    }
}

function handleWebSocketMessage(data) {
    switch (data.type) {
        case 'connected':
            showToast('Connected to Farnsworth Live Feed', 'success');
            break;
        case 'memory_stored':
            showToast('Memory stored!', 'success');
            break;
        case 'memory_recalled':
            showToast(`Found ${data.data?.count || 0} memories`, 'success');
            break;
        case 'note_added':
            loadNotes();
            break;
        case 'thinking_step':
            renderThinkingStep(data.data);
            break;
        case 'thinking_end':
            // Handled by the thinking modal
            break;
        case 'focus_start':
            showToast('Focus session started!', 'success');
            break;
        case 'focus_end':
            showToast('Focus session complete!', 'success');
            break;
        case 'error':
            showToast(data.data?.error || 'An error occurred', 'error');
            break;
        case 'heartbeat':
        case 'pong':
            // Silent heartbeat
            break;
        default:
            console.log('WS event:', data);
    }
}

// ============================================
// EVENT LISTENERS
// ============================================

function setupEventListeners() {
    // Connect button
    const connectBtn = document.getElementById('connect-btn');
    if (connectBtn) {
        connectBtn.addEventListener('click', connectWallet);
    }

    // Chat input
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');

    if (userInput) {
        userInput.addEventListener('input', () => {
            const count = userInput.value.length;
            document.getElementById('char-count').textContent = count;
            sendBtn.disabled = count === 0;
            autoResizeTextarea(userInput);
        });

        userInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
    }

    if (sendBtn) {
        sendBtn.addEventListener('click', sendMessage);
    }

    // Voice toggle
    const voiceToggle = document.getElementById('voice-toggle');
    if (voiceToggle) {
        voiceToggle.addEventListener('click', () => {
            state.voiceEnabled = !state.voiceEnabled;
            voiceToggle.classList.toggle('active', state.voiceEnabled);
        });
    }

    // Mic button for voice input
    const micBtn = document.getElementById('mic-btn');
    if (micBtn) {
        micBtn.addEventListener('click', toggleVoiceInput);
    }

    // Sidebar toggle
    const sidebarToggle = document.getElementById('sidebar-toggle');
    if (sidebarToggle) {
        sidebarToggle.addEventListener('click', () => {
            document.getElementById('left-sidebar')?.classList.toggle('active');
        });
    }

    // Profile switcher
    document.querySelectorAll('.profile-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const profile = btn.dataset.profile;
            switchProfile(profile);
        });
    });

    // Memory controls
    document.getElementById('remember-btn')?.addEventListener('click', rememberContent);
    document.getElementById('recall-btn')?.addEventListener('click', recallMemories);

    document.getElementById('memory-search')?.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') recallMemories();
    });

    // Notes controls
    document.getElementById('add-note-btn')?.addEventListener('click', addNote);
    document.getElementById('note-input')?.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') addNote();
    });

    // Snippets
    document.getElementById('new-snippet-btn')?.addEventListener('click', openSnippetModal);
    document.getElementById('save-snippet-btn')?.addEventListener('click', saveSnippet);

    // Focus timer
    document.getElementById('timer-start')?.addEventListener('click', startFocusTimer);
    document.getElementById('timer-stop')?.addEventListener('click', stopFocusTimer);
    document.getElementById('timer-reset')?.addEventListener('click', resetFocusTimer);

    document.querySelectorAll('.preset-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const minutes = parseInt(btn.dataset.minutes);
            setTimerPreset(minutes);
            document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
        });
    });

    // Health details
    document.getElementById('health-details-btn')?.addEventListener('click', openHealthModal);

    // Quick actions
    document.querySelectorAll('.quick-action-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const action = btn.dataset.action;
            handleQuickAction(action);
        });
    });

    // Thinking modal
    document.getElementById('start-thinking-btn')?.addEventListener('click', startThinking);

    // Close modals on overlay click
    document.querySelectorAll('.modal-overlay').forEach(overlay => {
        overlay.addEventListener('click', (e) => {
            if (e.target === overlay) {
                overlay.classList.add('hidden');
            }
        });
    });
}

function autoResizeTextarea(textarea) {
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 150) + 'px';
}

// ============================================
// WALLET CONNECTION
// ============================================

async function checkExistingWallet() {
    // Check if Phantom is available and already connected
    if (window.solana?.isPhantom && window.solana.isConnected) {
        try {
            const publicKey = window.solana.publicKey?.toString();
            if (publicKey) {
                await verifyToken(publicKey);
            }
        } catch (error) {
            console.log('No existing wallet connection');
        }
    }
}

async function connectWallet() {
    const statusEl = document.getElementById('wallet-status');
    const statusText = statusEl?.querySelector('.status-text');

    try {
        // Check if Phantom is installed
        if (!window.solana?.isPhantom) {
            // Demo mode - bypass wallet
            if (statusEl) statusEl.classList.remove('hidden');
            if (statusText) statusText.textContent = 'Demo mode - entering without wallet...';

            setTimeout(() => {
                enterChatInterface('demo-wallet');
            }, 1000);
            return;
        }

        if (statusEl) statusEl.classList.remove('hidden');
        if (statusText) statusText.textContent = 'Connecting to Phantom...';

        const response = await window.solana.connect();
        const publicKey = response.publicKey.toString();

        if (statusText) statusText.textContent = 'Verifying token balance...';
        await verifyToken(publicKey);

    } catch (error) {
        console.error('Wallet connection error:', error);
        if (statusText) statusText.textContent = 'Connection failed. Entering demo mode...';

        setTimeout(() => {
            enterChatInterface('demo-wallet');
        }, 1500);
    }
}

async function verifyToken(walletAddress) {
    try {
        const response = await fetch('/api/verify-token', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ wallet_address: walletAddress })
        });

        const data = await response.json();

        if (data.verified || data.demo_mode) {
            state.wallet = walletAddress;
            state.verified = true;
            enterChatInterface(walletAddress);
        } else {
            showToast('Insufficient token balance', 'error');
            document.getElementById('wallet-status')?.classList.add('hidden');
        }
    } catch (error) {
        console.error('Token verification error:', error);
        // Enter demo mode on error
        enterChatInterface(walletAddress);
    }
}

function enterChatInterface(walletAddress) {
    state.wallet = walletAddress;

    // Hide gate, show chat
    document.getElementById('token-gate')?.classList.add('hidden');
    document.getElementById('chat-app')?.classList.remove('hidden');

    // Update wallet display
    const walletBadge = document.getElementById('connected-wallet');
    const walletAddr = walletBadge?.querySelector('.wallet-addr');
    if (walletAddr) {
        walletAddr.textContent = walletAddress.slice(0, 4) + '...' + walletAddress.slice(-4);
    }

    // Add welcome message
    addWelcomeMessage();
}

function disconnectWallet() {
    state.wallet = null;
    state.verified = false;

    // Disconnect Phantom if connected
    if (window.solana?.isPhantom) {
        window.solana.disconnect();
    }

    // Show gate, hide chat
    document.getElementById('token-gate')?.classList.remove('hidden');
    document.getElementById('chat-app')?.classList.add('hidden');
    document.getElementById('wallet-status')?.classList.add('hidden');
}

// ============================================
// CHAT FUNCTIONALITY
// ============================================

function addWelcomeMessage() {
    const messagesContainer = document.getElementById('messages');
    if (!messagesContainer) return;
    messagesContainer.innerHTML = '';

    const welcomeMsg = `Good news, everyone! *adjusts spectacles*

Welcome to my Neural Interface v3.0! I'm Professor Farnsworth, your AI companion with FULL LOCAL FEATURES!

**What you can do right now:**
- 💾 **Memory** - Store and recall information
- 📝 **Notes** - Quick capture thoughts
- 💻 **Snippets** - Save code snippets
- ⏱️ **Focus Timer** - Pomodoro productivity
- 🎭 **Profiles** - Switch my personality
- 🏥 **Health** - Track your wellness
- 🤔 **Thinking** - Step-by-step reasoning

Try the sidebar panels or ask me anything! Now, what shall we work on? *rubs hands excitedly*`;

    addMessage(welcomeMsg, 'assistant');
}

function addMessage(content, role) {
    const messagesContainer = document.getElementById('messages');
    if (!messagesContainer) return;

    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;

    const avatar = role === 'user' ? '👤' : '🧠';
    const name = role === 'user' ? 'You' : 'Farnsworth';
    const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    // Convert markdown-like formatting
    const formattedContent = formatMessage(content);

    messageDiv.innerHTML = `
        <div class="message-avatar">${avatar}</div>
        <div class="message-body">
            <div class="message-meta">
                <span class="sender-name">${name}</span>
                <span class="message-time">${time}</span>
            </div>
            <div class="message-bubble glass-panel">
                ${formattedContent}
            </div>
        </div>
    `;

    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;

    // Text-to-speech for assistant messages
    if (role === 'assistant' && state.voiceEnabled) {
        speakText(content);
    }

    return messageDiv;
}

function formatMessage(content) {
    // Simple markdown-like formatting
    let formatted = content
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/`([^`]+)`/g, '<code>$1</code>')
        .replace(/^- (.+)$/gm, '<li>$1</li>')
        .replace(/\n\n/g, '</p><p>')
        .replace(/\n/g, '<br>');

    // Wrap in paragraph
    formatted = '<p>' + formatted + '</p>';

    // Fix list items
    formatted = formatted.replace(/(<li>.*<\/li>)+/g, '<ul>$&</ul>');

    return formatted;
}

async function sendMessage() {
    const input = document.getElementById('user-input');
    if (!input) return;
    const message = input.value.trim();

    if (!message) return;

    // Clear input
    input.value = '';
    document.getElementById('char-count').textContent = '0';
    document.getElementById('send-btn').disabled = true;
    input.style.height = 'auto';

    // Add user message
    addMessage(message, 'user');
    state.chatHistory.push({ role: 'user', content: message });

    // Show typing indicator
    const typingIndicator = document.getElementById('typing-indicator');
    typingIndicator?.classList.remove('hidden');

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: message,
                wallet: state.wallet,
                history: state.chatHistory.slice(-10)
            })
        });

        const data = await response.json();

        // Hide typing indicator
        typingIndicator?.classList.add('hidden');

        // Add assistant response
        addMessage(data.response, 'assistant');
        state.chatHistory.push({ role: 'assistant', content: data.response });

    } catch (error) {
        console.error('Chat error:', error);
        typingIndicator?.classList.add('hidden');
        addMessage('*wakes up startled* Wha? Oh my, something went wrong! Try again in a moment...', 'assistant');
    }
}

// ============================================
// VOICE FEATURES
// ============================================

let recognition = null;
let isListening = false;

function toggleVoiceInput() {
    const micBtn = document.getElementById('mic-btn');

    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
        showToast('Voice input not supported in this browser', 'error');
        return;
    }

    if (isListening) {
        stopVoiceInput();
    } else {
        startVoiceInput();
    }
}

function startVoiceInput() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    recognition = new SpeechRecognition();

    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.lang = 'en-US';

    const micBtn = document.getElementById('mic-btn');
    const input = document.getElementById('user-input');

    recognition.onstart = () => {
        isListening = true;
        micBtn?.classList.add('listening');
    };

    recognition.onresult = (event) => {
        let transcript = '';
        for (let i = event.resultIndex; i < event.results.length; i++) {
            transcript += event.results[i][0].transcript;
        }
        if (input) {
            input.value = transcript;
            document.getElementById('char-count').textContent = transcript.length;
            document.getElementById('send-btn').disabled = transcript.length === 0;
        }
    };

    recognition.onend = () => {
        isListening = false;
        micBtn?.classList.remove('listening');
    };

    recognition.onerror = (event) => {
        console.error('Speech recognition error:', event.error);
        isListening = false;
        micBtn?.classList.remove('listening');
        if (event.error !== 'aborted') {
            showToast('Voice recognition error: ' + event.error, 'error');
        }
    };

    recognition.start();
}

function stopVoiceInput() {
    if (recognition) {
        recognition.stop();
    }
}

function speakText(text) {
    if (!state.voiceEnabled) return;
    if (!('speechSynthesis' in window)) return;

    // Cancel any ongoing speech
    speechSynthesis.cancel();

    // Clean text for speech
    const cleanText = text
        .replace(/\*\*/g, '')
        .replace(/\*/g, '')
        .replace(/`/g, '')
        .replace(/\n/g, ' ')
        .slice(0, 500);

    const utterance = new SpeechSynthesisUtterance(cleanText);
    utterance.rate = 0.9;
    utterance.pitch = 0.8;

    speechSynthesis.speak(utterance);
}

// ============================================
// MEMORY SYSTEM
// ============================================

async function rememberContent() {
    const input = document.getElementById('memory-input');
    if (!input) return;
    const content = input.value.trim();

    if (!content) {
        showToast('Enter something to remember', 'warning');
        return;
    }

    try {
        const response = await fetch('/api/memory/remember', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                content: content,
                tags: [],
                importance: 0.5
            })
        });

        const data = await response.json();

        if (data.success) {
            showToast(data.message || 'Stored in memory!', 'success');
            input.value = '';
        } else {
            showToast(data.message || 'Failed to store memory', 'error');
        }
    } catch (error) {
        console.error('Remember error:', error);
        showToast('Failed to store memory', 'error');
    }
}

async function recallMemories() {
    const input = document.getElementById('memory-search');
    if (!input) return;
    const query = input.value.trim();

    if (!query) {
        showToast('Enter a search query', 'warning');
        return;
    }

    const resultsContainer = document.getElementById('memory-results');
    if (!resultsContainer) return;
    resultsContainer.innerHTML = '<div class="memory-item"><em>Searching...</em></div>';

    try {
        const response = await fetch('/api/memory/recall', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: query,
                limit: 10
            })
        });

        const data = await response.json();

        if (data.success && data.memories && data.memories.length > 0) {
            resultsContainer.innerHTML = data.memories.map(mem => `
                <div class="memory-item">
                    <div class="memory-content">${escapeHtml(typeof mem === 'string' ? mem : mem.content || '')}</div>
                    <div class="memory-meta">${mem.timestamp || ''}</div>
                </div>
            `).join('');
        } else {
            resultsContainer.innerHTML = '<div class="memory-item"><em>No memories found</em></div>';
        }
    } catch (error) {
        console.error('Recall error:', error);
        resultsContainer.innerHTML = '<div class="memory-item"><em>Search failed</em></div>';
    }
}

// ============================================
// NOTES SYSTEM
// ============================================

async function loadNotes() {
    try {
        const response = await fetch('/api/notes');
        const data = await response.json();

        const container = document.getElementById('notes-list');
        if (!container) return;

        if (data.success && data.notes && data.notes.length > 0) {
            container.innerHTML = data.notes.map(note => `
                <div class="note-item" data-id="${note.id || ''}">
                    <div class="note-content">${escapeHtml(typeof note === 'string' ? note : note.content || '')}</div>
                    <button class="note-delete" onclick="deleteNote('${note.id || ''}')">&times;</button>
                </div>
            `).join('');
        } else {
            container.innerHTML = '<div class="note-item"><em>No notes yet</em></div>';
        }
    } catch (error) {
        console.error('Load notes error:', error);
    }
}

async function addNote() {
    const input = document.getElementById('note-input');
    if (!input) return;
    const content = input.value.trim();

    if (!content) return;

    try {
        const response = await fetch('/api/notes', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                content: content,
                tags: []
            })
        });

        const data = await response.json();

        if (data.success) {
            input.value = '';
            loadNotes();
            showToast('Note added!', 'success');
        }
    } catch (error) {
        console.error('Add note error:', error);
        showToast('Failed to add note', 'error');
    }
}

async function deleteNote(noteId) {
    try {
        await fetch(`/api/notes/${noteId}`, { method: 'DELETE' });
        loadNotes();
    } catch (error) {
        console.error('Delete note error:', error);
    }
}

// ============================================
// SNIPPETS SYSTEM
// ============================================

async function loadSnippets() {
    try {
        const response = await fetch('/api/snippets');
        const data = await response.json();

        const container = document.getElementById('snippets-list');
        if (!container) return;

        if (data.success && data.snippets && data.snippets.length > 0) {
            container.innerHTML = data.snippets.map(snippet => `
                <div class="snippet-item" onclick="viewSnippet('${snippet.id || ''}')">
                    <div class="snippet-lang">${snippet.language || 'code'}</div>
                    <div class="snippet-desc">${escapeHtml(snippet.description || 'Untitled')}</div>
                </div>
            `).join('');
        } else {
            container.innerHTML = '<div class="snippet-item"><em>No snippets yet</em></div>';
        }
    } catch (error) {
        console.error('Load snippets error:', error);
    }
}

function openSnippetModal() {
    document.getElementById('snippet-modal')?.classList.remove('hidden');
    const descEl = document.getElementById('snippet-desc');
    const codeEl = document.getElementById('snippet-code');
    const tagsEl = document.getElementById('snippet-tags');
    if (descEl) descEl.value = '';
    if (codeEl) codeEl.value = '';
    if (tagsEl) tagsEl.value = '';
}

function closeSnippetModal() {
    document.getElementById('snippet-modal')?.classList.add('hidden');
}

async function saveSnippet() {
    const desc = document.getElementById('snippet-desc')?.value.trim() || '';
    const lang = document.getElementById('snippet-lang')?.value || 'python';
    const code = document.getElementById('snippet-code')?.value || '';
    const tagsInput = document.getElementById('snippet-tags')?.value || '';
    const tags = tagsInput.split(',').map(t => t.trim()).filter(Boolean);

    if (!code) {
        showToast('Enter some code', 'warning');
        return;
    }

    try {
        const response = await fetch('/api/snippets', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                code: code,
                language: lang,
                description: desc,
                tags: tags
            })
        });

        const data = await response.json();

        if (data.success) {
            closeSnippetModal();
            loadSnippets();
            showToast('Snippet saved!', 'success');
        }
    } catch (error) {
        console.error('Save snippet error:', error);
        showToast('Failed to save snippet', 'error');
    }
}

// ============================================
// FOCUS TIMER
// ============================================

function setTimerPreset(minutes) {
    state.focusTimer.duration = minutes * 60;
    state.focusTimer.remaining = minutes * 60;
    updateTimerDisplay();
}

function updateTimerDisplay() {
    const minutes = Math.floor(state.focusTimer.remaining / 60);
    const seconds = state.focusTimer.remaining % 60;
    const minEl = document.getElementById('timer-minutes');
    const secEl = document.getElementById('timer-seconds');
    if (minEl) minEl.textContent = minutes.toString().padStart(2, '0');
    if (secEl) secEl.textContent = seconds.toString().padStart(2, '0');
}

async function startFocusTimer() {
    if (state.focusTimer.active) return;

    state.focusTimer.active = true;
    state.focusTimer.task = 'Deep Work';

    document.getElementById('timer-start')?.classList.add('hidden');
    document.getElementById('timer-stop')?.classList.remove('hidden');
    const taskEl = document.getElementById('timer-task');
    if (taskEl) taskEl.textContent = 'Focus in progress...';

    // Notify server
    try {
        await fetch('/api/focus/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                task: state.focusTimer.task,
                duration_minutes: state.focusTimer.duration / 60
            })
        });
    } catch (error) {
        console.error('Focus start error:', error);
    }

    // Start countdown
    state.focusTimer.interval = setInterval(() => {
        state.focusTimer.remaining--;
        updateTimerDisplay();

        if (state.focusTimer.remaining <= 0) {
            completeFocusTimer();
        }
    }, 1000);
}

async function stopFocusTimer() {
    if (!state.focusTimer.active) return;

    clearInterval(state.focusTimer.interval);
    state.focusTimer.active = false;

    document.getElementById('timer-start')?.classList.remove('hidden');
    document.getElementById('timer-stop')?.classList.add('hidden');
    const taskEl = document.getElementById('timer-task');
    if (taskEl) taskEl.textContent = 'Session stopped';

    try {
        await fetch('/api/focus/stop', { method: 'POST' });
    } catch (error) {
        console.error('Focus stop error:', error);
    }
}

function resetFocusTimer() {
    stopFocusTimer();
    state.focusTimer.remaining = state.focusTimer.duration;
    updateTimerDisplay();
    const taskEl = document.getElementById('timer-task');
    if (taskEl) taskEl.textContent = 'Ready to focus';
}

function completeFocusTimer() {
    clearInterval(state.focusTimer.interval);
    state.focusTimer.active = false;

    document.getElementById('timer-start')?.classList.remove('hidden');
    document.getElementById('timer-stop')?.classList.add('hidden');
    const taskEl = document.getElementById('timer-task');
    if (taskEl) taskEl.textContent = 'Session complete! 🎉';

    // Play notification sound or show notification
    showToast('Focus session complete! Great work!', 'success');

    // Browser notification if allowed
    if (Notification.permission === 'granted') {
        new Notification('Farnsworth Focus Timer', {
            body: 'Your focus session is complete!',
            icon: '🧠'
        });
    }

    // Reset for next session
    state.focusTimer.remaining = state.focusTimer.duration;
}

// ============================================
// HEALTH SYSTEM
// ============================================

async function loadHealthSummary() {
    try {
        const response = await fetch('/api/health/summary');
        const data = await response.json();

        if (data.success) {
            const summary = data.summary;

            // Update wellness score ring
            const score = summary.wellness_score || 0;
            const circle = document.getElementById('wellness-circle');
            const circumference = 2 * Math.PI * 40; // radius = 40
            const offset = circumference - (score / 100) * circumference;
            if (circle) {
                circle.style.strokeDashoffset = offset;
            }
            const scoreEl = document.getElementById('wellness-score');
            if (scoreEl) scoreEl.textContent = score;

            // Update metrics
            const hrEl = document.getElementById('heart-rate');
            const stepsEl = document.getElementById('steps-count');
            const sleepEl = document.getElementById('sleep-hours');
            if (hrEl) hrEl.textContent = summary.heart_rate?.avg || '--';
            if (stepsEl) stepsEl.textContent = formatNumber(summary.steps?.today || 0);
            if (sleepEl) sleepEl.textContent = (summary.sleep?.hours || 0).toFixed(1) + 'h';
        }
    } catch (error) {
        console.error('Health summary error:', error);
    }
}

function openHealthModal() {
    document.getElementById('health-modal')?.classList.remove('hidden');
    loadHealthCharts();
}

function closeHealthModal() {
    document.getElementById('health-modal')?.classList.add('hidden');
}

async function loadHealthCharts() {
    try {
        // Load heart rate data
        const hrResponse = await fetch('/api/health/metrics/heart_rate?days=7');
        const hrData = await hrResponse.json();

        // Load steps data
        const stepsResponse = await fetch('/api/health/metrics/steps?days=7');
        const stepsData = await stepsResponse.json();

        // Render charts
        renderHealthChart('heart-rate-chart', hrData.data || [], 'Heart Rate', 'rgba(239, 68, 68, 0.8)');
        renderHealthChart('steps-chart', stepsData.data || [], 'Steps', 'rgba(16, 185, 129, 0.8)');

        // Load insights
        const insightsContainer = document.getElementById('health-insights');
        if (insightsContainer) {
            insightsContainer.innerHTML = `
                <h4>AI Insights</h4>
                <div class="insight-item">
                    <span class="insight-icon">💚</span>
                    <span class="insight-text">Your heart rate is within healthy range</span>
                </div>
                <div class="insight-item">
                    <span class="insight-icon">🚶</span>
                    <span class="insight-text">Try to reach 10,000 steps today!</span>
                </div>
                <div class="insight-item">
                    <span class="insight-icon">😴</span>
                    <span class="insight-text">Good sleep quality helps cognitive function</span>
                </div>
            `;
        }
    } catch (error) {
        console.error('Health charts error:', error);
    }
}

function renderHealthChart(canvasId, data, label, color) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || typeof Chart === 'undefined') return;

    // Destroy existing chart if any
    const existingChart = Chart.getChart(canvas);
    if (existingChart) {
        existingChart.destroy();
    }

    const labels = data.map(d => d.date?.slice(5) || '');
    const values = data.map(d => d.value || 0);

    new Chart(canvas, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: label,
                data: values,
                borderColor: color,
                backgroundColor: color.replace('0.8', '0.2'),
                tension: 0.4,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255,255,255,0.1)' },
                    ticks: { color: 'rgba(255,255,255,0.5)' }
                },
                y: {
                    grid: { color: 'rgba(255,255,255,0.1)' },
                    ticks: { color: 'rgba(255,255,255,0.5)' }
                }
            }
        }
    });
}

// ============================================
// PROFILE SWITCHING
// ============================================

async function switchProfile(profileId) {
    try {
        const response = await fetch('/api/profiles/switch', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ profile_id: profileId })
        });

        const data = await response.json();

        if (data.success) {
            state.currentProfile = profileId;

            // Update UI
            document.querySelectorAll('.profile-btn').forEach(btn => {
                btn.classList.toggle('active', btn.dataset.profile === profileId);
            });

            showToast(`Switched to ${profileId} mode`, 'success');
        }
    } catch (error) {
        console.error('Profile switch error:', error);
    }
}

// ============================================
// THINKING SYSTEM
// ============================================

function handleQuickAction(action) {
    switch (action) {
        case 'remember':
            document.getElementById('left-sidebar')?.classList.add('active');
            document.getElementById('memory-input')?.focus();
            break;
        case 'recall':
            document.getElementById('left-sidebar')?.classList.add('active');
            document.getElementById('memory-search')?.focus();
            break;
        case 'think':
            openThinkingModal();
            break;
        case 'tools':
            document.getElementById('right-sidebar')?.classList.add('active');
            break;
    }
}

function openThinkingModal() {
    document.getElementById('thinking-modal')?.classList.remove('hidden');
    const problemEl = document.getElementById('thinking-problem');
    const stepsEl = document.getElementById('thinking-steps');
    const conclusionEl = document.getElementById('thinking-conclusion');
    if (problemEl) problemEl.value = '';
    if (stepsEl) stepsEl.innerHTML = '';
    if (conclusionEl) conclusionEl.classList.add('hidden');
}

function closeThinkingModal() {
    document.getElementById('thinking-modal')?.classList.add('hidden');
}

async function startThinking() {
    const problemEl = document.getElementById('thinking-problem');
    const problem = problemEl?.value.trim();
    if (!problem) {
        showToast('Enter a problem to analyze', 'warning');
        return;
    }

    const stepsContainer = document.getElementById('thinking-steps');
    const conclusionContainer = document.getElementById('thinking-conclusion');

    if (stepsContainer) stepsContainer.innerHTML = '<div class="thinking-step"><em>Thinking...</em></div>';
    if (conclusionContainer) conclusionContainer.classList.add('hidden');

    try {
        const response = await fetch('/api/think', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                problem: problem,
                max_steps: 10
            })
        });

        const data = await response.json();

        if (data.success && stepsContainer) {
            // Render steps
            stepsContainer.innerHTML = (data.steps || []).map((step, i) => `
                <div class="thinking-step">
                    <div class="step-number">${step.step || i + 1}</div>
                    <div class="step-content">
                        <div class="step-thought">${escapeHtml(step.thought || '')}</div>
                        <div class="step-confidence">
                            <div class="confidence-bar">
                                <div class="confidence-fill" style="width: ${(step.confidence || 0) * 100}%"></div>
                            </div>
                            <span class="confidence-value">${Math.round((step.confidence || 0) * 100)}%</span>
                        </div>
                    </div>
                </div>
            `).join('');

            // Show conclusion
            if (data.conclusion && conclusionContainer) {
                conclusionContainer.innerHTML = `
                    <div class="conclusion-title">Conclusion</div>
                    <div class="conclusion-text">${escapeHtml(data.conclusion)}</div>
                `;
                conclusionContainer.classList.remove('hidden');
            }
        }
    } catch (error) {
        console.error('Thinking error:', error);
        if (stepsContainer) stepsContainer.innerHTML = '<div class="thinking-step"><em>Analysis failed</em></div>';
    }
}

function renderThinkingStep(stepData) {
    const container = document.getElementById('thinking-steps');
    if (!container) return;
    const stepDiv = document.createElement('div');
    stepDiv.className = 'thinking-step';
    stepDiv.innerHTML = `
        <div class="step-number">${stepData.step || '?'}</div>
        <div class="step-content">
            <div class="step-thought">${escapeHtml(stepData.thought || '')}</div>
            <div class="step-confidence">
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: ${(stepData.confidence || 0) * 100}%"></div>
                </div>
                <span class="confidence-value">${Math.round((stepData.confidence || 0) * 100)}%</span>
            </div>
        </div>
    `;
    container.appendChild(stepDiv);
}

// ============================================
// TRADING TOOLS
// ============================================

function openWhaleTracker() {
    openToolModal('🐋 Whale Tracker', `
        <form class="tool-form" id="whale-form">
            <label>
                Wallet Address
                <input type="text" id="whale-wallet" placeholder="Enter wallet address..." required>
            </label>
            <button type="submit" class="action-btn primary full-width">Track Whale</button>
        </form>
    `);

    document.getElementById('whale-form')?.addEventListener('submit', async (e) => {
        e.preventDefault();
        const walletEl = document.getElementById('whale-wallet');
        const wallet = walletEl?.value || '';
        const resultDiv = document.getElementById('tool-modal-result');
        if (resultDiv) {
            resultDiv.classList.remove('hidden');
            resultDiv.textContent = 'Tracking...';
        }

        const response = await fetch('/api/tools/whale-track', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ wallet_address: wallet })
        });
        const data = await response.json();
        if (resultDiv) resultDiv.textContent = JSON.stringify(data, null, 2);
    });
}

function openRugCheck() {
    openToolModal('🔍 Rug Check', `
        <form class="tool-form" id="rug-form">
            <label>
                Token Mint Address
                <input type="text" id="rug-mint" placeholder="Enter mint address..." required>
            </label>
            <button type="submit" class="action-btn primary full-width">Scan Token</button>
        </form>
    `);

    document.getElementById('rug-form')?.addEventListener('submit', async (e) => {
        e.preventDefault();
        const mintEl = document.getElementById('rug-mint');
        const mint = mintEl?.value || '';
        const resultDiv = document.getElementById('tool-modal-result');
        if (resultDiv) {
            resultDiv.classList.remove('hidden');
            resultDiv.textContent = 'Scanning...';
        }

        const response = await fetch('/api/tools/rug-check', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mint_address: mint })
        });
        const data = await response.json();
        if (resultDiv) resultDiv.textContent = JSON.stringify(data, null, 2);
    });
}

function openTokenScanner() {
    openToolModal('📈 Token Scanner', `
        <form class="tool-form" id="token-form">
            <label>
                Search Query
                <input type="text" id="token-query" placeholder="Token name or address..." required>
            </label>
            <button type="submit" class="action-btn primary full-width">Search</button>
        </form>
    `);

    document.getElementById('token-form')?.addEventListener('submit', async (e) => {
        e.preventDefault();
        const queryEl = document.getElementById('token-query');
        const query = queryEl?.value || '';
        const resultDiv = document.getElementById('tool-modal-result');
        if (resultDiv) {
            resultDiv.classList.remove('hidden');
            resultDiv.textContent = 'Searching...';
        }

        const response = await fetch('/api/tools/token-scan', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: query })
        });
        const data = await response.json();
        if (resultDiv) resultDiv.textContent = JSON.stringify(data, null, 2);
    });
}

async function openMarketSentiment() {
    openToolModal('🌡️ Market Sentiment', '<div class="tool-form">Loading sentiment data...</div>');

    const response = await fetch('/api/tools/market-sentiment');
    const data = await response.json();

    const contentEl = document.getElementById('tool-modal-content');
    if (contentEl) {
        contentEl.innerHTML = `
            <div class="tool-form">
                <h4 style="text-align: center; margin-bottom: 16px;">Fear & Greed Index</h4>
                <div style="text-align: center; font-size: 3rem; margin-bottom: 16px;">
                    ${data.data?.fear_greed_index || 'N/A'}
                </div>
                <div style="text-align: center; color: var(--text-secondary);">
                    ${data.data?.classification || 'Demo Mode'}
                </div>
            </div>
        `;
    }
}

function openToolModal(title, content) {
    const titleEl = document.getElementById('tool-modal-title');
    const contentEl = document.getElementById('tool-modal-content');
    const resultEl = document.getElementById('tool-modal-result');

    if (titleEl) titleEl.textContent = title;
    if (contentEl) contentEl.innerHTML = content;
    if (resultEl) resultEl.classList.add('hidden');
    document.getElementById('tool-modal')?.classList.remove('hidden');
}

function closeToolModal() {
    document.getElementById('tool-modal')?.classList.add('hidden');
}

function closeModal() {
    document.getElementById('modal-overlay')?.classList.add('hidden');
}

// ============================================
// NEURAL CANVAS
// ============================================

function initNeuralCanvas() {
    const canvas = document.getElementById('neural-canvas');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    let particles = [];
    let animationId;

    function resize() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    }

    function createParticles() {
        particles = [];
        const count = Math.floor((canvas.width * canvas.height) / 15000);

        for (let i = 0; i < count; i++) {
            particles.push({
                x: Math.random() * canvas.width,
                y: Math.random() * canvas.height,
                vx: (Math.random() - 0.5) * 0.5,
                vy: (Math.random() - 0.5) * 0.5,
                size: Math.random() * 2 + 1
            });
        }
    }

    function draw() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Draw connections
        ctx.strokeStyle = 'rgba(139, 92, 246, 0.1)';
        ctx.lineWidth = 1;

        for (let i = 0; i < particles.length; i++) {
            for (let j = i + 1; j < particles.length; j++) {
                const dx = particles[i].x - particles[j].x;
                const dy = particles[i].y - particles[j].y;
                const dist = Math.sqrt(dx * dx + dy * dy);

                if (dist < 150) {
                    ctx.beginPath();
                    ctx.moveTo(particles[i].x, particles[i].y);
                    ctx.lineTo(particles[j].x, particles[j].y);
                    ctx.globalAlpha = 1 - dist / 150;
                    ctx.stroke();
                }
            }
        }

        // Draw particles
        ctx.globalAlpha = 1;
        for (const p of particles) {
            ctx.beginPath();
            ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
            ctx.fillStyle = 'rgba(139, 92, 246, 0.5)';
            ctx.fill();

            // Update position
            p.x += p.vx;
            p.y += p.vy;

            // Wrap around
            if (p.x < 0) p.x = canvas.width;
            if (p.x > canvas.width) p.x = 0;
            if (p.y < 0) p.y = canvas.height;
            if (p.y > canvas.height) p.y = 0;
        }

        animationId = requestAnimationFrame(draw);
    }

    resize();
    createParticles();
    draw();

    window.addEventListener('resize', () => {
        resize();
        createParticles();
    });
}

// ============================================
// UTILITIES
// ============================================

function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;

    container.appendChild(toast);

    setTimeout(() => {
        toast.style.animation = 'toastIn 0.3s reverse';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatNumber(num) {
    if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
    if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
    return num.toString();
}

function copyToken() {
    const tokenAddr = document.getElementById('token-addr')?.textContent;
    if (tokenAddr) {
        navigator.clipboard.writeText(tokenAddr).then(() => {
            showToast('Token address copied!', 'success');
        });
    }
}

// Make functions globally available
window.copyToken = copyToken;
window.disconnectWallet = disconnectWallet;
window.openWhaleTracker = openWhaleTracker;
window.openRugCheck = openRugCheck;
window.openTokenScanner = openTokenScanner;
window.openMarketSentiment = openMarketSentiment;
window.closeModal = closeModal;
window.closeToolModal = closeToolModal;
window.closeThinkingModal = closeThinkingModal;
window.closeSnippetModal = closeSnippetModal;
window.closeHealthModal = closeHealthModal;
window.deleteNote = deleteNote;
