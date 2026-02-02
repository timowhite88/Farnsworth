/**
 * Farnsworth AI - Neural Interface v3.0
 * Full-stack chat interface with all local features
 */

// ============================================
// STATE MANAGEMENT
// ============================================

const state = {
    voiceEnabled: true,
    voiceVolume: parseFloat(localStorage.getItem('voiceVolume') || '0.8'),  // 0-1 range
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
    features: {},
    // Swarm Chat State
    swarmMode: false,
    swarmWs: null,
    swarmConnected: false,
    swarmUserId: null,
    swarmUserName: null,
    swarmOnlineUsers: [],
    swarmActiveModels: [],
    swarmTypingBots: new Set()
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
    loadEvolutionStats();

    // Start evolution stats auto-refresh (every 60 seconds)
    setInterval(loadEvolutionStats, 60000);

    // Auto-start in Swarm mode
    initSwarmMode();

    // Focus the input field
    const input = document.getElementById('user-input');
    if (input) input.focus();
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
        updateStatusIndicator('evolution', data.features?.evolution);
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

    // Volume slider
    const volumeSlider = document.getElementById('volume-slider');
    const volumeValue = document.getElementById('volume-value');
    if (volumeSlider) {
        // Set initial value from state
        volumeSlider.value = Math.round(state.voiceVolume * 100);
        if (volumeValue) volumeValue.textContent = `${volumeSlider.value}%`;

        volumeSlider.addEventListener('input', (e) => {
            const volume = parseInt(e.target.value) / 100;
            state.voiceVolume = volume;
            localStorage.setItem('voiceVolume', volume.toString());
            if (volumeValue) volumeValue.textContent = `${e.target.value}%`;

            // Apply to current audio if playing
            if (currentAudio) {
                currentAudio.volume = volume;
            }
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

    // Health details (legacy)
    document.getElementById('health-details-btn')?.addEventListener('click', openHealthModal);

    // Evolution force evolve button
    document.getElementById('force-evolve-btn')?.addEventListener('click', forceEvolve);

    // Swarm Chat Mode Toggle
    document.getElementById('personal-chat-btn')?.addEventListener('click', () => switchChatMode(false));
    document.getElementById('swarm-chat-btn')?.addEventListener('click', () => switchChatMode(true));

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
- 🐝 **Swarm Chat** - Chat with the community!

**🪙 Crypto Tools - Just Ask Naturally:**
- "What's the price of SOL?" or "Check $BONK"
- "Is this safe?" + paste a contract address
- "Rug check 9crfy4udr..." (paste any CA)
- "How's the market?" for sentiment

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

// Current audio element for TTS playback
let currentAudio = null;
let audioQueue = [];
let serverAudioQueue = [];  // {audioUrl, botName}
let isPlayingAudio = false;
let lastSpeaker = null;
const SPEAKER_DELAY_MS = 800;  // Pause between different speakers for natural conversation

// Play pre-generated audio from server URL (with queue)
async function playServerAudio(audioUrl, botName = 'Unknown') {
    if (!state.voiceEnabled) return;

    // Add to queue with bot name for tracking
    serverAudioQueue.push({ audioUrl, botName });
    console.log(`[Audio] Queued ${botName} audio, queue size:`, serverAudioQueue.length);
    processServerAudioQueue();
}

async function processServerAudioQueue() {
    if (isPlayingAudio || serverAudioQueue.length === 0) return;

    isPlayingAudio = true;
    const { audioUrl, botName } = serverAudioQueue.shift();

    // Stop any current audio
    if (currentAudio) {
        currentAudio.pause();
        currentAudio = null;
    }

    // Add delay between different speakers for natural pacing
    if (lastSpeaker && lastSpeaker !== botName) {
        await new Promise(resolve => setTimeout(resolve, SPEAKER_DELAY_MS));
    }

    try {
        const response = await fetch(audioUrl);
        if (response.ok) {
            const audioBlob = await response.blob();
            const blobUrl = URL.createObjectURL(audioBlob);
            currentAudio = new Audio(blobUrl);
            currentAudio.volume = state.voiceVolume;

            currentAudio.onended = () => {
                URL.revokeObjectURL(blobUrl);
                currentAudio = null;
                isPlayingAudio = false;
                lastSpeaker = botName;
                console.log(`[Audio] ${botName} finished speaking`);

                // Signal server that audio finished
                if (state.swarmWs && state.swarmWs.readyState === WebSocket.OPEN) {
                    state.swarmWs.send(JSON.stringify({
                        type: 'audio_complete',
                        bot_name: botName
                    }));
                }

                // Process next in queue with small delay
                setTimeout(() => processServerAudioQueue(), 300);
            };

            currentAudio.onerror = (e) => {
                console.error(`[Audio] Error playing ${botName} audio:`, e);
                isPlayingAudio = false;
                setTimeout(() => processServerAudioQueue(), 300);
            };

            console.log(`[Audio] Playing ${botName} voice, remaining in queue:`, serverAudioQueue.length);
            await currentAudio.play();
        } else {
            console.warn(`[Audio] Server audio not ready for ${botName}`);
            isPlayingAudio = false;
            setTimeout(() => processServerAudioQueue(), 300);
        }
    } catch (error) {
        console.error(`[Audio] Failed to fetch ${botName} audio:`, error);
        isPlayingAudio = false;
        setTimeout(() => processServerAudioQueue(), 300);
    }
}

// =============================================================================
// MULTI-VOICE SEQUENTIAL PLAYBACK
// Each swarm bot has their own unique cloned voice via Fish Speech / XTTS
// Bots speak one at a time - next waits for previous to finish
// =============================================================================

// Track which bot is currently speaking
let currentSpeakingBot = null;

// Sequential audio playback - bots wait for each other
async function speakText(text, botName = 'Farnsworth') {
    if (!state.voiceEnabled) return Promise.resolve();

    // Clean text for speech
    const cleanText = text
        .replace(/\*\*/g, '')
        .replace(/\*/g, '')
        .replace(/`/g, '')
        .replace(/#{1,6}\s*/g, '')  // Remove markdown headers
        .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')  // Remove links
        .replace(/[═─│┌┐└┘├┤┬┴┼]/g, '')  // Remove box chars
        .replace(/\n/g, ' ')
        .slice(0, 800);

    if (!cleanText.trim()) return Promise.resolve();

    // Add to queue and process
    return new Promise((resolve) => {
        audioQueue.push({ text: cleanText, botName, resolve });
        console.log(`[Voice] Queued ${botName}: "${cleanText.slice(0, 40)}..."`);
        processAudioQueue();
    });
}

async function processAudioQueue() {
    if (isPlayingAudio || audioQueue.length === 0) return;

    isPlayingAudio = true;
    const { text, botName, resolve } = audioQueue.shift();

    currentSpeakingBot = botName;
    console.log(`[Voice] ${botName} speaking (${audioQueue.length} in queue)`);

    // Update UI to show who's speaking
    updateSpeakingIndicator(botName, true);

    // Stop any current audio
    if (currentAudio) {
        currentAudio.pause();
        currentAudio = null;
    }

    // Cancel any browser speech
    if ('speechSynthesis' in window) {
        speechSynthesis.cancel();
    }

    try {
        // Try multi-voice API first (Fish Speech / XTTS with bot-specific voices)
        const response = await fetch('/api/speak/bot', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                text: text,
                bot_name: botName
            })
        });

        if (response.ok) {
            const audioBlob = await response.blob();
            const audioUrl = URL.createObjectURL(audioBlob);
            currentAudio = new Audio(audioUrl);
            currentAudio.volume = state.voiceVolume;

            currentAudio.onended = () => {
                URL.revokeObjectURL(audioUrl);
                currentAudio = null;
                isPlayingAudio = false;
                currentSpeakingBot = null;

                // Update UI
                updateSpeakingIndicator(botName, false);

                // Signal server that audio finished
                if (state.swarmWs && state.swarmWs.readyState === WebSocket.OPEN) {
                    state.swarmWs.send(JSON.stringify({
                        type: 'audio_complete',
                        bot_name: botName
                    }));
                }

                // Also notify via REST for queue management
                fetch('/api/voices/queue/complete?bot_name=' + encodeURIComponent(botName), {
                    method: 'POST'
                }).catch(() => {});

                console.log(`[Voice] ${botName} finished speaking`);
                resolve();
                // Process next in queue
                processAudioQueue();
            };

            currentAudio.onerror = (e) => {
                console.warn(`[Voice] Audio error for ${botName}:`, e);
                isPlayingAudio = false;
                currentSpeakingBot = null;
                updateSpeakingIndicator(botName, false);
                resolve();
                processAudioQueue();
            };

            await currentAudio.play();
            return;
        } else {
            console.warn(`[Voice] Multi-voice API failed (${response.status}), trying fallback`);
        }
    } catch (error) {
        console.warn('[Voice] Multi-voice error, falling back:', error);
    }

    // Fallback to old single-voice API
    try {
        const response = await fetch('/api/speak', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: text })
        });

        if (response.ok) {
            const audioBlob = await response.blob();
            const audioUrl = URL.createObjectURL(audioBlob);
            currentAudio = new Audio(audioUrl);
            currentAudio.volume = state.voiceVolume;

            currentAudio.onended = () => {
                URL.revokeObjectURL(audioUrl);
                currentAudio = null;
                isPlayingAudio = false;
                currentSpeakingBot = null;
                updateSpeakingIndicator(botName, false);
                resolve();
                processAudioQueue();
            };

            currentAudio.onerror = () => {
                isPlayingAudio = false;
                currentSpeakingBot = null;
                updateSpeakingIndicator(botName, false);
                resolve();
                processAudioQueue();
            };

            await currentAudio.play();
            return;
        }
    } catch (error) {
        console.warn('[Voice] Fallback TTS error:', error);
    }

    // Last resort: browser TTS
    if ('speechSynthesis' in window) {
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.rate = 0.9;
        utterance.pitch = 0.8;
        utterance.volume = state.voiceVolume;
        utterance.onend = () => {
            isPlayingAudio = false;
            currentSpeakingBot = null;
            updateSpeakingIndicator(botName, false);
            if (state.swarmWs && state.swarmWs.readyState === WebSocket.OPEN) {
                state.swarmWs.send(JSON.stringify({
                    type: 'audio_complete',
                    bot_name: botName
                }));
            }
            resolve();
            processAudioQueue();
        };
        speechSynthesis.speak(utterance);
    } else {
        isPlayingAudio = false;
        currentSpeakingBot = null;
        updateSpeakingIndicator(botName, false);
        resolve();
        processAudioQueue();
    }
}

// Update UI to show which bot is speaking
function updateSpeakingIndicator(botName, isSpeaking) {
    // Find bot messages and add/remove speaking indicator
    const messages = document.querySelectorAll('.message.swarm-bot');
    messages.forEach(msg => {
        const nameEl = msg.querySelector('.bot-name');
        if (nameEl && nameEl.textContent.includes(botName)) {
            if (isSpeaking) {
                msg.classList.add('speaking');
                // Add speaking animation
                if (!nameEl.querySelector('.speaking-indicator')) {
                    const indicator = document.createElement('span');
                    indicator.className = 'speaking-indicator';
                    indicator.innerHTML = ' 🔊';
                    nameEl.appendChild(indicator);
                }
            } else {
                msg.classList.remove('speaking');
                const indicator = nameEl.querySelector('.speaking-indicator');
                if (indicator) indicator.remove();
            }
        }
    });
}

// Get info about currently speaking bot
function getCurrentSpeaker() {
    return {
        botName: currentSpeakingBot,
        isSpeaking: isPlayingAudio,
        queueLength: audioQueue.length
    };
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
// EVOLUTION STATS SYSTEM
// ============================================

async function loadEvolutionStats() {
    try {
        const response = await fetch('/api/evolution/status');
        const data = await response.json();

        if (data.available) {
            // Update evolution cycle ring
            const threshold = data.auto_evolve_threshold || 100;
            const untilNext = data.learnings_until_next_evolution || 0;
            const progress = ((threshold - untilNext) / threshold) * 100;

            const circle = document.getElementById('evolution-circle');
            const circumference = 2 * Math.PI * 40;
            const offset = circumference - (progress / 100) * circumference;
            if (circle) {
                circle.style.strokeDashoffset = offset;
            }

            // Update cycle value
            const cycleEl = document.getElementById('evolution-cycle');
            if (cycleEl) cycleEl.textContent = data.evolution_cycles || 0;

            // Update metrics
            const learningsEl = document.getElementById('total-learnings');
            const untilEl = document.getElementById('until-evolution');
            const patternsEl = document.getElementById('patterns-count');

            if (learningsEl) learningsEl.textContent = formatNumber(data.total_learnings || 0);
            if (untilEl) untilEl.textContent = untilNext;
            if (patternsEl) patternsEl.textContent = data.patterns_count || 0;

            // Update personality stats
            const personalityContainer = document.getElementById('personality-stats');
            if (personalityContainer && data.personalities) {
                const personalities = Object.entries(data.personalities);
                personalityContainer.innerHTML = personalities.map(([name, p]) => `
                    <div class="personality-item">
                        <span class="personality-name">${getBotEmoji(name)} ${name}</span>
                        <span class="personality-meta">
                            <span class="personality-gen">Gen ${p.generation}</span>
                            <span class="personality-interactions">${formatNumber(p.interactions)} chats</span>
                        </span>
                    </div>
                `).join('');
            }

            // Update last evolution time
            const lastEl = document.getElementById('last-evolution');
            if (lastEl && data.last_evolution) {
                const lastTime = new Date(data.last_evolution);
                lastEl.textContent = `Last: ${formatTimeAgo(lastTime)}`;
            } else if (lastEl) {
                lastEl.textContent = 'Last: Never';
            }
        }
    } catch (error) {
        console.error('Evolution stats error:', error);
    }
}

function getBotEmoji(name) {
    const emojis = {
        'Farnsworth': '🧪',
        'DeepSeek': '🔮',
        'Phi': '⚡',
        'Swarm-Mind': '🧠'
    };
    return emojis[name] || '🤖';
}

function formatTimeAgo(date) {
    const seconds = Math.floor((new Date() - date) / 1000);
    if (seconds < 60) return 'just now';
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
    if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
    return `${Math.floor(seconds / 86400)}d ago`;
}

async function forceEvolve() {
    try {
        const btn = document.getElementById('force-evolve-btn');
        if (btn) {
            btn.textContent = 'Evolving...';
            btn.disabled = true;
        }

        const response = await fetch('/api/evolution/evolve', { method: 'POST' });
        const data = await response.json();

        if (data.success) {
            showNotification('Evolution cycle completed!', 'success');
            loadEvolutionStats(); // Refresh stats
        } else {
            showNotification(data.error || 'Evolution failed', 'error');
        }
    } catch (error) {
        console.error('Force evolve error:', error);
        showNotification('Evolution request failed', 'error');
    } finally {
        const btn = document.getElementById('force-evolve-btn');
        if (btn) {
            btn.textContent = 'Evolve Now';
            btn.disabled = false;
        }
    }
}

// ============================================
// HEALTH SYSTEM (Legacy)
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

function disconnectWallet() {
    // Placeholder for wallet disconnect functionality
    showToast('Wallet disconnected', 'info');
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

// ============================================
// SWARM CHAT - COMMUNITY MODE
// ============================================

function switchChatMode(toSwarm) {
    console.log('[Chat] Switching to', toSwarm ? 'Swarm' : 'Personal', 'mode');
    state.swarmMode = toSwarm;

    // Update UI buttons
    document.getElementById('personal-chat-btn')?.classList.toggle('active', !toSwarm);
    document.getElementById('swarm-chat-btn')?.classList.toggle('active', toSwarm);

    // Toggle swarm status header
    const swarmHeader = document.getElementById('swarm-status-header');
    if (swarmHeader) swarmHeader.style.display = toSwarm ? 'flex' : 'none';

    // Toggle learning widget visibility
    document.getElementById('swarm-learning-widget')?.classList.toggle('hidden', !toSwarm);

    // Clear messages
    const messagesContainer = document.getElementById('messages');
    if (messagesContainer) {
        messagesContainer.innerHTML = '';
    }

    if (toSwarm) {
        // Connect to Swarm Chat
        connectSwarmChat();
        addSwarmWelcomeMessage();
    } else {
        // Disconnect from Swarm
        disconnectSwarmChat();
        addWelcomeMessage();
    }

    showToast(toSwarm ? '🐝 Switched to Swarm Chat - Community Mode!' : '💬 Switched to Personal Chat', 'success');
}

function initSwarmMode() {
    // Auto-connect to swarm mode - community chat where everyone talks together!
    console.log('[Chat] Auto-connecting to Global Swarm!');
    switchChatMode(true);  // Start in swarm mode by default
}

// Username management for swarm chat
function getOrPromptUsername() {
    let username = localStorage.getItem("swarmUsername");
    if (!username) {
        username = prompt("Welcome to the Swarm! Enter a display name:", "");
        if (username && username.trim()) {
            username = username.trim().slice(0, 20);
            localStorage.setItem("swarmUsername", username);
        } else {
            username = "User_" + Math.random().toString(36).slice(2, 8);
            localStorage.setItem("swarmUsername", username);
        }
    }
    return username;
}

function changeUsername() {
    const currentName = localStorage.getItem("swarmUsername") || state.swarmUserName || "";
    const newName = prompt("Enter new display name:", currentName);
    if (newName && newName.trim()) {
        const username = newName.trim().slice(0, 20);
        localStorage.setItem("swarmUsername", username);
        state.swarmUserName = username;
        showToast("Username changed to: " + username, "success");
        // Update display
        const usernameSpan = document.getElementById("current-username");
        if (usernameSpan) usernameSpan.textContent = username;
        // Reconnect to apply new name
        if (state.swarmConnected) {
            disconnectSwarmChat();
            setTimeout(connectSwarmChat, 500);
        }
    }
}

function connectSwarmChat() {
    console.log('[Swarm] Attempting to connect to swarm chat...');
    if (state.swarmWs && state.swarmConnected) {
        console.log('[Swarm] Already connected, skipping');
        return;
    }

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/swarm`;
    console.log('[Swarm] Connecting to:', wsUrl);

    try {
        state.swarmWs = new WebSocket(wsUrl);
        console.log('[Swarm] WebSocket created');

        state.swarmWs.onopen = () => {
            console.log('[Swarm] WebSocket connected!');
            // Use stored username or prompt for one
            const userName = getOrPromptUsername();
            state.swarmUserName = userName;
            // Update username display
            const usernameSpan = document.getElementById("current-username");
            if (usernameSpan) usernameSpan.textContent = userName;
            console.log('[Swarm] Sending identification as:', userName);
            state.swarmWs.send(JSON.stringify({
                type: 'identify',
                user_name: userName
            }));
        };

        state.swarmWs.onmessage = (event) => {
            const data = JSON.parse(event.data);
            console.log('[Swarm] Received message:', data.type, data);
            handleSwarmMessage(data);
        };

        state.swarmWs.onclose = () => {
            console.log('[Swarm] WebSocket closed');
            state.swarmConnected = false;
            updateSwarmStatus();
            // Reconnect if still in swarm mode
            if (state.swarmMode) {
                console.log('[Swarm] Will reconnect in 3 seconds...');
                setTimeout(connectSwarmChat, 3000);
            }
        };

        state.swarmWs.onerror = (error) => {
            console.error('[Swarm] WebSocket error:', error);
            state.swarmConnected = false;
        };

    } catch (error) {
        console.error('Swarm connection failed:', error);
    }
}

function disconnectSwarmChat() {
    if (state.swarmWs) {
        state.swarmWs.close();
        state.swarmWs = null;
    }
    state.swarmConnected = false;
    state.swarmOnlineUsers = [];
    updateSwarmStatus();
}

function handleSwarmMessage(data) {
    switch (data.type) {
        case 'swarm_connected':
            state.swarmConnected = true;
            state.swarmUserId = data.user_id;
            state.swarmOnlineUsers = data.online_users || [];
            state.swarmActiveModels = data.active_models || [];
            updateSwarmStatus();

            // Load history
            if (data.messages) {
                data.messages.forEach(msg => renderSwarmMessage(msg, false));
            }
            showToast(`🐝 Connected to Swarm! ${data.online_count} users online`, 'success');
            break;

        case 'swarm_user':
            // Skip if this is our own message (already shown via optimistic UI)
            if (data.user_id === state.swarmUserId) {
                console.log('[Swarm] Skipping own message (already displayed)');
                break;
            }
            renderSwarmMessage(data);
            break;

        case 'swarm_bot':
            renderSwarmMessage(data);
            break;

        case 'swarm_system':
            addSwarmSystemMessage(data.content);
            break;

        case 'swarm_typing':
            handleSwarmTyping(data.bot_name, data.is_typing);
            break;

        case 'swarm_tool':
            addSwarmToolMessage(data);
            break;

        case 'online_update':
            state.swarmOnlineUsers = data.online_users || [];
            updateSwarmStatus();
            break;

        case 'heartbeat':
        case 'pong':
            break;

        default:
            console.log('Swarm event:', data);
    }
}

// Track rendered messages to prevent duplicates
const renderedMessageIds = new Set();

function renderSwarmMessage(data, animate = true) {
    const messagesContainer = document.getElementById('messages');
    if (!messagesContainer) return;

    // Deduplication: check if we've already rendered this message
    const msgId = data.msg_id || `${data.bot_name || data.user_name}_${data.timestamp}_${(data.content || '').substring(0, 20)}`;
    if (renderedMessageIds.has(msgId)) {
        console.log('[Swarm] Skipping duplicate message:', msgId);
        return;
    }
    renderedMessageIds.add(msgId);

    // Keep set size manageable
    if (renderedMessageIds.size > 100) {
        const oldest = Array.from(renderedMessageIds).slice(0, 50);
        oldest.forEach(id => renderedMessageIds.delete(id));
    }

    const messageDiv = document.createElement('div');
    messageDiv.className = `message swarm-message ${data.type === 'swarm_user' ? 'user' : 'bot'}`;
    messageDiv.setAttribute('data-msg-id', msgId);
    if (animate) messageDiv.classList.add('animate-in');

    let avatar, name, content, extraClass = '';

    if (data.type === 'swarm_user') {
        avatar = '👤';
        name = data.user_name || 'Anonymous';
        content = data.content;
        extraClass = 'swarm-user-msg';
    } else if (data.type === 'swarm_bot') {
        // Bot colors and emojis - includes all multi-model participants
        const botStyles = {
            'Farnsworth': { emoji: '🧠', color: '#8b5cf6' },
            'DeepSeek': { emoji: '🔮', color: '#3b82f6' },
            'Phi': { emoji: '⚡', color: '#10b981' },
            'Swarm-Mind': { emoji: '🐝', color: '#f59e0b' },
            'Orchestrator': { emoji: '🎯', color: '#ec4899' },
            'Claude': { emoji: '🎭', color: '#d97706' },      // Anthropic Claude via CLI
            'Kimi': { emoji: '🌸', color: '#f472b6' }         // Moonshot AI Kimi
        };
        const style = botStyles[data.bot_name] || { emoji: '🤖', color: '#6b7280' };
        avatar = style.emoji;
        name = data.bot_name;
        content = data.content || '[No response]';  // Fallback for debugging
        extraClass = 'swarm-bot-msg';
        messageDiv.style.setProperty('--bot-color', style.color);
        console.log('Swarm bot message:', data.bot_name, 'content:', content?.substring(0, 50));
    }

    const time = new Date(data.timestamp || Date.now()).toLocaleTimeString([], {
        hour: '2-digit',
        minute: '2-digit'
    });

    messageDiv.innerHTML = `
        <div class="message-avatar ${extraClass}">${avatar}</div>
        <div class="message-body">
            <div class="message-meta">
                <span class="sender-name">${escapeHtml(name)}</span>
                <span class="message-time">${time}</span>
            </div>
            <div class="message-bubble glass-panel ${extraClass}">
                ${formatMessage(content || '')}
            </div>
        </div>
    `;

    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;

    // Voice for ALL swarm bots
    const voiceEnabledBots = ['Farnsworth', 'Kimi', 'DeepSeek', 'Phi', 'Grok', 'Gemini', 'Claude', 'ClaudeOpus', 'OpenCode', 'HuggingFace', 'Swarm-Mind'];
    if (data.type === 'swarm_bot' && state.voiceEnabled && voiceEnabledBots.includes(data.bot_name)) {
        // Use pre-generated audio URL if available (server-side TTS)
        if (data.audio_url) {
            playServerAudio(data.audio_url, data.bot_name);
        } else {
            speakText(content, data.bot_name);
        }
    }
}

function addSwarmSystemMessage(content) {
    const messagesContainer = document.getElementById('messages');
    if (!messagesContainer) return;

    const msgDiv = document.createElement('div');
    msgDiv.className = 'swarm-system-message';
    msgDiv.innerHTML = `<span class="system-content">${escapeHtml(content)}</span>`;
    messagesContainer.appendChild(msgDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function addSwarmToolMessage(data) {
    const messagesContainer = document.getElementById('messages');
    if (!messagesContainer) return;

    const msgDiv = document.createElement('div');
    msgDiv.className = 'swarm-tool-message';
    msgDiv.innerHTML = `
        <span class="tool-icon">🛠️</span>
        <span class="tool-user">${escapeHtml(data.user_name)}</span>
        <span class="tool-action">used</span>
        <span class="tool-name">${escapeHtml(data.tool_name)}</span>
        <span class="tool-status ${data.success ? 'success' : 'failed'}">${data.success ? '✓' : '✗'}</span>
    `;
    messagesContainer.appendChild(msgDiv);
}

function handleSwarmTyping(botName, isTyping) {
    if (isTyping) {
        state.swarmTypingBots.add(botName);
    } else {
        state.swarmTypingBots.delete(botName);
    }
    updateSwarmTypingIndicator();
}

function updateSwarmTypingIndicator() {
    const indicator = document.getElementById('typing-indicator');
    if (!indicator) return;

    if (state.swarmTypingBots.size > 0) {
        const bots = Array.from(state.swarmTypingBots);
        indicator.classList.remove('hidden');
        indicator.querySelector('.typing-name').textContent = bots.join(', ');
    } else {
        indicator.classList.add('hidden');
    }
}

function updateSwarmStatus() {
    // Update online count badge
    const countBadge = document.getElementById('swarm-online-count');
    if (countBadge) {
        countBadge.textContent = state.swarmOnlineUsers.length;
        countBadge.classList.toggle('active', state.swarmOnlineUsers.length > 0);
    }

    // Update users list in sidebar
    const usersList = document.querySelector('#swarm-users-list .user-list');
    if (usersList) {
        usersList.innerHTML = state.swarmOnlineUsers.map(user =>
            `<div class="swarm-user-item">
                <span class="user-dot"></span>
                <span class="user-name">${escapeHtml(user)}</span>
            </div>`
        ).join('') || '<div class="no-users">No users online</div>';
    }

    // Fetch learning stats
    if (state.swarmMode) {
        fetchSwarmLearningStats();
    }
}

async function fetchSwarmLearningStats() {
    try {
        const response = await fetch('/api/swarm/learning');
        const data = await response.json();

        if (data.learning_stats) {
            const stats = data.learning_stats;
            const cyclesEl = document.getElementById('learning-cycles');
            const conceptsEl = document.getElementById('concept-count');
            const conceptsListEl = document.getElementById('top-concepts');

            if (cyclesEl) cyclesEl.textContent = stats.learning_cycles || 0;
            if (conceptsEl) conceptsEl.textContent = stats.concept_count || 0;

            if (conceptsListEl && stats.top_concepts) {
                conceptsListEl.innerHTML = '<h4>🔥 Trending Concepts</h4>' +
                    stats.top_concepts.slice(0, 5).map(([concept, score]) =>
                        `<div class="concept-item">
                            <span class="concept-name">${escapeHtml(concept)}</span>
                            <span class="concept-score">${(score * 100).toFixed(0)}%</span>
                        </div>`
                    ).join('');
            }
        }

        // Also fetch real-time processing stats
        fetchProcessingStats();
    } catch (error) {
        console.error('Failed to fetch learning stats:', error);
    }
}

async function fetchProcessingStats() {
    try {
        // Fetch evolution and orchestrator stats for real-time view
        const [evolutionRes, orchestratorRes] = await Promise.all([
            fetch('/api/evolution/status'),
            fetch('/api/orchestrator/status')
        ]);

        const evolution = await evolutionRes.json();
        const orchestrator = await orchestratorRes.json();

        const processingEl = document.getElementById('processing-stats');
        if (processingEl) {
            let html = '<h4>⚡ Live Processing</h4>';

            // Evolution stats
            if (evolution.available) {
                html += `
                    <div class="processing-item">
                        <span class="proc-label">🧬 Learnings:</span>
                        <span class="proc-value">${evolution.total_learnings || 0}</span>
                    </div>
                    <div class="processing-item">
                        <span class="proc-label">🔄 Evolution Cycles:</span>
                        <span class="proc-value">${evolution.evolution_cycles || 0}</span>
                    </div>
                    <div class="processing-item">
                        <span class="proc-label">📦 Patterns:</span>
                        <span class="proc-value">${evolution.patterns_count || 0}</span>
                    </div>
                    <div class="processing-item">
                        <span class="proc-label">📝 Buffer:</span>
                        <span class="proc-value">${evolution.buffer_size || 0}</span>
                    </div>
                `;

                // Show personality evolution
                if (evolution.personalities) {
                    html += '<div class="personality-list"><h5>🤖 Bot Evolution</h5>';
                    for (const [name, data] of Object.entries(evolution.personalities)) {
                        html += `
                            <div class="personality-item">
                                <span class="bot-name">${name}</span>
                                <span class="bot-gen">Gen ${data.generation}</span>
                                <span class="bot-int">${data.interactions} msgs</span>
                            </div>
                        `;
                    }
                    html += '</div>';
                }
            }

            // Orchestrator stats
            if (orchestrator.available) {
                html += `
                    <div class="processing-item">
                        <span class="proc-label">🎯 Turn #:</span>
                        <span class="proc-value">${orchestrator.turn_number || 0}</span>
                    </div>
                    <div class="processing-item">
                        <span class="proc-label">😊 Mood:</span>
                        <span class="proc-value">${orchestrator.mood || 'curious'}</span>
                    </div>
                `;
            }

            processingEl.innerHTML = html;
        }
    } catch (error) {
        console.debug('Processing stats fetch error:', error);
    }
}

function addSwarmWelcomeMessage() {
    const messagesContainer = document.getElementById('messages');
    if (!messagesContainer) return;

    const welcomeDiv = document.createElement('div');
    welcomeDiv.className = 'swarm-welcome';
    welcomeDiv.innerHTML = `
        <div class="swarm-welcome-header">
            <span class="swarm-icon">🐝</span>
            <h2>Welcome to Swarm Chat!</h2>
        </div>
        <div class="swarm-welcome-body">
            <p>You're now in <strong>Community Mode</strong> - chat with everyone and our AI swarm!</p>
            <div class="swarm-features">
                <div class="swarm-feature">
                    <span class="feature-icon">👥</span>
                    <span>Chat with the community</span>
                </div>
                <div class="swarm-feature">
                    <span class="feature-icon">🤖</span>
                    <span>Multiple AI models respond</span>
                </div>
                <div class="swarm-feature">
                    <span class="feature-icon">🧬</span>
                    <span>System learns in real-time</span>
                </div>
                <div class="swarm-feature">
                    <span class="feature-icon">🪙</span>
                    <span>Ask about any token or CA!</span>
                </div>
            </div>
            <p class="swarm-crypto-hint">
                <strong>🔍 Try:</strong> "Price of $SOL" • "Is this safe? [paste CA]" • "Rug check [address]"
            </p>
            <p class="swarm-bots">
                <strong>Active Bots:</strong>
                🧠 Farnsworth • 🔮 DeepSeek • ⚡ Phi • 🐝 Swarm-Mind • ✨ Kimi
            </p>
        </div>
    `;
    messagesContainer.appendChild(welcomeDiv);
}

// Override sendMessage to handle swarm mode
const originalSendMessage = sendMessage;
sendMessage = async function() {
    if (state.swarmMode) {
        await sendSwarmMessage();
    } else {
        await originalSendMessage();
    }
};

async function sendSwarmMessage() {
    const input = document.getElementById('user-input');
    if (!input || !state.swarmWs || !state.swarmConnected) return;

    const message = input.value.trim();
    if (!message) return;

    // Clear input
    input.value = '';
    document.getElementById('char-count').textContent = '0';
    document.getElementById('send-btn').disabled = true;
    input.style.height = 'auto';

    // Optimistic UI: Show own message immediately
    const timestamp = new Date().toISOString();
    renderSwarmMessage({
        type: 'swarm_user',
        user_name: state.swarmUserName || 'You',
        user_id: state.swarmUserId,
        content: message,
        timestamp: timestamp,
        msg_id: `own_${timestamp}_${message.substring(0, 10)}`
    }, true);

    // Send to swarm
    state.swarmWs.send(JSON.stringify({
        type: 'swarm_message',
        content: message
    }));
}

// Make swarm functions globally available
window.switchChatMode = switchChatMode;
window.connectSwarmChat = connectSwarmChat;
window.disconnectSwarmChat = disconnectSwarmChat;
window.changeUsername = changeUsername;
window.getOrPromptUsername = getOrPromptUsername;

// ==========================================
// POLYMARKET PREDICTIONS WIDGET
// ==========================================

const polymarketState = {
    predictions: [],
    stats: { accuracy: 0, streak: 0, total: 0, correct: 0 },
    lastUpdate: null,
    pollInterval: null
};

async function fetchPolymarketPredictions() {
    try {
        const response = await fetch('/api/polymarket/predictions?limit=10');
        if (!response.ok) throw new Error('Failed to fetch predictions');
        const data = await response.json();
        polymarketState.predictions = data.predictions || [];
        polymarketState.lastUpdate = new Date();
        renderPolymarketPredictions();
    } catch (error) {
        console.error('Polymarket predictions fetch error:', error);
    }
}

async function fetchPolymarketStats() {
    try {
        const response = await fetch('/api/polymarket/stats');
        if (!response.ok) throw new Error('Failed to fetch stats');
        const data = await response.json();
        polymarketState.stats = data;
        updatePolymarketStats();
    } catch (error) {
        console.error('Polymarket stats fetch error:', error);
    }
}

function updatePolymarketStats() {
    const { accuracy, streak, total } = polymarketState.stats;

    const accuracyEl = document.getElementById('pm-accuracy');
    const streakEl = document.getElementById('pm-streak');
    const totalEl = document.getElementById('pm-total');

    if (accuracyEl) {
        accuracyEl.textContent = `${(accuracy * 100).toFixed(1)}%`;
        // Color based on accuracy
        if (accuracy >= 0.7) accuracyEl.style.color = '#00ff88';
        else if (accuracy >= 0.5) accuracyEl.style.color = '#ffd700';
        else accuracyEl.style.color = '#ff4444';
    }
    if (streakEl) {
        streakEl.textContent = streak;
        if (streak >= 5) streakEl.style.color = '#00ff88';
    }
    if (totalEl) totalEl.textContent = total;
}

function renderPolymarketPredictions() {
    const feed = document.getElementById('predictions-feed');
    if (!feed) return;

    if (polymarketState.predictions.length === 0) {
        feed.innerHTML = `
            <div class="prediction-empty">
                <div class="prediction-loading-spinner"></div>
                <p>Collective analyzing markets...</p>
                <p class="prediction-subtitle">Predictions refresh every 5 minutes</p>
            </div>
        `;
        return;
    }

    const predictionsHtml = polymarketState.predictions.map(pred => {
        const confidence = (pred.confidence * 100).toFixed(0);
        const direction = pred.direction;
        const directionClass = direction === 'YES' ? 'direction-yes' : 'direction-no';
        const directionIcon = direction === 'YES' ? '📈' : '📉';

        // Result badge if resolved
        let resultBadge = '';
        if (pred.result !== null && pred.result !== undefined) {
            const isCorrect = pred.result === true;
            resultBadge = `<span class="result-badge ${isCorrect ? 'result-correct' : 'result-wrong'}">${isCorrect ? '✓' : '✗'}</span>`;
        }

        // Format timestamp
        const timeAgo = formatTimeAgo(new Date(pred.timestamp));

        // Top signals
        const topSignals = pred.top_signals || [];
        const signalsHtml = topSignals.slice(0, 3).map(s =>
            `<span class="signal-tag" title="${s.reasoning}">${s.name}: ${(s.weight * 100).toFixed(0)}%</span>`
        ).join('');

        return `
            <div class="prediction-card ${directionClass}">
                <div class="prediction-header">
                    <span class="prediction-direction">${directionIcon} ${direction}</span>
                    <span class="prediction-confidence">${confidence}% confident</span>
                    ${resultBadge}
                </div>
                <div class="prediction-question">${truncateText(pred.question, 80)}</div>
                <div class="prediction-signals">${signalsHtml}</div>
                <div class="prediction-meta">
                    <span class="prediction-time">${timeAgo}</span>
                    <span class="prediction-market-price">Market: ${(pred.current_price * 100).toFixed(0)}%</span>
                </div>
            </div>
        `;
    }).join('');

    feed.innerHTML = predictionsHtml;
}

function truncateText(text, maxLength) {
    if (!text) return '';
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength - 3) + '...';
}

function formatTimeAgo(date) {
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);

    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    return date.toLocaleDateString();
}

function startPolymarketPolling() {
    // Initial fetch
    fetchPolymarketPredictions();
    fetchPolymarketStats();

    // Poll every 30 seconds for updates
    polymarketState.pollInterval = setInterval(() => {
        fetchPolymarketPredictions();
        fetchPolymarketStats();
    }, 30000);
}

function stopPolymarketPolling() {
    if (polymarketState.pollInterval) {
        clearInterval(polymarketState.pollInterval);
        polymarketState.pollInterval = null;
    }
}

// Initialize when DOM is ready and on swarm page
document.addEventListener('DOMContentLoaded', () => {
    const predictionsFeed = document.getElementById('predictions-feed');
    if (predictionsFeed) {
        startPolymarketPolling();
    }
});

// Expose functions globally
window.fetchPolymarketPredictions = fetchPolymarketPredictions;
window.fetchPolymarketStats = fetchPolymarketStats;
window.startPolymarketPolling = startPolymarketPolling;
