/**
 * Farnsworth Neural Interface
 * Token-gated chat with Solana wallet verification
 */

// Configuration
const CONFIG = {
    REQUIRED_TOKEN: '9crfy4udrHQo8eP6mP393b5qwpGLQgcxVg9acmdwBAGS',
    MIN_TOKEN_BALANCE: 1,
    API_ENDPOINT: '/api/chat',
    TTS_ENABLED: true,
    MAX_MESSAGE_LENGTH: 500,
    DEMO_MODE: true
};

// State
let state = {
    wallet: null,
    walletAddress: null,
    tokenVerified: false,
    voiceEnabled: true,
    sidebarOpen: false,
    isTyping: false,
    conversationHistory: []
};

// DOM Elements
const elements = {};

// Initialize on load
document.addEventListener('DOMContentLoaded', init);

function init() {
    cacheElements();
    bindEvents();
    initNeuralCanvas();

    // Check if already connected
    checkExistingConnection();
}

function cacheElements() {
    elements.tokenGate = document.getElementById('token-gate');
    elements.chatApp = document.getElementById('chat-app');
    elements.connectBtn = document.getElementById('connect-btn');
    elements.walletStatus = document.getElementById('wallet-status');
    elements.connectedWallet = document.getElementById('connected-wallet');
    elements.messages = document.getElementById('messages');
    elements.userInput = document.getElementById('user-input');
    elements.sendBtn = document.getElementById('send-btn');
    elements.charCount = document.getElementById('char-count');
    elements.voiceToggle = document.getElementById('voice-toggle');
    elements.toolsToggle = document.getElementById('tools-toggle');
    elements.toolsSidebar = document.getElementById('tools-sidebar');
    elements.typingIndicator = document.getElementById('typing-indicator');
    elements.toastContainer = document.getElementById('toast-container');
}

function bindEvents() {
    // Wallet connection
    elements.connectBtn?.addEventListener('click', connectWallet);

    // Chat input
    elements.userInput?.addEventListener('input', handleInputChange);
    elements.userInput?.addEventListener('keydown', handleKeyDown);
    elements.sendBtn?.addEventListener('click', sendMessage);

    // Controls
    elements.voiceToggle?.addEventListener('click', toggleVoice);
    elements.toolsToggle?.addEventListener('click', toggleSidebar);

    // Quick actions (delegated)
    document.addEventListener('click', handleQuickAction);
}

// Neural Canvas Animation
function initNeuralCanvas() {
    const canvas = document.getElementById('neural-canvas');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    const particles = [];
    const particleCount = 50;

    for (let i = 0; i < particleCount; i++) {
        particles.push({
            x: Math.random() * canvas.width,
            y: Math.random() * canvas.height,
            vx: (Math.random() - 0.5) * 0.5,
            vy: (Math.random() - 0.5) * 0.5,
            radius: Math.random() * 2 + 1,
            opacity: Math.random() * 0.5 + 0.2
        });
    }

    function animate() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        particles.forEach((p, i) => {
            p.x += p.vx;
            p.y += p.vy;

            if (p.x < 0 || p.x > canvas.width) p.vx *= -1;
            if (p.y < 0 || p.y > canvas.height) p.vy *= -1;

            ctx.beginPath();
            ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(139, 92, 246, ${p.opacity})`;
            ctx.fill();

            // Connect nearby particles
            particles.slice(i + 1).forEach(p2 => {
                const dx = p.x - p2.x;
                const dy = p.y - p2.y;
                const dist = Math.sqrt(dx * dx + dy * dy);

                if (dist < 150) {
                    ctx.beginPath();
                    ctx.moveTo(p.x, p.y);
                    ctx.lineTo(p2.x, p2.y);
                    ctx.strokeStyle = `rgba(139, 92, 246, ${0.1 * (1 - dist / 150)})`;
                    ctx.stroke();
                }
            });
        });

        requestAnimationFrame(animate);
    }

    animate();

    window.addEventListener('resize', () => {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    });
}

// Wallet Functions
async function checkExistingConnection() {
    if (window.solana?.isPhantom) {
        try {
            const response = await window.solana.connect({ onlyIfTrusted: true });
            if (response.publicKey) {
                state.walletAddress = response.publicKey.toString();
                await verifyToken();
            }
        } catch (err) {
            // Not already connected, show gate
        }
    }
}

async function connectWallet() {
    if (!window.solana?.isPhantom) {
        showToast('Please install Phantom wallet', 'error');
        window.open('https://phantom.app/', '_blank');
        return;
    }

    try {
        elements.connectBtn.disabled = true;
        elements.connectBtn.innerHTML = '<span class="btn-content">Connecting...</span>';

        const response = await window.solana.connect();
        state.walletAddress = response.publicKey.toString();
        state.wallet = window.solana;

        await verifyToken();

    } catch (err) {
        console.error('Wallet connection failed:', err);
        showToast('Connection failed. Please try again.', 'error');
        resetConnectButton();
    }
}

async function verifyToken() {
    elements.walletStatus.classList.remove('hidden');
    elements.walletStatus.innerHTML = `
        <div class="status-spinner"></div>
        <span class="status-text">Verifying token balance...</span>
    `;

    try {
        // In demo mode, simulate verification
        if (CONFIG.DEMO_MODE) {
            await simulateDelay(1500);

            // For demo, always pass verification
            state.tokenVerified = true;
            showToast('Token verified! Welcome to Farnsworth.', 'success');
            enterChat();
            return;
        }

        // Real verification would go here
        const balance = await checkTokenBalance(state.walletAddress);

        if (balance >= CONFIG.MIN_TOKEN_BALANCE) {
            state.tokenVerified = true;
            showToast('Token verified! Welcome to Farnsworth.', 'success');
            enterChat();
        } else {
            elements.walletStatus.innerHTML = `
                <span class="status-text" style="color: var(--error);">
                    Insufficient token balance. You need at least ${CONFIG.MIN_TOKEN_BALANCE} token(s).
                </span>
            `;
            resetConnectButton();
        }

    } catch (err) {
        console.error('Token verification failed:', err);

        // In demo mode, still allow access
        if (CONFIG.DEMO_MODE) {
            state.tokenVerified = true;
            showToast('Demo mode: Access granted', 'warning');
            enterChat();
        } else {
            elements.walletStatus.innerHTML = `
                <span class="status-text" style="color: var(--error);">
                    Verification failed. Please try again.
                </span>
            `;
            resetConnectButton();
        }
    }
}

async function checkTokenBalance(walletAddress) {
    // This would connect to Solana RPC to check token balance
    // For now, return mock balance
    return 1;
}

function resetConnectButton() {
    elements.connectBtn.disabled = false;
    elements.connectBtn.innerHTML = `
        <span class="btn-content">
            <span class="phantom-icon">👻</span>
            <span>Connect Phantom</span>
        </span>
        <span class="btn-shimmer"></span>
    `;
}

function enterChat() {
    // Update wallet display
    const shortAddr = `${state.walletAddress.slice(0, 4)}...${state.walletAddress.slice(-4)}`;
    elements.connectedWallet.querySelector('.wallet-addr').textContent = shortAddr;

    // Transition to chat
    elements.tokenGate.style.opacity = '0';
    elements.tokenGate.style.transform = 'scale(0.95)';

    setTimeout(() => {
        elements.tokenGate.style.display = 'none';
        elements.chatApp.classList.remove('hidden');
        elements.chatApp.style.opacity = '0';

        requestAnimationFrame(() => {
            elements.chatApp.style.transition = 'opacity 0.5s ease';
            elements.chatApp.style.opacity = '1';
        });

        // Add welcome message
        addWelcomeMessage();

        // Focus input
        setTimeout(() => elements.userInput?.focus(), 500);
    }, 300);
}

function addWelcomeMessage() {
    const welcomeHTML = `
        <div class="message assistant" data-animate="true">
            <div class="message-avatar">🧠</div>
            <div class="message-body">
                <div class="message-meta">
                    <span class="sender-name">Farnsworth</span>
                    <span class="message-time">Now</span>
                </div>
                <div class="message-bubble glass-panel">
                    <p><em>"Good news, everyone!"</em></p>
                    <p>I'm Farnsworth, your Claude Companion AI. I possess persistent memory, can delegate to specialist agents, and evolve based on your feedback.</p>
                    <p>This is a <strong>limited demo interface</strong>. For the complete neural experience—including Solana trading, P2P networking, and model swarms—<a href="https://github.com/timowhite88/Farnsworth" target="_blank">install me locally</a>.</p>
                    <p>What would you like to explore?</p>
                </div>
                <div class="quick-actions">
                    <button class="quick-btn" data-prompt="What are your capabilities?">
                        <span class="quick-icon">🔮</span>
                        <span>Capabilities</span>
                    </button>
                    <button class="quick-btn" data-prompt="Tell me about your memory system">
                        <span class="quick-icon">🧠</span>
                        <span>Memory</span>
                    </button>
                    <button class="quick-btn" data-prompt="Show me a code example">
                        <span class="quick-icon">💻</span>
                        <span>Code Demo</span>
                    </button>
                    <button class="quick-btn" data-prompt="How do I install you locally?">
                        <span class="quick-icon">📦</span>
                        <span>Install</span>
                    </button>
                </div>
            </div>
        </div>
    `;

    elements.messages.innerHTML = welcomeHTML;

    // Speak welcome if voice enabled
    if (state.voiceEnabled) {
        speak("Good news, everyone! I'm Farnsworth, your Claude Companion AI. What would you like to explore?");
    }
}

// Chat Functions
function handleInputChange(e) {
    const length = e.target.value.length;
    elements.charCount.textContent = length;
    elements.sendBtn.disabled = length === 0 || length > CONFIG.MAX_MESSAGE_LENGTH;

    // Auto-resize textarea
    e.target.style.height = 'auto';
    e.target.style.height = Math.min(e.target.scrollHeight, 150) + 'px';
}

function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
}

async function sendMessage() {
    const content = elements.userInput.value.trim();
    if (!content || state.isTyping) return;

    // Add user message
    addMessage('user', content);

    // Clear input
    elements.userInput.value = '';
    elements.charCount.textContent = '0';
    elements.sendBtn.disabled = true;
    elements.userInput.style.height = 'auto';

    // Show typing indicator
    showTyping(true);

    try {
        // Get AI response
        const response = await getAIResponse(content);

        // Hide typing
        showTyping(false);

        // Add assistant message
        addMessage('assistant', response);

        // Speak response if voice enabled
        if (state.voiceEnabled) {
            speak(response);
        }

    } catch (err) {
        console.error('Failed to get response:', err);
        showTyping(false);
        addMessage('assistant', "I apologize, but I'm experiencing connectivity issues. This demo interface has limited capabilities. For full functionality, please install Farnsworth locally.");
    }
}

function addMessage(role, content) {
    const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const avatar = role === 'user' ? '👤' : '🧠';
    const name = role === 'user' ? 'You' : 'Farnsworth';

    const messageHTML = `
        <div class="message ${role}">
            <div class="message-avatar">${avatar}</div>
            <div class="message-body">
                <div class="message-meta">
                    <span class="sender-name">${name}</span>
                    <span class="message-time">${time}</span>
                </div>
                <div class="message-bubble glass-panel">
                    ${formatMessage(content)}
                </div>
            </div>
        </div>
    `;

    elements.messages.insertAdjacentHTML('beforeend', messageHTML);
    elements.messages.scrollTop = elements.messages.scrollHeight;

    // Store in history
    state.conversationHistory.push({ role, content });
}

function formatMessage(content) {
    // Convert markdown-like syntax to HTML
    return content
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/`([^`]+)`/g, '<code>$1</code>')
        .replace(/```(\w+)?\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>')
        .replace(/\n/g, '</p><p>')
        .replace(/^/, '<p>')
        .replace(/$/, '</p>');
}

async function getAIResponse(userMessage) {
    // Demo responses for common questions
    const demoResponses = {
        'capabilities': `I have many capabilities! Here's what I can do:

**Available in Demo:**
• 💾 **Memory** - I remember our conversations
• 🗣️ **Voice** - I can speak my responses
• 🤖 **Basic chat** - General conversation and questions

**Full Version Only (Install Locally):**
• 💰 **Solana Trading** - Jupiter swaps, DexScreener, wallet management
• 🐝 **Model Swarm** - 12+ models collaborating via PSO
• 🌐 **P2P Network** - Planetary memory sharing
• 👁️ **Vision** - Image analysis, OCR, scene understanding
• 📈 **Evolution** - Self-improvement from your feedback

To unlock everything, install locally: \`pip install farnsworth-ai\``,

        'memory': `My memory system is hierarchical, inspired by human cognition:

**Working Memory** - Current conversation context (~8,000 tokens)
**Recall Memory** - Searchable conversation history
**Archival Memory** - Permanent semantic storage (unlimited)
**Knowledge Graph** - Entities and relationships

I also have **Memory Dreaming** - during idle time, I consolidate memories, find patterns, and generate insights. It's like I'm literally dreaming about our conversations!

In this demo, I have limited memory. The full version stores everything permanently.`,

        'code': `Here's a quick example of using Farnsworth's MCP tools:

\`\`\`python
# Remember something
await farnsworth_remember(
    content="User prefers TypeScript",
    tags=["preference", "code"]
)

# Recall memories
results = await farnsworth_recall(
    query="coding preferences",
    limit=5
)

# Delegate to an agent
response = await farnsworth_delegate(
    task="Review this code for bugs",
    agent_type="code"
)
\`\`\`

Pretty cool, right? The full version has 20+ tools for memory, agents, Solana, vision, and more!`,

        'install': `Here's how to install Farnsworth locally:

**Option 1: pip (easiest)**
\`\`\`bash
pip install farnsworth-ai
farnsworth-server
\`\`\`

**Option 2: From source**
\`\`\`bash
git clone https://github.com/timowhite88/Farnsworth.git
cd Farnsworth
pip install -r requirements.txt
python main.py --setup
\`\`\`

**Option 3: Docker**
\`\`\`bash
docker-compose -f docker/docker-compose.yml up -d
\`\`\`

After installing, add Farnsworth to your Claude Desktop config and restart. You'll get infinite memory, model swarms, and all the premium features!`
    };

    // Check for matching demo response
    const lowerMsg = userMessage.toLowerCase();

    if (lowerMsg.includes('capabil') || lowerMsg.includes('what can you')) {
        return demoResponses.capabilities;
    }
    if (lowerMsg.includes('memory') || lowerMsg.includes('remember')) {
        return demoResponses.memory;
    }
    if (lowerMsg.includes('code') || lowerMsg.includes('example')) {
        return demoResponses.code;
    }
    if (lowerMsg.includes('install') || lowerMsg.includes('setup') || lowerMsg.includes('locally')) {
        return demoResponses.install;
    }

    // Try to call actual API
    try {
        const response = await fetch(CONFIG.API_ENDPOINT, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: userMessage,
                wallet: state.walletAddress,
                history: state.conversationHistory.slice(-10)
            })
        });

        if (response.ok) {
            const data = await response.json();
            return data.response;
        }
    } catch (err) {
        // API not available, use fallback
    }

    // Fallback response
    return `That's an interesting question! In this demo interface, my capabilities are limited. I can discuss my features, show code examples, and explain how to install the full version.

For deeper conversations, complex tasks, Solana trading, and all my advanced features, you'll want to **install Farnsworth locally**.

What else would you like to know about? Try asking about my *capabilities*, *memory system*, or *how to install*.`;
}

function showTyping(show) {
    state.isTyping = show;
    elements.typingIndicator.classList.toggle('hidden', !show);

    if (show) {
        elements.messages.scrollTop = elements.messages.scrollHeight;
    }
}

// Quick Actions
function handleQuickAction(e) {
    const btn = e.target.closest('.quick-btn');
    if (!btn) return;

    const prompt = btn.dataset.prompt;
    if (prompt) {
        elements.userInput.value = prompt;
        handleInputChange({ target: elements.userInput });
        sendMessage();
    }
}

// Voice / TTS
function toggleVoice() {
    state.voiceEnabled = !state.voiceEnabled;
    elements.voiceToggle.classList.toggle('active', state.voiceEnabled);

    if (!state.voiceEnabled) {
        window.speechSynthesis?.cancel();
    }

    showToast(state.voiceEnabled ? 'Voice enabled' : 'Voice disabled', 'success');
}

function speak(text) {
    if (!state.voiceEnabled || !window.speechSynthesis) return;

    // Cancel any ongoing speech
    window.speechSynthesis.cancel();

    // Clean text for speech
    const cleanText = text
        .replace(/\*\*/g, '')
        .replace(/\*/g, '')
        .replace(/`[^`]+`/g, 'code')
        .replace(/```[\s\S]*?```/g, 'code block')
        .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
        .substring(0, 500);

    const utterance = new SpeechSynthesisUtterance(cleanText);
    utterance.rate = 1.0;
    utterance.pitch = 0.9;
    utterance.volume = 0.8;

    // Try to get a good voice
    const voices = window.speechSynthesis.getVoices();
    const preferredVoice = voices.find(v =>
        v.name.includes('Daniel') ||
        v.name.includes('Google UK English Male') ||
        v.lang === 'en-GB'
    ) || voices.find(v => v.lang.startsWith('en'));

    if (preferredVoice) {
        utterance.voice = preferredVoice;
    }

    window.speechSynthesis.speak(utterance);
}

// Sidebar
function toggleSidebar() {
    state.sidebarOpen = !state.sidebarOpen;
    elements.toolsSidebar.classList.toggle('hidden', !state.sidebarOpen);

    // Animate in
    if (state.sidebarOpen) {
        requestAnimationFrame(() => {
            elements.toolsSidebar.classList.add('active');
        });
    } else {
        elements.toolsSidebar.classList.remove('active');
    }

    elements.toolsToggle.classList.toggle('active', state.sidebarOpen);
}

// Utility Functions
function copyToken() {
    navigator.clipboard.writeText(CONFIG.REQUIRED_TOKEN);
    showToast('Token address copied!', 'success');
}

function disconnectWallet() {
    if (window.solana?.disconnect) {
        window.solana.disconnect();
    }

    state.wallet = null;
    state.walletAddress = null;
    state.tokenVerified = false;

    // Return to gate
    elements.chatApp.classList.add('hidden');
    elements.tokenGate.style.display = 'flex';
    elements.tokenGate.style.opacity = '1';
    elements.tokenGate.style.transform = 'scale(1)';
    elements.walletStatus.classList.add('hidden');
    resetConnectButton();

    showToast('Wallet disconnected', 'success');
}

function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;

    elements.toastContainer.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(100px)';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

function simulateDelay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// Expose functions for HTML onclick handlers
window.copyToken = copyToken;
window.disconnectWallet = disconnectWallet;
window.toggleSidebar = toggleSidebar;
