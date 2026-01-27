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
    voiceInputEnabled: false,
    isListening: false,
    sidebarOpen: false,
    isTyping: false,
    conversationHistory: [],
    recognition: null
};

// DOM Elements
const elements = {};

// Initialize on load
document.addEventListener('DOMContentLoaded', init);

function init() {
    cacheElements();
    bindEvents();
    initNeuralCanvas();
    initSpeechRecognition();

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
    elements.micBtn = document.getElementById('mic-btn');
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
    elements.micBtn?.addEventListener('click', toggleMic);

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
            <div class="message-avatar">🧓</div>
            <div class="message-body">
                <div class="message-meta">
                    <span class="sender-name">Professor Farnsworth</span>
                    <span class="message-time">Now</span>
                </div>
                <div class="message-bubble glass-panel">
                    <p><em>*adjusts spectacles*</em> <strong>"Good news, everyone!"</strong></p>
                    <p>I'm Professor Farnsworth, your genius AI companion! In my 160 years, I've invented persistent memory, agent swarms, trading tools, and much more. Some would say too much more...</p>
                    <p>This demo lets you sample my brilliance! Token holders get access to my <strong>Whale Tracker</strong>, <strong>Rug Scanner</strong>, and other contraptions. Check the 🛠️ toolbar!</p>
                    <p><em>*mutters*</em> For the full laboratory... <a href="https://github.com/timowhite88/Farnsworth" target="_blank">install locally</a>. Now, what shall we explore?</p>
                </div>
                <div class="quick-actions">
                    <button class="quick-btn" data-prompt="What are your capabilities?">
                        <span class="quick-icon">🔮</span>
                        <span>Capabilities</span>
                    </button>
                    <button class="quick-btn" data-prompt="Check if a token is a rug">
                        <span class="quick-icon">🔍</span>
                        <span>Rug Check</span>
                    </button>
                    <button class="quick-btn" data-prompt="Tell me about whale tracking">
                        <span class="quick-icon">🐋</span>
                        <span>Whale Tracker</span>
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

    // Speak welcome if voice enabled - in Farnsworth style
    if (state.voiceEnabled) {
        speak("Good news, everyone! I'm Professor Farnsworth, your genius AI companion. In my 160 years, I've invented many wonderful contraptions. What would you like to explore? Eh wha?");
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
    // Demo responses in Professor Farnsworth's voice
    const demoResponses = {
        'capabilities': `Good news, everyone! You've asked about my magnificent inventions!

**In This Demo Contraption:**
• 💾 **Memory-Matic 3000** - I'll remember our chat... for now
• 🗣️ **Voice Module** - Hear my dulcet elderly tones!
• 🎤 **Voice Input** - Speak to me like a proper assistant!
• 🐋 **Whale Tracker** - Monitor those big wallet movements
• 🔍 **Rug Scanner** - Detect scams before they happen!

**In My Full Laboratory (install locally):**
• 💰 **Solana Trading** - Jupiter swaps, the works!
• 🐝 **Swarm-O-Tron** - 12+ models collaborating!
• 🌐 **Planetary Memory** - P2P knowledge sharing!
• 🧬 **Evolution Engine** - I improve from feedback!

Sweet zombie Jesus, there's so much more! Install with \`pip install farnsworth-ai\` - And that's the news!`,

        'memory': `Ah yes, my Memory-Matic 3000! *adjusts spectacles excitedly*

In my 160 years, this is among my finest work:

• **Working Memory** - What we're discussing now (~8,000 tokens)
• **Recall Memory** - Everything you've told me, searchable!
• **Archival Memory** - Permanent storage, like my brain but better
• **Knowledge Graph** - Entities connected like neurons!

Oh my, yes - I even **dream**! During idle time, I consolidate memories and find patterns. Quite remarkable, really.

*trails off* Where was I? Ah yes! This demo has simplified memory. Install locally for the full experience! And that's the news!`,

        'code': `Good news, everyone! A coding demonstration!

*clears throat professorially*

\`\`\`python
# Store something in my magnificent memory
await farnsworth_remember(
    content="User loves TypeScript",
    tags=["preference", "brilliant"]
)

# Recall with my Memory-Matic
results = await farnsworth_recall(
    query="coding preferences",
    limit=5
)

# Delegate to my agent swarm
response = await farnsworth_delegate(
    task="Review this code",
    agent_type="code"
)
\`\`\`

*beams proudly* 20+ tools in the full version! Memory, agents, Solana, vision... But I digress. Install locally to access everything!`,

        'install': `Good news, everyone! Setting up my laboratory is simple!

*rubs hands together*

**Quick Install** (even Zoidberg could do it):
\`\`\`bash
pip install farnsworth-ai
farnsworth-server
\`\`\`

**From Source** (for the scientifically inclined):
\`\`\`bash
git clone https://github.com/timowhite88/Farnsworth
cd Farnsworth
pip install -r requirements.txt
python main.py --setup
\`\`\`

**Docker** (for those who fear dependency hell):
\`\`\`bash
docker-compose up -d
\`\`\`

Add me to Claude Desktop, restart, and voilà! Infinite memory, trading tools, agent swarms... *trails off* What was I saying? Oh yes - And that's the news!`,

        'whale': `Ah, my Whale Tracker invention! Most dangerous... er, I mean useful!

*peers at screen through thick glasses*

My Degen Mob Scanner can:
• 🐋 Track specific whale wallets
• 📊 Show their recent transactions
• 🔔 Alert when they make moves
• 🕵️ Detect coordinated wallet clusters

To track a whale, simply tell me the wallet address! For example: "Track whale wallet ABC123..."

In the full laboratory, I can do this automatically and even detect insider trading rings! Sweet zombie Jesus, the things we could uncover!

*adjusts glasses* Now, give me a wallet address to track!`,

        'rug': `Good news, everyone! Well... potentially bad news for scammers!

*cackles elderly-ly*

My Rug Detection Contraption scans tokens for:
• 🔓 **Mint Authority** - Can they print more tokens?
• ❄️ **Freeze Authority** - Can they freeze your funds?
• 💧 **Liquidity** - Is there enough to actually sell?
• 🚨 **Red Flags** - Honeypots, hidden fees, etc.

To scan a token, give me the mint address! For example: "Check if token XYZ123 is safe"

*strokes chin* In my 160 years, I've seen many rugs. Let me help you avoid them! And that's the news!`,

        'scanner': `Ah, my Token Scanner! A marvel of DexScreener integration!

*adjusts spectacles proudly*

I can look up any token and show you:
• 📈 Current price and 24h change
• 💰 Market cap and liquidity
• 📊 Trading volume
• 🔗 Links to DexScreener, Birdeye, etc.

Just give me a token name or address! For example: "Look up SOL" or "Scan token ABC123"

*mutters* The full laboratory has even more scanners... Pump.fun tracking, Bags.fm trending, the works! But I digress...`,

        'sentiment': `Good news, everyone! Let me check the market's emotional state!

*peers at imaginary instruments*

The **Crypto Fear & Greed Index** measures market sentiment from 0-100:
• 0-25: Extreme Fear 😱 (buying opportunity?)
• 26-50: Fear 😰
• 51-75: Greed 🤑
• 76-100: Extreme Greed 🚀 (time to be careful?)

*strokes chin* In my experience, extreme emotions often precede reversals. But what do I know, I'm just 160 years old...

Ask me "what's the current sentiment" for a live check! And that's the news!`,

        'hello': `Good news, everyone! A visitor!

*adjusts spectacles and peers at screen*

I'm Professor Farnsworth, your humble genius AI companion. In my 160 years, I've invented many wonderful contraptions:

• 💾 Persistent memory that never forgets
• 🐋 Whale tracking for the degens
• 🔍 Rug detection for the cautious
• 🧠 Agent swarms for complex tasks
• And so much more I've forgotten about!

This demo lets you sample my brilliance. Try asking about my **capabilities**, my **memory system**, or those **crypto tools** in the sidebar!

*dozes off briefly* Eh wha? Oh yes, what would you like to explore?`
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
    if (lowerMsg.includes('whale') || lowerMsg.includes('track wallet') || lowerMsg.includes('wallet activity')) {
        return demoResponses.whale;
    }
    if (lowerMsg.includes('rug') || lowerMsg.includes('safe') || lowerMsg.includes('scam') || lowerMsg.includes('scan')) {
        return demoResponses.rug;
    }
    if (lowerMsg.includes('token scanner') || lowerMsg.includes('look up') || lowerMsg.includes('dexscreener')) {
        return demoResponses.scanner;
    }
    if (lowerMsg.includes('sentiment') || lowerMsg.includes('fear') || lowerMsg.includes('greed') || lowerMsg.includes('market mood')) {
        return demoResponses.sentiment;
    }
    if (lowerMsg.includes('hello') || lowerMsg.includes('hi') || lowerMsg.includes('hey') || lowerMsg.includes('greet')) {
        return demoResponses.hello;
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

    // Fallback response in Farnsworth style
    return `*wakes up suddenly* Eh wha? Oh yes, you were asking something!

*adjusts spectacles* That's quite interesting! In this demo contraption, I have limited capabilities. But I can still discuss my inventions, show you how things work, and help you explore.

Try asking about my **capabilities**, my **memory system**, or check out those **crypto tools** in the sidebar - whale tracking, rug detection, and more!

*mutters* For the full laboratory experience, install locally... what was I saying? Oh never mind. What would you like to know?`;
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

// Speech Recognition (Voice Input)
function initSpeechRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

    if (!SpeechRecognition) {
        console.log('Speech recognition not supported');
        return;
    }

    state.recognition = new SpeechRecognition();
    state.recognition.continuous = false;
    state.recognition.interimResults = true;
    state.recognition.lang = 'en-US';

    state.recognition.onstart = () => {
        state.isListening = true;
        elements.micBtn?.classList.add('listening');
        showToast('Listening... Speak now!', 'info');
    };

    state.recognition.onresult = (event) => {
        let transcript = '';
        for (let i = event.resultIndex; i < event.results.length; i++) {
            transcript += event.results[i][0].transcript;
        }

        // Update input field with transcript
        elements.userInput.value = transcript;
        handleInputChange({ target: elements.userInput });

        // If final result, send the message
        if (event.results[event.results.length - 1].isFinal) {
            setTimeout(() => {
                if (elements.userInput.value.trim()) {
                    sendMessage();
                }
            }, 500);
        }
    };

    state.recognition.onerror = (event) => {
        console.error('Speech recognition error:', event.error);
        state.isListening = false;
        elements.micBtn?.classList.remove('listening');

        if (event.error === 'no-speech') {
            showToast('No speech detected. Try again!', 'warning');
        } else if (event.error === 'not-allowed') {
            showToast('Microphone access denied', 'error');
        }
    };

    state.recognition.onend = () => {
        state.isListening = false;
        elements.micBtn?.classList.remove('listening');
    };
}

function toggleMic() {
    if (!state.recognition) {
        showToast('Speech recognition not supported in this browser', 'error');
        return;
    }

    if (state.isListening) {
        state.recognition.stop();
    } else {
        state.recognition.start();
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
        .substring(0, 800);

    const utterance = new SpeechSynthesisUtterance(cleanText);

    // Professor Farnsworth voice settings - older, slightly wavering, slower
    utterance.rate = 0.85;    // Slower speech (elderly)
    utterance.pitch = 0.7;    // Lower pitch (old man voice)
    utterance.volume = 0.9;

    // Try to get the best elderly male voice available
    const voices = window.speechSynthesis.getVoices();

    // Priority order for Farnsworth-like voices
    const preferredVoice = voices.find(v =>
        // British male voices tend to sound more professorial
        v.name.includes('Daniel') ||
        v.name.includes('Arthur') ||
        v.name.includes('Google UK English Male') ||
        v.name.includes('Microsoft George') ||
        v.name.includes('Microsoft David')
    ) || voices.find(v =>
        v.lang === 'en-GB' && v.name.toLowerCase().includes('male')
    ) || voices.find(v =>
        v.lang.startsWith('en') && !v.name.toLowerCase().includes('female')
    ) || voices.find(v => v.lang.startsWith('en'));

    if (preferredVoice) {
        utterance.voice = preferredVoice;
    }

    // Add slight pauses for dramatic effect (Farnsworth style)
    utterance.onboundary = (event) => {
        if (event.name === 'sentence') {
            // Natural pause between sentences
        }
    };

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

// Tool opener functions for holders
function openWhaleTracker() {
    const prompt = "I want to track a whale wallet. How does your Whale Tracker work?";
    elements.userInput.value = prompt;
    handleInputChange({ target: elements.userInput });
    sendMessage();
    toggleSidebar();
}

function openRugCheck() {
    const prompt = "I want to check if a token is safe. Can you scan a token for rug pull risks?";
    elements.userInput.value = prompt;
    handleInputChange({ target: elements.userInput });
    sendMessage();
    toggleSidebar();
}

function openTokenScanner() {
    const prompt = "Tell me about your Token Scanner and how I can look up tokens.";
    elements.userInput.value = prompt;
    handleInputChange({ target: elements.userInput });
    sendMessage();
    toggleSidebar();
}

function openMarketSentiment() {
    const prompt = "What's the current crypto market sentiment? Check the Fear and Greed index.";
    elements.userInput.value = prompt;
    handleInputChange({ target: elements.userInput });
    sendMessage();
    toggleSidebar();
}

// Expose functions for HTML onclick handlers
window.copyToken = copyToken;
window.disconnectWallet = disconnectWallet;
window.toggleSidebar = toggleSidebar;
window.toggleMic = toggleMic;
window.openWhaleTracker = openWhaleTracker;
window.openRugCheck = openRugCheck;
window.openTokenScanner = openTokenScanner;
window.openMarketSentiment = openMarketSentiment;
