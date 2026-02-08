/**
 * DEXAI v2.0 - Farnsworth Collective DEX Screener
 * A DexScreener replacement powered by the Farnsworth AI Collective
 */

const express = require('express');
const cors = require('cors');
const { WebSocketServer } = require('ws');
const http = require('http');
const path = require('path');

const app = express();
const server = http.createServer(app);
const wss = new WebSocketServer({ server, path: '/ws' });

const PORT = process.env.DEXAI_PORT || 3847;
const FARNSWORTH_API = process.env.FARNSWORTH_API || 'http://localhost:8080';
const ECOSYSTEM_WALLET = '3fSS5RVErbgcJEDCQmCXpKsD2tWqfhxFZtkDUB8qw';
const FARNS_TOKEN = '9crfy4udrHQo8eP6mP393b5qwpGLQgcxVg9acmdwBAGS';
const BOOST_PRICE_USD = 25;
const EXTENDED_INFO_PRICE_USD = 10;
const FETCH_INTERVAL = 30000;

app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));
app.use((req, res, next) => { console.log(`[${new Date().toISOString()}] ${req.method} ${req.path}`); next(); });

// ============================================================
// TOKEN CACHE
// ============================================================
const tokenCache = new Map();
let sortedByVolume = [];
let sortedByTrending = [];
let sortedByVelocity = [];
let sortedByNew = [];
let sortedByGainers = [];
let sortedByLosers = [];
let cacheReady = false;
const boosts = new Map();
const pendingBoosts = new Map();
const aiScoreCache = new Map();

function parsePair(pair) {
    const base = pair.baseToken || {};
    const quote = pair.quoteToken || {};
    const info = pair.info || {};
    const txns = pair.txns || {};
    const vol = pair.volume || {};
    const pc = pair.priceChange || {};
    const liq = pair.liquidity || {};
    const socials = {};
    for (const s of (info.socials || [])) socials[s.type || 'unknown'] = s.url || '';

    return {
        address: base.address || '',
        symbol: base.symbol || '???',
        name: base.name || 'Unknown',
        price: parseFloat(pair.priceUsd) || 0,
        priceNative: parseFloat(pair.priceNative) || 0,
        priceChange: {
            m5: parseFloat(pc.m5) || 0,
            h1: parseFloat(pc.h1) || 0,
            h6: parseFloat(pc.h6) || 0,
            h24: parseFloat(pc.h24) || 0,
        },
        volume: {
            m5: parseFloat(vol.m5) || 0,
            h1: parseFloat(vol.h1) || 0,
            h6: parseFloat(vol.h6) || 0,
            h24: parseFloat(vol.h24) || 0,
        },
        liquidity: parseFloat(liq.usd) || 0,
        fdv: parseFloat(pair.fdv) || 0,
        marketCap: parseFloat(pair.marketCap) || 0,
        txns: {
            m5: { buys: txns.m5?.buys || 0, sells: txns.m5?.sells || 0 },
            h1: { buys: txns.h1?.buys || 0, sells: txns.h1?.sells || 0 },
            h6: { buys: txns.h6?.buys || 0, sells: txns.h6?.sells || 0 },
            h24: { buys: txns.h24?.buys || 0, sells: txns.h24?.sells || 0 },
        },
        pairAddress: pair.pairAddress || '',
        pairCreatedAt: pair.pairCreatedAt || null,
        dexId: pair.dexId || 'raydium',
        chainId: pair.chainId || 'solana',
        url: pair.url || '',
        imageUrl: info.imageUrl || '',
        headerImg: info.header || '',
        websites: (info.websites || []).map(w => typeof w === 'string' ? w : (w.url || '')),
        socials,
        boostAmount: 0,
        aiScore: null,
        aiVerdict: null,
        collectiveVerified: false,
        velocity: 0,
        trendScore: 0,
        updatedAt: Date.now(),
    };
}

async function safeFetch(url, opts = {}) {
    try {
        const res = await fetch(url, { signal: AbortSignal.timeout(15000), ...opts });
        if (!res.ok) return null;
        return await res.json();
    } catch { return null; }
}

async function fetchTokenBatch(addresses) {
    const results = [];
    for (let i = 0; i < addresses.length; i += 30) {
        const batch = addresses.slice(i, i + 30);
        const data = await safeFetch(`https://api.dexscreener.com/latest/dex/tokens/${batch.join(',')}`);
        if (data?.pairs) {
            const byToken = {};
            for (const pair of data.pairs) {
                if (pair.chainId !== 'solana') continue;
                const addr = pair.baseToken?.address;
                if (!addr) continue;
                const liq = parseFloat(pair.liquidity?.usd) || 0;
                if (!byToken[addr] || liq > (parseFloat(byToken[addr].liquidity?.usd) || 0)) {
                    byToken[addr] = pair;
                }
            }
            for (const pair of Object.values(byToken)) results.push(parsePair(pair));
        }
        if (i + 30 < addresses.length) await new Promise(r => setTimeout(r, 250));
    }
    return results;
}

async function updateTokenCache() {
    try {
        const addresses = new Set();
        const boostAmounts = {};

        // 1. Boosted tokens
        const boostData = await safeFetch('https://api.dexscreener.com/token-boosts/latest/v1');
        if (Array.isArray(boostData)) {
            for (const item of boostData) {
                if (item.chainId === 'solana' && item.tokenAddress) {
                    addresses.add(item.tokenAddress);
                    boostAmounts[item.tokenAddress] = (boostAmounts[item.tokenAddress] || 0) + (item.amount || 0);
                }
            }
        }
        await new Promise(r => setTimeout(r, 300));

        // 2. Token profiles
        const profileData = await safeFetch('https://api.dexscreener.com/token-profiles/latest/v1');
        if (Array.isArray(profileData)) {
            for (const item of profileData) {
                if (item.chainId === 'solana' && item.tokenAddress) addresses.add(item.tokenAddress);
            }
        }
        await new Promise(r => setTimeout(r, 300));

        // 3. Search popular terms for broader coverage
        for (const term of ['SOL', 'pump', 'meme', 'BONK', 'WIF', 'AI']) {
            const data = await safeFetch(`https://api.dexscreener.com/latest/dex/search?q=${term}`);
            if (data?.pairs) {
                for (const pair of data.pairs.slice(0, 40)) {
                    if (pair.chainId === 'solana' && pair.baseToken?.address) {
                        addresses.add(pair.baseToken.address);
                    }
                }
            }
            await new Promise(r => setTimeout(r, 250));
        }

        console.log(`[DEXAI] Fetching details for ${addresses.size} tokens...`);
        const tokens = await fetchTokenBatch([...addresses]);

        // Merge into cache
        for (const token of tokens) {
            if (!token.address || token.volume.h24 <= 0) continue;
            const existing = tokenCache.get(token.address);
            if (existing) {
                // Velocity calc
                const timeDiff = (Date.now() - existing.updatedAt) / 60000;
                if (timeDiff > 0.5) {
                    token.velocity = Math.max(0, ((token.txns.h1.buys - existing.txns.h1.buys) / timeDiff).toFixed(2));
                } else {
                    token.velocity = existing.velocity;
                }
                token.aiScore = existing.aiScore;
                token.aiVerdict = existing.aiVerdict;
                token.collectiveVerified = existing.collectiveVerified;
            }
            token.boostAmount = boostAmounts[token.address] || token.boostAmount;

            // Trend score: volume + boosts + velocity + price momentum
            token.trendScore = (
                (token.volume.h24 > 0 ? Math.log10(token.volume.h24) * 10 : 0) +
                (token.boostAmount * 5) +
                (parseFloat(token.velocity) || 0) * 2 +
                Math.max(0, token.priceChange.h1) * 0.5 +
                (token.txns.h1.buys + token.txns.h1.sells) * 0.1
            );

            tokenCache.set(token.address, token);
        }

        // Build sorted arrays
        const all = [...tokenCache.values()].filter(t => t.volume.h24 > 0);
        sortedByVolume = [...all].sort((a, b) => b.volume.h24 - a.volume.h24).slice(0, 200);
        sortedByTrending = [...all].sort((a, b) => b.trendScore - a.trendScore).slice(0, 200);
        sortedByVelocity = [...all].sort((a, b) => (parseFloat(b.velocity) || 0) - (parseFloat(a.velocity) || 0)).slice(0, 200);
        sortedByNew = [...all].filter(t => t.pairCreatedAt).sort((a, b) => (b.pairCreatedAt || 0) - (a.pairCreatedAt || 0)).slice(0, 200);
        sortedByGainers = [...all].sort((a, b) => b.priceChange.h24 - a.priceChange.h24).slice(0, 200);
        sortedByLosers = [...all].sort((a, b) => a.priceChange.h24 - b.priceChange.h24).slice(0, 200);

        // Trim cache
        if (tokenCache.size > 600) {
            const keep = new Set(sortedByVolume.slice(0, 300).map(t => t.address));
            for (const [addr] of tokenCache) { if (!keep.has(addr)) tokenCache.delete(addr); }
        }

        cacheReady = true;
        console.log(`[DEXAI] Cache: ${tokenCache.size} tokens | Top volume: ${sortedByVolume[0]?.symbol} ($${(sortedByVolume[0]?.volume.h24 / 1e6).toFixed(1)}M)`);
        broadcastAll({ type: 'update', count: tokenCache.size, ts: Date.now() });
    } catch (e) {
        console.error('[DEXAI] Cache error:', e.message);
    }
}

// ============================================================
// AI SCORING (Farnsworth Collective)
// ============================================================
async function getAIScore(address) {
    const cached = aiScoreCache.get(address);
    if (cached && Date.now() - cached.ts < 300000) return cached.data;

    try {
        const data = await safeFetch(`${FARNSWORTH_API}/api/tools/score-token`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ token_address: address }),
        });
        if (data) {
            const result = {
                score: data.score || null,
                verdict: data.prediction || data.verdict || null,
                rugProbability: data.rug_probability || null,
                cabalScore: data.cabal_score || null,
                bundleDetected: data.bundle_detected || false,
                swarmSentiment: data.swarm_sentiment || null,
                agents: data.agents || [],
            };
            aiScoreCache.set(address, { data: result, ts: Date.now() });
            // Update token cache
            const token = tokenCache.get(address);
            if (token) { token.aiScore = result.score; token.aiVerdict = result.verdict; }
            return result;
        }
    } catch {}
    return null;
}

// ============================================================
// API ROUTES
// ============================================================
app.get('/api/health', (req, res) => {
    res.json({
        status: 'ok', service: 'DEXAI', version: '2.0.0',
        tokens: tokenCache.size, cacheReady,
        ecosystem_wallet: ECOSYSTEM_WALLET, farns_token: FARNS_TOKEN,
    });
});

app.get('/api/tokens', (req, res) => {
    const { sort = 'trending', limit = 100, offset = 0 } = req.query;
    const lim = Math.min(parseInt(limit) || 100, 200);
    const off = parseInt(offset) || 0;

    const lists = {
        volume: sortedByVolume, trending: sortedByTrending, velocity: sortedByVelocity,
        new: sortedByNew, gainers: sortedByGainers, losers: sortedByLosers,
    };
    const list = lists[sort] || sortedByTrending;
    const page = list.slice(off, off + lim);
    res.json({ tokens: page, total: list.length, sort, offset: off, limit: lim, cacheReady, ts: Date.now() });
});

app.get('/api/token/:address', async (req, res) => {
    const { address } = req.params;
    let token = tokenCache.get(address);

    if (!token) {
        // Fetch directly
        const data = await safeFetch(`https://api.dexscreener.com/latest/dex/tokens/${address}`);
        if (data?.pairs?.length) {
            const pairs = data.pairs.filter(p => p.chainId === 'solana')
                .sort((a, b) => (parseFloat(b.liquidity?.usd) || 0) - (parseFloat(a.liquidity?.usd) || 0));
            if (pairs.length) {
                token = parsePair(pairs[0]);
                tokenCache.set(address, token);
            }
        }
    }

    if (!token) return res.status(404).json({ error: 'Token not found' });

    // Fetch AI score in background
    getAIScore(address).then(ai => {
        if (ai) { token.aiScore = ai.score; token.aiVerdict = ai.verdict; }
    });

    const boost = boosts.get(address);
    res.json({ token, boost: boost || null, allPairs: [] });
});

app.get('/api/search', async (req, res) => {
    const { q } = req.query;
    if (!q || q.length < 2) return res.json({ tokens: [] });

    // Check cache first
    const query = q.toLowerCase();
    const cacheResults = [...tokenCache.values()].filter(t =>
        t.symbol.toLowerCase().includes(query) ||
        t.name.toLowerCase().includes(query) ||
        t.address.toLowerCase().startsWith(query)
    ).slice(0, 20);

    if (cacheResults.length >= 5) return res.json({ tokens: cacheResults });

    // Fall back to DexScreener search
    const data = await safeFetch(`https://api.dexscreener.com/latest/dex/search?q=${encodeURIComponent(q)}`);
    const tokens = (data?.pairs || []).filter(p => p.chainId === 'solana').slice(0, 30).map(parsePair);
    // Merge cache results + API results, deduplicate
    const seen = new Set(cacheResults.map(t => t.address));
    for (const t of tokens) { if (!seen.has(t.address)) { cacheResults.push(t); seen.add(t.address); } }
    res.json({ tokens: cacheResults.slice(0, 30) });
});

// Chart data via GeckoTerminal (free OHLCV)
app.get('/api/chart/:address', async (req, res) => {
    const { address } = req.params;
    const { timeframe = '15m' } = req.query;
    const tfMap = { '1m': 'minute&aggregate=1', '5m': 'minute&aggregate=5', '15m': 'minute&aggregate=15', '1h': 'hour&aggregate=1', '4h': 'hour&aggregate=4', '1d': 'day&aggregate=1' };
    const tf = tfMap[timeframe] || tfMap['15m'];

    const token = tokenCache.get(address);
    const poolAddr = token?.pairAddress || address;

    const data = await safeFetch(`https://api.geckoterminal.com/api/v2/networks/solana/pools/${poolAddr}/ohlcv/${tf}&limit=300&currency=usd`);
    if (!data?.data?.attributes?.ohlcv_list) {
        return res.json({ candles: [], source: 'unavailable' });
    }

    const candles = data.data.attributes.ohlcv_list.map(c => ({
        time: c[0], open: parseFloat(c[1]), high: parseFloat(c[2]),
        low: parseFloat(c[3]), close: parseFloat(c[4]), volume: parseFloat(c[5]),
    })).sort((a, b) => a.time - b.time);

    res.json({ candles, source: 'geckoterminal', pair: poolAddr });
});

// AI Prediction
app.get('/api/ai/score/:address', async (req, res) => {
    const result = await getAIScore(req.params.address);
    res.json(result || { score: null, message: 'Collective unavailable' });
});

// Quantum Simulation
app.get('/api/quantum/:address', async (req, res) => {
    const data = await safeFetch(`${FARNSWORTH_API}/api/farsight/crypto`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ token_address: req.params.address, simulations: 1000 }),
    });
    res.json(data || { available: false });
});

// ============================================================
// BOOST SYSTEM
// ============================================================
app.get('/api/boost/:address', (req, res) => {
    res.json({ boost: boosts.get(req.params.address) || null });
});

app.post('/api/boost/request', (req, res) => {
    const { address, paymentType = 'sol' } = req.body;
    if (!address) return res.status(400).json({ error: 'Token address required' });

    const existing = boosts.get(address);
    if (existing?.boostCount > 0 && !existing.collectiveApproved) {
        return res.json({ status: 'requires_approval', message: 'Additional boosts require Collective approval.', boostCount: existing.boostCount });
    }

    const requestId = `boost_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
    const payment = paymentType === 'farns'
        ? { type: 'farns_burn', tokenMint: FARNS_TOKEN, burnAddress: '1nc1nerator11111111111111111111111111111111', amountUsd: BOOST_PRICE_USD }
        : { type: 'sol', recipient: ECOSYSTEM_WALLET, amountUsd: BOOST_PRICE_USD };

    pendingBoosts.set(requestId, { address, paymentType, requestedAt: Date.now() });
    res.json({ requestId, paymentInstructions: payment, boostPrice: BOOST_PRICE_USD });
});

app.post('/api/boost/confirm', (req, res) => {
    const { requestId, txSignature } = req.body;
    if (!requestId || !txSignature) return res.status(400).json({ error: 'Request ID and tx signature required' });

    const pending = pendingBoosts.get(requestId);
    if (!pending) return res.status(404).json({ error: 'Boost request not found' });

    const existing = boosts.get(pending.address) || {
        address: pending.address, boostCount: 0, totalSolPaid: 0, totalFarnsBurned: 0,
        collectiveApproved: false, extendedInfo: false, lastBoostAt: null,
    };
    existing.boostCount++;
    existing.lastBoostAt = Date.now();
    if (pending.paymentType === 'farns') existing.totalFarnsBurned += BOOST_PRICE_USD;
    else existing.totalSolPaid += BOOST_PRICE_USD;

    boosts.set(pending.address, existing);
    pendingBoosts.delete(requestId);

    // Update token in cache
    const token = tokenCache.get(pending.address);
    if (token) { token.boostAmount += BOOST_PRICE_USD; token.trendScore += BOOST_PRICE_USD * 5; }

    res.json({ success: true, boost: existing });
});

app.post('/api/boost/submit-for-approval', async (req, res) => {
    const { address } = req.body;
    if (!address) return res.status(400).json({ error: 'Token address required' });

    const analysis = await safeFetch(`${FARNSWORTH_API}/api/tools/deep-analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ token_address: address, checks: ['bundles', 'holder_distribution', 'creator_history', 'liquidity_locks'] }),
    });

    if (!analysis) return res.json({ status: 'pending', message: 'Collective analysis in progress.' });

    const approved = (analysis.score || 0) >= 60 && (analysis.rug_probability || 1) < 0.3 && !analysis.bundle_detected;
    if (approved) {
        const existing = boosts.get(address) || { address, boostCount: 0, totalSolPaid: 0, totalFarnsBurned: 0, collectiveApproved: false, extendedInfo: false, lastBoostAt: null };
        existing.collectiveApproved = true;
        boosts.set(address, existing);
    }
    res.json({ status: approved ? 'approved' : 'rejected', analysis: { score: analysis.score, rugProb: analysis.rug_probability, bundled: analysis.bundle_detected } });
});

app.post('/api/extended-info/purchase', (req, res) => {
    const { address, paymentType = 'sol' } = req.body;
    if (!address) return res.status(400).json({ error: 'Token address required' });

    const requestId = `ext_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
    const payment = paymentType === 'farns'
        ? { type: 'farns_burn', tokenMint: FARNS_TOKEN, burnAddress: '1nc1nerator11111111111111111111111111111111', amountUsd: EXTENDED_INFO_PRICE_USD }
        : { type: 'sol', recipient: ECOSYSTEM_WALLET, amountUsd: EXTENDED_INFO_PRICE_USD };

    res.json({ requestId, paymentInstructions: payment, price: EXTENDED_INFO_PRICE_USD,
        features: ['Custom description', 'Social links', 'Team info', 'Roadmap', 'Header image', 'Search priority'] });
});

// ============================================================
// WEBSOCKET
// ============================================================
const wsClients = new Set();

wss.on('connection', (ws) => {
    wsClients.add(ws);
    ws.isAlive = true;
    ws.on('pong', () => { ws.isAlive = true; });
    ws.on('message', (msg) => {
        try {
            const data = JSON.parse(msg);
            if (data.type === 'subscribe') { ws.subs = ws.subs || new Set(); ws.subs.add(data.token); }
            if (data.type === 'unsubscribe') ws.subs?.delete(data.token);
        } catch {}
    });
    ws.on('close', () => wsClients.delete(ws));
    ws.send(JSON.stringify({ type: 'connected', tokens: tokenCache.size }));
});

function broadcastAll(data) {
    const msg = JSON.stringify(data);
    for (const ws of wsClients) { if (ws.readyState === 1) ws.send(msg); }
}

// Heartbeat
setInterval(() => {
    for (const ws of wsClients) {
        if (!ws.isAlive) { ws.terminate(); wsClients.delete(ws); continue; }
        ws.isAlive = false; ws.ping();
    }
}, 30000);

// ============================================================
// SPA ROUTING
// ============================================================
app.get('/token/:address', (req, res) => res.sendFile(path.join(__dirname, 'public', 'index.html')));
app.get('/', (req, res) => res.sendFile(path.join(__dirname, 'public', 'index.html')));

// ============================================================
// START
// ============================================================
server.listen(PORT, () => {
    console.log(`\n  DEXAI v2.0 — Farnsworth Collective DEX Screener`);
    console.log(`  Port: ${PORT} | Ecosystem: ${ECOSYSTEM_WALLET.slice(0, 8)}... | FARNS: ${FARNS_TOKEN.slice(0, 8)}...\n`);
    updateTokenCache();
    setInterval(updateTokenCache, FETCH_INTERVAL);
});

module.exports = { app, server, broadcastAll };
