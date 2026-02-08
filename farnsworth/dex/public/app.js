/**
 * DEXAI - Farnsworth Collective AI-Powered DEX Screener
 * Complete Frontend Application
 */

/* ============================================
   API BASE & CONSTANTS
   ============================================ */

const API = window.location.pathname.startsWith('/DEXAI') ? '/DEXAI' : (window.location.pathname.startsWith('/dex') ? '/dex' : '');

const PLACEHOLDER = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 40 40'%3E%3Ccircle cx='20' cy='20' r='20' fill='%231a1a28'/%3E%3Ctext x='20' y='25' text-anchor='middle' fill='%234a5568' font-size='16'%3E%3F%3C/text%3E%3C/svg%3E";

/* ============================================
   STATE
   ============================================ */

let currentSort = 'trending';
let currentOffset = 0;
let currentToken = null;
let chartInstance = null;
let allTokens = [];
let ws = null;
let refreshInterval = null;
let lastFetchTime = null;
let chartResizeObserver = null;

/* ============================================
   UTILITY FUNCTIONS
   ============================================ */

function formatPrice(price) {
    if (price === null || price === undefined || isNaN(price)) return '--';
    const p = Number(price);
    if (p === 0) return '$0.00';
    if (p < 0.0000001) return '$' + p.toExponential(2);
    if (p < 0.00001) return '$' + p.toFixed(9);
    if (p < 0.001) return '$' + p.toFixed(7);
    if (p < 0.01) return '$' + p.toFixed(6);
    if (p < 1) return '$' + p.toFixed(4);
    if (p < 1000) return '$' + p.toFixed(2);
    return '$' + p.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function formatNumber(num) {
    if (num === null || num === undefined || isNaN(num)) return '--';
    const n = Number(num);
    if (n === 0) return '$0';
    if (n >= 1e12) return '$' + (n / 1e12).toFixed(2) + 'T';
    if (n >= 1e9) return '$' + (n / 1e9).toFixed(2) + 'B';
    if (n >= 1e6) return '$' + (n / 1e6).toFixed(2) + 'M';
    if (n >= 1e3) return '$' + (n / 1e3).toFixed(1) + 'K';
    return '$' + n.toFixed(2);
}

function formatPercent(pct) {
    if (pct === null || pct === undefined || isNaN(pct)) return '--';
    const p = Number(pct);
    const sign = p >= 0 ? '+' : '';
    return sign + p.toFixed(2) + '%';
}

function formatAddr(addr) {
    if (!addr || addr.length < 10) return addr || '--';
    return addr.slice(0, 6) + '...' + addr.slice(-4);
}

function timeAgo(timestamp) {
    if (!timestamp) return 'never';
    const now = Date.now();
    const ts = typeof timestamp === 'string' ? new Date(timestamp).getTime() : timestamp;
    const diff = Math.max(0, now - ts);
    const seconds = Math.floor(diff / 1000);
    if (seconds < 5) return 'just now';
    if (seconds < 60) return seconds + 's ago';
    const minutes = Math.floor(seconds / 60);
    if (minutes < 60) return minutes + 'm ago';
    const hours = Math.floor(minutes / 60);
    if (hours < 24) return hours + 'h ago';
    const days = Math.floor(hours / 24);
    return days + 'd ago';
}

function debounce(fn, delay) {
    let timer;
    return function (...args) {
        clearTimeout(timer);
        timer = setTimeout(() => fn.apply(this, args), delay);
    };
}

function showToast(msg, type) {
    type = type || 'info';
    const container = document.getElementById('toasts');
    if (!container) return;
    const toast = document.createElement('div');
    toast.className = 'toast toast-' + type;
    toast.textContent = msg;
    container.appendChild(toast);
    requestAnimationFrame(() => { toast.classList.add('show'); });
    setTimeout(() => {
        toast.classList.remove('show');
        toast.classList.add('hide');
        setTimeout(() => { toast.remove(); }, 400);
    }, 4000);
}

function chgClass(val) {
    if (val === null || val === undefined || isNaN(val)) return '';
    return Number(val) >= 0 ? 'pos' : 'neg';
}

function escapeHtml(str) {
    if (!str) return '';
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

/* ============================================
   PARTICLES
   ============================================ */

function initParticles() {
    const canvas = document.getElementById('ambientCanvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let particles = [];
    const PARTICLE_COUNT = 100;

    function resize() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    }

    resize();
    window.addEventListener('resize', resize);

    // Create particles
    for (let i = 0; i < PARTICLE_COUNT; i++) {
        particles.push({
            x: Math.random() * canvas.width,
            y: Math.random() * canvas.height,
            r: Math.random() * 1.5 + 0.5,
            vx: (Math.random() - 0.5) * 0.3,
            vy: (Math.random() - 0.5) * 0.3,
            opacity: Math.random() * 0.15 + 0.05
        });
    }

    function draw() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        for (let i = 0; i < particles.length; i++) {
            const p = particles[i];
            p.x += p.vx;
            p.y += p.vy;

            // Wrap around edges
            if (p.x < -10) p.x = canvas.width + 10;
            if (p.x > canvas.width + 10) p.x = -10;
            if (p.y < -10) p.y = canvas.height + 10;
            if (p.y > canvas.height + 10) p.y = -10;

            ctx.beginPath();
            ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
            ctx.fillStyle = 'rgba(0, 255, 136, ' + p.opacity + ')';
            ctx.fill();
        }
        requestAnimationFrame(draw);
    }

    draw();
}

/* ============================================
   TICKER TAPE
   ============================================ */

function buildTicker(tokens) {
    const ticker = document.getElementById('ticker');
    if (!ticker || !tokens || tokens.length === 0) return;

    const items = tokens.slice(0, 30).map(t => {
        const chg = (t.priceChange && t.priceChange.h24 !== undefined) ? t.priceChange.h24 : null;
        const cls = chgClass(chg);
        const sym = escapeHtml(t.symbol || '???');
        const price = formatPrice(t.price);
        const chgStr = chg !== null ? formatPercent(chg) : '--';
        return '<span class="ticker-item">' +
            '<span class="ti-sym">' + sym + '</span>' +
            '<span class="ti-price">' + price + '</span>' +
            '<span class="ti-chg ' + cls + '">' + chgStr + '</span>' +
            '</span>';
    }).join('');

    // Duplicate content for seamless scrolling loop
    ticker.innerHTML = '<div class="ticker-track">' + items + items + '</div>';
}

/* ============================================
   SKELETON LOADING
   ============================================ */

function showSkeletons(count) {
    count = count || 12;
    const tbody = document.getElementById('tblBody');
    if (!tbody) return;
    let html = '';
    for (let i = 0; i < count; i++) {
        html += '<tr class="skel-row">';
        for (let j = 0; j < 12; j++) {
            html += '<td><div class="skel"></div></td>';
        }
        html += '</tr>';
    }
    tbody.innerHTML = html;
}

/* ============================================
   LOAD TOKENS (LIST VIEW)
   ============================================ */

async function loadTokens(sort, limit, offset) {
    sort = sort || currentSort;
    limit = limit || 100;
    offset = offset || 0;
    currentSort = sort;
    currentOffset = offset;

    // Update nav active state
    const navBtns = document.querySelectorAll('#mainNav .nav-btn');
    navBtns.forEach(btn => {
        btn.classList.toggle('active', btn.getAttribute('data-sort') === sort);
    });

    // Update view title
    const titles = {
        trending: 'Trending',
        volume: 'Top Volume',
        velocity: 'High Velocity',
        'new': 'New Pairs',
        gainers: 'Top Gainers',
        losers: 'Top Losers'
    };
    const viewTitle = document.getElementById('viewTitle');
    if (viewTitle) viewTitle.textContent = titles[sort] || 'Tokens';

    // Show skeletons while loading
    if (offset === 0) {
        showSkeletons(12);
    }

    try {
        const url = API + '/api/tokens?sort=' + encodeURIComponent(sort) +
            '&limit=' + limit + '&offset=' + offset;
        const res = await fetch(url);
        if (!res.ok) throw new Error('HTTP ' + res.status);
        const data = await res.json();

        const tokens = data.tokens || [];
        allTokens = offset === 0 ? tokens : allTokens.concat(tokens);

        renderTable(tokens, offset);

        // Update live count
        const liveCount = document.getElementById('liveCount');
        if (liveCount) liveCount.textContent = (data.total || allTokens.length).toLocaleString();

        // Update ticker on first load
        if (offset === 0 && tokens.length > 0) {
            buildTicker(tokens);
        }

        // Update "updated X ago"
        lastFetchTime = Date.now();
        updateMetaRefresh();

        // Show/hide load more
        const loadMoreEl = document.getElementById('loadMore');
        if (loadMoreEl) {
            const total = data.total || 0;
            loadMoreEl.style.display = (allTokens.length < total) ? 'flex' : 'none';
        }
    } catch (err) {
        console.error('Failed to load tokens:', err);
        const tbody = document.getElementById('tblBody');
        if (tbody && offset === 0) {
            tbody.innerHTML = '<tr><td colspan="12" style="text-align:center;padding:40px;color:#666680;">Failed to load tokens. Retrying...</td></tr>';
        }
        showToast('Failed to load tokens', 'error');
    }
}

function updateMetaRefresh() {
    const el = document.getElementById('metaRefresh');
    if (el && lastFetchTime) {
        el.textContent = 'Updated ' + timeAgo(lastFetchTime);
    }
}

/* ============================================
   RENDER TABLE
   ============================================ */

function renderTable(tokens, offset) {
    offset = offset || 0;
    const tbody = document.getElementById('tblBody');
    if (!tbody) return;

    if ((!tokens || tokens.length === 0) && offset === 0) {
        tbody.innerHTML = '<tr><td colspan="12" style="text-align:center;padding:40px;color:#666680;">No tokens found</td></tr>';
        return;
    }

    const html = tokens.map((t, i) => {
        const rank = offset + i + 1;
        const sym = escapeHtml(t.symbol || '???');
        const name = escapeHtml(t.name || '');
        const addr = t.address || '';
        const imgSrc = t.imageUrl || PLACEHOLDER;

        // Price changes
        const chg5m = t.priceChange ? t.priceChange.m5 : null;
        const chg1h = t.priceChange ? t.priceChange.h1 : null;
        const chg24h = t.priceChange ? t.priceChange.h24 : null;

        // Volume
        const vol24 = t.volume ? t.volume.h24 : null;

        // Txns for mini bar
        const txnH1 = t.txns && t.txns.h1 ? t.txns.h1 : { buys: 0, sells: 0 };
        const totalTxn = (txnH1.buys || 0) + (txnH1.sells || 0);
        const buyPct = totalTxn > 0 ? Math.round(((txnH1.buys || 0) / totalTxn) * 100) : 50;

        // AI badge
        let aiBadge = '<span class="ai-badge">--</span>';
        if (t.aiScore !== null && t.aiScore !== undefined) {
            const sc = Number(t.aiScore);
            const aCls = sc >= 70 ? 'high' : (sc >= 40 ? 'mid' : 'low');
            aiBadge = '<span class="ai-badge ' + aCls + '">' + sc + '</span>';
        }

        return '<tr onclick="viewToken(\'' + addr + '\')">' +
            '<td class="td-rank">' + rank + '</td>' +
            '<td class="td-token"><div class="tok-cell">' +
                '<img src="' + imgSrc + '" alt="' + sym + '" onerror="this.src=\'' + PLACEHOLDER + '\'" class="tok-img">' +
                '<div class="tok-info"><span class="tok-sym">' + sym + '</span><span class="tok-name">' + name + '</span></div>' +
            '</div></td>' +
            '<td class="td-price mono">' + formatPrice(t.price) + '</td>' +
            '<td class="td-chg mono ' + chgClass(chg5m) + '">' + formatPercent(chg5m) + '</td>' +
            '<td class="td-chg mono ' + chgClass(chg1h) + '">' + formatPercent(chg1h) + '</td>' +
            '<td class="td-chg mono ' + chgClass(chg24h) + '">' + formatPercent(chg24h) + '</td>' +
            '<td class="td-vol mono">' + formatNumber(vol24) + '</td>' +
            '<td class="td-liq mono">' + formatNumber(t.liquidity) + '</td>' +
            '<td class="td-mcap mono">' + formatNumber(t.marketCap || t.fdv) + '</td>' +
            '<td class="td-txn"><div class="txn-mini"><div class="txn-mini-buy" style="width:' + buyPct + '%"></div></div></td>' +
            '<td class="td-ai">' + aiBadge + '</td>' +
            '<td class="td-act"><button class="view-btn" onclick="event.stopPropagation();viewToken(\'' + addr + '\')">View</button></td>' +
            '</tr>';
    }).join('');

    if (offset === 0) {
        tbody.innerHTML = html;
    } else {
        tbody.insertAdjacentHTML('beforeend', html);
    }
}

/* ============================================
   LOAD MORE
   ============================================ */

function loadMore() {
    currentOffset += 100;
    loadTokens(currentSort, 100, currentOffset);
}

/* ============================================
   VIEW TOKEN (DETAIL VIEW)
   ============================================ */

async function viewToken(address) {
    if (!address) return;

    // Switch views
    const listView = document.getElementById('listView');
    const detailView = document.getElementById('detailView');
    if (listView) listView.style.display = 'none';
    if (detailView) detailView.classList.remove('hidden');

    // Clear previous chart
    destroyChart();

    // Reset panels to loading state
    resetDetailPanels();

    // Push state for browser back
    try {
        const path = API + '/token/' + address;
        window.history.pushState({ address: address }, '', path);
    } catch (e) { /* ignore */ }

    try {
        const res = await fetch(API + '/api/token/' + encodeURIComponent(address));
        if (!res.ok) throw new Error('HTTP ' + res.status);
        const data = await res.json();
        const token = data.token;
        if (!token) throw new Error('Token not found');

        currentToken = token;
        renderDetail(token);

        // Load supplementary data in parallel
        loadChart(address, '15m');
        loadAI(address);
        loadQuantum(address);
    } catch (err) {
        console.error('Failed to load token:', err);
        showToast('Failed to load token details', 'error');
    }
}

function resetDetailPanels() {
    const aiPanel = document.getElementById('aiPanel');
    if (aiPanel) {
        aiPanel.innerHTML = '<div class="ai-loading"><div class="ai-ring"></div><span>Querying the Collective...</span></div>';
    }
    const qPanel = document.getElementById('quantumPanel');
    if (qPanel) {
        qPanel.innerHTML = '<div class="q-loading"><div class="q-spinner"></div><span>Running simulations...</span></div>';
    }
}

/* ============================================
   RENDER DETAIL
   ============================================ */

function renderDetail(token) {
    if (!token) return;

    // Image
    const dImg = document.getElementById('dImg');
    if (dImg) {
        dImg.src = token.imageUrl || PLACEHOLDER;
        dImg.onerror = function () { this.src = PLACEHOLDER; };
    }

    // Symbol, name, address
    setText('dSym', token.symbol || '???');
    setText('dName', token.name || '');
    setText('dAddr', token.address || '');

    // DexScreener link
    const dDexLink = document.getElementById('dDexLink');
    if (dDexLink) {
        dDexLink.href = token.url || ('https://dexscreener.com/solana/' + (token.address || ''));
    }

    // Price
    setText('dPrice', formatPrice(token.price));

    // Main change (24h as default)
    const chg24 = token.priceChange ? token.priceChange.h24 : null;
    const dChg = document.getElementById('dChg');
    if (dChg) {
        dChg.textContent = formatPercent(chg24);
        dChg.className = 'price-chg ' + chgClass(chg24);
    }

    // Per-timeframe changes
    setChg('dChg5m', token.priceChange ? token.priceChange.m5 : null);
    setChg('dChg1h', token.priceChange ? token.priceChange.h1 : null);
    setChg('dChg6h', token.priceChange ? token.priceChange.h6 : null);
    setChg('dChg24h', token.priceChange ? token.priceChange.h24 : null);

    // Stats
    setText('dVol', formatNumber(token.volume ? token.volume.h24 : null));
    setText('dLiq', formatNumber(token.liquidity));
    setText('dFdv', formatNumber(token.fdv));
    setText('dMcap', formatNumber(token.marketCap));

    // Buy/Sell pressure (using h1 data)
    const txH1 = token.txns && token.txns.h1 ? token.txns.h1 : { buys: 0, sells: 0 };
    const totalTx = (txH1.buys || 0) + (txH1.sells || 0);
    const buyPct = totalTx > 0 ? ((txH1.buys || 0) / totalTx * 100) : 50;

    const pressureFill = document.getElementById('pressureFill');
    if (pressureFill) pressureFill.style.width = buyPct.toFixed(1) + '%';

    setText('dBuys', (txH1.buys || 0) + ' Buys');
    setText('dSells', (txH1.sells || 0) + ' Sells');

    // Transaction grid per timeframe
    setTxn('5m', token.txns ? token.txns.m5 : null);
    setTxn('1h', token.txns ? token.txns.h1 : null);
    setTxn('6h', token.txns ? token.txns.h6 : null);
    setTxn('24h', token.txns ? token.txns.h24 : null);

    // Links panel
    renderLinksPanel(token);

    // Pair info panel
    renderPairPanel(token);
}

function setText(id, text) {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
}

function setChg(id, val) {
    const el = document.getElementById(id);
    if (!el) return;
    el.textContent = formatPercent(val);
    el.className = 'ps-v ' + chgClass(val);
}

function setTxn(tf, data) {
    const buys = data ? (data.buys || 0) : 0;
    const sells = data ? (data.sells || 0) : 0;
    setText('txB' + tf, buys.toLocaleString());
    setText('txS' + tf, sells.toLocaleString());
}

function renderLinksPanel(token) {
    const panel = document.getElementById('linksPanel');
    if (!panel) return;

    let html = '';

    if (token.websites && token.websites.length > 0) {
        token.websites.forEach(w => {
            const url = typeof w === 'string' ? w : (w.url || w.label || '');
            if (url) {
                html += '<a href="' + escapeHtml(url) + '" target="_blank" rel="noopener" class="link-item">Website</a>';
            }
        });
    }

    if (token.socials && token.socials.length > 0) {
        token.socials.forEach(s => {
            const url = typeof s === 'string' ? s : (s.url || '');
            const label = (s.type || s.platform || 'Link');
            if (url) {
                html += '<a href="' + escapeHtml(url) + '" target="_blank" rel="noopener" class="link-item">' + escapeHtml(label) + '</a>';
            }
        });
    }

    if (!html) {
        html = '<span class="no-data">No verified links</span>';
    }

    panel.innerHTML = html;
}

function renderPairPanel(token) {
    const panel = document.getElementById('pairPanel');
    if (!panel) return;

    let html = '';

    if (token.pairAddress) {
        html += '<div class="pair-row"><span class="pair-label">Pair</span><span class="pair-val mono">' + formatAddr(token.pairAddress) + '</span></div>';
    }
    if (token.dexId) {
        html += '<div class="pair-row"><span class="pair-label">DEX</span><span class="pair-val">' + escapeHtml(token.dexId) + '</span></div>';
    }
    if (token.chainId) {
        html += '<div class="pair-row"><span class="pair-label">Chain</span><span class="pair-val">' + escapeHtml(token.chainId) + '</span></div>';
    }
    if (token.pairCreatedAt) {
        html += '<div class="pair-row"><span class="pair-label">Created</span><span class="pair-val">' + timeAgo(token.pairCreatedAt) + '</span></div>';
    }
    if (token.priceNative) {
        html += '<div class="pair-row"><span class="pair-label">Native Price</span><span class="pair-val mono">' + token.priceNative + '</span></div>';
    }

    if (!html) {
        html = '<span class="no-data">--</span>';
    }

    panel.innerHTML = html;
}

/* ============================================
   CHART (Lightweight Charts)
   ============================================ */

function destroyChart() {
    if (chartResizeObserver) {
        chartResizeObserver.disconnect();
        chartResizeObserver = null;
    }
    if (chartInstance) {
        try { chartInstance.remove(); } catch (e) { /* ignore */ }
        chartInstance = null;
    }
    const container = document.getElementById('chartContainer');
    if (container) container.innerHTML = '';
}

async function loadChart(address, timeframe) {
    timeframe = timeframe || '15m';
    const container = document.getElementById('chartContainer');
    if (!container) return;

    // Update active timeframe tab
    const tfBtns = document.querySelectorAll('#tfTabs .tf');
    tfBtns.forEach(btn => {
        btn.classList.toggle('active', btn.getAttribute('data-tf') === timeframe);
    });

    // Destroy previous chart
    destroyChart();

    // Show loading state
    container.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:450px;color:#666680;">Loading chart...</div>';

    try {
        const res = await fetch(API + '/api/chart/' + encodeURIComponent(address) + '?timeframe=' + timeframe);
        if (!res.ok) throw new Error('HTTP ' + res.status);
        const data = await res.json();
        const candles = data.candles || [];

        if (candles.length === 0) {
            container.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:450px;color:#666680;">No chart data available</div>';
            return;
        }

        // Clear container
        container.innerHTML = '';

        // Check that LightweightCharts is available
        if (typeof LightweightCharts === 'undefined') {
            container.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:450px;color:#666680;">Chart library not loaded</div>';
            return;
        }

        // Create chart
        const chart = LightweightCharts.createChart(container, {
            width: container.clientWidth,
            height: 450,
            layout: {
                background: { type: 'solid', color: '#0a0a14' },
                textColor: '#8892a4'
            },
            grid: {
                vertLines: { color: 'rgba(255,255,255,0.03)' },
                horzLines: { color: 'rgba(255,255,255,0.03)' }
            },
            crosshair: { mode: 0 },
            rightPriceScale: { borderColor: 'rgba(255,255,255,0.06)' },
            timeScale: {
                borderColor: 'rgba(255,255,255,0.06)',
                timeVisible: true,
                secondsVisible: false
            }
        });

        chartInstance = chart;

        // Candlestick series
        const candleSeries = chart.addCandlestickSeries({
            upColor: '#00ff88',
            downColor: '#ff3366',
            borderUpColor: '#00ff88',
            borderDownColor: '#ff3366',
            wickUpColor: '#00ff88',
            wickDownColor: '#ff3366'
        });

        // Prepare candle data (ensure time is ascending and unique)
        const candleData = candles
            .map(c => ({
                time: typeof c.time === 'number' ? c.time : Math.floor(new Date(c.time).getTime() / 1000),
                open: Number(c.open),
                high: Number(c.high),
                low: Number(c.low),
                close: Number(c.close)
            }))
            .sort((a, b) => a.time - b.time);

        // Deduplicate by time
        const dedupedCandles = [];
        const seenTimes = new Set();
        for (const c of candleData) {
            if (!seenTimes.has(c.time)) {
                seenTimes.add(c.time);
                dedupedCandles.push(c);
            }
        }

        candleSeries.setData(dedupedCandles);

        // Volume histogram series
        const volumeSeries = chart.addHistogramSeries({
            priceFormat: { type: 'volume' },
            priceScaleId: 'volume'
        });

        chart.priceScale('volume').applyOptions({
            scaleMargins: { top: 0.8, bottom: 0 }
        });

        const volumeData = candles
            .map(c => ({
                time: typeof c.time === 'number' ? c.time : Math.floor(new Date(c.time).getTime() / 1000),
                value: Number(c.volume || 0),
                color: Number(c.close) >= Number(c.open) ? 'rgba(0,255,136,0.3)' : 'rgba(255,51,102,0.3)'
            }))
            .sort((a, b) => a.time - b.time);

        // Deduplicate volume data by time
        const dedupedVolume = [];
        const seenVolTimes = new Set();
        for (const v of volumeData) {
            if (!seenVolTimes.has(v.time)) {
                seenVolTimes.add(v.time);
                dedupedVolume.push(v);
            }
        }

        volumeSeries.setData(dedupedVolume);

        // Fit content
        chart.timeScale().fitContent();

        // Resize observer
        chartResizeObserver = new ResizeObserver(entries => {
            for (const entry of entries) {
                const { width, height } = entry.contentRect;
                chart.applyOptions({ width: width, height: Math.max(height, 300) });
            }
        });
        chartResizeObserver.observe(container);

    } catch (err) {
        console.error('Failed to load chart:', err);
        container.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:450px;color:#666680;">Chart unavailable</div>';
    }
}

/* ============================================
   AI COLLECTIVE ANALYSIS
   ============================================ */

async function loadAI(address) {
    const panel = document.getElementById('aiPanel');
    if (!panel) return;

    panel.innerHTML = '<div class="ai-loading"><div class="ai-ring"></div><span>Querying the Collective...</span></div>';

    try {
        const res = await fetch(API + '/api/ai/score/' + encodeURIComponent(address));
        if (!res.ok) throw new Error('HTTP ' + res.status);
        const data = await res.json();

        if (data.score === null || data.score === undefined) {
            panel.innerHTML = '<div class="ai-unavailable"><span class="no-data">Collective analysis not yet available for this token.</span></div>';
            return;
        }

        const score = Number(data.score);
        const scoreCls = score >= 70 ? 'high' : (score >= 40 ? 'mid' : 'low');
        const verdict = data.verdict || 'ANALYZING';

        let html = '<div class="ai-result">';
        html += '<div class="ai-score-row">';
        html += '<span class="ai-score-big ' + scoreCls + '">' + score + '</span>';
        html += '<span class="ai-verdict ' + scoreCls + '">' + escapeHtml(verdict) + '</span>';
        html += '</div>';

        // Detail rows
        html += '<div class="ai-details">';

        if (data.rugProbability !== null && data.rugProbability !== undefined) {
            const rugPct = (Number(data.rugProbability) * 100).toFixed(1);
            const rugCls = Number(data.rugProbability) < 0.3 ? 'pos' : (Number(data.rugProbability) < 0.6 ? 'warn' : 'neg');
            html += '<div class="ai-detail-row"><span class="ai-dl">Rug Probability</span><span class="ai-dv ' + rugCls + '">' + rugPct + '%</span></div>';
        }

        if (data.cabalScore !== null && data.cabalScore !== undefined) {
            html += '<div class="ai-detail-row"><span class="ai-dl">Cabal Score</span><span class="ai-dv">' + Number(data.cabalScore).toFixed(0) + '/100</span></div>';
        }

        if (data.bundleDetected !== null && data.bundleDetected !== undefined) {
            const bundleTxt = data.bundleDetected ? 'DETECTED' : 'None';
            const bundleCls = data.bundleDetected ? 'neg' : 'pos';
            html += '<div class="ai-detail-row"><span class="ai-dl">Bundle</span><span class="ai-dv ' + bundleCls + '">' + bundleTxt + '</span></div>';
        }

        if (data.swarmSentiment) {
            html += '<div class="ai-detail-row"><span class="ai-dl">Swarm Sentiment</span><span class="ai-dv">' + escapeHtml(data.swarmSentiment) + '</span></div>';
        }

        if (data.agents && data.agents.length > 0) {
            html += '<div class="ai-agents"><span class="ai-dl">Agents</span><div class="ai-agent-list">';
            data.agents.forEach(a => {
                const agentName = typeof a === 'string' ? a : (a.name || a.id || 'Agent');
                html += '<span class="ai-agent-tag">' + escapeHtml(agentName) + '</span>';
            });
            html += '</div></div>';
        }

        html += '</div></div>';
        panel.innerHTML = html;

    } catch (err) {
        console.error('Failed to load AI analysis:', err);
        panel.innerHTML = '<div class="ai-unavailable"><span class="no-data">Collective analysis unavailable.</span></div>';
    }
}

/* ============================================
   QUANTUM SIMULATION
   ============================================ */

async function loadQuantum(address) {
    const panel = document.getElementById('quantumPanel');
    if (!panel) return;

    panel.innerHTML = '<div class="q-loading"><div class="q-spinner"></div><span>Running simulations...</span></div>';

    try {
        const res = await fetch(API + '/api/quantum/' + encodeURIComponent(address));
        if (!res.ok) throw new Error('HTTP ' + res.status);
        const data = await res.json();

        if (!data.available) {
            panel.innerHTML = '<div class="q-unavailable"><span class="no-data">Quantum simulation not available for this token.</span></div>';
            return;
        }

        let html = '<div class="q-result">';

        if (data.rugProbability !== null && data.rugProbability !== undefined) {
            const rugPct = (Number(data.rugProbability) * 100).toFixed(1);
            const rugCls = Number(data.rugProbability) < 0.3 ? 'pos' : (Number(data.rugProbability) < 0.6 ? 'warn' : 'neg');
            html += '<div class="q-row"><span class="q-label">Rug Risk</span><span class="q-val q-big ' + rugCls + '">' + rugPct + '%</span></div>';
        }

        if (data.confidence !== null && data.confidence !== undefined) {
            html += '<div class="q-row"><span class="q-label">Confidence</span><span class="q-val">' + (Number(data.confidence) * 100).toFixed(0) + '%</span></div>';
        }

        if (data.simulations) {
            html += '<div class="q-row"><span class="q-label">Simulations</span><span class="q-val">' + Number(data.simulations).toLocaleString() + '</span></div>';
        }

        if (data.priceTargets) {
            html += '<div class="q-targets"><span class="q-label">Price Targets</span>';
            if (data.priceTargets.low !== undefined) {
                html += '<div class="q-target"><span>Bear</span><span class="neg">' + formatPrice(data.priceTargets.low) + '</span></div>';
            }
            if (data.priceTargets.mid !== undefined) {
                html += '<div class="q-target"><span>Base</span><span>' + formatPrice(data.priceTargets.mid) + '</span></div>';
            }
            if (data.priceTargets.high !== undefined) {
                html += '<div class="q-target"><span>Bull</span><span class="pos">' + formatPrice(data.priceTargets.high) + '</span></div>';
            }
            html += '</div>';
        }

        html += '</div>';
        panel.innerHTML = html;

    } catch (err) {
        console.error('Failed to load quantum simulation:', err);
        panel.innerHTML = '<div class="q-unavailable"><span class="no-data">Quantum simulation unavailable.</span></div>';
    }
}

/* ============================================
   SEARCH
   ============================================ */

const handleSearch = debounce(async function (query) {
    const drop = document.getElementById('searchDrop');
    if (!drop) return;

    if (!query || query.length < 2) {
        drop.classList.add('hidden');
        drop.innerHTML = '';
        return;
    }

    try {
        const res = await fetch(API + '/api/search?q=' + encodeURIComponent(query));
        if (!res.ok) throw new Error('HTTP ' + res.status);
        const data = await res.json();
        const tokens = data.tokens || [];

        if (tokens.length === 0) {
            drop.innerHTML = '<div class="search-item search-empty">No results found</div>';
            drop.classList.remove('hidden');
            return;
        }

        drop.innerHTML = tokens.slice(0, 10).map(t => {
            const sym = escapeHtml(t.symbol || '???');
            const name = escapeHtml(t.name || '');
            const addr = t.address || '';
            const imgSrc = t.imageUrl || PLACEHOLDER;
            const chg = t.priceChange ? t.priceChange.h24 : null;

            return '<div class="search-item" onclick="viewToken(\'' + addr + '\')">' +
                '<img src="' + imgSrc + '" alt="' + sym + '" onerror="this.src=\'' + PLACEHOLDER + '\'" class="search-img">' +
                '<div class="search-info">' +
                    '<span class="search-sym">' + sym + '</span>' +
                    '<span class="search-name">' + name + '</span>' +
                '</div>' +
                '<div class="search-right">' +
                    '<span class="search-price">' + formatPrice(t.price) + '</span>' +
                    '<span class="search-chg ' + chgClass(chg) + '">' + formatPercent(chg) + '</span>' +
                '</div>' +
            '</div>';
        }).join('');

        drop.classList.remove('hidden');
    } catch (err) {
        console.error('Search failed:', err);
        drop.innerHTML = '<div class="search-item search-empty">Search failed</div>';
        drop.classList.remove('hidden');
    }
}, 300);

/* ============================================
   NAVIGATION & VIEW MANAGEMENT
   ============================================ */

function goBack() {
    destroyChart();
    currentToken = null;

    const listView = document.getElementById('listView');
    const detailView = document.getElementById('detailView');
    if (listView) listView.style.display = '';
    if (detailView) detailView.classList.add('hidden');

    // Update URL
    try {
        const path = API + '/';
        window.history.pushState({}, '', path);
    } catch (e) { /* ignore */ }
}

function goHome(event) {
    if (event) event.preventDefault();
    goBack();
    loadTokens('trending', 100, 0);
}

/* ============================================
   CLIPBOARD & BOOST
   ============================================ */

function copyAddr() {
    if (!currentToken || !currentToken.address) {
        showToast('No address to copy', 'error');
        return;
    }
    if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(currentToken.address).then(() => {
            showToast('Address copied to clipboard', 'success');
        }).catch(() => {
            fallbackCopy(currentToken.address);
        });
    } else {
        fallbackCopy(currentToken.address);
    }
}

function fallbackCopy(text) {
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.style.position = 'fixed';
    ta.style.left = '-9999px';
    document.body.appendChild(ta);
    ta.select();
    try {
        document.execCommand('copy');
        showToast('Address copied to clipboard', 'success');
    } catch (e) {
        showToast('Failed to copy address', 'error');
    }
    document.body.removeChild(ta);
}

function openBoost() {
    const modal = document.getElementById('boostModal');
    if (modal) modal.classList.remove('hidden');
}

function closeBoost() {
    const modal = document.getElementById('boostModal');
    if (modal) modal.classList.add('hidden');
}

function selectBoostType(el) {
    if (!el) return;
    const opts = document.querySelectorAll('.boost-opt');
    opts.forEach(o => o.classList.remove('selected'));
    el.classList.add('selected');
}

async function confirmBoost() {
    if (!currentToken || !currentToken.address) {
        showToast('No token selected', 'error');
        return;
    }

    // Determine selected payment type
    const selected = document.querySelector('.boost-opt.selected');
    const paymentType = selected ? (selected.getAttribute('data-type') || 'sol') : 'sol';

    try {
        const res = await fetch(API + '/api/boost/request', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                address: currentToken.address,
                paymentType: paymentType
            })
        });

        if (!res.ok) throw new Error('HTTP ' + res.status);
        const data = await res.json();

        if (data.paymentInstructions) {
            showToast('Boost requested! Check payment instructions.', 'success');
        } else if (data.error) {
            showToast(data.error, 'error');
        } else {
            showToast('Boost request submitted', 'info');
        }

        closeBoost();
    } catch (err) {
        console.error('Boost request failed:', err);
        showToast('Boost request failed', 'error');
    }
}

/* ============================================
   WEBSOCKET
   ============================================ */

function connectWebSocket() {
    try {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsPath = API ? API + '/ws' : '/ws';
        const url = protocol + '//' + window.location.host + wsPath;

        ws = new WebSocket(url);

        ws.onopen = function () {
            console.log('[DEXAI] WebSocket connected');
        };

        ws.onmessage = function (event) {
            try {
                const msg = JSON.parse(event.data);
                handleWsMessage(msg);
            } catch (e) {
                // Ignore malformed messages
            }
        };

        ws.onclose = function () {
            console.log('[DEXAI] WebSocket disconnected, reconnecting in 3s...');
            ws = null;
            setTimeout(connectWebSocket, 3000);
        };

        ws.onerror = function (err) {
            console.error('[DEXAI] WebSocket error:', err);
        };

    } catch (err) {
        console.error('[DEXAI] WebSocket connection failed:', err);
        setTimeout(connectWebSocket, 3000);
    }
}

function handleWsMessage(msg) {
    if (!msg || !msg.type) return;

    switch (msg.type) {
        case 'update':
            // If on list view, refresh data
            const listView = document.getElementById('listView');
            if (listView && listView.style.display !== 'none') {
                loadTokens(currentSort, 100, 0);
            }
            break;

        case 'price':
            // Real-time price update for current token
            if (currentToken && msg.address === currentToken.address && msg.price) {
                setText('dPrice', formatPrice(msg.price));
                if (msg.priceChange) {
                    const dChg = document.getElementById('dChg');
                    if (dChg) {
                        dChg.textContent = formatPercent(msg.priceChange);
                        dChg.className = 'price-chg ' + chgClass(msg.priceChange);
                    }
                }
            }
            break;

        case 'ticker':
            if (msg.tokens) {
                buildTicker(msg.tokens);
            }
            break;

        default:
            break;
    }
}

/* ============================================
   INITIALIZATION
   ============================================ */

document.addEventListener('DOMContentLoaded', function () {
    // 1. Init particles
    initParticles();

    // 2. Setup nav button clicks
    const navBtns = document.querySelectorAll('#mainNav .nav-btn');
    navBtns.forEach(function (btn) {
        btn.addEventListener('click', function () {
            const sort = btn.getAttribute('data-sort');
            if (sort) {
                // Ensure we're showing list view
                goBack();
                loadTokens(sort, 100, 0);
            }
        });
    });

    // 3. Setup search
    const searchInput = document.getElementById('searchInput');
    if (searchInput) {
        searchInput.addEventListener('input', function () {
            handleSearch(searchInput.value.trim());
        });

        searchInput.addEventListener('focus', function () {
            if (searchInput.value.trim().length >= 2) {
                const drop = document.getElementById('searchDrop');
                if (drop && drop.innerHTML) drop.classList.remove('hidden');
            }
        });
    }

    // Close search dropdown on outside click
    document.addEventListener('click', function (e) {
        const searchBox = e.target.closest('.search-box');
        const drop = document.getElementById('searchDrop');
        if (!searchBox && drop) {
            drop.classList.add('hidden');
        }
    });

    // "/" key focuses search input
    document.addEventListener('keydown', function (e) {
        if (e.key === '/' && document.activeElement !== searchInput) {
            e.preventDefault();
            if (searchInput) searchInput.focus();
        }
        // Escape closes search dropdown
        if (e.key === 'Escape') {
            const drop = document.getElementById('searchDrop');
            if (drop) drop.classList.add('hidden');
            if (searchInput) searchInput.blur();
        }
    });

    // 4. Setup timeframe tab clicks
    const tfTabs = document.getElementById('tfTabs');
    if (tfTabs) {
        tfTabs.addEventListener('click', function (e) {
            const btn = e.target.closest('.tf');
            if (!btn) return;
            const tf = btn.getAttribute('data-tf');
            if (tf && currentToken) {
                loadChart(currentToken.address, tf);
            }
        });
    }

    // 5. Load initial tokens
    loadTokens('trending', 100, 0);

    // 6. Connect WebSocket
    connectWebSocket();

    // 7. URL routing: check if path includes /token/
    const path = window.location.pathname;
    const tokenMatch = path.match(/\/token\/([A-Za-z0-9]+)/);
    if (tokenMatch && tokenMatch[1]) {
        viewToken(tokenMatch[1]);
    }

    // 8. Handle browser back/forward
    window.addEventListener('popstate', function (e) {
        if (e.state && e.state.address) {
            viewToken(e.state.address);
        } else {
            goBack();
        }
    });

    // 9. Auto-refresh list every 15 seconds
    refreshInterval = setInterval(function () {
        const listView = document.getElementById('listView');
        if (listView && listView.style.display !== 'none') {
            loadTokens(currentSort, 100, 0);
        }
        updateMetaRefresh();
    }, 15000);

    // 10. Wallet button placeholder
    const walletBtn = document.getElementById('walletBtn');
    if (walletBtn) {
        walletBtn.addEventListener('click', function () {
            showToast('Wallet connection coming soon', 'info');
        });
    }
});
