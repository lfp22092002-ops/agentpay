/* ═══════════════════════════════════════
   AgentPay Mini App — Main JS
   ═══════════════════════════════════════ */

const API_BASE = window.location.hostname === 'localhost'
    ? 'http://localhost:8080'
    : `${window.location.protocol}//${window.location.hostname}:8080`;

const tg = window.Telegram?.WebApp;

// ── State ──────────────────────────────
let authToken = null;
let agents = [];
let selectedAgent = null;

// ── Init ───────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
    // Telegram SDK setup
    if (tg) {
        tg.ready();
        tg.expand();
        tg.enableClosingConfirmation?.();
    }

    setupTabs();
    setupFilters();

    try {
        await authenticate();
        document.getElementById('loading').classList.add('hidden');
        document.getElementById('app').classList.remove('hidden');
        await loadDashboard();
    } catch (err) {
        console.error('Auth failed:', err);
        document.querySelector('#loading p').textContent = 'Authentication failed. Please reopen.';
    }
});

// ── Authentication ─────────────────────
async function authenticate() {
    const initData = tg?.initData || '';
    const user = tg?.initDataUnsafe?.user;

    const res = await fetch(`${API_BASE}/v1/auth/telegram`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ init_data: initData }),
    });

    if (!res.ok) throw new Error(`Auth failed: ${res.status}`);
    const data = await res.json();
    authToken = data.token;

    // Display user info
    const nameEl = document.getElementById('user-name');
    if (user) {
        nameEl.textContent = user.first_name || user.username || '';
    } else if (data.user_name) {
        nameEl.textContent = data.user_name;
    }
}

// ── API Helper ─────────────────────────
async function api(path, opts = {}) {
    const headers = {
        'Content-Type': 'application/json',
        ...(authToken ? { 'Authorization': `Bearer ${authToken}` } : {}),
        ...(opts.headers || {}),
    };

    const res = await fetch(`${API_BASE}${path}`, {
        ...opts,
        headers,
    });

    if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || `API error ${res.status}`);
    }

    return res.json();
}

// ── Tabs ───────────────────────────────
function setupTabs() {
    document.querySelectorAll('.tab').forEach(tab => {
        tab.addEventListener('click', () => {
            // Update tab buttons
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            tab.classList.add('active');

            // Update tab content
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            const target = tab.dataset.tab;
            document.getElementById(`tab-${target}`).classList.add('active');

            // Load data for the tab if agent selected
            if (selectedAgent) {
                if (target === 'transactions') loadTransactions();
                if (target === 'settings') loadSettings();
                if (target === 'card') loadCard();
                if (target === 'wallet') loadWallet();
            }
        });
    });
}

// ── Filters ────────────────────────────
function setupFilters() {
    document.getElementById('filter-type').addEventListener('change', loadTransactions);
    document.getElementById('filter-date').addEventListener('change', loadTransactions);
}

// ── Dashboard ──────────────────────────
async function loadDashboard() {
    try {
        const data = await api('/v1/miniapp/agents');
        agents = data.agents || [];

        const container = document.getElementById('agents-list');
        if (agents.length === 0) {
            container.innerHTML = '<div class="empty-state">No agents yet. Create one via the bot!</div>';
            return;
        }

        container.innerHTML = agents.map(agent => {
            const spendPercent = agent.daily_limit_usd > 0
                ? Math.min(100, (agent.daily_spent_usd / agent.daily_limit_usd) * 100)
                : 0;
            const barClass = spendPercent > 80 ? 'danger' : spendPercent > 50 ? 'warning' : '';

            return `
                <div class="agent-card ${selectedAgent?.id === agent.id ? 'selected' : ''}"
                     data-agent-id="${agent.id}" onclick="selectAgent('${agent.id}')">
                    <div class="agent-card-header">
                        <span class="agent-name">${esc(agent.name)}</span>
                        <span class="agent-status ${agent.is_active ? 'active' : 'inactive'}">
                            ${agent.is_active ? 'Active' : 'Inactive'}
                        </span>
                    </div>
                    <div class="agent-stats">
                        <div class="stat">
                            <span class="stat-label">Balance</span>
                            <span class="stat-value balance">$${fmt(agent.balance_usd)}</span>
                        </div>
                        <div class="stat">
                            <span class="stat-label">Daily Limit</span>
                            <span class="stat-value">$${fmt(agent.daily_limit_usd)}</span>
                        </div>
                        <div class="stat">
                            <span class="stat-label">Tx Limit</span>
                            <span class="stat-value">$${fmt(agent.tx_limit_usd)}</span>
                        </div>
                    </div>
                    <div class="spend-bar">
                        <div class="spend-bar-track">
                            <div class="spend-bar-fill ${barClass}" style="width: ${spendPercent}%"></div>
                        </div>
                        <div class="spend-bar-label">
                            <span>$${fmt(agent.daily_spent_usd)} spent today</span>
                            <span>$${fmt(agent.daily_limit_usd)} limit</span>
                        </div>
                    </div>
                </div>
            `;
        }).join('');
    } catch (err) {
        document.getElementById('agents-list').innerHTML =
            `<div class="empty-state">Error: ${esc(err.message)}</div>`;
    }
}

function selectAgent(agentId) {
    selectedAgent = agents.find(a => a.id === agentId) || null;
    // Re-render dashboard to show selection
    document.querySelectorAll('.agent-card').forEach(card => {
        card.classList.toggle('selected', card.dataset.agentId === agentId);
    });
    // Haptic feedback
    tg?.HapticFeedback?.selectionChanged?.();
    toast(`Selected: ${selectedAgent?.name}`);
}

// ── Transactions ───────────────────────
async function loadTransactions() {
    const container = document.getElementById('transactions-list');
    if (!selectedAgent) {
        container.innerHTML = '<div class="empty-state">Select an agent from the dashboard first</div>';
        return;
    }

    container.innerHTML = '<div class="empty-state">Loading…</div>';

    try {
        const typeFilter = document.getElementById('filter-type').value;
        const dateFilter = document.getElementById('filter-date').value;

        let url = `/v1/miniapp/agents/${selectedAgent.id}/transactions?limit=50`;
        if (typeFilter) url += `&type=${typeFilter}`;
        if (dateFilter) url += `&date=${dateFilter}`;

        const data = await api(url);
        const txs = data.transactions || [];

        if (txs.length === 0) {
            container.innerHTML = '<div class="empty-state">No transactions found</div>';
            return;
        }

        container.innerHTML = txs.map(tx => {
            const amountClass = tx.type === 'spend' ? 'spend' :
                                tx.type === 'deposit' ? 'deposit' :
                                tx.type === 'refund' ? 'refund' : '';
            const prefix = tx.type === 'spend' ? '-' : tx.type === 'deposit' ? '+' : '';
            const time = new Date(tx.created_at).toLocaleString('en-US', {
                month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit'
            });

            return `
                <div class="tx-item">
                    <div class="tx-left">
                        <span class="tx-type">${esc(tx.type)}</span>
                        <span class="tx-desc">${esc(tx.description || '—')}</span>
                        <span class="tx-time">${time}</span>
                    </div>
                    <div class="tx-right">
                        <span class="tx-amount ${amountClass}">${prefix}$${fmt(tx.amount)}</span>
                        <div class="tx-status">${esc(tx.status)}</div>
                    </div>
                </div>
            `;
        }).join('');
    } catch (err) {
        container.innerHTML = `<div class="empty-state">Error: ${esc(err.message)}</div>`;
    }
}

// ── Settings ───────────────────────────
async function loadSettings() {
    const panel = document.getElementById('settings-panel');
    if (!selectedAgent) {
        panel.innerHTML = '<div class="empty-state">Select an agent from the dashboard first</div>';
        return;
    }

    const a = selectedAgent;
    panel.innerHTML = `
        <div class="settings-group">
            <div class="settings-group-title">Spending Limits</div>
            <div class="setting-row">
                <span class="setting-label">Daily Limit (USD)</span>
                <div class="setting-value">
                    <input type="number" id="set-daily-limit" value="${a.daily_limit_usd}" min="0" max="10000" step="1">
                </div>
            </div>
            <div class="setting-row">
                <span class="setting-label">Per-Tx Limit (USD)</span>
                <div class="setting-value">
                    <input type="number" id="set-tx-limit" value="${a.tx_limit_usd}" min="0" max="10000" step="1">
                </div>
            </div>
            <div class="setting-row">
                <span class="setting-label">Auto-Approve (USD)</span>
                <div class="setting-value">
                    <input type="number" id="set-auto-approve" value="${a.auto_approve_usd}" min="0" max="10000" step="0.5">
                </div>
            </div>
        </div>

        <div class="settings-group">
            <div class="settings-group-title">Agent Status</div>
            <div class="setting-row">
                <span class="setting-label">Active</span>
                <label class="toggle">
                    <input type="checkbox" id="set-active" ${a.is_active ? 'checked' : ''}>
                    <span class="toggle-slider"></span>
                </label>
            </div>
        </div>

        <button class="btn btn-primary btn-full" onclick="saveSettings()">Save Changes</button>
    `;
}

async function saveSettings() {
    if (!selectedAgent) return;

    const payload = {
        daily_limit_usd: parseFloat(document.getElementById('set-daily-limit').value),
        tx_limit_usd: parseFloat(document.getElementById('set-tx-limit').value),
        auto_approve_usd: parseFloat(document.getElementById('set-auto-approve').value),
        is_active: document.getElementById('set-active').checked,
    };

    try {
        await api(`/v1/miniapp/agents/${selectedAgent.id}/settings`, {
            method: 'PATCH',
            body: JSON.stringify(payload),
        });

        // Update local state
        Object.assign(selectedAgent, payload);
        tg?.HapticFeedback?.notificationOccurred?.('success');
        toast('Settings saved!');
        await loadDashboard();
    } catch (err) {
        tg?.HapticFeedback?.notificationOccurred?.('error');
        toast(`Error: ${err.message}`, true);
    }
}

// ── Card ───────────────────────────────
async function loadCard() {
    const panel = document.getElementById('card-panel');
    if (!selectedAgent) {
        panel.innerHTML = '<div class="empty-state">Select an agent from the dashboard first</div>';
        return;
    }

    panel.innerHTML = '<div class="empty-state">Loading card…</div>';

    try {
        const data = await api(`/v1/miniapp/agents/${selectedAgent.id}/card`);
        const card = data.card;

        if (!card || !card.last4) {
            panel.innerHTML = '<div class="empty-state">No virtual card assigned to this agent</div>';
            return;
        }

        const stateClass = card.state === 'OPEN' ? 'open' : 'paused';
        const stateLabel = card.state === 'OPEN' ? '● Active' : '⏸ Paused';

        let txHtml = '';
        if (data.transactions && data.transactions.length > 0) {
            txHtml = `
                <div class="card-txns-title">Recent Transactions</div>
                ${data.transactions.map(t => `
                    <div class="tx-item">
                        <div class="tx-left">
                            <span class="tx-type">${esc(t.merchant)}</span>
                            <span class="tx-time">${new Date(t.created).toLocaleDateString()}</span>
                        </div>
                        <div class="tx-right">
                            <span class="tx-amount spend">-$${(t.amount_cents / 100).toFixed(2)}</span>
                            <div class="tx-status">${esc(t.status)}</div>
                        </div>
                    </div>
                `).join('')}
            `;
        }

        panel.innerHTML = `
            <div class="card-state ${stateClass}">${stateLabel}</div>
            <div class="card-visual">
                <div class="card-brand">AgentPay Virtual</div>
                <div class="card-number">•••• •••• •••• ${card.last4}</div>
                <div class="card-bottom">
                    <div>
                        <div class="card-exp-label">Expires</div>
                        <div class="card-exp-value">${card.exp_month}/${card.exp_year}</div>
                    </div>
                    <div>
                        <div class="card-limit-label">Spend Limit</div>
                        <div class="card-limit-value">$${(card.spend_limit_cents / 100).toFixed(2)}</div>
                    </div>
                </div>
            </div>
            ${txHtml}
            <button class="btn ${card.state === 'OPEN' ? 'btn-danger' : 'btn-primary'} btn-full"
                    onclick="toggleCard('${card.state === 'OPEN' ? 'pause' : 'resume'}')">
                ${card.state === 'OPEN' ? '⏸ Pause Card' : '▶ Resume Card'}
            </button>
        `;
    } catch (err) {
        panel.innerHTML = `<div class="empty-state">Error: ${esc(err.message)}</div>`;
    }
}

async function toggleCard(action) {
    if (!selectedAgent) return;
    try {
        await api(`/v1/miniapp/agents/${selectedAgent.id}/card/${action}`, { method: 'POST' });
        tg?.HapticFeedback?.notificationOccurred?.('success');
        toast(`Card ${action}d!`);
        await loadCard();
    } catch (err) {
        toast(`Error: ${err.message}`, true);
    }
}

// ── Wallet ─────────────────────────────
async function loadWallet() {
    const panel = document.getElementById('wallet-panel');
    if (!selectedAgent) {
        panel.innerHTML = '<div class="empty-state">Select an agent from the dashboard first</div>';
        return;
    }

    panel.innerHTML = '<div class="empty-state">Loading wallet…</div>';

    try {
        const data = await api(`/v1/miniapp/agents/${selectedAgent.id}/wallet`);

        if (!data.address) {
            panel.innerHTML = '<div class="empty-state">No on-chain wallet configured for this agent</div>';
            return;
        }

        panel.innerHTML = `
            <div class="wallet-address-box">
                <div class="wallet-address-label">Address</div>
                <div class="wallet-address" onclick="copyAddress('${data.address}')">${data.address}</div>
            </div>
            <div class="wallet-balances">
                <div class="wallet-balance-card">
                    <div class="wallet-token">ETH</div>
                    <div class="wallet-amount">${data.balance_eth || '0.00'}</div>
                    <div class="wallet-network">${data.network || 'Unknown'}</div>
                </div>
                <div class="wallet-balance-card">
                    <div class="wallet-token">USDC</div>
                    <div class="wallet-amount">${data.balance_usdc || '0.00'}</div>
                    <div class="wallet-network">${data.network || 'Unknown'}</div>
                </div>
            </div>
        `;
    } catch (err) {
        panel.innerHTML = `<div class="empty-state">Error: ${esc(err.message)}</div>`;
    }
}

function copyAddress(address) {
    navigator.clipboard.writeText(address).then(() => {
        tg?.HapticFeedback?.notificationOccurred?.('success');
        toast('Address copied!');
    }).catch(() => {
        toast('Copy failed', true);
    });
}

// ── Toast ──────────────────────────────
function toast(msg, isError = false) {
    // Remove existing toast
    document.querySelectorAll('.toast').forEach(t => t.remove());

    const el = document.createElement('div');
    el.className = `toast ${isError ? 'error' : ''}`;
    el.textContent = msg;
    document.body.appendChild(el);

    requestAnimationFrame(() => {
        el.classList.add('show');
        setTimeout(() => {
            el.classList.remove('show');
            setTimeout(() => el.remove(), 300);
        }, 2500);
    });
}

// ── Helpers ────────────────────────────
function esc(str) {
    if (!str) return '';
    const d = document.createElement('div');
    d.textContent = str;
    return d.innerHTML;
}

function fmt(num) {
    return Number(num || 0).toFixed(2);
}
