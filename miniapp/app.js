/* ═══════════════════════════════════════
   AgentPay Mini App — Main JS
   Premium Telegram Web App Experience
   ═══════════════════════════════════════ */

const API_BASE = window.location.hostname === 'localhost'
    ? 'http://localhost:8080'
    : `${window.location.protocol}//${window.location.hostname}:8080`;

const tg = window.Telegram?.WebApp;

// ── State ──────────────────────────────
let authToken = null;
let agents = [];
let selectedAgent = null;
let currentFilter = '';

// ── Init ───────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
    if (tg) {
        tg.ready();
        tg.expand();
        tg.enableClosingConfirmation?.();
        tg.setHeaderColor?.('#0f0f14');
        tg.setBackgroundColor?.('#0f0f14');
    }

    setupTabs();
    setupFilterChips();

    try {
        await authenticate();
        document.getElementById('loading').classList.add('hidden');
        document.getElementById('app').classList.remove('hidden');
        await loadDashboard();
    } catch (err) {
        console.error('Auth failed:', err);
        document.querySelector('.loading-text').textContent = 'Authentication failed';
        document.querySelector('.loading-spinner').style.display = 'none';
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

    const nameEl = document.getElementById('user-name');
    const avatarEl = document.getElementById('user-avatar');
    const firstName = user?.first_name || data.user_name || 'User';
    nameEl.textContent = firstName;
    avatarEl.textContent = firstName.charAt(0).toUpperCase();
}

// ── API Helper ─────────────────────────
async function api(path, opts = {}) {
    const headers = {
        'Content-Type': 'application/json',
        ...(authToken ? { 'Authorization': `Bearer ${authToken}` } : {}),
        ...(opts.headers || {}),
    };

    const res = await fetch(`${API_BASE}${path}`, { ...opts, headers });

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
            switchTab(tab.dataset.tab);
        });
    });
}

function switchTab(tabName) {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelector(`.tab[data-tab="${tabName}"]`)?.classList.add('active');

    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    document.getElementById(`tab-${tabName}`)?.classList.add('active');

    // Show/hide balance overview on dashboard
    const balanceOverview = document.getElementById('balance-overview');
    if (tabName === 'dashboard') {
        balanceOverview.style.display = '';
    } else {
        balanceOverview.style.display = 'none';
    }

    if (selectedAgent) {
        if (tabName === 'transactions') loadTransactions();
        if (tabName === 'settings') loadSettings();
        if (tabName === 'card') loadCard();
        if (tabName === 'wallet') loadWallet();
    }

    tg?.HapticFeedback?.selectionChanged?.();
}

// ── Filter Chips ───────────────────────
function setupFilterChips() {
    document.querySelectorAll('.chip').forEach(chip => {
        chip.addEventListener('click', () => {
            document.querySelectorAll('.chip').forEach(c => c.classList.remove('active'));
            chip.classList.add('active');
            currentFilter = chip.dataset.filter;
            loadTransactions();
            tg?.HapticFeedback?.selectionChanged?.();
        });
    });
}

// ── Dashboard ──────────────────────────
async function loadDashboard() {
    try {
        const data = await api('/v1/miniapp/agents');
        agents = data.agents || [];

        // Update balance overview
        const totalBalance = agents.reduce((sum, a) => sum + (a.balance_usd || 0), 0);
        document.getElementById('total-balance').textContent = `$${fmt(totalBalance)}`;
        document.getElementById('agent-count').textContent = agents.length;

        const container = document.getElementById('agents-list');

        if (agents.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <div class="empty-icon">
                        <svg width="56" height="56" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" opacity="0.3"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></svg>
                    </div>
                    <p class="empty-title">No agents yet</p>
                    <p class="empty-desc">Create your first agent via the bot to get started</p>
                </div>
            `;
            return;
        }

        container.innerHTML = agents.map((agent, i) => {
            const spendPercent = agent.daily_limit_usd > 0
                ? Math.min(100, (agent.daily_spent_usd / agent.daily_limit_usd) * 100)
                : 0;
            const barClass = spendPercent > 80 ? 'danger' : spendPercent > 50 ? 'warning' : '';
            const agentEmoji = ['🤖', '⚡', '🧠', '🔮', '🚀'][i % 5];

            return `
                <div class="agent-card ${selectedAgent?.id === agent.id ? 'selected' : ''}"
                     data-agent-id="${agent.id}" onclick="selectAgent('${agent.id}')"
                     style="animation: fadeIn 0.3s ease ${i * 0.05}s both">
                    <div class="agent-card-header">
                        <div class="agent-name-group">
                            <div class="agent-icon">${agentEmoji}</div>
                            <span class="agent-name">${esc(agent.name)}</span>
                        </div>
                        <span class="agent-status ${agent.is_active ? 'active' : 'inactive'}">
                            ${agent.is_active ? 'Live' : 'Off'}
                        </span>
                    </div>
                    <div class="agent-stats">
                        <div class="stat">
                            <span class="stat-label">Balance</span>
                            <span class="stat-value balance">$${fmt(agent.balance_usd)}</span>
                        </div>
                        <div class="stat">
                            <span class="stat-label">Daily Cap</span>
                            <span class="stat-value">$${fmt(agent.daily_limit_usd)}</span>
                        </div>
                        <div class="stat">
                            <span class="stat-label">Per Tx</span>
                            <span class="stat-value">$${fmt(agent.tx_limit_usd)}</span>
                        </div>
                    </div>
                    <div class="spend-bar">
                        <div class="spend-bar-track">
                            <div class="spend-bar-fill ${barClass}" style="width: ${spendPercent}%"></div>
                        </div>
                        <div class="spend-bar-label">
                            <span>$${fmt(agent.daily_spent_usd)} today</span>
                            <span>${Math.round(spendPercent)}% used</span>
                        </div>
                    </div>
                </div>
            `;
        }).join('');
    } catch (err) {
        document.getElementById('agents-list').innerHTML =
            `<div class="empty-state">
                <p class="empty-title">Connection error</p>
                <p class="empty-desc">${esc(err.message)}</p>
            </div>`;
    }
}

function selectAgent(agentId) {
    selectedAgent = agents.find(a => a.id === agentId) || null;
    document.querySelectorAll('.agent-card').forEach(card => {
        card.classList.toggle('selected', card.dataset.agentId === agentId);
    });
    tg?.HapticFeedback?.impactOccurred?.('light');
    toast(`${selectedAgent?.name} selected`, 'success');
}

// ── Fund Flow ──────────────────────────
function openFundFlow() {
    if (agents.length === 0) {
        toast('Create an agent first via the bot', 'error');
        return;
    }

    const modal = document.getElementById('fund-modal');
    modal.classList.remove('hidden');

    const agentSelect = document.getElementById('fund-agent-select');
    const amountSelect = document.getElementById('fund-amount-select');
    amountSelect.classList.add('hidden');

    agentSelect.innerHTML = agents.map(agent => `
        <div class="fund-option" onclick="selectFundAgent('${agent.id}')">
            <div class="fund-option-icon">🤖</div>
            <div class="fund-option-info">
                <div class="fund-option-name">${esc(agent.name)}</div>
                <div class="fund-option-detail">Balance: $${fmt(agent.balance_usd)}</div>
            </div>
            <span class="fund-option-arrow">›</span>
        </div>
    `).join('');

    tg?.HapticFeedback?.impactOccurred?.('light');
}

const STAR_OPTIONS = [
    { stars: 50, usd: '$0.65' },
    { stars: 200, usd: '$2.60' },
    { stars: 500, usd: '$6.50' },
    { stars: 1000, usd: '$13.00' },
    { stars: 2500, usd: '$32.50' },
];

function selectFundAgent(agentId) {
    const agentSelect = document.getElementById('fund-agent-select');
    const amountSelect = document.getElementById('fund-amount-select');

    agentSelect.classList.add('hidden');
    amountSelect.classList.remove('hidden');

    amountSelect.innerHTML = `
        <p style="font-size: 13px; color: var(--text-hint); margin-bottom: 16px; text-align: center;">
            Choose amount to add
        </p>
        ${STAR_OPTIONS.map(opt => `
            <div class="fund-amount-option" onclick="executeFund('${agentId}', ${opt.stars})">
                <span class="fund-amount-stars">⭐ ${opt.stars} Stars</span>
                <span class="fund-amount-usd">${opt.usd}</span>
            </div>
        `).join('')}
    `;

    tg?.HapticFeedback?.selectionChanged?.();
}

async function executeFund(agentId, stars) {
    closeFundModal();
    toast('Opening payment…');

    // This would trigger the Telegram Stars payment via bot API
    // For now, we signal to the bot
    if (tg) {
        tg.sendData(JSON.stringify({ action: 'fund', agent_id: agentId, stars }));
    }
}

function closeFundModal() {
    document.getElementById('fund-modal').classList.add('hidden');
}

// ── Transactions ───────────────────────
async function loadTransactions() {
    const container = document.getElementById('transactions-list');
    if (!selectedAgent) {
        container.innerHTML = `
            <div class="empty-state">
                <div class="empty-icon">
                    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" opacity="0.3"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
                </div>
                <p class="empty-title">Select an agent first</p>
                <p class="empty-desc">Tap an agent on the dashboard to view its activity</p>
            </div>
        `;
        return;
    }

    container.innerHTML = `
        <div class="skeleton-card"><div class="skeleton-line w80"></div><div class="skeleton-line w50"></div></div>
        <div class="skeleton-card"><div class="skeleton-line w60"></div><div class="skeleton-line w40"></div></div>
    `;

    try {
        let url = `/v1/miniapp/agents/${selectedAgent.id}/transactions?limit=50`;
        if (currentFilter) url += `&type=${currentFilter}`;

        const data = await api(url);
        const txs = data.transactions || [];

        if (txs.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <div class="empty-icon">
                        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" opacity="0.3"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
                    </div>
                    <p class="empty-title">No transactions</p>
                    <p class="empty-desc">${currentFilter ? 'No matching transactions found' : 'Activity will show up here'}</p>
                </div>
            `;
            return;
        }

        container.innerHTML = txs.map((tx, i) => {
            const icons = { spend: '↗', deposit: '↙', refund: '↩', fee: '⚡' };
            const icon = icons[tx.type] || '•';
            const amountClass = tx.type === 'spend' ? 'spend' :
                                tx.type === 'deposit' ? 'deposit' :
                                tx.type === 'refund' ? 'refund' : '';
            const prefix = tx.type === 'spend' ? '-' : tx.type === 'deposit' ? '+' : '';
            const time = new Date(tx.created_at).toLocaleString('en-US', {
                month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit'
            });

            return `
                <div class="tx-item" style="animation: fadeIn 0.3s ease ${i * 0.03}s both">
                    <div class="tx-icon-wrap ${tx.type}">${icon}</div>
                    <div class="tx-details">
                        <div class="tx-type">${esc(tx.description || tx.type)}</div>
                        <div class="tx-desc">${time}</div>
                    </div>
                    <div class="tx-right">
                        <div class="tx-amount ${amountClass}">${prefix}$${fmt(tx.amount)}</div>
                        <div class="tx-time">${esc(tx.status)}</div>
                    </div>
                </div>
            `;
        }).join('');
    } catch (err) {
        container.innerHTML = `
            <div class="empty-state">
                <p class="empty-title">Error loading</p>
                <p class="empty-desc">${esc(err.message)}</p>
            </div>
        `;
    }
}

// ── Settings ───────────────────────────
async function loadSettings() {
    const panel = document.getElementById('settings-panel');
    if (!selectedAgent) {
        panel.innerHTML = `
            <div class="empty-state">
                <div class="empty-icon">
                    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" opacity="0.3"><circle cx="12" cy="12" r="3"/><path d="M12 1v2m0 18v2M4.22 4.22l1.42 1.42m12.72 12.72l1.42 1.42M1 12h2m18 0h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/></svg>
                </div>
                <p class="empty-title">Select an agent</p>
                <p class="empty-desc">Tap an agent to manage its settings</p>
            </div>
        `;
        return;
    }

    const a = selectedAgent;
    panel.innerHTML = `
        <div class="settings-group">
            <div class="settings-group-title">Spending Controls</div>
            <div class="setting-row">
                <span class="setting-label">Daily limit</span>
                <div class="setting-value">
                    <input type="number" id="set-daily-limit" value="${a.daily_limit_usd}" min="0" max="10000" step="1">
                </div>
            </div>
            <div class="setting-row">
                <span class="setting-label">Per transaction</span>
                <div class="setting-value">
                    <input type="number" id="set-tx-limit" value="${a.tx_limit_usd}" min="0" max="10000" step="1">
                </div>
            </div>
            <div class="setting-row">
                <span class="setting-label">Auto-approve up to</span>
                <div class="setting-value">
                    <input type="number" id="set-auto-approve" value="${a.auto_approve_usd}" min="0" max="10000" step="0.5">
                </div>
            </div>
        </div>

        <div class="settings-group">
            <div class="settings-group-title">Status</div>
            <div class="setting-row" style="border-radius: var(--radius-sm);">
                <span class="setting-label">Agent active</span>
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

        Object.assign(selectedAgent, payload);
        tg?.HapticFeedback?.notificationOccurred?.('success');
        toast('Settings saved', 'success');
        await loadDashboard();
    } catch (err) {
        tg?.HapticFeedback?.notificationOccurred?.('error');
        toast(err.message, 'error');
    }
}

// ── Card ───────────────────────────────
async function loadCard() {
    const panel = document.getElementById('card-panel');
    if (!selectedAgent) {
        panel.innerHTML = `
            <div class="empty-state">
                <div class="empty-icon">
                    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" opacity="0.3"><rect x="1" y="4" width="22" height="16" rx="2"/><line x1="1" y1="10" x2="23" y2="10"/></svg>
                </div>
                <p class="empty-title">Select an agent</p>
                <p class="empty-desc">Tap an agent to view or create a virtual card</p>
            </div>
        `;
        return;
    }

    panel.innerHTML = '<div class="skeleton-card"><div class="skeleton-line w80"></div><div class="skeleton-line w60"></div><div class="skeleton-line w40"></div></div>';

    try {
        const data = await api(`/v1/miniapp/agents/${selectedAgent.id}/card`);
        const card = data.card;

        if (!card || !card.last4) {
            panel.innerHTML = `
                <div class="empty-state">
                    <div class="empty-icon">
                        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" opacity="0.3"><rect x="1" y="4" width="22" height="16" rx="2"/><line x1="1" y1="10" x2="23" y2="10"/></svg>
                    </div>
                    <p class="empty-title">No card yet</p>
                    <p class="empty-desc">This agent doesn't have a virtual card</p>
                </div>
            `;
            return;
        }

        const stateClass = card.state === 'OPEN' ? 'open' : 'paused';
        const stateLabel = card.state === 'OPEN' ? '● Active' : '⏸ Paused';

        let txHtml = '';
        if (data.transactions && data.transactions.length > 0) {
            txHtml = `
                <div class="card-txns-title">Recent</div>
                ${data.transactions.map(t => `
                    <div class="tx-item">
                        <div class="tx-icon-wrap spend">↗</div>
                        <div class="tx-details">
                            <div class="tx-type">${esc(t.merchant)}</div>
                            <div class="tx-desc">${new Date(t.created).toLocaleDateString()}</div>
                        </div>
                        <div class="tx-right">
                            <div class="tx-amount spend">-$${(t.amount_cents / 100).toFixed(2)}</div>
                            <div class="tx-time">${esc(t.status)}</div>
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
        panel.innerHTML = `
            <div class="empty-state">
                <p class="empty-title">Error</p>
                <p class="empty-desc">${esc(err.message)}</p>
            </div>
        `;
    }
}

async function toggleCard(action) {
    if (!selectedAgent) return;
    try {
        await api(`/v1/miniapp/agents/${selectedAgent.id}/card/${action}`, { method: 'POST' });
        tg?.HapticFeedback?.notificationOccurred?.('success');
        toast(`Card ${action}d`, 'success');
        await loadCard();
    } catch (err) {
        toast(err.message, 'error');
    }
}

// ── Wallet ─────────────────────────────
async function loadWallet() {
    const panel = document.getElementById('wallet-panel');
    if (!selectedAgent) {
        panel.innerHTML = `
            <div class="empty-state">
                <div class="empty-icon">
                    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" opacity="0.3"><circle cx="12" cy="12" r="10"/><path d="M2 12h20M12 2a15.3 15.3 0 014 10 15.3 15.3 0 01-4 10 15.3 15.3 0 01-4-10 15.3 15.3 0 014-10z"/></svg>
                </div>
                <p class="empty-title">Select an agent</p>
                <p class="empty-desc">Tap an agent to view its on-chain wallet</p>
            </div>
        `;
        return;
    }

    panel.innerHTML = '<div class="skeleton-card"><div class="skeleton-line w80"></div><div class="skeleton-line w50"></div></div>';

    try {
        const data = await api(`/v1/miniapp/agents/${selectedAgent.id}/wallet`);

        if (!data.address) {
            panel.innerHTML = `
                <div class="empty-state">
                    <div class="empty-icon">
                        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" opacity="0.3"><circle cx="12" cy="12" r="10"/><path d="M2 12h20M12 2a15.3 15.3 0 014 10 15.3 15.3 0 01-4 10 15.3 15.3 0 01-4-10 15.3 15.3 0 014-10z"/></svg>
                    </div>
                    <p class="empty-title">No wallet yet</p>
                    <p class="empty-desc">Create an on-chain wallet via the bot with /wallet</p>
                </div>
            `;
            return;
        }

        panel.innerHTML = `
            <div class="wallet-address-box" onclick="copyAddress('${data.address}')">
                <div class="wallet-address-label">Address</div>
                <div class="wallet-address">${data.address}</div>
            </div>
            <div class="wallet-balances">
                <div class="wallet-balance-card">
                    <div class="wallet-token">ETH</div>
                    <div class="wallet-amount">${data.balance_eth || '0.00'}</div>
                    <div class="wallet-network">${data.network || 'Base'}</div>
                </div>
                <div class="wallet-balance-card">
                    <div class="wallet-token">USDC</div>
                    <div class="wallet-amount">${data.balance_usdc || '0.00'}</div>
                    <div class="wallet-network">${data.network || 'Base'}</div>
                </div>
            </div>
        `;
    } catch (err) {
        panel.innerHTML = `
            <div class="empty-state">
                <p class="empty-title">Error</p>
                <p class="empty-desc">${esc(err.message)}</p>
            </div>
        `;
    }
}

function copyAddress(address) {
    navigator.clipboard.writeText(address).then(() => {
        tg?.HapticFeedback?.notificationOccurred?.('success');
        toast('Address copied', 'success');
    }).catch(() => {
        toast('Copy failed', 'error');
    });
}

// ── Toast ──────────────────────────────
function toast(msg, type = '') {
    const container = document.getElementById('toast-container');
    const el = document.createElement('div');
    el.className = `toast ${type}`;
    el.innerHTML = `<span>${esc(msg)}</span>`;
    container.appendChild(el);

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
