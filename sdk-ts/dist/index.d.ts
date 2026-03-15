/** Agent balance information. */
interface Balance {
    agent_id: string;
    agent_name: string;
    balance_usd: number;
    daily_limit_usd: number;
    daily_spent_usd: number;
    daily_remaining_usd: number;
    tx_limit_usd: number;
    is_active: boolean;
}
/** A single transaction record. */
interface Transaction {
    id: string;
    type: string;
    amount: number;
    fee: number;
    description?: string;
    status: string;
    created_at: string;
    transaction_id?: string;
    remaining_balance?: number;
}
/** On-chain wallet details. */
interface Wallet {
    address?: string;
    network?: string;
    chain?: string;
    balance?: string;
    balance_eth?: string;
    balance_usdc?: string;
    balance_native?: string;
    balance_sol?: string;
    native_token?: string;
}
/** Supported blockchain network. */
interface Chain {
    id: string;
    name: string;
    type: string;
    native_token: string;
    usdc_supported: boolean;
    explorer: string;
}
/** Webhook registration details. */
interface WebhookInfo {
    url?: string;
    secret?: string;
    events?: string[];
}
/** Response from the spend endpoint. */
interface SpendResponse {
    success: boolean;
    transaction_id?: string;
    amount: number;
    fee: number;
    remaining_balance: number;
    approval_id?: string;
    status: string;
    error?: string;
}
/** Response from the transfer endpoint. */
interface TransferResponse {
    success: boolean;
    transaction_id?: string;
    amount: number;
    from_balance: number;
    error?: string;
}
/** Response from the refund endpoint. */
interface RefundResponse {
    success: boolean;
    refund_transaction_id?: string;
    amount_refunded: number;
}
/** Response from the x402 pay endpoint. */
interface X402Response {
    success: boolean;
    data?: unknown;
    amount_paid?: number;
    error?: string;
}
/** Agent identity / KYA profile. */
interface AgentIdentity {
    display_name?: string;
    description?: string;
    trust_score?: number;
    verified?: boolean;
    [key: string]: unknown;
}
/** Trust score breakdown. */
interface TrustScore {
    total: number;
    [key: string]: unknown;
}
/** Client configuration options. */
interface AgentPayConfig {
    /** Base URL of the AgentPay API server. */
    baseUrl?: string;
    /** Request timeout in milliseconds. */
    timeoutMs?: number;
    /** Max retries for transient failures (429, 5xx, network). */
    maxRetries?: number;
}

/**
 * AgentPay TypeScript SDK — Client
 *
 * @example
 * ```ts
 * import { AgentPayClient } from "agentpay";
 *
 * const client = new AgentPayClient("ap_your_api_key");
 * const balance = await client.getBalance();
 * console.log(`Balance: $${balance.balance_usd}`);
 * ```
 */

declare class AgentPayClient {
    private readonly apiKey;
    private readonly baseUrl;
    private readonly timeoutMs;
    private readonly maxRetries;
    constructor(apiKey: string, config?: AgentPayConfig);
    private request;
    /** Get the current agent balance. */
    getBalance(): Promise<Balance>;
    /** Get the agent's on-chain wallet details. */
    getWallet(chain?: string): Promise<Wallet>;
    /** List all supported blockchain networks. */
    listChains(): Promise<Chain[]>;
    /** Spend funds from the agent balance. */
    spend(amount: number, description: string, idempotencyKey?: string): Promise<SpendResponse>;
    /** Refund a previous transaction. */
    refund(transactionId: string): Promise<RefundResponse>;
    /** Transfer funds to another agent. */
    transfer(toAgentId: string, amount: number, description?: string): Promise<TransferResponse>;
    /** Retrieve recent transactions. */
    getTransactions(limit?: number): Promise<Transaction[]>;
    /** Register a webhook endpoint. */
    registerWebhook(url: string, events?: string[]): Promise<WebhookInfo>;
    /** Pay for an x402-gated resource using the agent's wallet. */
    x402Pay(url: string, maxAmount?: number): Promise<X402Response>;
    /** Get the agent's identity profile (KYA). */
    getIdentity(): Promise<AgentIdentity>;
    /** Get the agent's trust score breakdown. */
    getTrustScore(): Promise<TrustScore>;
}

/** Base error for all AgentPay SDK errors. */
declare class AgentPayError extends Error {
    readonly statusCode: number;
    readonly detail: string;
    constructor(statusCode: number, detail: string);
}
/** Thrown when the API key is invalid or missing (HTTP 401). */
declare class AuthenticationError extends AgentPayError {
    constructor(detail: string);
}
/** Thrown when the agent has insufficient balance (HTTP 402 / 400 with balance hint). */
declare class InsufficientBalanceError extends AgentPayError {
    constructor(detail: string);
}
/** Thrown when rate limited (HTTP 429) and retries exhausted. */
declare class RateLimitError extends AgentPayError {
    constructor(detail: string);
}

export { type AgentIdentity, AgentPayClient, type AgentPayConfig, AgentPayError, AuthenticationError, type Balance, type Chain, InsufficientBalanceError, RateLimitError, type RefundResponse, type SpendResponse, type Transaction, type TransferResponse, type TrustScore, type Wallet, type WebhookInfo, type X402Response };
