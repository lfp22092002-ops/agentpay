// AgentPay TypeScript SDK — Response types

/** Agent balance information. */
export interface Balance {
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
export interface Transaction {
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
export interface Wallet {
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
export interface Chain {
  id: string;
  name: string;
  type: string;
  native_token: string;
  usdc_supported: boolean;
  explorer: string;
}

/** Webhook registration details. */
export interface WebhookInfo {
  url?: string;
  secret?: string;
  events?: string[];
}

/** Response from the spend endpoint. */
export interface SpendResponse {
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
export interface TransferResponse {
  success: boolean;
  transaction_id?: string;
  amount: number;
  from_balance: number;
  error?: string;
}

/** Response from the refund endpoint. */
export interface RefundResponse {
  success: boolean;
  refund_transaction_id?: string;
  amount_refunded: number;
}

/** Response from the x402 pay endpoint. */
export interface X402Response {
  success: boolean;
  data?: unknown;
  amount_paid?: number;
  error?: string;
}

/** Agent identity / KYA profile. */
export interface AgentIdentity {
  display_name?: string;
  description?: string;
  trust_score?: number;
  verified?: boolean;
  [key: string]: unknown;
}

/** Trust score breakdown. */
export interface TrustScore {
  total: number;
  [key: string]: unknown;
}

/** Client configuration options. */
export interface AgentPayConfig {
  /** Base URL of the AgentPay API server. */
  baseUrl?: string;
  /** Request timeout in milliseconds. */
  timeoutMs?: number;
  /** Max retries for transient failures (429, 5xx, network). */
  maxRetries?: number;
}
