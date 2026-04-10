// AgentPay TypeScript SDK — response types

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
export interface Webhook {
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
  new_balance: number;
  error?: string;
}

/** Response from the x402 pay endpoint. */
export interface X402Response {
  success: boolean;
  status?: number;
  data?: string;
  paid_usd: number;
  error?: string;
}

/** Options for creating an AgentPay client. */
export interface AgentPayOptions {
  /** Base URL of the AgentPay API (default: https://leofundmybot.dev). */
  baseUrl?: string;
  /** Request timeout in milliseconds (default: 30000). */
  timeoutMs?: number;
  /** Maximum retry attempts for 429/5xx errors (default: 3, set 0 to disable). */
  maxRetries?: number;
}

/** Agent identity profile (KYA — Know Your Agent). */
export interface AgentIdentity {
  agent_id: string;
  display_name: string;
  description?: string;
  homepage_url?: string;
  logo_url?: string;
  category?: string;
  verified: boolean;
  trust_score: number;
  total_transactions: number;
  total_volume_usd: number;
  first_seen: string;
  last_active: string;
  metadata_json?: string;
}

/** Trust score breakdown. */
export interface TrustScoreBreakdown {
  total: number;
  account_age_pts: number;
  account_age_max: number;
  transaction_count_pts: number;
  transaction_count_max: number;
  volume_pts: number;
  volume_max: number;
  profile_completeness_pts: number;
  profile_completeness_max: number;
  verified_pts: number;
  verified_max: number;
  details: Record<string, unknown>;
}

/** A payee rule (allow or deny). */
export interface PayeeRule {
  id: string;
  rule_type: "allow" | "deny";
  payee_type: "agent_id" | "domain" | "category" | "address";
  payee_value: string;
  max_amount_usd?: number | null;
  note?: string | null;
  is_active: boolean;
  created_at: string;
}

/** Response from the list payee rules endpoint. */
export interface PayeeRulesResponse {
  rules: PayeeRule[];
  total: number;
}
