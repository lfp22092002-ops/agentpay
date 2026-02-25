"""Pydantic response models for AgentPay SDK."""

from __future__ import annotations

from datetime import datetime
from typing import Any, List, Optional

from pydantic import BaseModel, Field


class Balance(BaseModel):
    """Agent balance information."""

    agent_id: str
    agent_name: str
    balance_usd: float
    daily_limit_usd: float
    daily_spent_usd: float
    daily_remaining_usd: float
    tx_limit_usd: float
    is_active: bool


class Transaction(BaseModel):
    """A single transaction record."""

    id: str
    type: str
    amount: float
    fee: float
    description: Optional[str] = None
    status: str
    created_at: str

    # Aliases for fields that may come back with different names
    transaction_id: Optional[str] = Field(None, alias="transaction_id")
    remaining_balance: Optional[float] = None


class Wallet(BaseModel):
    """On-chain wallet details."""

    address: Optional[str] = None
    network: Optional[str] = None
    chain: Optional[str] = None
    balance: Optional[str] = None
    balance_eth: Optional[str] = None
    balance_usdc: Optional[str] = None
    balance_native: Optional[str] = None
    balance_sol: Optional[str] = None
    native_token: Optional[str] = None


class Chain(BaseModel):
    """Supported blockchain network."""

    id: str
    name: str
    type: str
    native_token: str
    usdc_supported: bool
    explorer: str


class Webhook(BaseModel):
    """Webhook registration details."""

    url: Optional[str] = None
    secret: Optional[str] = None
    events: Optional[List[str]] = None


class SpendResponse(BaseModel):
    """Response from the spend endpoint."""

    success: bool
    transaction_id: Optional[str] = None
    amount: float
    fee: float = 0
    remaining_balance: float = 0
    approval_id: Optional[str] = None
    status: str = "completed"
    error: Optional[str] = None


class TransferResponse(BaseModel):
    """Response from the transfer endpoint."""

    success: bool
    transaction_id: Optional[str] = None
    amount: float = 0
    from_balance: float = 0
    error: Optional[str] = None


class RefundResponse(BaseModel):
    """Response from the refund endpoint."""

    success: bool
    refund_transaction_id: Optional[str] = None
    amount_refunded: float = 0
    new_balance: float = 0
    error: Optional[str] = None


class X402Response(BaseModel):
    """Response from the x402 pay endpoint."""

    success: bool
    status: Optional[int] = None
    data: Optional[str] = None
    paid_usd: float = 0
    error: Optional[str] = None
