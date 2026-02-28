import uuid
from datetime import datetime
from decimal import Decimal
from sqlalchemy import String, BigInteger, Numeric, DateTime, Boolean, ForeignKey, Text, Enum as SQLEnum, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship
from models.database import Base
import enum


class TransactionType(str, enum.Enum):
    DEPOSIT = "deposit"
    SPEND = "spend"
    REFUND = "refund"
    FEE = "fee"


class TransactionStatus(str, enum.Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


class PaymentMethod(str, enum.Enum):
    TELEGRAM_STARS = "telegram_stars"
    STRIPE = "stripe"
    USDC = "usdc"
    MANUAL = "manual"


class AgentIdentity(Base):
    """Agent identity/reputation — the "Know Your Agent" (KYA) system."""
    __tablename__ = "agent_identities"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    agent_id: Mapped[str] = mapped_column(String(36), ForeignKey("agents.id"), unique=True, index=True)
    display_name: Mapped[str] = mapped_column(String(255))
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    homepage_url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    logo_url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    category: Mapped[str | None] = mapped_column(String(50), nullable=True)
    verified: Mapped[bool] = mapped_column(Boolean, default=False)
    trust_score: Mapped[int] = mapped_column(BigInteger, default=0)
    total_transactions: Mapped[int] = mapped_column(BigInteger, default=0)
    total_volume_usd: Mapped[Decimal] = mapped_column(Numeric(12, 4), default=Decimal("0.0000"))
    first_seen: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    last_active: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    agent: Mapped["Agent"] = relationship(back_populates="identity")


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    telegram_id: Mapped[int] = mapped_column(BigInteger, unique=True, index=True)
    username: Mapped[str | None] = mapped_column(String(255), nullable=True)
    first_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    is_pro: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    agents: Mapped[list["Agent"]] = relationship(back_populates="user", cascade="all, delete-orphan")


class PlatformRevenue(Base):
    """Tracks platform fee revenue from agent spends."""
    __tablename__ = "platform_revenue"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    transaction_id: Mapped[str] = mapped_column(String(36), ForeignKey("transactions.id"), index=True)
    agent_id: Mapped[str] = mapped_column(String(36), ForeignKey("agents.id"), index=True)
    amount_usd: Mapped[Decimal] = mapped_column(Numeric(12, 4))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Agent(Base):
    __tablename__ = "agents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.id"), index=True)
    name: Mapped[str] = mapped_column(String(255))
    api_key_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)  # SHA-256 hex
    api_key_prefix: Mapped[str] = mapped_column(String(12), default="")  # "ap_xxxx..." for display
    balance_usd: Mapped[Decimal] = mapped_column(Numeric(12, 4), default=Decimal("0.0000"))
    daily_limit_usd: Mapped[Decimal] = mapped_column(Numeric(12, 4), default=Decimal("50.0000"))
    tx_limit_usd: Mapped[Decimal] = mapped_column(Numeric(12, 4), default=Decimal("25.0000"))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    auto_approve_usd: Mapped[Decimal] = mapped_column(Numeric(12, 4), default=Decimal("10.0000"))
    webhook_url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    webhook_secret: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    user: Mapped["User"] = relationship(back_populates="agents")
    wallet: Mapped["Wallet | None"] = relationship(back_populates="agent", uselist=False, cascade="all, delete-orphan")
    transactions: Mapped[list["Transaction"]] = relationship(back_populates="agent", cascade="all, delete-orphan")
    identity: Mapped["AgentIdentity | None"] = relationship(back_populates="agent", uselist=False, cascade="all, delete-orphan")


class Wallet(Base):
    __tablename__ = "wallets"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    agent_id: Mapped[str] = mapped_column(String(36), ForeignKey("agents.id"), unique=True)
    wallet_type: Mapped[str] = mapped_column(String(50), default="internal")  # internal, usdc, virtual_card
    chain: Mapped[str] = mapped_column(String(20), default="base")  # base, polygon, bnb, solana
    external_id: Mapped[str | None] = mapped_column(String(255), nullable=True)  # Coinbase/Lithic ID
    address: Mapped[str | None] = mapped_column(String(255), nullable=True)  # crypto address
    card_last4: Mapped[str | None] = mapped_column(String(4), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    agent: Mapped["Agent"] = relationship(back_populates="wallet")


class Transaction(Base):
    __tablename__ = "transactions"
    __table_args__ = (
        Index("idx_tx_agent_created", "agent_id", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    agent_id: Mapped[str] = mapped_column(String(36), ForeignKey("agents.id"), index=True)
    tx_type: Mapped[TransactionType] = mapped_column(SQLEnum(TransactionType))
    status: Mapped[TransactionStatus] = mapped_column(SQLEnum(TransactionStatus), default=TransactionStatus.PENDING)
    amount_usd: Mapped[Decimal] = mapped_column(Numeric(12, 4))
    fee_usd: Mapped[Decimal] = mapped_column(Numeric(12, 4), default=Decimal("0.0000"))
    payment_method: Mapped[PaymentMethod | None] = mapped_column(SQLEnum(PaymentMethod), nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    external_ref: Mapped[str | None] = mapped_column(String(255), nullable=True)  # Stripe/Stars charge ID
    idempotency_key: Mapped[str | None] = mapped_column(String(64), nullable=True, unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    agent: Mapped["Agent"] = relationship(back_populates="transactions")
