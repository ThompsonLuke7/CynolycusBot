from enum import Enum


class Direction(str, Enum):
    UNKNOWN = "UNKNOWN"
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"


class MarketRegime(str, Enum):
    UNKNOWN = "UNKNOWN"
    STRONG_RISK_ON = "STRONG_RISK_ON"
    RISK_ON = "RISK_ON"
    NEUTRAL = "NEUTRAL"
    DETERIORATING = "DETERIORATING"
    RISK_OFF = "RISK_OFF"
    CRISIS = "CRISIS"


class ThemeRegime(str, Enum):
    UNKNOWN = "UNKNOWN"
    LEADERSHIP = "LEADERSHIP"
    ACCUMULATION = "ACCUMULATION"
    HEALTHY = "HEALTHY"
    NEUTRAL = "NEUTRAL"
    DETERIORATING = "DETERIORATING"
    DISTRIBUTION = "DISTRIBUTION"
    LIQUIDATION = "LIQUIDATION"


class TickerSetup(str, Enum):
    UNKNOWN = "UNKNOWN"
    BREAKOUT = "BREAKOUT"
    PULLBACK_IN_UPTREND = "PULLBACK_IN_UPTREND"
    MOMENTUM_CONTINUATION = "MOMENTUM_CONTINUATION"
    CONFIRMED_REVERSAL = "CONFIRMED_REVERSAL"
    COUNTERTREND_BOUNCE = "COUNTERTREND_BOUNCE"
    FAILED_RECLAIM = "FAILED_RECLAIM"
    BREAKDOWN = "BREAKDOWN"
    EXHAUSTION = "EXHAUSTION"
    RANGE_BOUND = "RANGE_BOUND"


class DealerRegime(str, Enum):
    UNKNOWN = "UNKNOWN"
    POSITIVE_GAMMA = "POSITIVE_GAMMA"
    NEUTRAL_GAMMA = "NEUTRAL_GAMMA"
    SHORT_GAMMA = "SHORT_GAMMA"
    PINNING = "PINNING"
    UPSIDE_ACCELERATION = "UPSIDE_ACCELERATION"
    DOWNSIDE_ACCELERATION = "DOWNSIDE_ACCELERATION"


class PolicyAction(str, Enum):
    APPROVE = "APPROVE"
    APPROVE_REDUCED = "APPROVE_REDUCED"
    REJECT = "REJECT"
    DEFER = "DEFER"
    EXIT = "EXIT"
    REDUCE = "REDUCE"
    HEDGE = "HEDGE"


class RuntimeEnvironment(str, Enum):
    DEVELOPMENT = "DEVELOPMENT"
    QA_PAPER = "QA_PAPER"
    PRODUCTION_LIVE = "PRODUCTION_LIVE"


class PolicyMode(str, Enum):
    OFF = "OFF"
    SHADOW = "SHADOW"
    ENFORCE = "ENFORCE"


class DecisionKind(str, Enum):
    ENTRY = "ENTRY"
    ADJUSTMENT = "ADJUSTMENT"
    EXIT = "EXIT"


class SizeUnit(str, Enum):
    """The unit `TradeIntent.position_size_requested` is denominated in.

    An entry is a money budget because the policy engine sizes risk in dollars.
    A reduction is a typed quantity because only an exact share or contract
    count can close a position; a dollar figure cannot.
    """

    UNKNOWN = "UNKNOWN"
    NOTIONAL_USD = "NOTIONAL_USD"
    SHARES = "SHARES"
    CONTRACTS = "CONTRACTS"


class StateType(str, Enum):
    MARKET = "MARKET"
    SECTOR = "SECTOR"
    THEME = "THEME"
    THEME_MEMBERSHIP = "THEME_MEMBERSHIP"
    TICKER = "TICKER"
    CATALYST_EVENT = "CATALYST_EVENT"
    CATALYST_PRESSURE = "CATALYST_PRESSURE"
    DEALER = "DEALER"
    PORTFOLIO = "PORTFOLIO"
    READINESS = "READINESS"


class AssetClass(str, Enum):
    EQUITY = "EQUITY"
    OPTION = "OPTION"


class OptionType(str, Enum):
    CALL = "CALL"
    PUT = "PUT"


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class PositionIntent(str, Enum):
    BUY_TO_OPEN = "BUY_TO_OPEN"
    BUY_TO_CLOSE = "BUY_TO_CLOSE"
    SELL_TO_OPEN = "SELL_TO_OPEN"
    SELL_TO_CLOSE = "SELL_TO_CLOSE"


class QuoteAssurance(str, Enum):
    """Whether an order's legs carry the market they were priced against.

    Entries are always QUOTED: opening risk we cannot price is never
    acceptable. A close may be DEGRADED, because a position we are trying to
    exit must not be trapped by a failed quote fetch — but the degradation is
    recorded rather than hidden, so an unpriced close is never mistaken for a
    priced one.
    """

    QUOTED = "QUOTED"
    DEGRADED = "DEGRADED"


class DebitCredit(str, Enum):
    DEBIT = "DEBIT"
    CREDIT = "CREDIT"


class ExecutionStatus(str, Enum):
    PLANNED = "PLANNED"
    SUBMISSION_PENDING = "SUBMISSION_PENDING"
    ACCEPTED = "ACCEPTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    CANCELED = "CANCELED"
    EXPIRED = "EXPIRED"
    UNKNOWN = "UNKNOWN"
    RECONCILIATION_REQUIRED = "RECONCILIATION_REQUIRED"


class SubmissionAttemptStatus(str, Enum):
    RESERVED = "RESERVED"
    JOURNALED = "JOURNALED"
    SUBMITTING = "SUBMITTING"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    AMBIGUOUS = "AMBIGUOUS"
    RECONCILIATION_REQUIRED = "RECONCILIATION_REQUIRED"


class InstrumentFamily(str, Enum):
    EQUITY = "EQUITY"
    SINGLE_OPTION = "SINGLE_OPTION"
    VERTICAL = "VERTICAL"
    CALENDAR = "CALENDAR"
    DIAGONAL = "DIAGONAL"
    STRADDLE = "STRADDLE"
    STRANGLE = "STRANGLE"
    BUTTERFLY = "BUTTERFLY"
    IRON_BUTTERFLY = "IRON_BUTTERFLY"
    CONDOR = "CONDOR"
    IRON_CONDOR = "IRON_CONDOR"
    COVERED_CALL = "COVERED_CALL"
    CASH_SECURED_PUT = "CASH_SECURED_PUT"
    PROTECTIVE_PUT = "PROTECTIVE_PUT"
    COLLAR = "COLLAR"
    ROLL = "ROLL"


class OwnershipStatus(str, Enum):
    ASSIGNED = "ASSIGNED"
    UNASSIGNED = "UNASSIGNED"
    CLOSED = "CLOSED"


class ReconciliationStatus(str, Enum):
    MATCHED = "MATCHED"
    PARTIAL = "PARTIAL"
    UNASSIGNED = "UNASSIGNED"
    ORPHANED_OWNERSHIP = "ORPHANED_OWNERSHIP"
    QUANTITY_MISMATCH = "QUANTITY_MISMATCH"


class MissingStateAction(str, Enum):
    REJECT = "REJECT"
    WARN = "WARN"
    OMIT = "OMIT"


class ModifierOperation(str, Enum):
    MULTIPLY = "MULTIPLY"
    CAP = "CAP"


class DataQualitySeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
