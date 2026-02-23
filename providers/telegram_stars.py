from decimal import Decimal
from config.settings import STARS_TO_USD_RATE


def stars_to_usd(stars: int) -> Decimal:
    """Convert Telegram Stars to USD amount."""
    return Decimal(str(stars * STARS_TO_USD_RATE)).quantize(Decimal("0.0001"))


def usd_to_stars(usd: Decimal) -> int:
    """Convert USD to approximate Stars count."""
    return max(1, int(float(usd) / STARS_TO_USD_RATE))
