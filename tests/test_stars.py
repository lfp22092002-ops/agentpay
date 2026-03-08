"""
Tests for providers/telegram_stars.py — Stars ↔ USD conversion.
"""
from decimal import Decimal



class TestStarsToUsd:
    def test_basic_conversion(self):
        from providers.telegram_stars import stars_to_usd
        result = stars_to_usd(100)
        assert result == Decimal("1.3000")

    def test_zero_stars(self):
        from providers.telegram_stars import stars_to_usd
        assert stars_to_usd(0) == Decimal("0.0000")

    def test_large_amount(self):
        from providers.telegram_stars import stars_to_usd
        result = stars_to_usd(10000)
        assert result == Decimal("130.0000")

    def test_returns_decimal(self):
        from providers.telegram_stars import stars_to_usd
        result = stars_to_usd(50)
        assert isinstance(result, Decimal)

    def test_precision_four_decimals(self):
        from providers.telegram_stars import stars_to_usd
        result = stars_to_usd(1)
        assert result == Decimal("0.0130")


class TestUsdToStars:
    def test_basic_conversion(self):
        from providers.telegram_stars import usd_to_stars
        result = usd_to_stars(Decimal("1.30"))
        assert result == 100

    def test_minimum_one_star(self):
        from providers.telegram_stars import usd_to_stars
        result = usd_to_stars(Decimal("0.001"))
        assert result == 1

    def test_large_amount(self):
        from providers.telegram_stars import usd_to_stars
        result = usd_to_stars(Decimal("130.00"))
        assert result == 10000

    def test_returns_int(self):
        from providers.telegram_stars import usd_to_stars
        result = usd_to_stars(Decimal("5.00"))
        assert isinstance(result, int)

    def test_rounds_down(self):
        """Fractional stars round down, minimum 1."""
        from providers.telegram_stars import usd_to_stars
        result = usd_to_stars(Decimal("0.02"))
        assert result == 1


class TestRoundTrip:
    def test_round_trip_approximate(self):
        """Converting stars→usd→stars should be approximately consistent."""
        from providers.telegram_stars import stars_to_usd, usd_to_stars
        original = 500
        usd = stars_to_usd(original)
        back = usd_to_stars(usd)
        assert back == original
