import pytest

from filing_analyst.tools.compute_ratio import RatioError, compute_ratio

# Real rows from data/facts.duckdb (NVIDIA, label = revenue / gross_profit).
NVDA_REVENUE_FY2026 = 215_938_000_000.0
NVDA_GROSS_PROFIT_FY2026 = 153_463_000_000.0
NVDA_REVENUE_FY2025 = 130_497_000_000.0


def test_gross_margin_from_real_nvidia_rows() -> None:
    result = compute_ratio(
        "gross_margin",
        {"gross_profit": NVDA_GROSS_PROFIT_FY2026, "revenue": NVDA_REVENUE_FY2026},
    )
    assert result.value == pytest.approx(153_463 / 215_938, abs=1e-6)
    assert result.percent == "71.07%"
    assert result.formula == "gross_profit / revenue = 153,463,000,000 / 215,938,000,000"


def test_gross_margin_falls_back_to_cost_of_revenue() -> None:
    result = compute_ratio("gross_margin", {"revenue": 1_000.0, "cost_of_revenue": 400.0})
    assert result.value == 0.6 and result.percent == "60.00%"
    assert result.formula.startswith("(revenue - cost_of_revenue) / revenue")


def test_growth_from_real_nvidia_revenue() -> None:
    result = compute_ratio("growth", {"old": NVDA_REVENUE_FY2025, "new": NVDA_REVENUE_FY2026})
    assert result.value == pytest.approx(215_938 / 130_497 - 1, abs=1e-6)
    assert result.percent == "65.47%"


def test_growth_also_gives_the_dollar_change_from_code() -> None:
    """Nov 19: asked 'by how much did Amazon's revenue grow', the model subtracted itself."""
    result = compute_ratio("growth", {"old": 637_959_000_000.0, "new": 716_924_000_000.0})
    assert result.percent == "12.38%"
    assert result.amount == 78_965_000_000
    assert result.formula.endswith("change = new - old = 78,965,000,000")
    shrink = compute_ratio("growth", {"old": 97_690_000_000.0, "new": 94_827_000_000.0})  # TSLA
    assert shrink.amount == -2_863_000_000 and shrink.percent == "-2.93%"


def test_margins_have_no_dollar_change() -> None:
    result = compute_ratio("net_margin", {"net_income": 1.0, "revenue": 4.0})
    assert result.amount is None


@pytest.mark.parametrize(
    ("name", "inputs", "message"),
    [
        ("gross_margin", {"revenue": 100.0}, "missing input"),
        ("net_margin", {"net_income": 5.0, "revenue": 0.0}, "revenue must be positive"),
        ("growth", {"old": 0.0, "new": 10.0}, "positive starting value"),
        ("growth", {"old": -50.0, "new": 10.0}, "positive starting value"),
        ("ebitda_margin", {"revenue": 100.0}, "unknown ratio"),
        ("net_margin", {"net_income": 5.0, "revenue": 100.0, "assets": 9.0}, "unexpected input"),
    ],
)
def test_bad_inputs_raise_instead_of_returning_a_number(
    name: str, inputs: dict[str, float], message: str
) -> None:
    with pytest.raises(RatioError, match=message):
        compute_ratio(name, inputs)


def test_operating_and_net_margin() -> None:
    assert (
        compute_ratio("operating_margin", {"operating_income": 25.0, "revenue": 100.0}).value
        == 0.25
    )
    assert compute_ratio("net_margin", {"net_income": -10.0, "revenue": 100.0}).percent == "-10.00%"
