"""Financial ratios in plain Python, so the LLM never does arithmetic.

The agent passes numbers it got from SQL rows; this returns the value plus the formula with the
actual inputs, so every computed number in an answer can be checked by hand.
"""

from collections.abc import Callable

from pydantic import BaseModel

Inputs = dict[str, float]


class RatioError(ValueError):
    """The ratio cannot be computed from these inputs; never return a guessed number."""


class RatioResult(BaseModel):
    name: str
    value: float  # a fraction: 0.7107 means 71.07%
    percent: str
    formula: str  # with the actual numbers, e.g. "gross_profit / revenue = 153,463 / 215,938"
    inputs: Inputs


def _need(inputs: Inputs, *names: str) -> list[float]:
    missing = [n for n in names if n not in inputs]
    if missing:
        raise RatioError(f"missing input(s): {', '.join(missing)}")
    return [inputs[n] for n in names]


def _margin(part_name: str, part: float, revenue: float) -> tuple[float, str]:
    if revenue <= 0:
        raise RatioError(f"revenue must be positive for a margin, got {revenue:,.0f}")
    return part / revenue, f"{part_name} / revenue = {part:,.0f} / {revenue:,.0f}"


def gross_margin(inputs: Inputs) -> tuple[float, str]:
    """From gross_profit, or from cost_of_revenue when a company does not report gross
    profit (GOOGL, META, QCOM and others in our gap list)."""
    if "gross_profit" in inputs:
        gross_profit, revenue = _need(inputs, "gross_profit", "revenue")
        return _margin("gross_profit", gross_profit, revenue)
    revenue, cost = _need(inputs, "revenue", "cost_of_revenue")
    value, _ = _margin("gross_profit", revenue - cost, revenue)
    return (
        value,
        f"(revenue - cost_of_revenue) / revenue = ({revenue:,.0f} - {cost:,.0f}) / {revenue:,.0f}",
    )


def operating_margin(inputs: Inputs) -> tuple[float, str]:
    return _margin("operating_income", *_need(inputs, "operating_income", "revenue"))


def net_margin(inputs: Inputs) -> tuple[float, str]:
    return _margin("net_income", *_need(inputs, "net_income", "revenue"))


def growth(inputs: Inputs) -> tuple[float, str]:
    old, new = _need(inputs, "old", "new")
    if old <= 0:
        raise RatioError(f"growth needs a positive starting value, got {old:,.0f}")
    return (new - old) / old, f"(new - old) / old = ({new:,.0f} - {old:,.0f}) / {old:,.0f}"


# name -> (function, inputs it accepts)
RATIOS: dict[str, tuple[Callable[[Inputs], tuple[float, str]], set[str]]] = {
    "gross_margin": (gross_margin, {"revenue", "gross_profit", "cost_of_revenue"}),
    "operating_margin": (operating_margin, {"revenue", "operating_income"}),
    "net_margin": (net_margin, {"revenue", "net_income"}),
    "growth": (growth, {"old", "new"}),
}


def compute_ratio(name: str, inputs: Inputs) -> RatioResult:
    if name not in RATIOS:
        raise RatioError(f"unknown ratio {name!r}; choose from {', '.join(sorted(RATIOS))}")
    function, allowed = RATIOS[name]
    unexpected = set(inputs) - allowed
    if unexpected:  # usually a sign the model mixed up metrics: fail loudly
        raise RatioError(f"unexpected input(s) for {name}: {', '.join(sorted(unexpected))}")
    value, formula = function(inputs)
    return RatioResult(
        name=name,
        value=round(value, 6),
        percent=f"{value * 100:.2f}%",
        formula=formula,
        inputs=inputs,
    )
