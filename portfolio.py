"""Compare three portfolio baselines using decimal returns and volatilities."""

import math


# EDITABLE INPUTS: enter percentages as decimals (3.5% = 0.035).
SLEEVES = [
    {"name": "Treasury bills", "return": 0.035, "volatility": 0.005,
     "reserve_eligible": True},
    {"name": "Treasury 10-year note", "return": 0.043, "volatility": 0.070,
     "reserve_eligible": True},
    {"name": "Investment-grade bond fund", "return": 0.050, "volatility": 0.060,
     "reserve_eligible": False},
    {"name": "Global equity index fund", "return": 0.080, "volatility": 0.160,
     "reserve_eligible": False},
]
# Rows and columns follow the SLEEVES order above.
CORRELATIONS = [
    [1.00, 0.10, 0.05, 0.00],
    [0.10, 1.00, 0.80, -0.10],
    [0.05, 0.80, 1.00, 0.25],
    [0.00, -0.10, 0.25, 1.00],
]
CPI_ESTIMATE = 0.025  # Labelled estimate, not observed CPI.
REAL_RETURN_TARGET = 0.040
OBJECTIVE_RETURN = CPI_ESTIMATE + REAL_RETURN_TARGET
BENCHMARK_ALLOCATIONS = {
    "Investment-grade bond fund": 0.40,
    "Global equity index fund": 0.60,
}


def validate_inputs(sleeves, correlations):
    """Reject invalid inputs before calculating any portfolio."""
    n = len(sleeves)
    if not n or len(correlations) != n or any(len(row) != n for row in correlations):
        raise ValueError("Correlation table must be square and match the sleeves.")
    for i, row in enumerate(correlations):
        for j, value in enumerate(row):
            if not math.isfinite(value) or not -1 <= value <= 1:
                raise ValueError(
                    f"Correlation at row {i + 1}, column {j + 1} must be between -1 and 1."
                )
            if value != correlations[j][i]:
                raise ValueError(
                    f"Correlation table is not symmetric at row {i + 1}, column {j + 1}."
                )
        if row[i] != 1:
            raise ValueError("Every diagonal correlation must equal 1.")
    for sleeve in sleeves:
        if not math.isfinite(sleeve["return"]):
            raise ValueError("Expected returns must be finite.")
        if not math.isfinite(sleeve["volatility"]) or sleeve["volatility"] <= 0:
            raise ValueError("Volatilities must be finite and greater than zero.")
    if len({s["name"] for s in sleeves}) != n:
        raise ValueError("Sleeve names must be unique.")


def covariance_matrix(sleeves, correlations):
    """Compute covariance[i][j] = correlation[i][j] * vol[i] * vol[j]."""
    validate_inputs(sleeves, correlations)
    return [
        [correlations[i][j] * left["volatility"] * right["volatility"]
         for j, right in enumerate(sleeves)]
        for i, left in enumerate(sleeves)
    ]


def benchmark_weights(sleeves, allocations):
    """Map named benchmark allocations to the sleeve order."""
    if set(allocations) - {s["name"] for s in sleeves}:
        raise ValueError("Benchmark allocation refers to an unknown sleeve.")
    weights = [allocations.get(s["name"], 0.0) for s in sleeves]
    if any(w < 0 for w in weights) or not math.isclose(sum(weights), 1.0):
        raise ValueError("Benchmark weights must be nonnegative and sum to 100%.")
    return weights


def equal_weights(sleeves):
    return [1.0 / len(sleeves)] * len(sleeves)


def inverse_volatility_weights(sleeves):
    inverse_vols = [1.0 / s["volatility"] for s in sleeves]
    total = sum(inverse_vols)
    return [value / total for value in inverse_vols]


def expected_return(weights, sleeves):
    return sum(w * s["return"] for w, s in zip(weights, sleeves))


def portfolio_volatility(weights, covariance):
    variance = sum(
        weights[i] * weights[j] * covariance[i][j]
        for i in range(len(weights)) for j in range(len(weights))
    )
    if variance < -1e-12:
        raise ValueError("Correlation inputs produce a negative portfolio variance.")
    return math.sqrt(max(0.0, variance))


def reserve_eligible_weight(weights, sleeves):
    return sum(w for w, s in zip(weights, sleeves) if s["reserve_eligible"])


def meets_objective(portfolio_return, objective):
    return portfolio_return >= objective


def print_portfolio(name, weights, sleeves, covariance, objective):
    result_return = expected_return(weights, sleeves)
    print(f"\n{name}")
    for sleeve, weight in zip(sleeves, weights):
        print(f"  {sleeve['name']}: {weight:.1%}")
    print(f"  Expected return: {result_return:.2%}")
    print(f"  Volatility: {portfolio_volatility(weights, covariance):.2%}")
    print(f"  Reserve-eligible weight: {reserve_eligible_weight(weights, sleeves):.1%}")
    print(f"  Meets objective: {'Yes' if meets_objective(result_return, objective) else 'No'}")


def main():
    try:
        covariance = covariance_matrix(SLEEVES, CORRELATIONS)
        portfolios = [
            ("Benchmark", benchmark_weights(SLEEVES, BENCHMARK_ALLOCATIONS)),
            ("Equal weight", equal_weights(SLEEVES)),
            ("Inverse volatility", inverse_volatility_weights(SLEEVES)),
        ]
        print(f"Objective: {OBJECTIVE_RETURN:.2%} "
              f"(CPI estimate {CPI_ESTIMATE:.2%} + {REAL_RETURN_TARGET:.2%})")
        for name, weights in portfolios:
            print_portfolio(name, weights, SLEEVES, covariance, OBJECTIVE_RETURN)
    except ValueError as error:
        raise SystemExit(f"Cannot run portfolio analysis: {error}") from None


if __name__ == "__main__":
    main()
