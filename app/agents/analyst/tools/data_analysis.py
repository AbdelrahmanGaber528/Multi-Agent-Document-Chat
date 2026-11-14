import math
import statistics
from typing import Optional

from app.agents.analyst.tools.table_extractor import (
    TableExtractorError,
    get_column,
    parse_number,
)
from app.core.contract import ExtractedTable, MetricAnalysis
from app.core.errors import DataAnalysisError


def _r(value: float) -> float:
    return round(value, 6)


def _as_number(value: object, what: str) -> float:
    """Validate that value is a finite number (rejects bool, NaN, inf, strings)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DataAnalysisError(f"{what} must be a number, got {value!r}.")
    number = float(value)
    if not math.isfinite(number):
        raise DataAnalysisError(f"{what} must be finite, got {value!r}.")
    return number


def _label(raw: object, idx: int, prefix: str) -> str:
    text = str(raw).strip() if raw is not None else ""
    return text or f"{prefix}_{idx}"


def find_table_with_columns(tables: list[ExtractedTable], *columns: str) -> ExtractedTable:
    """Return the first table that contains ALL the named columns (case-insensitive)."""
    wanted = {c.strip().lower() for c in columns}
    for table in tables:
        if wanted <= {h.strip().lower() for h in table.headers}:
            return table
    available = [t.headers for t in tables]
    raise DataAnalysisError(f"No table has all of {list(columns)}. Tables found: {available}")


def _build_ranking(
    items: list[tuple[str, float]],
    skipped: list[str],
    metric: str,
    chunk_ids: list[str],
    higher_is_better: bool,
) -> MetricAnalysis:
    if not items:
        raise DataAnalysisError(f"No valid numeric values found for '{metric}'.")

    ranked = sorted(items, key=lambda pair: pair[1], reverse=higher_is_better)
    nums = [value for _, value in ranked]
    best_val, worst_val = ranked[0][1], ranked[-1][1]
    best_labels = [label for label, value in ranked if value == best_val]
    worst_labels = [label for label, value in ranked if value == worst_val]

    rankings = []
    prev_value: Optional[float] = None
    prev_rank = 0
    for position, (label, value) in enumerate(ranked, start=1):
        rank = prev_rank if value == prev_value else position  # ties share a rank: 1, 1, 3
        prev_value, prev_rank = value, rank
        gap = abs(value - best_val)  # always >= 0, whichever direction is "better"
        rankings.append(
            {
                "rank": rank,
                "label": label,
                "value": _r(value),
                "gap_from_best": _r(gap),
                "pct_gap_from_best": _r(gap / abs(best_val) * 100.0) if best_val != 0 else None,
            }
        )

    mean = statistics.fmean(nums)
    spread = abs(best_val - worst_val)
    direction = "highest" if higher_is_better else "lowest"
    tie_note = " (tie)" if len(best_labels) > 1 else ""
    summary = (
        f"{metric}: best is {', '.join(best_labels)}{tie_note} at {_r(best_val):g} "
        f"({direction} of {len(nums)} items); mean {_r(mean):g}, spread {_r(spread):g}."
    )
    if skipped:
        summary += f" Skipped (non-numeric): {', '.join(skipped)}."

    return MetricAnalysis(
        analysis_type="ranking",
        metric=metric,
        summary=summary,
        chunk_ids=chunk_ids,
        details={
            "higher_is_better": higher_is_better,
            "count": len(nums),
            "skipped": skipped,
            "best": {"labels": best_labels, "value": _r(best_val)},
            "worst": {"labels": worst_labels, "value": _r(worst_val)},
            "statistics": {
                "mean": _r(mean),
                "median": _r(statistics.median(nums)),
                "min": _r(min(nums)),
                "max": _r(max(nums)),
                "spread": _r(spread),
                # sample standard deviation; undefined for a single value
                "stdev": _r(statistics.stdev(nums)) if len(nums) >= 2 else None,
            },
            "rankings": rankings,
        },
    )


def analyze_table_metric(
    table: ExtractedTable,
    metric_column: str,
    label_column: Optional[str] = None,
    higher_is_better: bool = True,
) -> MetricAnalysis:
    """Rank the rows of a table by a numeric column and compute summary statistics.

    Use this when an evidence table holds metrics (accuracy, F1, latency, cost,
    revenue) and you need the best performer, averages or spread. Pass a table
    produced by extract_tables, not raw text.

    Args:
        table: An ExtractedTable (use find_table_with_columns if a chunk has several).
        metric_column: Header of the numeric column to analyze, e.g. "F1".
        label_column: Header that names each row, e.g. "Model". Defaults to the
            first column.
        higher_is_better: True for accuracy/revenue, False for error rate/loss/latency.

    Returns:
        MetricAnalysis (analysis_type "ranking"). details holds count, skipped
        rows, best/worst (ties list every label), statistics (mean, median, min,
        max, spread, sample stdev) and rankings with gap_from_best (always >= 0).
        Rows whose cell is not numeric are listed in details["skipped"].

    Raises:
        DataAnalysisError: Unknown column or no numeric values.
    """
    try:
        values = get_column(table, metric_column, numeric=True)
        if label_column is not None:
            labels = get_column(table, label_column)
        elif table.headers and table.headers[0].strip().lower() != metric_column.strip().lower():
            labels = [row[0] if row else "" for row in table.rows]
        else:
            labels = [""] * len(table.rows)
    except TableExtractorError as exc:
        raise DataAnalysisError(str(exc)) from exc

    items: list[tuple[str, float]] = []
    skipped: list[str] = []
    for idx, (raw_label, value) in enumerate(zip(labels, values), start=1):
        label = _label(raw_label, idx, "row")
        if value is None:
            skipped.append(label)
        else:
            items.append((label, value))

    return _build_ranking(items, skipped, metric_column.strip(), [table.chunk_id], higher_is_better)


def analyze_numbers(
    values: list[float],
    metric_name: str,
    labels: Optional[list[str]] = None,
    chunk_ids: Optional[list[str]] = None,
    higher_is_better: bool = True,
) -> MetricAnalysis:
    """Rank and summarize numbers that come from separate chunks (not one table).

    Args:
        values: Numbers to analyze, e.g. [92.0, 95.0, 89.0].
        metric_name: What they measure, e.g. "CNN accuracy (%)".
        labels: Optional names matching values in order, e.g. ["Paper A", "Paper B"].
        chunk_ids: IDs of the chunks the numbers came from, for citation.
        higher_is_better: False when smaller is better (error, latency).

    Returns:
        MetricAnalysis (analysis_type "ranking"), same details as analyze_table_metric.

    Raises:
        DataAnalysisError: Empty values, non-finite numbers, or labels of a different length.
    """
    if not values:
        raise DataAnalysisError("values list cannot be empty.")
    if labels is not None and len(labels) != len(values):
        raise DataAnalysisError(
            f"labels length ({len(labels)}) must match values length ({len(values)})."
        )

    numbers = [_as_number(v, "value") for v in values]
    names = [_label(labels[i] if labels else None, i + 1, "item") for i in range(len(numbers))]
    return _build_ranking(
        list(zip(names, numbers)),
        [],
        metric_name.strip() or "value",
        list(chunk_ids or []),
        higher_is_better,
    )


def analyze_trend(
    values: list[float],
    periods: list[str],
    metric_name: str,
    chunk_ids: Optional[list[str]] = None,
    higher_is_better: bool = True,
) -> MetricAnalysis:
    """Describe how a metric changes across ordered periods (years, epochs, versions).

    Args:
        values: Metric values in chronological order.
        periods: Labels for each value in the same order, e.g. ["2021", "2022", "2023"].
        metric_name: Name of the metric.
        chunk_ids: IDs of the chunks the numbers came from.
        higher_is_better: Used only to say whether the trend is favorable.

    Returns:
        MetricAnalysis (analysis_type "trend"). details holds first/last, change,
        pct_change, avg_change_per_period, slope_per_period (least-squares, assumes
        equally spaced periods), direction ("increasing"/"decreasing"/"flat"),
        trend_is_favorable, peak and low.

    Raises:
        DataAnalysisError: Fewer than 2 points, mismatched lengths, or non-finite values.
    """
    if len(values) < 2:
        raise DataAnalysisError("A trend needs at least 2 values.")
    if len(periods) != len(values):
        raise DataAnalysisError(
            f"periods length ({len(periods)}) must match values length ({len(values)})."
        )

    ys = [_as_number(v, "value") for v in values]
    labels = [_label(p, i, "period") for i, p in enumerate(periods, start=1)]
    n = len(ys)

    slope, _ = statistics.linear_regression(list(range(n)), ys)
    change = ys[-1] - ys[0]
    pct_change = _r(change / abs(ys[0]) * 100.0) if ys[0] != 0 else None

    if math.isclose(slope, 0.0, abs_tol=1e-12):
        direction, favorable = "flat", None
    else:
        direction = "increasing" if slope > 0 else "decreasing"
        favorable = (slope > 0) == higher_is_better

    peak_idx = max(range(n), key=lambda k: ys[k])
    low_idx = min(range(n), key=lambda k: ys[k])

    summary = (
        f"{metric_name} is {direction} over {n} periods "
        f"({labels[0]} {_r(ys[0]):g} -> {labels[-1]} {_r(ys[-1]):g}, change {_r(change):+g}"
        + (f", {pct_change:+.2f}%" if pct_change is not None else "")
        + ")."
    )

    return MetricAnalysis(
        analysis_type="trend",
        metric=metric_name.strip() or "value",
        summary=summary,
        chunk_ids=list(chunk_ids or []),
        details={
            "periods": labels,
            "values": [_r(y) for y in ys],
            "first": _r(ys[0]),
            "last": _r(ys[-1]),
            "change": _r(change),
            "pct_change": pct_change,
            "avg_change_per_period": _r(change / (n - 1)),
            "slope_per_period": _r(slope),
            "direction": direction,
            "trend_is_favorable": favorable,
            "peak": {"period": labels[peak_idx], "value": _r(ys[peak_idx])},
            "low": {"period": labels[low_idx], "value": _r(ys[low_idx])},
        },
    )


def analyze_table_trend(
    table: ExtractedTable,
    period_column: str,
    metric_column: str,
    higher_is_better: bool = True,
) -> MetricAnalysis:
    """Trend of a table column over a period column (e.g. Year vs Revenue).

    Rows are sorted by period when every period is a plain number (such as a
    year), because tables are often listed newest first. Otherwise table order
    is kept. Rows whose metric is not numeric are dropped.

    Raises:
        DataAnalysisError: Unknown column, or fewer than 2 usable rows.
    """
    try:
        periods = get_column(table, period_column)
        values = get_column(table, metric_column, numeric=True)
    except TableExtractorError as exc:
        raise DataAnalysisError(str(exc)) from exc

    points = [(p, v) for p, v in zip(periods, values) if v is not None]
    keys = [parse_number(p) for p, _ in points]
    if points and all(k is not None for k in keys):
        order = sorted(range(len(points)), key=lambda k: keys[k])
        points = [points[k] for k in order]

    return analyze_trend(
        values=[v for _, v in points],
        periods=[p for p, _ in points],
        metric_name=metric_column.strip(),
        chunk_ids=[table.chunk_id],
        higher_is_better=higher_is_better,
    )


def analyze_correlation(
    x_values: list[float],
    y_values: list[float],
    x_name: str,
    y_name: str,
    chunk_ids: Optional[list[str]] = None,
) -> MetricAnalysis:
    """Measure the linear relationship between two variables (Pearson correlation).

    Use for questions like "does accuracy rise with model size?".

    Args:
        x_values: First variable, e.g. parameter counts.
        y_values: Second variable, same length and order, e.g. accuracy.
        x_name: Name of the first variable.
        y_name: Name of the second variable.
        chunk_ids: IDs of the chunks the numbers came from.

    Returns:
        MetricAnalysis (analysis_type "correlation"). details holds n, r,
        r_squared, slope and intercept (y on x), strength and direction.
        Correlation does not prove causation, and with very few points it is
        only indicative (the summary says so when n < 5).

    Raises:
        DataAnalysisError: Fewer than 3 pairs, mismatched lengths, non-finite
            values, or one variable is constant (correlation undefined).
    """
    if len(x_values) != len(y_values):
        raise DataAnalysisError(
            f"x_values length ({len(x_values)}) must match y_values length ({len(y_values)})."
        )
    if len(x_values) < 3:
        raise DataAnalysisError("Correlation needs at least 3 pairs of values.")

    xs = [_as_number(v, x_name) for v in x_values]
    ys = [_as_number(v, y_name) for v in y_values]
    try:
        r = statistics.correlation(xs, ys)
        slope, intercept = statistics.linear_regression(xs, ys)
    except statistics.StatisticsError as exc:
        raise DataAnalysisError(f"Correlation undefined (is one variable constant?): {exc}") from exc

    strength = (
        "strong" if abs(r) >= 0.8 else "moderate" if abs(r) >= 0.5
        else "weak" if abs(r) >= 0.3 else "negligible"
    )
    direction = "positive" if r > 0 else "negative"
    summary = f"{x_name} vs {y_name}: {strength} {direction} correlation (r={r:.3f}, n={len(xs)})."
    if len(xs) < 5:
        summary += " Very few points: treat as indicative only."

    return MetricAnalysis(
        analysis_type="correlation",
        metric=f"{x_name} vs {y_name}",
        summary=summary,
        chunk_ids=list(chunk_ids or []),
        details={
            "n": len(xs),
            "r": _r(r),
            "r_squared": _r(r * r),
            "slope": _r(slope),
            "intercept": _r(intercept),
            "strength": strength,
            "direction": direction,
        },
    )


def analyze_table_correlation(
    table: ExtractedTable,
    x_column: str,
    y_column: str,
) -> MetricAnalysis:
    """Correlation between two numeric columns of a table. Rows where either cell
    is not numeric are dropped.

    Raises:
        DataAnalysisError: Unknown column, or fewer than 3 usable rows.
    """
    try:
        xs = get_column(table, x_column, numeric=True)
        ys = get_column(table, y_column, numeric=True)
    except TableExtractorError as exc:
        raise DataAnalysisError(str(exc)) from exc

    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    return analyze_correlation(
        x_values=[x for x, _ in pairs],
        y_values=[y for _, y in pairs],
        x_name=x_column.strip(),
        y_name=y_column.strip(),
        chunk_ids=[table.chunk_id],
    )