import math
from typing import Optional
from simpleeval import InvalidExpression, simple_eval
from app.core.contract import Calculation

MAX_EXPRESSION_LENGTH = 200

# Explicit whitelist. Passing a NON-EMPTY dict replaces simpleeval's defaults, which would otherwise include rand()/randint() and make results non-deterministic.
_SAFE_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
}


def calculate(
    expression: str,
    description: str,
    chunk_ids: Optional[list[str]] = None,
) -> Calculation:
    """Evaluate an arithmetic expression and record which evidence chunks the numbers came from.

    Use this tool for ANY numeric computation (sums, differences, ratios,
    percentages, averages, growth rates). Never compute numbers yourself.

    Args:
        expression: Plain arithmetic expression using numeric literals only,
            e.g. "(120.5 - 98) / 98 * 100". Supports + - * / // % **, parentheses,
            and functions: abs, round, min, max, sqrt. Strip currency symbols and
            commas ("1200", not "$1,200") and write percentages as fractions
            ("15%" -> "0.15"). No variables or other functions allowed.
        description: One short sentence explaining what is being calculated,
            e.g. "Revenue growth from 2022 to 2023 (%)".
        chunk_ids: Source evidence chunk IDs containing the input numbers.
    """
    
    if not expression or not expression.strip():
        raise CalculatorError("Expression is empty.")
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise CalculatorError(f"Expression is longer than {MAX_EXPRESSION_LENGTH} characters.")

    try:
        raw = simple_eval(expression, functions=_SAFE_FUNCTIONS)
    except InvalidExpression as exc:
        raise CalculatorError(f"Not allowed or not understood: {exc}") from exc
    except ZeroDivisionError as exc:
        raise CalculatorError("Division by zero.") from exc
    except (SyntaxError, OverflowError, ValueError, TypeError) as exc:
        raise CalculatorError(f"Could not evaluate '{expression}': {exc}") from exc

    # bool is a subclass of int, so reject it explicitly (e.g. "1 < 2" gives True).
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise CalculatorError(f"Expression must produce a number, got {type(raw).__name__}.")

    value = float(raw)
    if not math.isfinite(value):
        raise CalculatorError("Result is not a finite number.")

    return Calculation(
        description=description,
        expression=expression,
        result=round(value, 6),
        chunk_ids=list(chunk_ids or []),
    )