from typing import Any, Dict, List

import numpy as np
import pandas as pd


def validate_line_items_math(
    line_items: List[Dict[str, Any]],
    tax_included: bool = True,
    tolerance: float = 0.01,
) -> Dict[str, Any]:
    if not line_items:
        return {
            "line_items_checked": [],
            "all_line_items_ok": True,
            "avg_abs_error": 0.0,
        }

    df = pd.DataFrame(line_items)

    for col in ["quantity", "unit_price", "line_total", "tax_rate"]:
        if col not in df:
            df[col] = np.nan

    if tax_included:
        df["expected_line_total"] = (
            df["quantity"] * df["unit_price"] * (1 + df["tax_rate"].fillna(0))
        )
    else:
        df["expected_line_total"] = df["quantity"] * df["unit_price"]

    df["diff"] = df["line_total"] - df["expected_line_total"]
    df["math_ok"] = df["diff"].abs() <= tolerance

    all_ok = bool(df["math_ok"].fillna(False).all())
    avg_abs_error = float(df["diff"].abs().mean() or 0.0)

    return {
        "line_items_checked": df.to_dict(orient="records"),
        "all_line_items_ok": all_ok,
        "avg_abs_error": avg_abs_error,
    }
