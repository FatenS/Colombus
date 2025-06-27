# ─── templates.py (new small helper) ───────────────────────────────
import io
import pandas as pd
from flask import send_file

def _make_excel(headers: list[str], example: dict[str, str] | None = None):
    """
    Build an in-memory XLSX with one header row and -- optionally – a demo row.

    Parameters
    ----------
    headers : list[str]
        Ordered column names expected by the corresponding upload route.
    example : dict[str, Any] | None
        A single illustrative record. Keys **must** match the headers above.
    """
    df = (
        pd.DataFrame([example], columns=headers)    # keeps column order
        if example else
        pd.DataFrame(columns=headers)
    )
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as xls:
        df.to_excel(xls, index=False, sheet_name="Template")
    bio.seek(0)
    return send_file(
        bio,
        as_attachment=True,
        download_name="template.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
