"""
Query the LMSAL HEK/HCR "search-her" endpoint (AIA flares with correlated
IRIS / SOT / SOTSP / XRT / EIS observations) and load the result into pandas.

Returns two tables:
  events : one row per HEK event (all scalar fields, nested dicts flattened
           as "parent.child" columns)
  corr   : one row per correlated observation (from any nested list-of-dicts
           field, e.g. IRIS/XRT coverage), with the parent event's index,
           ID and start time attached so you can join back to `events`.

The parser does not hard-code the response layout: it finds the list of
event records and splits out any nested list fields on its own.
"""

import json
import requests
import pandas as pd

URL = "https://www.lmsal.com/hek/hcr"

# Same query as the browser URL. A list of tuples keeps the repeated
# `optionalcorr` keys (a dict would drop all but the last one).
PARAMS = [
    ("cosec", 2),
    ("cmd", "search-her"),
    ("type", "column"),
    ("event_coordsys", "helioprojective"),
    ("event_region", "all"),
    ("event_type", "FL"),
    ("x1", -5000), ("x2", 5000), ("y1", -5000), ("y2", 5000),
    ("event_starttime", "2024-05-07T00:00"),
    ("event_endtime", "2024-05-10T00:00"),
    ("result_limit", 80),
    ("sparam0", "Search_FRM_Name"), ("op0", "="), ("value0", "SSW Latest Events"),
    ("sparam1", "Search_Instrument"), ("op1", "like"), ("value1", "AIA"),
    ("optionalcorr", "IRIS"),
    ("optionalcorr", "SOT"),
    ("optionalcorr", "SOTSP"),
    ("optionalcorr", "XRT"),
    ("optionalcorr", "EIS"),
    ("sort_by", "sum_overlap_scores"),
    ("sort_order", "desc"),
]


def _fmt_time(t):
    """Accept a str, datetime, pandas/numpy timestamp or astropy Time -> 'YYYY-MM-DDTHH:MM:SS'."""
    if hasattr(t, "isot"):                      # astropy.time.Time
        t = t.isot
    return pd.Timestamp(t).strftime("%Y-%m-%dT%H:%M:%S")


def build_params(start, end, result_limit=80):
    """Copy of PARAMS with the requested time window (and result limit)."""
    new = {"event_starttime": _fmt_time(start),
           "event_endtime": _fmt_time(end),
           "result_limit": result_limit}
    if pd.Timestamp(new["event_endtime"]) <= pd.Timestamp(new["event_starttime"]):
        raise ValueError("end must be after start")
    return [(k, new.get(k, v)) for k, v in PARAMS]


def fetch(start="2024-05-07T00:00", end="2024-05-10T00:00",
          result_limit=80, timeout=60):
    """GET the endpoint for the given time window and return the parsed JSON."""
    params = build_params(start, end, result_limit)
    r = requests.get(URL, params=params, timeout=timeout,
                     headers={"User-Agent": "python-requests (HEK query)"})
    r.raise_for_status()
    text = r.text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Some LMSAL endpoints wrap JSON in a JSONP callback: cb({...});
        if "(" in text and text.endswith((")", ");")):
            return json.loads(text[text.index("(") + 1: text.rindex(")")])
        raise ValueError(f"Response is not JSON. First 300 chars:\n{text[:300]}")


def _find_records(obj):
    """Locate the list of event dicts inside the response."""
    if isinstance(obj, list):
        return obj
    for key in ("result", "Events", "events", "results", "data"):
        if isinstance(obj.get(key), list):
            return obj[key]
    # fall back to the longest list-of-dicts found at the top level
    lists = [v for v in obj.values()
             if isinstance(v, list) and v and isinstance(v[0], dict)]
    if not lists:
        raise KeyError(f"No list of records found; top-level keys: {list(obj)}")
    return max(lists, key=len)


def _is_list_of_dicts(v):
    return isinstance(v, list) and any(isinstance(x, dict) for x in v)


def to_frames(data):
    """Split the response into (events, corr) DataFrames."""
    records = _find_records(data)

    # Any field that is a list of dicts in at least one record is treated as
    # nested correlation data rather than an event column.
    nested_keys = sorted({k for rec in records for k, v in rec.items()
                          if _is_list_of_dicts(v)})

    flat, corr_rows = [], []
    for i, rec in enumerate(records):
        flat.append({k: v for k, v in rec.items() if k not in nested_keys})
        for k in nested_keys:
            for item in rec.get(k) or []:
                if isinstance(item, dict):
                    corr_rows.append({"event_idx": i,
                                      "corr_field": k,
                                      "event_id": rec.get("kb_archivid", rec.get("SOL_standard")),
                                      "event_starttime": rec.get("event_starttime"),
                                      **item})

    events = pd.json_normalize(flat)
    corr = pd.json_normalize(corr_rows) if corr_rows else pd.DataFrame()

    # Convert obvious time columns to datetimes.
    for df in (events, corr):
        for c in df.columns:
            if "time" in c.lower() and (df[c].dtype == object
                                        or pd.api.types.is_string_dtype(df[c])):
                df[c] = pd.to_datetime(df[c], errors="coerce")
    return events, corr


def query(start, end, result_limit=80):
    """One call: fetch the window [start, end] and return (events, corr)."""
    return to_frames(fetch(start, end, result_limit))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="HEK AIA flares + IRIS/Hinode correlations")
    ap.add_argument("start", nargs="?", default="2024-05-07T00:00",
                    help="event_starttime, e.g. 2024-05-07T00:00")
    ap.add_argument("end", nargs="?", default="2024-05-10T00:00",
                    help="event_endtime, e.g. 2024-05-10T00:00")
    ap.add_argument("--limit", type=int, default=80, help="result_limit")
    args = ap.parse_args()

    raw = fetch(args.start, args.end, args.limit)
    with open("hek_raw.json", "w") as f:        # keep the raw response
        json.dump(raw, f, indent=1)

    events, corr = to_frames(raw)
    print(f"{len(events)} events, {events.shape[1]} columns")
    print(f"{len(corr)} correlated-observation rows")

    cols = [c for c in ("event_starttime", "event_peaktime", "event_endtime",
                        "fl_goescls", "hpc_x", "hpc_y", "ar_noaanum")
            if c in events.columns]
    print(events[cols].head(10) if cols else events.head())
    if not corr.empty:
        print(corr["corr_field"].value_counts())

    events.to_csv("hek_events.csv", index=False)
    if not corr.empty:
        corr.to_csv("hek_correlations.csv", index=False)
