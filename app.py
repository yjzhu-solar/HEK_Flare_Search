"""
Minimal Streamlit front end for the HEK flare query.

Run with:  streamlit run app.py
"""

import sys
from datetime import date, datetime, time
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "example_notebook"))
from hek_query import query  # noqa: E402

# HEK column -> display name, in table order.
COLUMNS = {
    "event_starttime": "Flare start time",
    "event_peaktime": "Flare peak time",
    "event_endtime": "Flare end time",
    "fl_goescls": "GOES class",
    "ar_noaanum": "NOAA active region",
    "gs_movieurl": "Flare movie",
}


def parse_ar(text):
    """'3662', '13662' or 'AR 13662' -> 3662 (HEK drops the leading 1); '' -> None."""
    digits = "".join(ch for ch in str(text) if ch.isdigit())
    if not digits:
        return None
    return int(digits) % 10000


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_events(start, end, result_limit):
    events, _ = query(start, end, result_limit=result_limit)
    return events


def filter_flares(events, ar):
    """Keep flares from the given AR (any of 3662 / 13662), sorted by start time."""
    if events.empty:
        return events
    if ar is not None:
        noaa = pd.to_numeric(events["ar_noaanum"], errors="coerce")
        events = events[noaa % 10000 == ar]
    cols = [c for c in COLUMNS if c in events.columns]
    events = events[cols].sort_values("event_starttime").reset_index(drop=True)
    if "ar_noaanum" in events:
        # Restore the leading 1 dropped by HEK (NOAA numbering passed 10000 in June 2002).
        noaa = pd.to_numeric(events["ar_noaanum"], errors="coerce").astype("Int64")
        wrapped = (noaa > 0) & (noaa < 10000) & (events["event_starttime"] >= "2002-06-14")
        noaa[wrapped] += 10000
        events["ar_noaanum"] = noaa.where(noaa > 0)
    if "gs_movieurl" in events:
        events["gs_movieurl"] = events["gs_movieurl"].replace("", None)
    return events.rename(columns=COLUMNS)


st.set_page_config(page_title="HEK Flare Search", layout="wide")
st.title("HEK Flare Search")
st.caption("SSW Latest Events flares (AIA) from the LMSAL HEK. All times UTC.")

with st.form("search"):
    c1, c2, c3, c4 = st.columns(4)
    start_date = c1.date_input("Start date", date(2024, 5, 8))
    start_time = c2.time_input("Start time", time(0, 0), step=60)
    end_date = c3.date_input("End date", date(2024, 5, 9))
    end_time = c4.time_input("End time", time(12, 30), step=60)
    c5, c6 = st.columns([3, 1])
    ar_text = c5.text_input("NOAA active region (e.g. 3664 or 13664; blank = all)", "13664")
    limit = c6.number_input("Max events", 10, 2000, 500, step=50)
    submitted = st.form_submit_button("Search")

if submitted:
    start = datetime.combine(start_date, start_time)
    end = datetime.combine(end_date, end_time)
    ar = parse_ar(ar_text)
    if end <= start:
        st.error("End time must be after start time.")
        st.stop()
    if ar_text.strip() and ar is None:
        st.error(f"Could not read an AR number from {ar_text!r}.")
        st.stop()

    try:
        with st.spinner("Querying HEK..."):
            events = fetch_events(start, end, int(limit))
    except Exception as e:
        st.error(f"HEK query failed: {e}")
        st.stop()

    if len(events) >= limit:
        st.warning(f"HEK returned the maximum of {limit} events; results may be "
                   "incomplete. Increase 'Max events' or shorten the time window.")

    flares = filter_flares(events, ar)
    label = f"AR {ar_text.strip()}" if ar is not None else "all regions"
    st.subheader(f"{len(flares)} flare(s) for {label}")
    if flares.empty:
        st.info("No flares match the criteria.")
    else:
        st.dataframe(flares, width="stretch", hide_index=True, column_config={
            "Flare movie": st.column_config.LinkColumn(display_text="▶ Movie"),
        })
        st.download_button("Download CSV", flares.to_csv(index=False),
                           file_name="hek_flares.csv", mime="text/csv")
