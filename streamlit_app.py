import io
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# -----------------------------
# Helpers: parsing & normalization
# -----------------------------

DATE_CANDIDATES = ["date", "day", "timestamp", "time", "datetime"]
HABIT_CANDIDATES = ["habit", "name", "title", "habit_name", "habitid", "habit_id", "id"]
VALUE_CANDIDATES = ["value", "completed", "check", "status", "score", "amount"]

def _parse_possible_date(col: pd.Series) -> pd.Series:
    """Best-effort date parsing (supports ms/seconds epoch, common strings)."""
    s = col.copy()
    # If numeric: try epoch
    if pd.api.types.is_integer_dtype(s) or pd.api.types.is_float_dtype(s):
        # Heuristic: ms if big, else seconds
        try:
            s_dt = pd.to_datetime(s, unit="ms", utc=True, errors="coerce")
            if s_dt.notna().mean() > 0.8:
                return s_dt.dt.tz_convert(None).dt.normalize()
        except Exception:
            pass
        try:
            s_dt = pd.to_datetime(s, unit="s", utc=True, errors="coerce")
            if s_dt.notna().mean() > 0.8:
                return s_dt.dt.tz_convert(None).dt.normalize()
        except Exception:
            pass

    # Otherwise try generic parse
    s_dt = pd.to_datetime(s, errors="coerce", utc=False)
    if s_dt.notna().mean() > 0.0:
        return s_dt.dt.normalize()
    return pd.to_datetime(pd.NaT)

def _find_column(df: pd.DataFrame, candidates) -> str | None:
    cols = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in cols:
            return cols[c]
    # fuzzy: exact case-insensitive contains
    for c in df.columns:
        low = c.lower().strip()
        for cand in candidates:
            if cand in low:
                return c
    return None

def _detect_wide_date_col(df: pd.DataFrame) -> str | None:
    """In 'wide' format (one column = one habit), detect the date index/column."""
    # common names
    for c in df.columns:
        if c.lower().strip() in DATE_CANDIDATES:
            return c
    # otherwise if index looks like dates
    if isinstance(df.index, pd.DatetimeIndex):
        return None
    try:
        probe = pd.to_datetime(df.iloc[:, 0], errors="coerce")
        if probe.notna().mean() > 0.5:
            return df.columns[0]
    except Exception:
        pass
    return None

def _tidy_from_long(df: pd.DataFrame) -> pd.DataFrame:
    """Expecting columns containing: date, habit, (optional) value/completed."""
    work = df.copy()
    # standardize column names map
    name_map = {c: c for c in df.columns}
    date_col = _find_column(df, DATE_CANDIDATES)
    habit_col = _find_column(df, HABIT_CANDIDATES)
    value_col = _find_column(df, VALUE_CANDIDATES)

    if date_col is None:
        raise ValueError("Couldn't find a date/timestamp column.")
    if habit_col is None:
        # If no habit col, maybe file represents a single habit (infer from filename upstream)
        habit_col = "__habit__"
        work[habit_col] = "Habit"

    # Parse date
    work["date"] = _parse_possible_date(work[date_col])
    work = work.dropna(subset=["date"])

    # Value/completion handling
    completed = None
    if value_col and value_col in work:
        v = work[value_col]
        # Normalize common truthy/score-like values to 0/1
        if pd.api.types.is_numeric_dtype(v):
            # If score is 0..100, treat >0 as done; else >0 as done
            completed = (v.fillna(0) > 0).astype(int)
        else:
            completed = v.astype(str).str.lower().isin(
                ["1", "true", "yes", "y", "done", "complete", "x", "✓"]
            ).astype(int)
    else:
        # If no explicit value column, assume presence of a row = done
        completed = pd.Series(1, index=work.index)

    tidy = work[[habit_col, "date"]].copy()
    tidy.rename(columns={habit_col: "habit"}, inplace=True)
    tidy["completed"] = completed.values
    return tidy

def _tidy_from_wide(df: pd.DataFrame) -> pd.DataFrame:
    """Rows=dates, columns=habits (0/1/✓/empty)."""
    work = df.copy()
    date_col = _detect_wide_date_col(work)
    if date_col is not None:
        work["__date__"] = _parse_possible_date(work[date_col])
        work = work.dropna(subset=["__date__"])
        work = work.drop(columns=[date_col], errors="ignore")
    else:
        # try to parse first column as date
        work["__date__"] = _parse_possible_date(work.iloc[:, 0])
        work = work.dropna(subset=["__date__"])
        work = work.drop(columns=[work.columns[0]], errors="ignore")

    # melt remaining columns as habits
    habit_cols = [c for c in work.columns if c != "__date__"]
    long = work.melt(id_vars="__date__", value_vars=habit_cols, var_name="habit", value_name="val")
    # Normalize values to completion
    v = long["val"]
    if pd.api.types.is_numeric_dtype(v):
        comp = (v.fillna(0) > 0).astype(int)
    else:
        comp = v.astype(str).str.lower().isin(
            ["1", "true", "yes", "y", "done", "complete", "x", "✓"]
        ).astype(int)
    out = long.rename(columns={"__date__": "date"})[["habit", "date"]].copy()
    out["completed"] = comp.values
    return out

def tidy_from_any(df: pd.DataFrame) -> pd.DataFrame | None:
    """Attempt long-format first, then wide-format."""
    try:
        return _tidy_from_long(df)
    except Exception:
        pass
    try:
        return _tidy_from_wide(df)
    except Exception:
        pass
    return None

def load_loop_zip(file_bytes: bytes) -> list[pd.DataFrame]:
    """Read all CSVs from a zip and return as dataframes."""
    frames = []
    with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:
        for name in z.namelist():
            if name.lower().endswith(".csv"):
                with z.open(name) as f:
                    try:
                        df = pd.read_csv(f)
                    except UnicodeDecodeError:
                        df = pd.read_csv(f, encoding="latin-1")
                # If file appears to be a single-habit 'repetitions' style (date-only), tag it
                if "habit" not in {c.lower() for c in df.columns}:
                    df = df.copy()
                    df["__file_habit__"] = Path(name).stem
                frames.append(df)
    return frames

def unify_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Convert multiple raw frames into one tidy: [habit, date, completed]."""
    tidies = []
    for df in frames:
        t = tidy_from_any(df)
        if t is None:
            # Last attempt: if file had no habit column, attach from filename
            if "__file_habit__" in df.columns:
                t = _tidy_from_long(df.rename(columns={"__file_habit__": "habit"}))
        if t is not None and not t.empty:
            tidies.append(t)
    if not tidies:
        raise ValueError("Could not extract habit data from the provided file(s).")
    out = pd.concat(tidies, ignore_index=True)
    # Normalize habit names
    out["habit"] = out["habit"].astype(str).str.strip()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    # Deduplicate (if multiple rows for same day/habit, treat any completion as done)
    out = (
        out.groupby(["habit", "date"], as_index=False)["completed"]
        .max()
        .sort_values(["habit", "date"])
        .reset_index(drop=True)
    )
    return out

def build_calendar_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Add week & weekday for a calendar heatmap."""
    d = df.copy()
    d["year"] = d["date"].dt.year
    d["week"] = d["date"].dt.isocalendar().week.astype(int)
    d["dow"] = d["date"].dt.weekday  # 0=Mon
    return d

def streaks_for_habit(d: pd.DataFrame) -> dict:
    """Compute current & longest streak for a single habit tidy frame."""
    x = d.sort_values("date").reset_index(drop=True)
    longest = 0
    current = 0
    prev_date = None
    for _, row in x.iterrows():
        if row["completed"] != 1:
            # break streak
            longest = max(longest, current)
            current = 0
            prev_date = row["date"]
            continue
        if prev_date is None:
            current = 1
        else:
            delta = (row["date"] - prev_date).days
            if delta == 1:
                current += 1
            elif delta == 0:
                # same day duplicate (already deduped to max, but guard)
                pass
            else:
                longest = max(longest, current)
                current = 1
        prev_date = row["date"]
    longest = max(longest, current)
    # Current streak counts up to the last date in the data. If dataset doesn't include "today",
    # we still report the streak based on the last consecutive days present.
    return {"longest": int(longest), "current": int(current)}

def completion_rate(d: pd.DataFrame) -> float:
    return float(d["completed"].mean()) if len(d) else 0.0

def ensure_daily_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing days with 0 for each habit (for accurate streaks/heatmaps)."""
    if df.empty:
        return df
    min_d, max_d = df["date"].min(), df["date"].max()
    all_days = pd.date_range(min_d, max_d, freq="D")
    habits = df["habit"].unique()
    grid = (
        pd.MultiIndex.from_product([habits, all_days], names=["habit", "date"])
        .to_frame(index=False)
        .merge(df, how="left", on=["habit", "date"])
        .fillna({"completed": 0})
    )
    grid["completed"] = grid["completed"].astype(int)
    return grid

# -----------------------------
# UI
# -----------------------------

st.set_page_config(page_title="Loop Habits Analyzer", page_icon="✅", layout="wide")
st.title("✅ Loop Habits Analyzer")

with st.sidebar:
    st.markdown("### Upload your data")
    uploaded = st.file_uploader("ZIP or CSV from Loop Habit Tracker", type=["zip", "csv"])
    st.caption("Tip: You can drop the raw Loop export `.zip` directly here.")

# Load data
raw_frames = []
tidy = None
error = None

if uploaded is not None:
    try:
        if uploaded.name.lower().endswith(".zip"):
            raw_frames = load_loop_zip(uploaded.read())
        else:
            # single CSV
            b = uploaded.read()
            try:
                df = pd.read_csv(io.BytesIO(b))
            except UnicodeDecodeError:
                df = pd.read_csv(io.BytesIO(b), encoding="latin-1")
            raw_frames = [df]
        tidy = unify_frames(raw_frames)
    except Exception as e:
        error = str(e)

# Demo data if nothing uploaded or parse failed
if tidy is None:
    if error:
        st.warning(f"Could not parse the file: {error}")
    st.info("No file? Using demo data so you can explore the app.")
    rng = pd.date_range(datetime.now().date() - pd.Timedelta(days=180), periods=181, freq="D")
    habits = ["Meditate", "Workout", "Read", "Sleep ≥7h"]
    rng_df = (
        pd.MultiIndex.from_product([habits, rng], names=["habit", "date"])
        .to_frame(index=False)
    )
    np.random.seed(7)
    # create semi-realistic patterns
    probs = {"Meditate": 0.7, "Workout": 0.45, "Read": 0.6, "Sleep ≥7h": 0.55}
    rng_df["completed"] = rng_df["habit"].map(probs)
    rng_df["completed"] = (np.random.rand(len(rng_df)) < rng_df["completed"]).astype(int)
    tidy = rng_df

# Normalize & enrich
tidy["date"] = pd.to_datetime(tidy["date"]).dt.normalize()
full = ensure_daily_grid(tidy)
min_date, max_date = full["date"].min(), full["date"].max()

# Sidebar filters
with st.sidebar:
    st.markdown("### Filters")
    habits_all = sorted(full["habit"].unique().tolist())
    selected_habits = st.multiselect("Habits", habits_all, default=habits_all[: min(5, len(habits_all))])
    date_range = st.date_input("Date range", value=(min_date.date(), max_date.date()))
    agg_unit = st.selectbox("Aggregate", ["Daily", "Weekly", "Monthly"], index=1)

# Apply filters
f = full.copy()
if selected_habits:
    f = f[f["habit"].isin(selected_habits)]
if isinstance(date_range, tuple) and len(date_range) == 2:
    d0, d1 = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    f = f[(f["date"] >= d0) & (f["date"] <= d1)]

# -----------------------------
# Top KPIs
# -----------------------------
col1, col2, col3, col4 = st.columns(4)
overall_rate = completion_rate(f)
total_habits = f["habit"].nunique()
total_days = f["date"].nunique()

# Longest streak across selected
streak_longest = 0
current_best = 0
if total_habits > 0:
    for h in f["habit"].unique():
        s = streaks_for_habit(f[f["habit"] == h].sort_values("date"))
        streak_longest = max(streak_longest, s["longest"])
        current_best = max(current_best, s["current"])

col1.metric("Habits", total_habits)
col2.metric("Days", total_days)
col3.metric("Overall completion", f"{overall_rate*100:.1f}%")
col4.metric("Longest streak (any habit)", streak_longest)

st.caption(f"Data window: {min_date.date()} → {max_date.date()}")

# -----------------------------
# Charts
# -----------------------------
# 1) Completion over time (aggregated)
df_time = f.copy()
if agg_unit == "Weekly":
    df_time["bucket"] = df_time["date"].dt.to_period("W").apply(lambda r: r.start_time)
elif agg_unit == "Monthly":
    df_time["bucket"] = df_time["date"].dt.to_period("M").dt.to_timestamp()
else:
    df_time["bucket"] = df_time["date"]

df_time = (
    df_time.groupby("bucket", as_index=False)["completed"]
    .mean()
    .rename(columns={"completed": "completion_rate"})
)

line = (
    alt.Chart(df_time)
    .mark_line(point=True)
    .encode(
        x=alt.X("bucket:T", title="Date"),
        y=alt.Y("completion_rate:Q", title="Completion rate", axis=alt.Axis(format="%")),
        tooltip=[alt.Tooltip("bucket:T", title="Date"), alt.Tooltip("completion_rate:Q", title="Completion", format=".1%")],
    )
    .properties(height=280)
)
st.subheader("Completion trend")
st.altair_chart(line, use_container_width=True)

# 2) Habit leaderboard
leader = (
    f.groupby("habit", as_index=False)["completed"]
    .mean()
    .sort_values("completed", ascending=False)
    .rename(columns={"completed": "rate"})
)
bar = (
    alt.Chart(leader)
    .mark_bar()
    .encode(
        x=alt.X("rate:Q", title="Completion rate", axis=alt.Axis(format="%")),
        y=alt.Y("habit:N", sort="-x", title="Habit"),
        tooltip=[alt.Tooltip("habit:N"), alt.Tooltip("rate:Q", format=".1%")],
    )
    .properties(height=28 * max(4, len(leader)))
)
st.subheader("Habit leaderboard")
st.altair_chart(bar, use_container_width=True)

# 3) Calendar heatmap (selected habit)
st.subheader("Calendar heatmap")
habit_for_heatmap = st.selectbox("Pick a habit for calendar heatmap", sorted(f["habit"].unique()))
cal = build_calendar_grid(f[f["habit"] == habit_for_heatmap])

# Build a compact calendar: week vs weekday
cal_heat = (
    alt.Chart(cal)
    .mark_rect()
    .encode(
        x=alt.X("week:O", title="ISO week"),
        y=alt.Y("dow:O", title="Day of week", sort=[0,1,2,3,4,5,6],
                axis=alt.Axis(values=[0,1,2,3,4,5,6],
                              labelExpr="['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][datum.value]")),
        color=alt.Color("completed:Q", title="Done", scale=alt.Scale(domain=[0,1], range=["#e6e6e6","#4CAF50"])),
        tooltip=[alt.Tooltip("date:T"), alt.Tooltip("completed:Q")],
    )
    .properties(height=180)
)
st.altair_chart(cal_heat, use_container_width=True)

# 4) Weekday pattern (what days are strongest?)
weekday = (
    f.assign(weekday=f["date"].dt.day_name())
    .groupby("weekday", as_index=False)["completed"].mean()
)
weekday["weekday_num"] = pd.to_datetime(weekday["weekday"], format="%A").dt.weekday
weekday = weekday.sort_values("weekday_num")
wd_bar = (
    alt.Chart(weekday)
    .mark_bar()
    .encode(
        x=alt.X("weekday:N", title="Weekday"),
        y=alt.Y("completed:Q", title="Completion rate", axis=alt.Axis(format="%")),
        tooltip=[alt.Tooltip("weekday:N"), alt.Tooltip("completed:Q", format=".1%")],
    )
    .properties(height=220)
)
st.subheader("Weekday pattern")
st.altair_chart(wd_bar, use_container_width=True)

# -----------------------------
# Per-habit details & streaks
# -----------------------------
st.subheader("Per-habit streaks & stats")
rows = []
for h in sorted(f["habit"].unique()):
    dfh = f[f["habit"] == h]
    s = streaks_for_habit(dfh)
    rows.append(
        {
            "Habit": h,
            "Completion": dfh["completed"].mean(),
            "Current streak (days)": s["current"],
            "Longest streak (days)": s["longest"],
            "Active days": int(dfh["completed"].sum()),
            "Total days": int(dfh.shape[0]),
        }
    )
table = pd.DataFrame(rows).sort_values("Completion", ascending=False)
table["Completion"] = (table["Completion"] * 100).round(1).astype(str) + "%"
st.dataframe(table, use_container_width=True, hide_index=True)

# -----------------------------
# Export
# -----------------------------
st.subheader("Download tidy data")
csv = f[["habit", "date", "completed"]].sort_values(["habit", "date"])
st.download_button(
    "Download as CSV",
    data=csv.to_csv(index=False).encode("utf-8"),
    file_name="loop_habits_tidy.csv",
    mime="text/csv",
)
st.caption("Tidy columns: habit, date (YYYY-MM-DD), completed (0/1).")

# Footer
st.markdown("---")
st.markdown("Built for Loop Habit Tracker exports. Upload your `.zip` or `.csv` and explore!")
