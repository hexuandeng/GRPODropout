from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from live_metric_cache import DEFAULT_LOG_DIR, DEFAULT_ROLLOUT_ROOT, update_cache


DEFAULT_OUTPUT_CSV = Path(__file__).resolve().parent / "outputs" / "live_step_summary.csv"


st.set_page_config(page_title="Live GRPO Metrics", layout="wide")
st.title("Live GRPO Metrics")

with st.sidebar:
    st.header("Inputs")
    log_dir = Path(st.text_input("Log directory", str(DEFAULT_LOG_DIR)))
    rollout_root = Path(st.text_input("Rollout root", str(DEFAULT_ROLLOUT_ROOT)))
    output_csv = Path(st.text_input("Cache CSV", str(DEFAULT_OUTPUT_CSV)))
    refresh_seconds = st.number_input("Auto refresh seconds", min_value=0, max_value=300, value=30, step=5)
    rebuild = st.button("Refresh now", type="primary")

if refresh_seconds:
    try:
        from streamlit_autorefresh import st_autorefresh

        st_autorefresh(interval=int(refresh_seconds) * 1000, key="live_refresh")
    except Exception:
        st.caption("Install streamlit-autorefresh for browser-side auto refresh, or use Refresh now.")

if rebuild or refresh_seconds or not output_csv.exists():
    with st.spinner("Updating metric cache from logs and rollout JSONL files..."):
        rows = update_cache(log_dir=log_dir, rollout_root=rollout_root, output_csv=output_csv)
    if rebuild:
        st.success(f"Updated cache: {len(rows)} rows")

if not output_csv.exists():
    st.warning("No cache CSV yet. Click Refresh now.")
    st.stop()

df = pd.read_csv(output_csv)
if df.empty:
    st.warning("Cache CSV is empty.")
    st.stop()

df["series"] = (
    df.get("run_name", "").fillna("").astype(str)
    + " | "
    + df.get("dataset", "").fillna("").astype(str)
    + " | "
    + df.get("method", "").fillna("").astype(str)
)
df["step"] = pd.to_numeric(df["step"], errors="coerce")
df = df.dropna(subset=["step"]).sort_values(["series", "step"])

with st.sidebar:
    st.header("Filters")
    if "source_log" in df.columns:
        log_options = sorted([item for item in df["source_log"].dropna().astype(str).unique().tolist() if item])
        selected_logs = st.multiselect("Source logs", log_options, default=[])
        if selected_logs:
            df = df[df["source_log"].astype(str).isin(selected_logs)].copy()
    series_options = sorted(df["series"].dropna().unique().tolist())
    selected_series = st.multiselect("Series", series_options, default=series_options[:6])
    plot_mode = st.radio(
        "Plot mode",
        ["Separate charts", "Overlay comparison"],
        index=0,
        help="Separate charts draws one small chart per training run. Overlay comparison puts selected runs in one chart.",
    )
    if selected_series:
        df_view = df[df["series"].isin(selected_series)].copy()
    else:
        df_view = df.copy()

st.subheader("Coverage")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", len(df_view))
c2.metric("Log steps", int(df_view.get("log_has_step", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()))
c3.metric("Rollout steps", int(df_view.get("rollout_has_step", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()))
c4.metric("Series", df_view["series"].nunique())

missing_log = df_view[
    (~df_view.get("log_has_step", pd.Series(False, index=df_view.index)).fillna(False).astype(bool))
    & (df_view.get("rollout_has_step", pd.Series(False, index=df_view.index)).fillna(False).astype(bool))
]
if not missing_log.empty:
    st.caption(f"{len(missing_log)} rows are filled from rollout because the corresponding log step is missing.")


def plot_metric(metric: str, title: str) -> None:
    if metric not in df_view.columns:
        st.info(f"Missing metric: {metric}")
        return
    plot_df = df_view[["step", "series", metric]].copy()
    plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")
    plot_df = plot_df.dropna(subset=[metric])
    if plot_df.empty:
        st.info(f"No numeric values for {metric}")
        return
    st.markdown(f"**{title}**")
    if plot_mode == "Overlay comparison":
        st.line_chart(plot_df, x="step", y=metric, color="series", height=320)
        return

    series_names = sorted(plot_df["series"].dropna().unique().tolist())
    for start in range(0, len(series_names), 2):
        cols = st.columns(2)
        for col, series_name in zip(cols, series_names[start : start + 2], strict=False):
            item_df = plot_df[plot_df["series"] == series_name]
            with col:
                st.caption(series_name)
                st.line_chart(item_df, x="step", y=metric, height=220)


left, right = st.columns(2)
with left:
    plot_metric("actor/entropy", "Actor Entropy")
    plot_metric("critic/score/mean", "Train Score Mean")
    plot_metric("val-aux/amc23/score/mean@16", "Validation Score Mean@16")
    plot_metric("selection/group_cov_before_mean", "Covariance: group_cov_before_select")
with right:
    plot_metric("critic/rewards/mean", "Train Reward Mean")
    plot_metric("val-aux/amc23/reward/mean@16", "Validation Reward Mean@16")
    plot_metric("selected_entropy_push_sum_frozen_total", "Selected Samples Entropy Increase")
    plot_metric("selection/group_selected_entropy_push_mean", "Selected Entropy Push Mean")

st.subheader("Selection Counts")
count_cols = [
    "selected_adv_pos_count",
    "selected_adv_neg_count",
    "deleted_adv_pos_count",
    "deleted_adv_neg_count",
]
available_count_cols = [col for col in count_cols if col in df_view.columns]
if available_count_cols:
    long_df = df_view[["step", "series", *available_count_cols]].melt(
        id_vars=["step", "series"], var_name="count_type", value_name="count"
    )
    long_df["count"] = pd.to_numeric(long_df["count"], errors="coerce")
    long_df = long_df.dropna(subset=["count"])
    selected_count_type = st.selectbox("Count type", available_count_cols)
    count_plot_df = long_df[long_df["count_type"] == selected_count_type]
    if plot_mode == "Overlay comparison":
        st.line_chart(count_plot_df, x="step", y="count", color="series", height=320)
    else:
        series_names = sorted(count_plot_df["series"].dropna().unique().tolist())
        for start in range(0, len(series_names), 2):
            cols = st.columns(2)
            for col, series_name in zip(cols, series_names[start : start + 2], strict=False):
                item_df = count_plot_df[count_plot_df["series"] == series_name]
                with col:
                    st.caption(series_name)
                    st.line_chart(item_df, x="step", y="count", height=220)
else:
    st.info("Selection count columns are available only for rollout-backed steps.")

st.subheader("Raw Data")
shown_cols = [
    "series",
    "run_name",
    "step",
    "metric_source",
    "actor/entropy",
    "critic/score/mean",
    "critic/rewards/mean",
    "val-aux/amc23/score/mean@16",
    "val-aux/amc23/reward/mean@16",
    "selection/group_cov_before_mean",
    "selected_entropy_push_sum_frozen_total",
]
shown_cols = [col for col in shown_cols if col in df_view.columns]
st.dataframe(df_view[shown_cols].tail(300), use_container_width=True, height=420)
