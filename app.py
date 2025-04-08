import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd

st.set_page_config(layout="wide")

# --- Data Preparation ---
data_dsm_dsl = {
    'Entity': ['BP', 'TK', 'PP', 'UB'],
    'DSM Progress': [80, 70, 90, 85],
    'DSL Progress': [75, 65, 80, 80]
}

data_latest_progress = {
    'Year': list(range(2020, 2026)) * 4,
    'Entity': ['BP'] * 6 + ['TK'] * 6 + ['PP'] * 6 + ['UB'] * 6,
    'Type': ['DSM'] * 24,
    'Progress': [
        60, 65, 70, 75, 80, 85,  # BP
        50, 55, 60, 65, 70, 75,  # TK
        70, 75, 80, 85, 90, 95,  # PP
        65, 70, 75, 80, 85, 90   # UB
    ]
}

data_latest_progress_dsl = {
    'Year': list(range(2020, 2026)) * 4,
    'Entity': ['BP'] * 6 + ['TK'] * 6 + ['PP'] * 6 + ['UB'] * 6,
    'Type': ['DSL'] * 24,
    'Progress': [
        55, 60, 65, 70, 75, 80,  # BP
        45, 50, 55, 60, 65, 70,  # TK
        65, 70, 75, 80, 85, 90,  # PP
        60, 65, 70, 75, 80, 85   # UB
    ]
}

data_entity_progress = {
    'Entity': ['BP', 'TK', 'PP', 'UB'],
    'Progress %': [80, 70, 90, 85]
}

# Convert to DataFrames
df_dsm_dsl = pd.DataFrame(data_dsm_dsl)
df_entity_progress = pd.DataFrame(data_entity_progress)
df_combined_progress = pd.concat([
    pd.DataFrame(data_latest_progress),
    pd.DataFrame(data_latest_progress_dsl)
])

# --- Dashboard ---
st.title("📊 DSM & DSL Dashboard")

tab_overview, tab_time, tab_compare = st.tabs(["🔍 Overview", "📈 Progress Over Time", "📊 Comparison"])

# --- Tab 1: Overview ---
with tab_overview:
    st.subheader("Overall Progress Summary")

    # KPI Cards
    avg_dsm = df_dsm_dsl['DSM Progress'].mean()
    avg_dsl = df_dsm_dsl['DSL Progress'].mean()
    top_entity = df_entity_progress.loc[df_entity_progress['Progress %'].idxmax()]

    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
    col_kpi1.metric("📈 Avg DSM Progress", f"{avg_dsm:.1f}%")
    col_kpi2.metric("📉 Avg DSL Progress", f"{avg_dsl:.1f}%")
    col_kpi3.metric("🏆 Top Entity", f"{top_entity['Entity']} ({top_entity['Progress %']}%)")

    col1, col2 = st.columns([1, 1])

    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(df_dsm_dsl['Entity'], df_dsm_dsl['DSM Progress'], label='DSM')
        ax.bar(df_dsm_dsl['Entity'], df_dsm_dsl['DSL Progress'],
               bottom=df_dsm_dsl['DSM Progress'], label='DSL')
        ax.set_xlabel('Entity')
        ax.set_ylabel('Progress (%)')
        ax.set_title("DSM + DSL Stacked Progress")
        ax.legend()
        st.pyplot(fig)

    with col2:
        fig2 = px.bar(df_entity_progress, x='Entity', y='Progress %',
                      title="Overall Progress by Entity", height=400)
        st.plotly_chart(fig2, use_container_width=True)

# --- Tab 2: Progress Over Time ---
with tab_time:
    st.subheader("📈 Progress Over Time")
    selected_entity = st.selectbox("Select an Entity:", df_dsm_dsl['Entity'].unique(), key="time_entity")

    filtered_df = df_combined_progress[df_combined_progress['Entity'] == selected_entity]

    fig_line = px.line(filtered_df,
                       x='Year', y='Progress',
                       color='Type',
                       markers=True,
                       title=f"{selected_entity} - DSM vs DSL Progress Over Time",
                       labels={'Progress': 'Progress (%)'},
                       height=500)
    st.plotly_chart(fig_line, use_container_width=True)

# --- Tab 3: Comparison ---
with tab_compare:
    st.subheader("📊 DSM vs DSL Comparison")
    selected_entity_bar = st.selectbox("Select an Entity:", df_dsm_dsl['Entity'].unique(), key="compare_entity")

    filtered_df_bar = df_combined_progress[df_combined_progress['Entity'] == selected_entity_bar]

    fig_bar = px.bar(filtered_df_bar,
                     x='Year', y='Progress', color='Type',
                     barmode='group',
                     title=f"{selected_entity_bar} - DSM vs DSL Grouped Comparison",
                     labels={'Progress': 'Progress (%)'},
                     height=500)
    st.plotly_chart(fig_bar, use_container_width=True)
