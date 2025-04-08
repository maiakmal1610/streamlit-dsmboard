import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd

# Sample data (same as before)
data_dsm_dsl = {
    'Entity': ['BP', 'TK', 'PP', 'UB'],
    'DSM Progress': [80, 70, 90, 85],
    'DSL Progress': [75, 65, 80, 80]
}

data_latest_progress = {
    'Year': [2020, 2021, 2022, 2023, 2024, 2025,
             2020, 2021, 2022, 2023, 2024, 2025,
             2020, 2021, 2022, 2023, 2024, 2025,
             2020, 2021, 2022, 2023, 2024, 2025],
    'Entity': ['BP'] * 6 + ['TK'] * 6 + ['PP'] * 6 + ['UB'] * 6,
    'Type': ['DSM'] * 24,
    'Progress': [60, 65, 70, 75, 80, 85,  # BP
                 50, 55, 60, 65, 70, 75,  # TK
                 70, 75, 80, 85, 90, 95,  # PP
                 65, 70, 75, 80, 85, 90]   # UB
}

data_latest_progress_dsl = {
    'Year': [2020, 2021, 2022, 2023, 2024, 2025,
             2020, 2021, 2022, 2023, 2024, 2025,
             2020, 2021, 2022, 2023, 2024, 2025,
             2020, 2021, 2022, 2023, 2024, 2025],
    'Entity': ['BP'] * 6 + ['TK'] * 6 + ['PP'] * 6 + ['UB'] * 6,
    'Type': ['DSL'] * 24,
    'Progress': [55, 60, 65, 70, 75, 80,  # BP
                 45, 50, 55, 60, 65, 70,  # TK
                 65, 70, 75, 80, 85, 90,  # PP
                 60, 65, 70, 75, 80, 85]   # UB
}

# Convert data to DataFrame
df_dsm_dsl = pd.DataFrame(data_dsm_dsl)
df_entity_progress = pd.DataFrame(data_dsm_dsl)

# Combine DSM and DSL data
data_latest_progress_combined = pd.concat([
    pd.DataFrame(data_latest_progress),
    pd.DataFrame(data_latest_progress_dsl)
])

# Streamlit app
st.title("Dashboard")

# Tab for combined view
tab1, tab2, tab3 = st.tabs(["Combined View", "Latest Progress Comparison", "Bar Chart for Comparison"])

with tab1:
    st.subheader("Current Progress - DSM vs DSL")

    # Dropdown for choosing Entity
    selected_entity = st.selectbox("Choose Entity", df_dsm_dsl['Entity'].unique())

    # Dropdown for choosing Date
    selected_year = st.selectbox("Choose Year", sorted(list(data_latest_progress['Year'].unique())))


    # Filter data based on the selected entity and year
    dsm_data = df_dsm_dsl[df_dsm_dsl['Entity'] == selected_entity]
    dsl_data = df_entity_progress[df_entity_progress['Entity'] == selected_entity]

    # Get the latest progress for DSM and DSL for the selected year
    latest_dsm = data_latest_progress_combined[(data_latest_progress_combined['Entity'] == selected_entity) &
                                               (data_latest_progress_combined['Year'] == selected_year) &
                                               (data_latest_progress_combined['Type'] == 'DSM')]['Progress'].values[0]
    latest_dsl = data_latest_progress_combined[(data_latest_progress_combined['Entity'] == selected_entity) &
                                               (data_latest_progress_combined['Year'] == selected_year) &
                                               (data_latest_progress_combined['Type'] == 'DSL')]['Progress'].values[0]

    # Plotting the horizontal bar chart for DSM vs DSL
    fig, ax = plt.subplots(figsize=(8, 4))

    # Bar chart with horizontal bars
    ax.barh(selected_entity, latest_dsm, color='blue', label='DSM', align='center')
    ax.barh(selected_entity, latest_dsl, color='orange', left=latest_dsm, label='DSL', align='center')

    # Labels and title
    ax.set_xlabel('Progress (%)')
    ax.set_title(f"DSM vs DSL Progress for {selected_entity} in {selected_year}")
    ax.legend()

    # Display the plot
    st.pyplot(fig)

with tab2:
    st.write("### Latest Progress Comparison")
    fig_line = px.line(data_latest_progress_combined, x='Year', y='Progress', color='Type', facet_row='Entity')
    st.plotly_chart(fig_line, use_container_width=True)

with tab3:
    st.write("### Bar Chart for Comparison")
    fig_bar = px.bar(
        data_latest_progress_combined,
        x='Year',
        y='Progress',
        color='Type',
        barmode='group',  
        facet_row='Entity',  
        title="DSM vs DSL Progress Over Time",
        labels={'Progress': 'Progress (%)'}
    )
    st.plotly_chart(fig_bar, use_container_width=True)
