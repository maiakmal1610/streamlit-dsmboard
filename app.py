import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import numpy as np

# Set page configuration
st.set_page_config(layout="wide", page_title="DSM & DSL Dashboard")

# Function to load data - better for organization and error handling
@st.cache_data
def load_data():
    try:
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

        # Convert to DataFrames
        df_dsm_dsl = pd.DataFrame(data_dsm_dsl)
        
        # Create a derived dataframe for entity total progress (average of DSM and DSL)
        df_entity_progress = pd.DataFrame({
            'Entity': df_dsm_dsl['Entity'],
            'Progress %': np.round((df_dsm_dsl['DSM Progress'] + df_dsm_dsl['DSL Progress']) / 2, 1)
        })
        
        # Combine time series data
        df_combined_progress = pd.concat([
            pd.DataFrame(data_latest_progress),
            pd.DataFrame(data_latest_progress_dsl)
        ])
        
        return df_dsm_dsl, df_entity_progress, df_combined_progress
    
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

# Load data
df_dsm_dsl, df_entity_progress, df_combined_progress = load_data()

# --- Dashboard ---
st.title("📊 DSM & DSL Dashboard")

# Create tabs
tab_overview, tab_time, tab_compare = st.tabs(["🔍 Overview", "📈 Progress Over Time", "📊 Comparison"])

# --- Tab 1: Overview ---
with tab_overview:
    st.subheader("Overall Progress Summary")

    # Error handling
    if df_dsm_dsl is not None and df_entity_progress is not None:
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
            # Create a dataframe format suitable for side-by-side bars in Plotly
            df_comparison = pd.melt(
                df_dsm_dsl, 
                id_vars=['Entity'], 
                value_vars=['DSM Progress', 'DSL Progress'],
                var_name='Type', 
                value_name='Progress'
            )
            # Clean up the "Type" column
            df_comparison['Type'] = df_comparison['Type'].str.replace(' Progress', '')
            
            # Using Plotly for consistency
            fig1 = px.bar(
                df_comparison,
                x='Entity', 
                y='Progress',
                color='Type',
                barmode='group',
                title="DSM vs DSL Progress by Entity",
                labels={'Progress': 'Progress (%)'},
                height=400
            )
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            fig2 = px.bar(
                df_entity_progress, 
                x='Entity', 
                y='Progress %',
                title="Overall Progress by Entity", 
                height=400,
                color='Entity'
            )
            fig2.update_layout(showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.error("Data not available. Please check the data sources.")

# --- Tab 2: Progress Over Time ---
with tab_time:
    st.subheader("📈 Progress Over Time")
    
    if df_combined_progress is not None:
        # Adding an "All Entities" option
        entity_options = ['All Entities'] + list(df_dsm_dsl['Entity'].unique())
        selected_entity = st.selectbox("Select an Entity:", entity_options, key="time_entity")

        # Filter based on selection
        if selected_entity == 'All Entities':
            filtered_df = df_combined_progress
            title = "All Entities - DSM vs DSL Progress Over Time"
        else:
            filtered_df = df_combined_progress[df_combined_progress['Entity'] == selected_entity]
            title = f"{selected_entity} - DSM vs DSL Progress Over Time"

        # Group by Year and Type to get averages when showing all entities
        if selected_entity == 'All Entities':
            chart_data = filtered_df.groupby(['Year', 'Type'], as_index=False)['Progress'].mean()
        else:
            chart_data = filtered_df

        fig_line = px.line(
            chart_data,
            x='Year', 
            y='Progress',
            color='Type',
            markers=True,
            title=title,
            labels={'Progress': 'Progress (%)'},
            height=500
        )
        
        # Add dots at data points
        fig_line.update_traces(mode='lines+markers')
        
        # Improve layout
        fig_line.update_layout(
            xaxis=dict(tickmode='linear'),
            yaxis=dict(range=[0, 100]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.error("Time series data not available.")

# --- Tab 3: Comparison ---
with tab_compare:
    st.subheader("📊 DSM vs DSL Comparison")
    
    if df_combined_progress is not None:
        # Adding an "All Entities" option
        entity_options = ['All Entities'] + list(df_dsm_dsl['Entity'].unique())
        selected_entity_bar = st.selectbox("Select an Entity:", entity_options, key="compare_entity")

        # Year selection
        years = sorted(df_combined_progress['Year'].unique())
        selected_years = st.multiselect(
            "Select Years to Compare:", 
            years, 
            default=[years[-2], years[-1]]  # Default to last two years
        )
        
        if not selected_years:
            st.warning("Please select at least one year to display data.")
        else:
            # Filter based on selections
            if selected_entity_bar == 'All Entities':
                filtered_df_bar = df_combined_progress[df_combined_progress['Year'].isin(selected_years)]
                title = f"All Entities - DSM vs DSL Comparison ({', '.join(map(str, selected_years))})"
                
                # Group by Year and Type for "All Entities"
                chart_data = filtered_df_bar.groupby(['Year', 'Type'], as_index=False)['Progress'].mean()
            else:
                filtered_df_bar = df_combined_progress[
                    (df_combined_progress['Entity'] == selected_entity_bar) & 
                    (df_combined_progress['Year'].isin(selected_years))
                ]
                title = f"{selected_entity_bar} - DSM vs DSL Comparison ({', '.join(map(str, selected_years))})"
                chart_data = filtered_df_bar
            
            if not chart_data.empty:
                fig_bar = px.bar(
                    chart_data,
                    x='Year', 
                    y='Progress', 
                    color='Type',
                    barmode='group',
                    title=title,
                    labels={'Progress': 'Progress (%)'},
                    height=500
                )
                
                # Format the chart
                fig_bar.update_layout(
                    yaxis=dict(range=[0, 100]),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.warning("No data available for the selected filters.")
    else:
        st.error("Comparison data not available.")

# Add footer information
st.markdown("---")
st.markdown("**Dashboard created with Streamlit** | Last updated: April 2025")
