import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime

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

        # Create exact dates for more granular date selection
        # Create a list of dates (monthly for each year)
        all_dates = []
        for year in range(2020, 2026):
            for month in range(1, 13):
                # Skip future months in 2025
                if year == 2025 and month > 4:  # Assuming current date is April 2025
                    continue
                all_dates.append(datetime(year, month, 1))
        
        # Create time-based data with proper dates
        time_based_data = []
        
        entities = ['BP', 'TK', 'PP', 'UB']
        
        # Starting values
        start_values_dsm = {'BP': 55, 'TK': 45, 'PP': 65, 'UB': 60}
        start_values_dsl = {'BP': 50, 'TK': 40, 'PP': 60, 'UB': 55}
        
        # Generate monthly progress data
        for entity in entities:
            dsm_progress = start_values_dsm[entity]
            dsl_progress = start_values_dsl[entity]
            
            for date in all_dates:
                # Add small monthly increases with some randomness
                dsm_progress += np.random.uniform(0.3, 0.7)
                dsl_progress += np.random.uniform(0.2, 0.6)
                
                # Add DSM record
                time_based_data.append({
                    'Date': date,
                    'Year': date.year,
                    'Month': date.month,
                    'Entity': entity,
                    'Type': 'DSM',
                    'Progress': min(round(dsm_progress, 1), 100)  # Cap at 100%
                })
                
                # Add DSL record
                time_based_data.append({
                    'Date': date,
                    'Year': date.year,
                    'Month': date.month,
                    'Entity': entity,
                    'Type': 'DSL',
                    'Progress': min(round(dsl_progress, 1), 100)  # Cap at 100%
                })
        
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
        
        # Add the time-based detailed dataframe
        df_time_based = pd.DataFrame(time_based_data)
        
        return df_dsm_dsl, df_entity_progress, df_combined_progress, df_time_based
    
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None

# Function to get data for a specific date
def get_data_for_date(df, selected_date):
    # Filter data up to the selected date
    filtered_df = df[df['Date'] <= selected_date]
    
    # Get the latest record for each Entity and Type combination
    latest_data = filtered_df.sort_values('Date').groupby(['Entity', 'Type']).last().reset_index()
    
    # Create a comparison dataframe in the format we need
    dsm_data = latest_data[latest_data['Type'] == 'DSM'].copy()
    dsl_data = latest_data[latest_data['Type'] == 'DSL'].copy()
    
    # Rename for merging
    dsm_data = dsm_data.rename(columns={'Progress': 'DSM Progress'})
    dsl_data = dsl_data.rename(columns={'Progress': 'DSL Progress'})
    
    # Keep only needed columns
    dsm_data = dsm_data[['Entity', 'DSM Progress']]
    dsl_data = dsl_data[['Entity', 'DSL Progress']]
    
    # Merge
    result_df = pd.merge(dsm_data, dsl_data, on='Entity')
    
    return result_df

# Load data
df_dsm_dsl, df_entity_progress, df_combined_progress, df_time_based = load_data()

# --- Dashboard ---
st.title("📊 DSM & DSL Dashboard")

# Create tabs
tab_overview, tab_time, tab_compare = st.tabs(["🔍 Overview", "📈 Progress Over Time", "📊 Comparison"])

# --- Tab 1: Overview ---
with tab_overview:
    st.subheader("Overall Progress Summary")

    # Error handling
    if df_dsm_dsl is not None and df_entity_progress is not None and df_time_based is not None:
        # Date selection
        col_view_type, col_date = st.columns([1, 2])
        
        with col_view_type:
            view_type = st.radio("View Progress:", ["Latest Progress", "Historical Progress"])
        
        if view_type == "Historical Progress":
            with col_date:
                # Get min and max dates from the dataframe
                min_date = df_time_based['Date'].min()
                max_date = df_time_based['Date'].max()
                
                # Default to 6 months ago
                default_date = max_date - pd.DateOffset(months=6)
                if default_date < min_date:
                    default_date = min_date
                
                selected_date = st.date_input(
                    "Select Date:", 
                    value=default_date,
                    min_value=min_date.date(),
                    max_value=max_date.date()
                )
                
                # Convert to datetime for filtering
                selected_datetime = datetime.combine(selected_date, datetime.min.time())
                
                # Get data for the selected date
                historical_df = get_data_for_date(df_time_based, selected_datetime)
                
                # Use the historical data
                working_df = historical_df
                
                # Create entity progress dataframe for historical data
                working_entity_progress = pd.DataFrame({
                    'Entity': working_df['Entity'],
                    'Progress %': np.round((working_df['DSM Progress'] + working_df['DSL Progress']) / 2, 1)
                })
                
                st.info(f"Showing progress as of {selected_date.strftime('%B %d, %Y')}")
        else:
            # Use the latest data
            working_df = df_dsm_dsl
            working_entity_progress = df_entity_progress
            
        # KPI Cards
        avg_dsm = working_df['DSM Progress'].mean()
        avg_dsl = working_df['DSL Progress'].mean()
        top_entity = working_entity_progress.loc[working_entity_progress['Progress %'].idxmax()]

        col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
        col_kpi1.metric("📈 Avg DSM Progress", f"{avg_dsm:.1f}%")
        col_kpi2.metric("📉 Avg DSL Progress", f"{avg_dsl:.1f}%")
        col_kpi3.metric("🏆 Top Entity", f"{top_entity['Entity']} ({top_entity['Progress %']}%)")

        col1, col2 = st.columns([1, 1])

        with col1:
            # Create a dataframe format suitable for side-by-side bars in Plotly
            df_comparison = pd.melt(
                working_df, 
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
            # Set y-axis to go from 0 to 100
            fig1.update_layout(yaxis=dict(range=[0, 100]))
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            fig2 = px.bar(
                working_entity_progress, 
                x='Entity', 
                y='Progress %',
                title="Overall Progress by Entity", 
                height=400,
                color='Entity'
            )
            fig2.update_layout(showlegend=False, yaxis=dict(range=[0, 100]))
            st.plotly_chart(fig2, use_container_width=True)
            
        # If showing historical data, add comparison with current
        if view_type == "Historical Progress":
            st.subheader("Historical vs. Current Comparison")
            
            # Create comparison dataframe
            hist_values = {entity: progress for entity, progress in zip(working_entity_progress['Entity'], working_entity_progress['Progress %'])}
            current_values = {entity: progress for entity, progress in zip(df_entity_progress['Entity'], df_entity_progress['Progress %'])}
            
            comparison_data = []
            for entity in hist_values.keys():
                comparison_data.append({
                    'Entity': entity,
                    'Historical': hist_values[entity],
                    'Current': current_values[entity],
                    'Change': current_values[entity] - hist_values[entity]
                })
            
            df_comparison = pd.DataFrame(comparison_data)
            
            # Display comparison
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Create comparison chart
                df_melt = pd.melt(
                    df_comparison,
                    id_vars=['Entity'],
                    value_vars=['Historical', 'Current'],
                    var_name='Period',
                    value_name='Progress'
                )
                
                fig = px.bar(
                    df_melt,
                    x='Entity',
                    y='Progress',
                    color='Period',
                    barmode='group',
                    title=f"Progress Comparison: {selected_date.strftime('%B %Y')} vs Current",
                    height=400
                )
                fig.update_layout(yaxis=dict(range=[0, 100]))
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Display comparison table with metrics
                st.markdown("### Progress Change")
                
                for _, row in df_comparison.iterrows():
                    delta_color = "normal"
                    if row['Change'] > 5:
                        delta_color = "good"
                    elif row['Change'] < 0:
                        delta_color = "inverse"
                        
                    st.metric(
                        f"{row['Entity']}",
                        f"{row['Current']:.1f}%",
                        f"{row['Change']:.1f}%",
                        delta_color=delta_color
                    )
    else:
        st.error("Data not available. Please check the data sources.")

# --- Tab 2: Progress Over Time ---
with tab_time:
    st.subheader("📈 Progress Over Time")
    
    if df_time_based is not None:
        # Adding an "All Entities" option
        entity_options = ['All Entities'] + list(df_time_based['Entity'].unique())
        selected_entity = st.selectbox("Select an Entity:", entity_options, key="time_entity")
        
        # Add time granularity option
        time_granularity = st.radio("Time Granularity:", ["Monthly", "Yearly"], horizontal=True)

        # Filter based on selection
        if selected_entity == 'All Entities':
            filtered_df = df_time_based
            title = "All Entities - DSM vs DSL Progress Over Time"
        else:
            filtered_df = df_time_based[df_time_based['Entity'] == selected_entity]
            title = f"{selected_entity} - DSM vs DSL Progress Over Time"

        # Apply time granularity
        if time_granularity == "Yearly":
            # Group by Year and Type
            if selected_entity == 'All Entities':
                chart_data = filtered_df.groupby(['Year', 'Type'], as_index=False)['Progress'].mean()
                chart_data['Date'] = pd.to_datetime(chart_data['Year'], format='%Y')
            else:
                chart_data = filtered_df.groupby(['Year', 'Entity', 'Type'], as_index=False)['Progress'].mean()
                chart_data['Date'] = pd.to_datetime(chart_data['Year'], format='%Y')
            
            x_axis = 'Year'
        else:  # Monthly
            # Use the actual dates
            if selected_entity == 'All Entities':
                # Group by month and year
                chart_data = filtered_df.groupby([filtered_df['Date'].dt.to_period('M'), 'Type'], as_index=False)['Progress'].mean()
                chart_data['Date'] = chart_data['Date'].dt.to_timestamp()
            else:
                chart_data = filtered_df
            
            x_axis = 'Date'

        fig_line = px.line(
            chart_data,
            x=x_axis, 
            y='Progress',
            color='Type',
            markers=True,
            title=title,
            labels={'Progress': 'Progress (%)', 'Date': 'Date'},
            height=500
        )
        
        # Add dots at data points
        fig_line.update_traces(mode='lines+markers')
        
        # Improve layout
        fig_line.update_layout(
            yaxis=dict(range=[0, 100]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.error("Time series data not available.")

# --- Tab 3: Comparison ---
with tab_compare:
    st.subheader("📊 DSM vs DSL Comparison")
    
    if df_time_based is not None:
        # Adding an "All Entities" option
        entity_options = ['All Entities'] + list(df_time_based['Entity'].unique())
        selected_entity_bar = st.selectbox("Select an Entity:", entity_options, key="compare_entity")

        # Date range selection
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date:", 
                value=df_time_based['Date'].min().date(),
                min_value=df_time_based['Date'].min().date(),
                max_value=df_time_based['Date'].max().date()
            )
        
        with col2:
            end_date = st.date_input(
                "End Date:", 
                value=df_time_based['Date'].max().date(),
                min_value=df_time_based['Date'].min().date(),
                max_value=df_time_based['Date'].max().date()
            )
            
        # Convert to datetime
        start_datetime = datetime.combine(start_date, datetime.min.time())
        end_datetime = datetime.combine(end_date, datetime.min.time())
        
        if start_datetime > end_datetime:
            st.warning("Start date cannot be after end date.")
        else:
            # Filter based on selections
            time_mask = (df_time_based['Date'] >= start_datetime) & (df_time_based['Date'] <= end_datetime)
            
            if selected_entity_bar == 'All Entities':
                filtered_df_bar = df_time_based[time_mask]
                title = f"All Entities - DSM vs DSL Comparison ({start_date.strftime('%b %Y')} to {end_date.strftime('%b %Y')})"
                
                # Group by Year and Type for "All Entities"
                chart_data = filtered_df_bar.groupby(['Year', 'Type'], as_index=False)['Progress'].mean()
            else:
                filtered_df_bar = df_time_based[
                    (df_time_based['Entity'] == selected_entity_bar) & time_mask
                ]
                title = f"{selected_entity_bar} - DSM vs DSL Comparison ({start_date.strftime('%b %Y')} to {end_date.strftime('%b %Y')})"
                
                # Group by Year for single entity
                chart_data = filtered_df_bar.groupby(['Year', 'Type'], as_index=False)['Progress'].mean()
            
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
