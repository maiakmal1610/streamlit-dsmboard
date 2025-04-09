import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set page configuration
st.set_page_config(layout="wide", page_title="Entity Tracking Dashboard")

# Function to load data - in a real scenario, this would pull from your database
@st.cache_data
def load_data():
    try:
        # --- Generate sample data ---
        # Create entities
        entities = ['BP', 'UB', 'TK', 'PP']
        
        # Create base entity counts
        entity_counts = {
            'BP': {'total': 1250, 'dsl': 1200, 'dsm': 1050},
            'UB': {'total': 950, 'dsl': 900, 'dsm': 820},
            'TK': {'total': 1500, 'dsl': 1450, 'dsm': 1100},  # Significant DSM shortfall
            'PP': {'total': 780, 'dsl': 760, 'dsm': 730}
        }
        
        # Create history data (daily for last 30 days)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        history_data = []
        
        # Starting values - percentage of current totals
        starting_pct = 0.80  # Starting with 80% of current values
        
        for entity in entities:
            total_count = entity_counts[entity]['total']
            dsl_count = entity_counts[entity]['dsl']
            dsm_count = entity_counts[entity]['dsm']
            
            # Entity growth rate - slightly different for each entity
            growth_rates = {'BP': 0.0012, 'UB': 0.0010, 'TK': 0.0015, 'PP': 0.0008}
            
            # DSL vs DSM ratios - TK has more divergence
            dsm_ratios = {'BP': 0.9, 'UB': 0.92, 'TK': 0.76, 'PP': 0.95}
            
            current_total = total_count * starting_pct
            current_dsl = dsl_count * starting_pct
            current_dsm = dsm_count * starting_pct
            
            for date in date_range:
                # Add daily growth with some randomness
                growth_factor = 1 + growth_rates[entity] + np.random.uniform(-0.0005, 0.0005)
                current_total *= growth_factor
                
                # DSL grows at the same rate
                current_dsl *= growth_factor
                
                # DSM grows similarly but with entity-specific ratio differences
                # For TK specifically, make DSM growth lag behind significantly
                if entity == 'TK':
                    dsm_growth = growth_factor * dsm_ratios[entity] * (0.95 + 0.05 * np.random.random())
                else:
                    dsm_growth = growth_factor * dsm_ratios[entity] * (0.98 + 0.02 * np.random.random())
                
                current_dsm *= dsm_growth
                
                # Record values (rounded to integers)
                history_data.append({
                    'Date': date,
                    'Entity': entity,
                    'Total': int(current_total),
                    'DSL': int(current_dsl),
                    'DSM': int(current_dsm),
                    'DSL_DSM_Gap': int(current_dsl - current_dsm)
                })
        
        # Create current snapshot
        current_data = []
        for entity in entities:
            current_data.append({
                'Entity': entity,
                'Total': entity_counts[entity]['total'],
                'DSL': entity_counts[entity]['dsl'],
                'DSM': entity_counts[entity]['dsm'],
                'DSL_DSM_Gap': entity_counts[entity]['dsl'] - entity_counts[entity]['dsm']
            })
        
        # Convert to DataFrames
        df_current = pd.DataFrame(current_data)
        df_history = pd.DataFrame(history_data)
        
        # Create melt version for some charts
        df_current_melted = pd.melt(
            df_current,
            id_vars=['Entity'],
            value_vars=['DSL', 'DSM'],
            var_name='Interface',
            value_name='Count'
        )
        
        return df_current, df_history, df_current_melted
    
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

# Load data
df_current, df_history, df_current_melted = load_data()

# --- Dashboard ---
st.title("📊 Entity Tracking Dashboard")
st.caption("Monitoring entities (BP, UB, TK, PP) across DSL (Listing) and DSM (Map) interfaces")

# Create tabs
tab_overview, tab_comparison, tab_history = st.tabs([
    "📋 Overall Counts", 
    "🔄 DSL vs DSM", 
    "📈 Historical Trends"
])

# --- Tab 1: Overall Entity Counts ---
with tab_overview:
    st.header("Entity Counts Overview")

    if df_current is not None:
        # MODIFIED: Replace KPI cards with entity count breakdowns
        # Get entity counts
        bp_count = df_current[df_current['Entity'] == 'BP']['Total'].values[0]
        ub_count = df_current[df_current['Entity'] == 'UB']['Total'].values[0]
        tk_count = df_current[df_current['Entity'] == 'TK']['Total'].values[0]
        pp_count = df_current[df_current['Entity'] == 'PP']['Total'].values[0]
        
        # KPI metrics - replaced with entity-specific counts
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("BP (Blok Penghitungan)", f"{bp_count:,}")
        col2.metric("UB (Unit Bangunan)", f"{ub_count:,}")
        col3.metric("TK (Tempat Kediaman)", f"{tk_count:,}")
        col4.metric("PP (Pertubuhan Perniagaan)", f"{pp_count:,}")
        
        # Entity counts chart
        st.subheader("Entity Count Distribution")
        
        fig = px.bar(
            df_current, 
            x='Entity', 
            y='Total',
            color='Entity',
            text='Total',
            title="Total Entity Counts",
            height=500
        )
        
        fig.update_traces(texttemplate='%{text:,}', textposition='outside')
        fig.update_layout(uniformtext_minsize=10, uniformtext_mode='hide')
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Entity percentage distribution
        st.subheader("Entity Distribution")
        
        fig = px.pie(
            df_current,
            values='Total',
            names='Entity',
            title="Entity Distribution (%)",
            height=500
        )
        
        fig.update_traces(textinfo='percent+label')
        
        col1, col2 = st.columns([2, 1])
        col1.plotly_chart(fig, use_container_width=True)
        
        # Entity count table
        col2.subheader("Entity Count Details")
        col2.dataframe(
            df_current[['Entity', 'Total']].sort_values('Total', ascending=False),
            hide_index=True,
            column_config={
                "Entity": "Entity Type",
                "Total": st.column_config.NumberColumn("Total Count", format="%d")
            },
            use_container_width=True
        )
        
    else:
        st.error("Error loading entity count data.")

# --- Tab 2: DSL vs DSM Comparison ---
with tab_comparison:
    st.header("DSL vs DSM Interface Comparison")
    
    if df_current is not None and df_current_melted is not None:
        # Calculate gap metrics
        total_gap = df_current['DSL_DSM_Gap'].sum()
        max_gap_entity = df_current.loc[df_current['DSL_DSM_Gap'].idxmax()]['Entity']
        max_gap_value = df_current.loc[df_current['DSL_DSM_Gap'].idxmax()]['DSL_DSM_Gap']
        gap_percentage = (total_gap / df_current['DSL'].sum()) * 100
        
        # Display overall metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total DSL-DSM Gap", f"{total_gap:,}")
        col2.metric("Biggest Gap Entity", f"{max_gap_entity} ({max_gap_value:,})")
        col3.metric("Gap Percentage", f"{gap_percentage:.1f}%")
        
        # DSL vs DSM comparison by entity
        st.subheader("DSL vs DSM Counts by Entity")
        
        fig = px.bar(
            df_current_melted,
            x='Entity',
            y='Count',
            color='Interface',
            barmode='group',
            text='Count',
            title="DSL vs DSM Entity Counts",
            height=500
        )
        
        fig.update_traces(texttemplate='%{text:,}', textposition='outside')
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display gap analysis
        st.subheader("DSL-DSM Gap Analysis")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Create gap chart
            fig = px.bar(
                df_current,
                x='Entity',
                y='DSL_DSM_Gap',
                color='Entity',
                text='DSL_DSM_Gap',
                title="Gap Between DSL and DSM Counts",
                height=400
            )
            
            fig.update_traces(texttemplate='%{text:,}', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Calculate gap percentages
            df_gap = df_current.copy()
            df_gap['Gap_Percentage'] = (df_gap['DSL_DSM_Gap'] / df_gap['DSL'] * 100).round(1)
            
            # Sort by gap percentage
            df_gap = df_gap.sort_values('Gap_Percentage', ascending=False)
            
            # Create KPI cards for each entity
            st.markdown("### Gap Percentage by Entity")
            
            for _, row in df_gap.iterrows():
                gap = row["Gap_Percentage"]
                delta_color = "normal"
    
                # Color logic
                if gap > 15:
                    delta_color = "normal"  # Red (↑ is bad)
                elif gap < 5:
                    gap = -gap       
                
                st.metric(
                    label=f"{row['Entity']} Gap ({row['Gap_Percentage']}%)",
                    value=f"{row['DSL_DSM_Gap']:,}",
                    delta=row["Gap_Percentage"],  # still a number here
                    delta_color=delta_color,
                    help=f"Difference between DSL ({row['DSL']:,}) and DSM ({row['DSM']:,}) counts"
)
        
        # DSL vs DSM detailed comparison
        st.subheader("Detailed Interface Comparison")
        
        # Create comparison table
        comparison_df = df_current[['Entity', 'DSL', 'DSM', 'DSL_DSM_Gap']].copy()
        comparison_df['Gap_Percentage'] = (comparison_df['DSL_DSM_Gap'] / comparison_df['DSL'] * 100).round(1)
        comparison_df['Completion_Rate'] = ((comparison_df['DSM'] / comparison_df['DSL']) * 100).round(1)
        
        # Highlight cells based on values
        def highlight_gaps(val):
            if isinstance(val, (int, float)):
                if 'Gap_Percentage' in comparison_df.columns and val > 15:
                    return 'background-color: rgba(255, 0, 0, 0.2)'
                elif 'Completion_Rate' in comparison_df.columns and val < 85:
                    return 'background-color: rgba(255, 0, 0, 0.2)'
            return ''
        
        # Display styled table
        st.dataframe(
            comparison_df.style.applymap(highlight_gaps),
            column_config={
                "Entity": "Entity Type",
                "DSL": st.column_config.NumberColumn("DSL Count", format="%d"),
                "DSM": st.column_config.NumberColumn("DSM Count", format="%d"),
                "DSL_DSM_Gap": st.column_config.NumberColumn("Count Gap", format="%d"),
                "Gap_Percentage": st.column_config.NumberColumn("Gap %", format="%.1f%%"),
                "Completion_Rate": st.column_config.NumberColumn("DSM Completion", format="%.1f%%")
            },
            use_container_width=True
        )
        
    else:
        st.error("Error loading comparison data.")

# --- Tab 3: Historical Trends ---
with tab_history:
    st.header("Historical Trends Analysis")
    
    if df_history is not None:
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Entity filter
            entity_options = list(df_history['Entity'].unique())
            selected_entity = st.selectbox("Select Entity:", entity_options, key="history_entity")
        
        with col2:
            # Time period selection
            time_period = st.selectbox(
                "Time Period:", 
                ["Last 30 Days", "Last 90 Days", "Last 180 Days", "All Time"],
                key="history_time"
            )
        
        with col3:
            # Display options
            display_option = st.radio(
                "Display:",
                ["Counts", "Gap Analysis"],
                horizontal=True,
                key="history_display"
            )
        
        # Filter based on selection
        entity_mask = (df_history['Entity'] == selected_entity)
        
        # Apply time filter
        end_date = df_history['Date'].max()
        
        if time_period == "Last 30 Days":
            start_date = end_date - timedelta(days=30)
        elif time_period == "Last 90 Days":
            start_date = end_date - timedelta(days=90)
        elif time_period == "Last 180 Days":
            start_date = end_date - timedelta(days=180)
        else:  # All Time
            start_date = df_history['Date'].min()
        
        date_mask = (df_history['Date'] >= start_date) & (df_history['Date'] <= end_date)
        
        # Combine filters
        filtered_df = df_history[entity_mask & date_mask]
        
        # Create charts based on display option
        if display_option == "Counts":
            st.subheader(f"{selected_entity} Count Trends")
            
            # Melt the dataframe for plotting
            plot_columns = ['Total', 'DSL', 'DSM']
            df_plot = filtered_df.melt(
                id_vars=['Date', 'Entity'],
                value_vars=plot_columns,
                var_name='Metric',
                value_name='Count'
            )
            
            # Create line chart
            fig = px.line(
                df_plot,
                x='Date',
                y='Count',
                color='Metric',
                title=f"{selected_entity} Counts Over Time",
                labels={'Count': 'Entity Count', 'Date': 'Date'},
                height=500
            )
            
            fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate current values
            current_total = filtered_df.iloc[-1]['Total']
            current_dsl = filtered_df.iloc[-1]['DSL']
            current_dsm = filtered_df.iloc[-1]['DSM']
            
            # Calculate change over period
            start_total = filtered_df.iloc[0]['Total']
            start_dsl = filtered_df.iloc[0]['DSL']
            start_dsm = filtered_df.iloc[0]['DSM']
            
            total_change = ((current_total / start_total) - 1) * 100
            dsl_change = ((current_dsl / start_dsl) - 1) * 100
            dsm_change = ((current_dsm / start_dsm) - 1) * 100
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Count", f"{current_total:,}", f"{total_change:.1f}%")
            col2.metric("DSL Count", f"{current_dsl:,}", f"{dsl_change:.1f}%")
            col3.metric("DSM Count", f"{current_dsm:,}", f"{dsm_change:.1f}%")
            
        else:  # Gap Analysis
            st.subheader(f"{selected_entity} DSL-DSM Gap Trends")
            
            # Create line chart for gap
            fig = px.line(
                filtered_df,
                x='Date',
                y='DSL_DSM_Gap',
                title=f"{selected_entity} DSL-DSM Gap Over Time",
                labels={'DSL_DSM_Gap': 'Count Gap', 'Date': 'Date'},
                height=500
            )
            
            # Add DSL and DSM lines for reference
            fig.add_trace(
                go.Scatter(
                    x=filtered_df['Date'],
                    y=filtered_df['DSL'],
                    name='DSL Count',
                    line=dict(dash='dot')
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=filtered_df['Date'],
                    y=filtered_df['DSM'],
                    name='DSM Count',
                    line=dict(dash='dot')
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate gap percentage over time
            filtered_df['Gap_Percentage'] = (filtered_df['DSL_DSM_Gap'] / filtered_df['DSL'] * 100)
            
            # Create gap percentage chart
            fig = px.line(
                filtered_df,
                x='Date',
                y='Gap_Percentage',
                title=f"{selected_entity} DSL-DSM Gap Percentage Over Time",
                labels={'Gap_Percentage': 'Gap Percentage (%)', 'Date': 'Date'},
                height=400
            )
            
            # Add reference line at 10%
            fig.add_hline(
                y=10, 
                line_dash="dash", 
                line_color="red",
                annotation_text="10% Threshold",
                annotation_position="bottom right"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display current gap metrics
            current_gap = filtered_df.iloc[-1]['DSL_DSM_Gap']
            current_gap_pct = filtered_df.iloc[-1]['Gap_Percentage']
            
            # Calculate change in gap over period
            start_gap = filtered_df.iloc[0]['DSL_DSM_Gap']
            start_gap_pct = filtered_df.iloc[0]['Gap_Percentage']
            
            gap_change = current_gap - start_gap
            gap_pct_change = current_gap_pct - start_gap_pct
            
            col1, col2 = st.columns(2)
            
            delta_color = "normal"
            if gap_change > 0:
                delta_color = "inverse"  # Gap increased (bad)
            elif gap_change < 0:
                delta_color = "good"     # Gap decreased (good)
                
            col1.metric(
                "Current DSL-DSM Gap", 
                f"{current_gap:,}", 
                f"{gap_change:+,}", 
                delta_color=delta_color
            )
            
            delta_color = "normal"
            if gap_pct_change > 0:
                delta_color = "inverse"  # Gap percentage increased (bad)
            elif gap_pct_change < 0:
                delta_color = "good"     # Gap percentage decreased (good)
                
            col2.metric(
                "Current Gap Percentage", 
                f"{current_gap_pct:.1f}%", 
                f"{gap_pct_change:+.1f}%", 
                delta_color=delta_color
            )
            
    else:
        st.error("Error loading historical data.")

# Add footer information
st.markdown("---")
st.markdown("**Entity Tracking Dashboard** | Last updated: April 2025")
