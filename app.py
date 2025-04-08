import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import folium
from streamlit_folium import folium_static

# Set page configuration
st.set_page_config(layout="wide", page_title="Field Update Tracking Dashboard")

# Function to load data - better for organization and error handling
@st.cache_data
def load_data():
    try:
        # --- Data Preparation ---
        # Base entity data
        data_dsm_dsl = {
            'Entity': ['BP', 'TK', 'PP', 'UB'],
            'DSM Progress': [80, 70, 90, 85],
            'DSL Progress': [75, 65, 80, 80]
        }

        # Create exact dates for more granular date selection
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
        
        # Generate user activity data
        user_data = []
        users = [f"User_{i}" for i in range(1, 11)]  # 10 users
        
        # Generate random update locations (lat/lng)
        regions = {
            'North': {'lat_range': (34.0, 36.0), 'lng_range': (-118.0, -116.0)},
            'South': {'lat_range': (32.0, 34.0), 'lng_range': (-118.0, -116.0)},
            'East': {'lat_range': (33.0, 35.0), 'lng_range': (-116.0, -114.0)},
            'West': {'lat_range': (33.0, 35.0), 'lng_range': (-120.0, -118.0)}
        }
        
        # Generate user update data
        for date in all_dates:
            for user in users:
                # Randomly assign user to a region
                region = np.random.choice(list(regions.keys()))
                region_data = regions[region]
                
                # Generate random location within region
                lat = np.random.uniform(region_data['lat_range'][0], region_data['lat_range'][1])
                lng = np.random.uniform(region_data['lng_range'][0], region_data['lng_range'][1])
                
                # Random entity and interface
                entity = np.random.choice(entities)
                interface = np.random.choice(['DSM', 'DSL'])
                
                # Random update metrics
                updates_count = np.random.randint(1, 20)
                time_spent = np.random.randint(5, 60)  # minutes
                fields_updated = np.random.randint(1, 10)
                completion_rate = np.random.uniform(70, 100)
                error_rate = np.random.uniform(0, 30)
                
                # Update types
                update_types = ['New', 'Modify', 'Delete']
                update_type = np.random.choice(update_types, p=[0.4, 0.5, 0.1])
                
                user_data.append({
                    'Date': date,
                    'Year': date.year,
                    'Month': date.month,
                    'User': user,
                    'Entity': entity,
                    'Interface': interface,
                    'Region': region,
                    'Latitude': lat,
                    'Longitude': lng,
                    'Updates_Count': updates_count,
                    'Time_Spent': time_spent,
                    'Fields_Updated': fields_updated,
                    'Completion_Rate': completion_rate,
                    'Error_Rate': error_rate,
                    'Update_Type': update_type
                })
        
        # Convert to DataFrames
        df_dsm_dsl = pd.DataFrame(data_dsm_dsl)
        
        # Create a derived dataframe for entity total progress (average of DSM and DSL)
        df_entity_progress = pd.DataFrame({
            'Entity': df_dsm_dsl['Entity'],
            'Progress %': np.round((df_dsm_dsl['DSM Progress'] + df_dsm_dsl['DSL Progress']) / 2, 1)
        })
        
        df_time_based = pd.DataFrame(time_based_data)
        df_user_activity = pd.DataFrame(user_data)
        
        # Calculate total updates by entity and interface
        total_updates = df_user_activity.groupby(['Entity', 'Interface']).agg({
            'Updates_Count': 'sum',
            'Time_Spent': 'mean',
            'Completion_Rate': 'mean',
            'Error_Rate': 'mean'
        }).reset_index()
        
        return df_dsm_dsl, df_entity_progress, df_time_based, df_user_activity, total_updates
    
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None, None

# Load data
df_dsm_dsl, df_entity_progress, df_time_based, df_user_activity, total_updates = load_data()

# --- Dashboard ---
st.title("🗺️ Field Update Tracking Dashboard")
st.caption("Monitoring DSM (Map) & DSL (List) Updates for Field Entities")

# Create tabs
tab_overview, tab_user, tab_geo, tab_quality, tab_time = st.tabs([
    "🔍 Overview", 
    "👤 User Activity", 
    "🌍 Geographic", 
    "📊 Data Quality", 
    "📈 Time Analysis"
])

# --- Tab 1: Overview ---
with tab_overview:
    st.subheader("Update Activity Summary")

    # Error handling
    if df_dsm_dsl is not None and df_entity_progress is not None and df_user_activity is not None:
        # Date range selection
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date:", 
                value=(datetime.now() - timedelta(days=30)).date(),
                min_value=df_user_activity['Date'].min().date(),
                max_value=df_user_activity['Date'].max().date()
            )
        
        with col2:
            end_date = st.date_input(
                "End Date:", 
                value=datetime.now().date(),
                min_value=df_user_activity['Date'].min().date(),
                max_value=df_user_activity['Date'].max().date()
            )
        
        # Convert to datetime
        start_datetime = datetime.combine(start_date, datetime.min.time())
        end_datetime = datetime.combine(end_date, datetime.min.time())
        
        # Filter data based on date range
        filtered_activity = df_user_activity[
            (df_user_activity['Date'] >= start_datetime) & 
            (df_user_activity['Date'] <= end_datetime)
        ]
        
        # Calculate metrics for the selected period
        total_dsm_updates = filtered_activity[filtered_activity['Interface'] == 'DSM']['Updates_Count'].sum()
        total_dsl_updates = filtered_activity[filtered_activity['Interface'] == 'DSL']['Updates_Count'].sum()
        total_users = filtered_activity['User'].nunique()
        avg_completion = filtered_activity['Completion_Rate'].mean()
        
        # KPI Cards
        col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
        col_kpi1.metric("🗺️ DSM Updates", f"{total_dsm_updates:,}")
        col_kpi2.metric("📋 DSL Updates", f"{total_dsl_updates:,}")
        col_kpi3.metric("👥 Active Users", f"{total_users}")
        col_kpi4.metric("✅ Avg Completion Rate", f"{avg_completion:.1f}%")
        
        # Create charts
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Updates by entity and interface
            updates_by_entity = filtered_activity.groupby(['Entity', 'Interface'])['Updates_Count'].sum().reset_index()
            
            fig1 = px.bar(
                updates_by_entity,
                x='Entity', 
                y='Updates_Count',
                color='Interface',
                barmode='group',
                title="Updates by Entity and Interface",
                labels={'Updates_Count': 'Total Updates', 'Interface': 'Interface Type'},
                height=400
            )
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            # Update type distribution
            update_types = filtered_activity.groupby(['Update_Type', 'Entity'])['Updates_Count'].sum().reset_index()
            
            fig2 = px.bar(
                update_types, 
                x='Entity', 
                y='Updates_Count',
                color='Update_Type',
                title="Update Types by Entity", 
                height=400
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        # Interface comparison
        st.subheader("DSM vs DSL Interface Comparison")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Calculate metrics by interface
            interface_metrics = filtered_activity.groupby('Interface').agg({
                'Updates_Count': 'sum',
                'Time_Spent': 'mean',
                'Completion_Rate': 'mean',
                'Error_Rate': 'mean'
            }).reset_index()
            
            # Prepare data for radar chart
            categories = ['Update Volume', 'Speed (Less Time)', 'Completion Rate', 'Accuracy (Less Errors)']
            
            # Normalize values for radar chart
            dsm_values = [
                interface_metrics[interface_metrics['Interface'] == 'DSM']['Updates_Count'].values[0],
                100 - interface_metrics[interface_metrics['Interface'] == 'DSM']['Time_Spent'].values[0],  # Invert so less time is better
                interface_metrics[interface_metrics['Interface'] == 'DSM']['Completion_Rate'].values[0],
                100 - interface_metrics[interface_metrics['Interface'] == 'DSM']['Error_Rate'].values[0]  # Invert so less errors is better
            ]
            
            dsl_values = [
                interface_metrics[interface_metrics['Interface'] == 'DSL']['Updates_Count'].values[0],
                100 - interface_metrics[interface_metrics['Interface'] == 'DSL']['Time_Spent'].values[0],
                interface_metrics[interface_metrics['Interface'] == 'DSL']['Completion_Rate'].values[0],
                100 - interface_metrics[interface_metrics['Interface'] == 'DSL']['Error_Rate'].values[0]
            ]
            
            # Normalize to 0-100 scale
            max_update = max(dsm_values[0], dsl_values[0])
            dsm_values[0] = (dsm_values[0] / max_update) * 100
            dsl_values[0] = (dsl_values[0] / max_update) * 100
            
            # Create radar chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=dsm_values,
                theta=categories,
                fill='toself',
                name='DSM (Map)'
            ))
            
            fig.add_trace(go.Scatterpolar(
                r=dsl_values,
                theta=categories,
                fill='toself',
                name='DSL (List)'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )
                ),
                title="Interface Performance Comparison",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Interface preference by entity
            interface_pref = filtered_activity.groupby(['Entity', 'Interface'])['Updates_Count'].sum().reset_index()
            interface_pref_pivot = interface_pref.pivot(index='Entity', columns='Interface', values='Updates_Count').reset_index()
            interface_pref_pivot['DSM_Ratio'] = interface_pref_pivot['DSM'] / (interface_pref_pivot['DSM'] + interface_pref_pivot['DSL']) * 100
            interface_pref_pivot['DSL_Ratio'] = 100 - interface_pref_pivot['DSM_Ratio']
            
            # Melt for plotting
            interface_melt = pd.melt(
                interface_pref_pivot, 
                id_vars=['Entity'], 
                value_vars=['DSM_Ratio', 'DSL_Ratio'],
                var_name='Interface_Ratio', 
                value_name='Percentage'
            )
            
            # Clean names
            interface_melt['Interface_Ratio'] = interface_melt['Interface_Ratio'].str.replace('_Ratio', '')
            
            # Create stacked bar chart
            fig = px.bar(
                interface_melt,
                x='Entity',
                y='Percentage',
                color='Interface_Ratio',
                title="Interface Preference by Entity",
                labels={'Percentage': 'Usage %', 'Interface_Ratio': 'Interface'},
                height=500
            )
            
            fig.update_layout(barmode='stack')
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Data not available. Please check the data sources.")

# --- Tab 2: User Activity ---
with tab_user:
    st.subheader("👤 User Activity Analysis")
    
    if df_user_activity is not None:
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Entity filter
            entity_options = ['All Entities'] + list(df_user_activity['Entity'].unique())
            selected_entity = st.selectbox("Select Entity:", entity_options, key="user_entity")
        
        with col2:
            # Interface filter
            interface_options = ['All Interfaces'] + list(df_user_activity['Interface'].unique())
            selected_interface = st.selectbox("Select Interface:", interface_options, key="user_interface")
        
        with col3:
            # Date range
            time_period = st.selectbox(
                "Time Period:", 
                ["Last 30 Days", "Last 3 Months", "Last 6 Months", "Last Year", "All Time"],
                key="user_time"
            )
        
        # Process filters
        end_date = df_user_activity['Date'].max()
        
        if time_period == "Last 30 Days":
            start_date = end_date - timedelta(days=30)
        elif time_period == "Last 3 Months":
            start_date = end_date - timedelta(days=90)
        elif time_period == "Last 6 Months":
            start_date = end_date - timedelta(days=180)
        elif time_period == "Last Year":
            start_date = end_date - timedelta(days=365)
        else:  # All Time
            start_date = df_user_activity['Date'].min()
        
        # Apply filters
        mask = (df_user_activity['Date'] >= start_date) & (df_user_activity['Date'] <= end_date)
        
        if selected_entity != 'All Entities':
            mask &= (df_user_activity['Entity'] == selected_entity)
        
        if selected_interface != 'All Interfaces':
            mask &= (df_user_activity['Interface'] == selected_interface)
        
        filtered_data = df_user_activity[mask]
        
        if not filtered_data.empty:
            # User activity metrics
            st.subheader("User Performance Metrics")
            
            # Calculate metrics by user
            user_metrics = filtered_data.groupby('User').agg({
                'Updates_Count': 'sum',
                'Time_Spent': 'mean',
                'Fields_Updated': 'sum',
                'Completion_Rate': 'mean',
                'Error_Rate': 'mean'
            }).reset_index()
            
            # Sort by updates
            user_metrics = user_metrics.sort_values('Updates_Count', ascending=False)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # User performance chart
                fig = px.bar(
                    user_metrics,
                    x='User',
                    y='Updates_Count',
                    color='Updates_Count',
                    title="Updates by User",
                    labels={'Updates_Count': 'Total Updates'},
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Top users table
                st.markdown("### Top Users")
                
                # Format table data
                top_users = user_metrics.head(5).copy()
                top_users['Avg Time (min)'] = top_users['Time_Spent'].round(1)
                top_users['Completion %'] = top_users['Completion_Rate'].round(1)
                top_users['Error %'] = top_users['Error_Rate'].round(1)
                
                # Display table
                st.dataframe(
                    top_users[['User', 'Updates_Count', 'Avg Time (min)', 'Completion %', 'Error %']],
                    use_container_width=True
                )
            
            # User activity over time
            st.subheader("User Activity Timeline")
            
            # Group by date and calculate daily updates
            user_timeline = filtered_data.groupby(['Date', 'User'])['Updates_Count'].sum().reset_index()
            
            # Create line chart
            fig = px.line(
                user_timeline,
                x='Date',
                y='Updates_Count',
                color='User',
                title="User Activity Over Time",
                labels={'Updates_Count': 'Updates', 'Date': 'Date'},
                height=500
            )
            
            # Add markers
            fig.update_traces(mode='lines+markers')
            
            st.plotly_chart(fig, use_container_width=True)
            
            # User Entity Preference
            st.subheader("User Entity & Interface Preferences")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Entity preference by user
                entity_pref = filtered_data.groupby(['User', 'Entity'])['Updates_Count'].sum().reset_index()
                
                fig = px.bar(
                    entity_pref,
                    x='User',
                    y='Updates_Count',
                    color='Entity',
                    title="Entity Preference by User",
                    labels={'Updates_Count': 'Updates'},
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Interface preference by user
                interface_pref = filtered_data.groupby(['User', 'Interface'])['Updates_Count'].sum().reset_index()
                
                # Calculate percentage
                user_totals = interface_pref.groupby('User')['Updates_Count'].sum().reset_index()
                user_totals.rename(columns={'Updates_Count': 'Total'}, inplace=True)
                
                interface_pref = pd.merge(interface_pref, user_totals, on='User')
                interface_pref['Percentage'] = (interface_pref['Updates_Count'] / interface_pref['Total']) * 100
                
                fig = px.bar(
                    interface_pref,
                    x='User',
                    y='Percentage',
                    color='Interface',
                    title="Interface Preference by User (%)",
                    labels={'Percentage': 'Usage %'},
                    height=400
                )
                
                fig.update_layout(barmode='stack', yaxis=dict(range=[0, 100]))
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No data available for the selected filters.")
    else:
        st.error("User activity data not available.")

# --- Tab 3: Geographic ---
with tab_geo:
    st.subheader("🌍 Geographic Update Distribution")
    
    if df_user_activity is not None:
        # Filters
        col1, col2 = st.columns(2)
        
        with col1:
            # Entity filter
            entity_options = ['All Entities'] + list(df_user_activity['Entity'].unique())
            selected_entity = st.selectbox("Select Entity:", entity_options, key="geo_entity")
        
        with col2:
            # Interface filter
            interface_options = ['All Interfaces'] + list(df_user_activity['Interface'].unique())
            selected_interface = st.selectbox("Select Interface:", interface_options, key="geo_interface")
        
        # Apply filters
        filtered_data = df_user_activity.copy()
        
        if selected_entity != 'All Entities':
            filtered_data = filtered_data[filtered_data['Entity'] == selected_entity]
        
        if selected_interface != 'All Interfaces':
            filtered_data = filtered_data[filtered_data['Interface'] == selected_interface]
        
        # Create map
        st.subheader("Update Location Map")
        
        # Group by coordinates and sum updates
        geo_data = filtered_data.groupby(['Latitude', 'Longitude', 'Region']).agg({
            'Updates_Count': 'sum',
            'Entity': 'first',  # Just get one if filtered to single entity
            'Interface': 'first'  # Just get one if filtered to single interface
        }).reset_index()
        
        # Create map centered at average coordinates
        center_lat = geo_data['Latitude'].mean()
        center_lng = geo_data['Longitude'].mean()
        
        m = folium.Map(location=[center_lat, center_lng], zoom_start=8)
        
        # Add markers for each location
        for _, row in geo_data.iterrows():
            # Size circle based on number of updates
            radius = min(50, max(5, row['Updates_Count'] / 10))
            
            # Determine color by entity or interface
            if selected_entity != 'All Entities':
                color = 'blue'  # Use interface color if entity is fixed
                popup_text = f"Interface: {row['Interface']}<br>Updates: {row['Updates_Count']}<br>Region: {row['Region']}"
            elif selected_interface != 'All Interfaces':
                color = 'green'  # Use entity color if interface is fixed
                popup_text = f"Entity: {row['Entity']}<br>Updates: {row['Updates_Count']}<br>Region: {row['Region']}"
            else:
                color = 'red'  # Default
                popup_text = f"Entity: {row['Entity']}<br>Interface: {row['Interface']}<br>Updates: {row['Updates_Count']}<br>Region: {row['Region']}"
            
            folium.CircleMarker(
                location=[row['Latitude'], row['Longitude']],
                radius=radius,
                color=color,
                fill=True,
                fill_opacity=0.7,
                popup=popup_text
            ).add_to(m)
        
        # Display map
        folium_static(m)
        
        # Regional analysis
        st.subheader("Regional Update Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Updates by region
            region_data = filtered_data.groupby('Region')['Updates_Count'].sum().reset_index()
            
            fig = px.pie(
                region_data,
                values='Updates_Count',
                names='Region',
                title="Updates by Region",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Entity distribution by region
            if selected_entity == 'All Entities':
                entity_region = filtered_data.groupby(['Region', 'Entity'])['Updates_Count'].sum().reset_index()
                
                fig = px.bar(
                    entity_region,
                    x='Region',
                    y='Updates_Count',
                    color='Entity',
                    title="Entity Distribution by Region",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Interface distribution by region
                interface_region = filtered_data.groupby(['Region', 'Interface'])['Updates_Count'].sum().reset_index()
                
                fig = px.bar(
                    interface_region,
                    x='Region',
                    y='Updates_Count',
                    color='Interface',
                    title=f"{selected_entity} Updates by Region and Interface",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Geographic data not available.")

# --- Tab 4: Data Quality ---
with tab_quality:
    st.subheader("📊 Update Quality Analysis")
    
    if df_user_activity is not None:
        # Entity selection
        entity_options = ['All Entities'] + list(df_user_activity['Entity'].unique())
        selected_entity = st.selectbox("Select Entity:", entity_options, key="quality_entity")
        
        # Filter data
        filtered_data = df_user_activity.copy()
        
        if selected_entity != 'All Entities':
            filtered_data = filtered_data[filtered_data['Entity'] == selected_entity]
        
        # Calculate quality metrics
        quality_by_interface = filtered_data.groupby('Interface').agg({
            'Completion_Rate': 'mean',
            'Error_Rate': 'mean',
            'Updates_Count': 'sum'
        }).reset_index()
        
        # Quality metrics by update type
        quality_by_type = filtered_data.groupby(['Update_Type', 'Interface']).agg({
            'Completion_Rate': 'mean',
            'Error_Rate': 'mean'
        }).reset_index()
        
        # Display quality metrics
        col1, col2 = st.columns(2)
        
        with col1:
            # Completion rate by interface
            fig = px.bar(
                quality_by_interface,
                x='Interface',
                y='Completion_Rate',
                color='Interface',
                title="Field Completion Rate by Interface",
                labels={'Completion_Rate': 'Completion Rate (%)'},
                height=400
            )
            
            fig.update_layout(yaxis=dict(range=[0, 100]))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Error rate by interface
            fig = px.bar(
                quality_by_interface,
                x='Interface',
                y='Error_Rate',
                color='Interface',
                title="Error Rate by Interface",
                labels={'Error_Rate': 'Error Rate (%)'},
                height=400
            )
            
            fig.update_layout(yaxis=dict(range=[0, 100]))
            st.plotly_chart(fig, use_container_width=True)
        
        # Quality metrics by update type
        st.subheader("Quality by Update Type")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Completion rate by update type
            fig = px.bar(
                quality_by_type,
                x='Update_Type',
                y='Completion_Rate',
                color='Interface',
                barmode='group',
                title="Completion Rate by Update Type",
                labels={'Completion_Rate': 'Completion Rate (%)'},
                height=400
            )
            
            fig.update_layout(yaxis=dict(range=[0, 100]))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Error rate by update type
            fig = px.bar(
                quality_by_type,
