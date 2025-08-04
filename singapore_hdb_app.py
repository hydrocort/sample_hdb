import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Singapore HDB Resale Prices Dashboard",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache data for performance
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_hdb_data(limit=None):
    """
    Fetch HDB resale flat prices data from Singapore's data.gov.sg API
    Dataset: Resale flat prices based on registration date from Jan-2017 onwards
    """
    dataset_id = "d_8b84c4ee58e3cfc0ece0d773c8ca6abc"
    base_url = "https://data.gov.sg/api/action/datastore_search"
    
    params = {
        "resource_id": dataset_id,
        "limit": limit if limit else 10000  # Start with 10K records to avoid API limits
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if data.get("success"):
            records = data["result"]["records"]
            df = pd.DataFrame(records)
            
            # Data cleaning and preprocessing
            if not df.empty:
                # Convert data types
                df['resale_price'] = pd.to_numeric(df['resale_price'], errors='coerce')
                df['floor_area_sqm'] = pd.to_numeric(df['floor_area_sqm'], errors='coerce')
                df['lease_commence_date'] = pd.to_numeric(df['lease_commence_date'], errors='coerce')
                df['month'] = pd.to_datetime(df['month'], errors='coerce')
                
                # Calculate price per sqm
                df['price_per_sqm'] = df['resale_price'] / df['floor_area_sqm']
                
                # Extract year for easier filtering
                df['year'] = df['month'].dt.year
                
                # Calculate flat age
                df['flat_age'] = df['year'] - df['lease_commence_date']
                
                # Remove rows with missing essential data
                df = df.dropna(subset=['resale_price', 'month'])
                
            return df
        else:
            st.error(f"Failed to fetch data: {data.get('error', 'Unknown error')}")
            return pd.DataFrame()
            
    except requests.exceptions.RequestException as e:
        st.error(f"Network error: {str(e)}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        return pd.DataFrame()

def main():
    # Title and description
    st.title("🏠 Singapore HDB Resale Prices Dashboard")
    st.markdown("""
    **Explore Singapore's HDB resale flat prices from January 2017 onwards**
    
    This dashboard provides interactive analysis of Housing Development Board (HDB) resale transactions 
    based on registration dates. Data is sourced from Singapore's open data portal (data.gov.sg).
    """)
    
    # Sidebar for filters
    st.sidebar.header("🔍 Filters")
    
    # Load data
    with st.spinner("Loading HDB resale data..."):
        # Allow users to choose data size
        data_size = st.sidebar.selectbox(
            "Dataset Size",
            options=[5000, 10000, 20000],
            index=1,
            help="Larger datasets may take longer to load and analyze"
        )
        df = fetch_hdb_data(limit=data_size)
    
    if df.empty:
        st.error("Unable to load data. Please check your internet connection and try again.")
        return
    
    # Display data info
    st.sidebar.success(f"📊 Loaded {len(df):,} records")
    st.sidebar.info(f"📅 Data from {df['month'].min().strftime('%b %Y')} to {df['month'].max().strftime('%b %Y')}")
    
    # Sidebar filters
    years = sorted(df['year'].unique())
    
    # Handle year selection based on available data
    if len(years) == 1:
        # Only one year available, show as info
        st.sidebar.info(f"📅 Data available for: {years[0]}")
        selected_years = (years[0], years[0])
    else:
        # Multiple years available, use slider
        selected_years = st.sidebar.slider(
            "Select Year Range",
            min_value=int(min(years)),
            max_value=int(max(years)),
            value=(int(min(years)), int(max(years))),
            step=1
        )
    
    towns = sorted(df['town'].unique())
    selected_towns = st.sidebar.multiselect(
        "Select Towns",
        options=towns,
        default=towns[:5] if len(towns) > 5 else towns
    )
    
    flat_types = sorted(df['flat_type'].unique())
    selected_flat_types = st.sidebar.multiselect(
        "Select Flat Types",
        options=flat_types,
        default=flat_types
    )
    
    # Filter data based on selections
    filtered_df = df[
        (df['year'] >= selected_years[0]) & 
        (df['year'] <= selected_years[1]) &
        (df['town'].isin(selected_towns)) &
        (df['flat_type'].isin(selected_flat_types))
    ]
    
    if filtered_df.empty:
        st.warning("No data available for the selected filters. Please adjust your selection.")
        return
    
    # Main dashboard layout
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_price = filtered_df['resale_price'].mean()
        st.metric("Average Resale Price", f"${avg_price:,.0f}")
    
    with col2:
        median_price = filtered_df['resale_price'].median()
        st.metric("Median Resale Price", f"${median_price:,.0f}")
    
    with col3:
        avg_psf = filtered_df['price_per_sqm'].mean()
        st.metric("Average Price per sqm", f"${avg_psf:,.0f}")
    
    with col4:
        total_transactions = len(filtered_df)
        st.metric("Total Transactions", f"{total_transactions:,}")
    
    st.divider()
    
    # Charts section
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Price Trends", "🏘️ By Location", "🏠 By Flat Type", "📊 Detailed Analysis"])
    
    with tab1:
        st.subheader("Price Trends Over Time")
        
        # Monthly average price trend
        monthly_avg = filtered_df.groupby(filtered_df['month'].dt.to_period('M'))['resale_price'].mean().reset_index()
        monthly_avg['month'] = monthly_avg['month'].dt.to_timestamp()
        
        fig_trend = px.line(
            monthly_avg, 
            x='month', 
            y='resale_price',
            title="Average Resale Price Trend",
            labels={'resale_price': 'Average Resale Price ($)', 'month': 'Month'}
        )
        fig_trend.update_layout(height=400)
        st.plotly_chart(fig_trend, use_container_width=True)
        
        # Price distribution
        st.subheader("Price Distribution")
        fig_hist = px.histogram(
            filtered_df, 
            x='resale_price', 
            nbins=50,
            title="Distribution of Resale Prices",
            labels={'resale_price': 'Resale Price ($)', 'count': 'Number of Transactions'}
        )
        fig_hist.update_layout(height=400)
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with tab2:
        st.subheader("Analysis by Town")
        
        # Average price by town
        town_avg = filtered_df.groupby('town').agg({
            'resale_price': 'mean',
            'price_per_sqm': 'mean',
            'flat_type': 'count'
        }).round(0).sort_values('resale_price', ascending=False)
        town_avg.columns = ['Avg Price ($)', 'Avg Price per sqm ($)', 'Transactions']
        
        fig_town = px.bar(
            town_avg.reset_index(), 
            x='town', 
            y='Avg Price ($)',
            title="Average Resale Price by Town",
            labels={'town': 'Town', 'Avg Price ($)': 'Average Resale Price ($)'}
        )
        fig_town.update_layout(height=500, xaxis_tickangle=-45)
        st.plotly_chart(fig_town, use_container_width=True)
        
        # Show data table
        st.subheader("Town Statistics")
        st.dataframe(town_avg, use_container_width=True)
    
    with tab3:
        st.subheader("Analysis by Flat Type")
        
        # Box plot by flat type
        fig_box = px.box(
            filtered_df, 
            x='flat_type', 
            y='resale_price',
            title="Price Distribution by Flat Type",
            labels={'flat_type': 'Flat Type', 'resale_price': 'Resale Price ($)'}
        )
        fig_box.update_layout(height=500)
        st.plotly_chart(fig_box, use_container_width=True)
        
        # Average metrics by flat type
        flat_type_stats = filtered_df.groupby('flat_type').agg({
            'resale_price': ['mean', 'median', 'count'],
            'floor_area_sqm': 'mean',
            'price_per_sqm': 'mean'
        }).round(0)
        
        flat_type_stats.columns = ['Avg Price', 'Median Price', 'Transactions', 'Avg Area (sqm)', 'Avg Price per sqm']
        
        st.subheader("Flat Type Statistics")
        st.dataframe(flat_type_stats, use_container_width=True)
    
    with tab4:
        st.subheader("Detailed Analysis")
        
        # Correlation between floor area and price
        fig_scatter = px.scatter(
            filtered_df.sample(min(5000, len(filtered_df))),  # Sample for performance
            x='floor_area_sqm', 
            y='resale_price',
            color='flat_type',
            title="Resale Price vs Floor Area",
            labels={'floor_area_sqm': 'Floor Area (sqm)', 'resale_price': 'Resale Price ($)'},
            hover_data=['town', 'year']
        )
        fig_scatter.update_layout(height=500)
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Price per sqm by flat age
        if 'flat_age' in filtered_df.columns and not filtered_df['flat_age'].isna().all():
            age_groups = filtered_df.copy()
            age_groups = age_groups[age_groups['flat_age'] > 0]  # Remove invalid ages
            
            fig_age = px.scatter(
                age_groups.sample(min(3000, len(age_groups))),
                x='flat_age', 
                y='price_per_sqm',
                color='flat_type',
                title="Price per sqm vs Flat Age",
                labels={'flat_age': 'Flat Age (years)', 'price_per_sqm': 'Price per sqm ($)'},
                hover_data=['town', 'resale_price']
            )
            fig_age.update_layout(height=500)
            st.plotly_chart(fig_age, use_container_width=True)
    
    # Raw data viewer (optional)
    if st.sidebar.checkbox("Show Raw Data"):
        st.subheader("Raw Data")
        st.dataframe(filtered_df.head(1000), use_container_width=True)
    
    # Footer
    st.divider()
    st.markdown("""
    **Data Source:** Singapore Government Open Data Portal (data.gov.sg)  
    **Dataset:** Resale flat prices based on registration date from Jan-2017 onwards  
    **Last Updated:** Data is refreshed every 15 hours from the source
    """)

if __name__ == "__main__":
    main()