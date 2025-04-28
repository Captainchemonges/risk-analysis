# Aviation Accident Risk Dashboard
# Enhanced version with data validation, error handling, and visualization improvements

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

sns.set(style="whitegrid")

@st.cache_data
def load_data():
    try:
        # FIX: Add multiple fallback encodings and column standardization
        encodings = ['utf-8', 'latin1', 'cp1252']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv('aviation_data.csv', encoding=encoding)
                # Standardize column names first
                df.columns = df.columns.str.replace('[^a-zA-Z0-9]', '_', regex=True).str.upper()
                break
            except UnicodeDecodeError:
                continue
                
        if df is None:
            st.error("Failed to read file with common encodings")
            return pd.DataFrame()

        # CRITICAL FIX: Validate required columns
        required_columns = {'EVENT_ID', 'MAKE', 'EVENT_DATE', 'TOTAL_FATAL_INJURIES', 
                           'WEATHER_CONDITION', 'PHASE_OF_FLIGHT'}
        
        missing_cols = required_columns - set(df.columns)
        if missing_cols:
            st.error(f"Missing critical columns: {', '.join(missing_cols)}")
            return pd.DataFrame()

        # Data cleaning pipeline
        df['MAKE'] = df['MAKE'].astype(str).str.upper().str.strip()
        df['EVENT_DATE'] = pd.to_datetime(df['EVENT_DATE'], errors='coerce')
        df['TOTAL_FATAL_INJURIES'] = pd.to_numeric(df['TOTAL_FATAL_INJURIES'], errors='coerce').fillna(0)
        df['WEATHER_CONDITION'] = df['WEATHER_CONDITION'].fillna('UNKNOWN').str.upper().str.strip()
        df['PHASE_OF_FLIGHT'] = df['PHASE_OF_FLIGHT'].fillna('UNKNOWN').str.upper().str.strip()
        
        # Add derived features
        df['YEAR'] = df['EVENT_DATE'].dt.year
        df['DECADE'] = (df['YEAR'] // 10) * 10
        
        return df

    except Exception as e:
        st.error(f"Data loading error: {str(e)}")
        return pd.DataFrame()

def calculate_risk_scores(df):
    try:
        risk_df = df.groupby('MAKE').agg(
            TOTAL_ACCIDENTS=('EVENT_ID', 'count'),
            TOTAL_FATALITIES=('TOTAL_FATAL_INJURIES', 'sum')
        ).reset_index()
        
        risk_df['AVG_FATALITIES_PER_ACCIDENT'] = (
            risk_df['TOTAL_FATALITIES'] / risk_df['TOTAL_ACCIDENTS']
        ).fillna(0)
        
        # Normalize with smoothing for zero values
        scaler = MinMaxScaler()
        features = risk_df[['TOTAL_ACCIDENTS', 'TOTAL_FATALITIES', 'AVG_FATALITIES_PER_ACCIDENT']]
        risk_df[['NORM_ACCIDENTS', 'NORM_FATALITIES', 'NORM_AVG_FATALITIES']] = (
            scaler.fit_transform(features + 1e-6)  # Add small epsilon to avoid division issues
        )
        
        risk_df['RISK_SCORE'] = (
            0.4 * risk_df['NORM_ACCIDENTS'] +
            0.4 * risk_df['NORM_FATALITIES'] +
            0.2 * risk_df['NORM_AVG_FATALITIES']
        )
        
        return risk_df.sort_values('RISK_SCORE', ascending=False)
    
    except Exception as e:
        st.error(f"Risk calculation error: {str(e)}")
        return pd.DataFrame()

def plot_top_makes(risk_df, top_n=10):
    try:
        plt.figure(figsize=(12, 8))
        sns.barplot(
            data=risk_df.head(top_n), 
            y='MAKE', 
            x='RISK_SCORE', 
            palette='rocket_r',
            edgecolor='black'
        )
        plt.title(f'Top {top_n} Aircraft Makes by Risk Score', pad=20)
        plt.xlabel('Risk Score', labelpad=15)
        plt.ylabel('Aircraft Make', labelpad=15)
        plt.grid(True, axis='x', alpha=0.3)
        st.pyplot(plt.gcf())
    except Exception as e:
        st.error(f"Plotting error: {str(e)}")

def plot_accidents_by_weather_phase(df, selected_weathers):
    try:
        filtered_df = df[df['WEATHER_CONDITION'].isin(selected_weathers)]
        grouped = filtered_df.groupby(['WEATHER_CONDITION', 'PHASE_OF_FLIGHT']).size().reset_index(name='COUNT')
        
        plt.figure(figsize=(16, 8))
        sns.barplot(
            data=grouped, 
            x='PHASE_OF_FLIGHT', 
            y='COUNT', 
            hue='WEATHER_CONDITION',
            palette='viridis',
            edgecolor='white'
        )
        plt.title('Accidents by Weather & Flight Phase', pad=20)
        plt.xlabel('Phase of Flight', labelpad=15)
        plt.ylabel('Accident Count', labelpad=15)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Weather Condition', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, axis='y', alpha=0.3)
        st.pyplot(plt.gcf())
    except Exception as e:
        st.error(f"Plotting error: {str(e)}")

def show_data_summary(df):
    st.subheader("Data Summary")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Accidents", df['EVENT_ID'].nunique())
        st.metric("Unique Manufacturers", df['MAKE'].nunique())
        
    with col2:
        st.metric("Time Range", f"{df['YEAR'].min()} - {df['YEAR'].max()}")
        st.metric("Fatal Injuries", int(df['TOTAL_FATAL_INJURIES'].sum()))

def main():
    st.set_page_config(layout="wide", page_title="Aviation Safety Dashboard")
    st.title(" Aviation Accident Risk Dashboard")
    
    with st.spinner("Loading aviation data..."):
        df = load_data()
    
    if df.empty:
        st.error("No data loaded - check source file and format")
        return
    
    # Sidebar controls
    st.sidebar.header("Dashboard Controls")
    top_n = st.sidebar.slider("Top Manufacturers to Show", 5, 25, 15)
    weather_options = sorted(df['WEATHER_CONDITION'].unique())
    selected_weather = st.sidebar.multiselect(
        "Weather Conditions", 
        weather_options, 
        default=['VMC', 'IMC', 'UNKNOWN']
    )
    
    # Main dashboard layout
    show_data_summary(df)
    
    tab1, tab2, tab3 = st.tabs(["Risk Analysis", "Accident Patterns", "Recommendations"])
    
    with tab1:
        st.header("Aircraft Manufacturer Risk Analysis")
        risk_df = calculate_risk_scores(df)
        plot_top_makes(risk_df, top_n)
        
        with st.expander("Raw Risk Data"):
            st.dataframe(risk_df.style.background_gradient(cmap='Reds', subset=['RISK_SCORE']))
    
    with tab2:
        st.header("Accident Patterns Analysis")
        plot_accidents_by_weather_phase(df, selected_weather)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Phase of Flight Distribution")
            phase_counts = df['PHASE_OF_FLIGHT'].value_counts()
            st.dataframe(phase_counts)
        
        with col2:
            st.subheader("Weather Condition Distribution")
            weather_counts = df['WEATHER_CONDITION'].value_counts()
            st.dataframe(weather_counts)
    
    with tab3:
        st.header("Safety Recommendations")
        st.markdown("""
        ### Operational Recommendations
        - **Manufacturer Focus**: Prioritize maintenance for high-risk aircraft makes
        - **Phase-Specific Training**: Enhance training for approach/landing phases
        - **Weather Protocols**: Strengthen IMC operational procedures
        
        ### Data Improvements
        - **Standardize Categories**: Implement consistent phase of flight taxonomy
        - **Enrich Data**: Add maintenance history and pilot experience data
        - **Real-time Monitoring**: Integrate with live weather/ATC systems
        """)

if __name__ == "__main__":
    main()
