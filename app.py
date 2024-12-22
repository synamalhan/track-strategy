import streamlit as st
import pandas as pd
from fastf1 import get_session  # Simulated API fetch for telemetry data

def fetch_telemetry_data(driver_code, race, season):
    """
    Fetch telemetry data using an API or generate simulated data.
    """
    try:
        # Example using fastf1 for live data fetching
        session = get_session(season, race, 'R')
        session.load()
        lap_data = session.laps.pick_driver(driver_code)
        telemetry_data = lap_data.get_telemetry().to_pandas()

        return telemetry_data

    except Exception as e:
        st.error(f"Error fetching telemetry data: {e}")
        return None

def show_home_page():
    st.header("Welcome to the F1 Track Strategy App!")
    st.write("""
        Fetch telemetry data, analyze track performance, and optimize race strategies.
    """)

    st.subheader("Generate Telemetry Data")

    # Input Fields
    driver_code = st.text_input("Enter Driver Code (e.g., VER for Verstappen):")
    race = st.text_input("Enter Race Name (e.g., Monaco):")
    season = st.number_input("Enter Season (e.g., 2023):", min_value=1950, max_value=2024, step=1)

    if st.button("Fetch Data"):
        if driver_code and race and season:
            # Fetch or simulate telemetry data
            telemetry_data = fetch_telemetry_data(driver_code, race, season)

            if telemetry_data is not None:
                st.success("Telemetry data fetched successfully!")
                
                # Display a sample of the data
                st.write("Telemetry Data Sample:")
                st.dataframe(telemetry_data.head())

                # Download link for the data
                csv = telemetry_data.to_csv(index=False).encode('utf-8')
                st.download_button("Download Data", csv, f"{driver_code}_telemetry.csv", "text/csv")

                st.markdown("---")

                # Buttons to Navigate to Other Pages
                st.subheader("Navigate to Analysis Pages")
                st.write("Choose an analysis page to continue:")
                st.button("Go to Overtaking Zones", key="overtaking")
                st.button("Go to Lap Performance Optimization", key="lap_optimization")
                st.button("Go to Strategy Insights", key="strategy")
                st.button("Go to Interpretability", key="interpretability")

        else:
            st.warning("Please fill in all fields before fetching data.")
