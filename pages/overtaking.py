import streamlit as st
import matplotlib.pyplot as plt

def show_overtaking_zones_page():
    st.header("Overtaking Zones")
    st.write("""
        This page visualizes overtaking clusters and helps identify overtaking opportunities on the track.
    """)

    # Example plot or data visualization
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [10, 20, 25], label='Overtaking Zones')  # Sample data
    ax.set_title('Overtaking Zone Clusters')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()

    st.pyplot(fig)

# Call this function in the main app.py file
show_overtaking_zones_page()
