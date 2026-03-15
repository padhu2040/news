import streamlit as st
import pandas as pd

# Set up the page
st.set_page_config(page_title="News Aggregator", page_icon="📰", layout="wide")

st.title("📰 Open News Aggregator")
st.markdown("Tracking bias and clustering global events.")

st.divider()

# Mock Event Data (Simulating a clustered story)
st.header("Event: New Tech Regulations Announced")
st.write("**Neutral Summary:** Government officials have proposed a new bill aimed at regulating artificial intelligence development. Tech companies are divided on the implications.")

# Bias Bar Visualization (Using Streamlit progress/columns)
st.subheader("Coverage Bias")
col1, col2, col3 = st.columns(3)
with col1:
    st.info("Left: 30%")
with col2:
    st.warning("Center: 20%")
with col3:
    st.error("Right: 50%")

st.divider()

# Mock Articles for this event
st.subheader("Sources covering this event:")

# Creating columns for Left, Center, Right
left_col, center_col, right_col = st.columns(3)

with left_col:
    st.markdown("### Left Leaning")
    st.write("**Tech Blog A**")
    st.caption("New regulations might stifle open-source innovation.")
    st.button("Read", key="L1")

with center_col:
    st.markdown("### Center")
    st.write("**Global News Wire**")
    st.caption("AI regulatory bill proposed in congress today.")
    st.button("Read", key="C1")

with right_col:
    st.markdown("### Right Leaning")
    st.write("**Business Daily**")
    st.caption("Government overreach threatens tech sector growth.")
    st.button("Read", key="R1")
