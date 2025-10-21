import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
import os
import warnings

# Suppress warnings for cleaner UI
warnings.filterwarnings("ignore")

# Title and description
st.set_page_config(page_title="Credit Risk Dashboard", layout="wide")
st.title("📊 Credit Risk Prediction Dashboard")
st.markdown("AI-powered insights for Ghanaian microfinance institutions. Upload borrower data to assess risk levels and guide lending decisions.")

# File upload
uploaded_file = st.file_uploader("Upload borrower data (.csv)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Basic validation
    required_cols = ["age", "income", "loan_amount"]
    if not all(col in df.columns for col in required_cols):
        st.error(
            f"Missing required columns: {', '.join([col for col in required_cols if col not in df.columns])}")
    else:
        # Feature engineering
        df["income_to_loan_ratio"] = df["income"] / df["loan_amount"]
        df["risk_score"] = np.clip(
            # Dummy model logic
            (df["income_to_loan_ratio"] * 0.3 + df["age"] * 0.01), 0, 1)
        df["risk_level"] = pd.cut(df["risk_score"], bins=[0, 0.3, 0.7, 1], labels=[
                                  "Low", "Medium", "High"])

        # Layout: Columns for summary and filters
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("📋 Borrower Risk Overview")
            st.dataframe(df[["age", "income", "loan_amount",
                         "income_to_loan_ratio", "risk_score", "risk_level"]])

        with col2:
            st.subheader("📊 Summary Stats")
            st.metric("Total Borrowers", len(df))
            st.metric("High Risk Borrowers",
                      df[df["risk_level"] == "High"].shape[0])
            st.metric("Average Risk Score", round(df["risk_score"].mean(), 2))

        # Risk distribution chart
        st.subheader("📈 Risk Score Distribution")
        fig = px.histogram(df, x="risk_score", color="risk_level", nbins=30,
                           title="Borrower Risk Score Distribution", labels={"risk_score": "Predicted Risk Score"})
        st.plotly_chart(fig, use_container_width=True)

        # Filter by risk level
        st.subheader("🔍 Filter Borrowers by Risk Level")
        selected_level = st.selectbox("Select Risk Level", options=[
                                      "All", "Low", "Medium", "High"])
        if selected_level != "All":
            filtered_df = df[df["risk_level"] == selected_level]
            st.write(f"Showing {selected_level} risk borrowers:")
            st.dataframe(filtered_df)

        # Top high-risk borrowers
        st.subheader("🚨 Top 5 High-Risk Borrowers")
        top_risk = df.sort_values(by="risk_score", ascending=False).head(5)
        st.dataframe(top_risk)

        # Optional decision logic
        st.subheader("🧠 Loan Decision Recommendation")
        selected_index = st.selectbox(
            "Select borrower index", options=df.index)
        borrower = df.loc[selected_index]
        recommendation = "Review manually" if borrower["risk_level"] == "High" else "Likely Approve"
        st.write(
            f"Borrower #{selected_index} is **{borrower['risk_level']} risk**. Recommended action: **{recommendation}**.")

        # 📍 Regional Risk Analysis
        st.subheader("📍 Risk Prediction by Region")
        if "region" in df.columns:
            region_summary = df.groupby("region").agg(
                total_borrowers=("risk_score", "count"),
                avg_risk_score=("risk_score", "mean"),
                high_risk_pct=("risk_level", lambda x: (x == "High").mean())
            ).reset_index()

            fig_region = px.bar(region_summary, x="region", y="avg_risk_score", color="high_risk_pct",
                                title="Average Risk Score by Region",
                                labels={"avg_risk_score": "Avg Risk",
                                        "high_risk_pct": "High Risk %"},
                                color_continuous_scale="Reds")
            st.plotly_chart(fig_region, use_container_width=True)
            st.dataframe(region_summary)
        else:
            st.warning(
                "Region column not found in uploaded data. Add a 'region' column to enable regional analysis.")

        # Downloadable report
        st.subheader("📥 Export Results")
        st.download_button("Download Risk Report", df.to_csv(
            index=False), file_name="credit_risk_report.csv")

else:
    st.info("Please upload a borrower dataset to begin.")
