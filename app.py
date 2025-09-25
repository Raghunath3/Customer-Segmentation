import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load models with joblib
kmeans_model = joblib.load("kmeans_model.pkl")
final_model = joblib.load("final_model.sav")   # Decision Tree

st.title("Customer Segmentation App")
st.write("Enter customer details to predict which cluster they belong to.")

# Input fields (adjust as per your dataset features)
age = st.number_input("Age", 18, 100, 25)
income = st.number_input("Annual Income (in thousands)", 10, 200, 50)
spending = st.number_input("Spending Score (1-100)", 1, 100, 50)

if st.button("Predict Cluster"):
    features = np.array([[age, income, spending]])

    # First: clustering
    cluster_label = kmeans_model.predict(features)[0]

    # Second: supervised model
    final_label = final_model.predict(features)[0]

    st.success(f"Cluster (KMeans): {cluster_label}")
    st.success(f"Predicted Segment (Decision Tree): {final_label}")

# Optional: show cluster centers
if st.checkbox("Show Cluster Centers (KMeans)"):
    centers = kmeans_model.cluster_centers_
    fig, ax = plt.subplots()
    ax.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    st.pyplot(fig)
