import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
from utils import parse_csv_matrices, compute_symmetry_score

st.set_page_config(page_title="CSV Analysis", layout="wide")
st.title("CSV Analysis")

csv_file = st.file_uploader("Upload a CSV containing flattened matrices (one per row)", type=["csv"])
c1, c2 = st.columns(2)
show_mean_btn = c1.button("Show mean matrix and its symmetry score")
plot_dist_btn = c2.button("Plot symmetry score distribution")

if csv_file is not None and (show_mean_btn or plot_dist_btn):
    mats, n = parse_csv_matrices(csv_file.getvalue())
    if mats is None:
        st.error("Unsupported CSV shape. Each row must be a flattened n×n matrix with n in {3,5,7,9,11}.")
    else:
        if show_mean_btn:
            mean_mat = mats.mean(axis=0)
            score = compute_symmetry_score(mean_mat)
            st.subheader(f"Mean {n}x{n} matrix")
            st.dataframe(pd.DataFrame(mean_mat), use_container_width=True)
            st.success(f"Symmetry score of mean matrix: {score:.6f}")
        if plot_dist_btn:
            scores = np.array([compute_symmetry_score(m) for m in mats])
            mean_val = float(scores.mean())
            median_val = float(np.median(scores))
            fig = plt.figure(figsize=(10,6))
            bins = 12
            counts, bin_edges, _ = plt.hist(scores, bins=bins, edgecolor='black', alpha=0.7)
            plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f"Mean = {mean_val:.2f}")
            plt.axvline(median_val, color='green', linestyle='-', linewidth=2, label=f"Median = {median_val:.2f}")
            centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            labels_x = [f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}" for i in range(len(bin_edges)-1)]
            plt.xticks(centers, labels_x, rotation=45, ha="right")
            ax = plt.gca()
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=8, integer=True))
            ax.set_ylim(0, counts.max() * 1.10 if counts.size else 1)
            plt.xlabel("Symmetry Score Range", fontsize=12)
            plt.ylabel("Frequency", fontsize=12)
            plt.title(f"Distribution of Symmetry Scores (n={scores.size})", fontsize=14)
            plt.legend()
            plt.tight_layout()
            st.pyplot(fig)
elif (show_mean_btn or plot_dist_btn) and csv_file is None:
    st.error("Please upload a CSV file first")
