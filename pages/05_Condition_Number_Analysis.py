import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker

st.set_page_config(page_title="Condition Number Analysis", layout="wide")
st.title("Condition Number Analysis")

st.markdown(
    "This page contains two tools:\n"
    "1. **Condition Number Calculator** – Upload a CSV of flattened 3×3 kernels to compute the condition number of each matrix.\n"
    "2. **Distribution Plotter** – Upload a CSV containing condition numbers to visualize their frequency distribution, including mean and median markers."
)

tabs = st.tabs(["Compute condition numbers", "Plot condition number distribution"])

# ============================================================
# TAB 1 — Compute condition numbers from 3×3 kernel CSV
# ============================================================

with tabs[0]:
    st.subheader("Compute Condition Numbers from 3×3 Kernels")

    csv_kn = st.file_uploader(
        "Upload CSV of flattened 3×3 matrices (9 values per row)",
        type=["csv"],
        key="cond_calc_uploader"
    )

    calc_btn = st.button("Compute condition numbers")

    if csv_kn is not None and calc_btn:
        df = pd.read_csv(csv_kn, header=None)

        if df.shape[1] != 9:
            st.error("CSV must have exactly 9 columns.")
        else:
            mats = df.values.reshape(-1, 3, 3)
            conds = np.array([np.linalg.cond(m) for m in mats])

            df_out = pd.DataFrame({"condition_number": conds})

            st.subheader("Preview")
            st.dataframe(df_out.head(20))

            st.download_button(
                "Download Condition Numbers CSV",
                df_out.to_csv(index=False).encode("utf-8"),
                file_name="condition_numbers.csv",
                mime="text/csv"
            )

    elif calc_btn and csv_kn is None:
        st.error("Please upload a CSV file first.")


# ============================================================
# TAB 2 — Plot distribution from condition number CSV
# ============================================================

with tabs[1]:
    st.subheader("Plot Distribution of Condition Numbers")

    csv_cond = st.file_uploader(
        "Upload CSV containing a column of condition numbers",
        type=["csv"],
        key="cond_dist_uploader"
    )

    plot_btn = st.button("Plot distribution")

    if csv_cond is not None and plot_btn:
        df = pd.read_csv(csv_cond)

        # find the column with numbers
        if df.shape[1] != 1:
            st.error("CSV must contain exactly one column of condition numbers.")
        else:
            conds = df.iloc[:, 0].values.astype(float)

            mean_val = float(np.mean(conds))
            median_val = float(np.median(conds))

            fig = plt.figure(figsize=(10, 6))
            bins = 12

            counts, bin_edges, _ = plt.hist(conds, bins=bins, edgecolor='black', alpha=0.7)
            plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f"Mean = {mean_val:.2f}")
            plt.axvline(median_val, color='green', linestyle='-', linewidth=2, label=f"Median = {median_val:.2f}")

            centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            labels = [f"{bin_edges[i]:.2f}–{bin_edges[i+1]:.2f}" for i in range(len(bin_edges)-1)]
            plt.xticks(centers, labels, rotation=45, ha="right")

            ax = plt.gca()
            ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            ax.set_ylim(0, counts.max() * 1.12)

            plt.title("Condition Number Distribution")
            plt.xlabel("Condition Number Range")
            plt.ylabel("Frequency")
            plt.legend()
            plt.tight_layout()

            st.pyplot(fig)

    elif plot_btn and csv_cond is None:
        st.error("Please upload a CSV file first.")
