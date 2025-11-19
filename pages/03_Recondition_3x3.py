import streamlit as st
import numpy as np
import pandas as pd
from io import StringIO
from utils import recondition_kernel

st.set_page_config(page_title="Recondition 3×3", layout="wide")
st.title("Recondition 3×3 by Condition Number")

rec_csv = st.file_uploader("Upload a CSV with flattened 3×3 matrices (each row has 9 values)", type=["csv"])
C_val = st.number_input("Threshold C (max allowed condition number)", min_value=1.0, value=5.0, step=0.5)
rec_btn = st.button("Run reconditioning on CSV")

if rec_csv is not None and rec_btn:
    df_in = pd.read_csv(rec_csv, header=None)
    if df_in.shape[1] != 9:
        st.error("The CSV must have exactly 9 columns (flattened 3×3 matrices).")
    else:
        mats = df_in.values.reshape(-1, 3, 3)
        conds = np.array([np.linalg.cond(m) for m in mats])
        status = np.where(conds > C_val, "higher", "lower_or_equal")
        rec_out = []
        for m, c in zip(mats, conds):
            if c > C_val:
                m_rec, _, _, _, _ = recondition_kernel(m, C_val)
                rec_out.append(m_rec.reshape(-1))
            else:
                rec_out.append(m.reshape(-1))
        rec_out = np.vstack(rec_out)
        df_out = df_in.copy()
        df_out["cond_number"] = conds
        df_out["threshold_flag"] = status
        rec_cols = [f"rec_{i+1}" for i in range(9)]
        for i in range(9):
            df_out[rec_cols[i]] = rec_out[:, i]
        st.subheader("Preview of reconditioned output")
        st.dataframe(df_out.head(20), use_container_width=True)
        csv_buf = StringIO()
        df_out.to_csv(csv_buf, index=False, header=False)
        st.download_button(
            "Download reconditioned CSV",
            csv_buf.getvalue().encode("utf-8"),
            file_name="reconditioned_3x3_matrices.csv",
            mime="text/csv"
        )
elif rec_btn and rec_csv is None:
    st.error("Please upload a 3×3 matrices CSV first")
