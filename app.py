import streamlit as st

st.set_page_config(page_title="ConvNet Kernel Lab", layout="wide")
st.markdown(
    """
    <style>
    .app-header {
        padding-top: 0.5rem;
        padding-bottom: 1.5rem;
        border-bottom: 1px solid rgba(128,128,128,0.3);
        margin-bottom: 1.2rem;
    }
    .app-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .app-subtitle {
        font-size: 1.0rem;
        color: rgba(200,200,200,0.85);
    }
    .feature-card {
        padding: 1.0rem 1.2rem;
        border-radius: 0.8rem;
        border: 1px solid rgba(128,128,128,0.3);
        backdrop-filter: blur(6px);
        margin-bottom: 0.8rem;
    }
    .feature-title {
        font-weight: 600;
        font-size: 1.05rem;
        margin-bottom: 0.2rem;
    }
    .feature-tag {
        font-size: 0.82rem;
        opacity: 0.8;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="app-header">
      <div class="app-title">ConvNet Kernel Lab</div>
      <div class="app-subtitle">
        A small toolkit for exploring convolution kernels in Keras <code>.h5</code> models,
        analyzing symmetry and condition numbers, and applying SVD-based reconditioning.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "Use the **sidebar** to navigate between tools. Below is a quick overview of what each page does."
)

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
        <div class="feature-card">
          <div class="feature-title">📦 Model Layer Inspector</div>
          <div class="feature-tag">Upload a Keras <code>.h5</code> model</div>
          <p style="margin-top:0.4rem;">
          Inspect all layers of a model, filter convolution kernels by spatial size
          (3×3, 5×5, 7×7, …), and export the kernels of a selected layer to a CSV file
          for further analysis.
          </p>
        </div>

        <div class="feature-card">
          <div class="feature-title">📊 Symmetry & Distribution Analysis</div>
          <div class="feature-tag">Work with kernel CSV files</div>
          <p style="margin-top:0.4rem;">
          Load a CSV of flattened kernels, compute the mean kernel and its symmetry score,
          and visualize the distribution of symmetry scores with mean and median markers.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <div class="feature-card">
          <div class="feature-title">🧮 Condition Number Reconditioning 3×3</div>
          <div class="feature-tag">SVD-based reconditioning</div>
          <p style="margin-top:0.4rem;">
          Process a CSV of flattened 3×3 kernels, compute condition numbers, compare them
          with a threshold <code>C</code>, and apply SVD reconditioning when needed.
          The output contains the final kernels, their condition numbers, and flags.
          </p>
        </div>

        <div class="feature-card">
          <div class="feature-title">🔬 Random & Input Kernel Lab</div>
          <div class="feature-tag">Single-kernel sandbox</div>
          <p style="margin-top:0.4rem;">
          Generate random n×n kernels or paste your own matrix, then evaluate symmetry scores,
          condition numbers, and the effect of reconditioning on a single kernel.
          </p>
        </div>

        <div class="feature-card">
          <div class="feature-title">📈 Condition Number Analysis</div>
          <div class="feature-tag">Bulk condition statistics</div>
          <p style="margin-top:0.4rem;">
          From a CSV of 3×3 kernels, compute condition numbers for each kernel, or load a CSV
          of condition numbers and plot their empirical distribution with mean and median.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    ---
    **Tip:** Start with <strong>Model Layer Inspector</strong> if you want to go directly from a
    Keras model to kernel CSVs, then move to the analysis pages for deeper inspection.
    """
)
