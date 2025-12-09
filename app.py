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
    .feature-card-button {
        padding: 1.0rem 1.2rem;
        border-radius: 0.8rem;
        border: 1px solid rgba(128,128,128,0.3);
        backdrop-filter: blur(6px);
        margin-bottom: 0.8rem;
        transition: all 0.15s ease-in-out;
        text-align: left;
        width: 100%;
        background: transparent;
        color: inherit;
        text-decoration: none;
        outline: none;
        font: inherit;
    }
    .feature-card-button:hover {
        border-color: rgba(0,200,255,0.8);
        background: rgba(255,255,255,0.02);
        box-shadow: 0 4px 12px rgba(0,0,0,0.35);
        transform: translateY(-2px);
        cursor: pointer;
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
        A toolkit for exploring convolution kernels in Keras <code>.h5</code> models,
        analyzing symmetry and condition numbers, generating statistical plots,
        and applying SVD-based reconditioning techniques.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("Use the <strong>sidebar</strong> to navigate between tools. Below is an overview of each page.", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
        <form action="/Model_Layer_Inspector" method="get">
          <button class="feature-card-button" type="submit">
            <div class="feature-title">📦 Model Layer Inspector</div>
            <div class="feature-tag">Upload a Keras <code>.h5</code> model</div>
            <p style="margin-top:0.4rem;">
              Inspect all layers of a neural network, filter convolution kernels by spatial size,
              and export the kernels of any chosen layer to CSV for deeper analysis.
            </p>
          </button>
        </form>

        <form action="/Symmetry_and_Distribution_Analysis" method="get">
          <button class="feature-card-button" type="submit">
            <div class="feature-title">📊 Symmetry &amp; Distribution Analysis</div>
            <div class="feature-tag">Analyze kernel CSV files</div>
            <p style="margin-top:0.4rem;">
              Load CSV files that contain flattened kernels, compute their mean kernel and symmetry score,
              and visualize symmetry score distributions with mean and median markers.
            </p>
          </button>
        </form>

        <form action="/Symmetry_Map_From_Image" method="get">
          <button class="feature-card-button" type="submit">
            <div class="feature-title">🖼️ Symmetry Map From Image</div>
            <div class="feature-tag">Patch-based symmetry visualization</div>
            <p style="margin-top:0.4rem;">
              Upload an image and compute a dense symmetry map using 3×3 patches.
              A symmetry threshold slider lets you interactively highlight regions with high structural symmetry,
              and a histogram summarizes the symmetry score distribution.
            </p>
          </button>
        </form>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <form action="/Condition_Number_Reconditioning_3x3" method="get">
          <button class="feature-card-button" type="submit">
            <div class="feature-title">🧮 Condition Number Reconditioning 3×3</div>
            <div class="feature-tag">SVD-based kernel repair</div>
            <p style="margin-top:0.4rem;">
              From a CSV of 3×3 kernels, compute the condition number of each kernel and apply
              SVD-based reconditioning to those exceeding a chosen threshold <code>C</code>.
              The output CSV contains the reconditioned kernels and condition metrics.
            </p>
          </button>
        </form>

        <form action="/Random_and_Input_Kernel_Lab" method="get">
          <button class="feature-card-button" type="submit">
            <div class="feature-title">🔬 Random &amp; Input Kernel Lab</div>
            <div class="feature-tag">Single-kernel sandbox</div>
            <p style="margin-top:0.4rem;">
              Generate random kernels or input your own matrix, then compute symmetry scores,
              condition numbers, and observe how reconditioning changes the structure.
            </p>
          </button>
        </form>

        <form action="/Condition_Number_Analysis" method="get">
          <button class="feature-card-button" type="submit">
            <div class="feature-title">📈 Condition Number Analysis</div>
            <div class="feature-tag">Bulk condition statistics</div>
            <p style="margin-top:0.4rem;">
              From a CSV of 3×3 kernels, compute condition numbers for each kernel, or load a CSV
              of condition numbers and plot their empirical distribution with mean and median.
            </p>
          </button>
        </form>
        """,
        unsafe_allow_html=True,
    )
