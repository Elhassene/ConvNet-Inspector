import numpy as np
import pandas as pd
import tempfile
from io import StringIO
from tensorflow.keras.models import load_model


def kernel_distance(F_in: np.ndarray, F_out: np.ndarray) -> float:
    return float(np.linalg.norm(F_in - F_out, 'fro'))


def normalize_fro(mat):
    f = np.linalg.norm(mat, "fro")
    return mat if f == 0 else (mat / f)

def rotate_90(mat):  return np.rot90(mat, k=1)
def rotate_180(mat): return np.rot90(mat, k=2)
def rotate_270(mat): return np.rot90(mat, k=3)
def reflect_vertical(mat):   return np.fliplr(mat)
def reflect_horizontal(mat): return np.flipud(mat)
def reflect_diagonal_tl_br(mat): return mat.T
def reflect_diagonal_tr_bl(mat): return np.fliplr(mat).T

transformations = [
    rotate_90, rotate_180, rotate_270,
    reflect_vertical, reflect_horizontal,
    reflect_diagonal_tl_br, reflect_diagonal_tr_bl
]

def compute_symmetry_score(mat):
    nk = normalize_fro(mat)
    d = [np.linalg.norm(tf(nk) - nk, "fro") for tf in transformations]
    avg = float(np.mean(d))
    s = 1.0 - 0.5 * avg
    return float(np.clip(s, 0.0, 1.0))

def parse_csv_matrices(csv_bytes):
    df = pd.read_csv(StringIO(csv_bytes.decode("utf-8")), header=None)
    n2 = df.shape[1]
    n = int(np.sqrt(n2))
    if n * n != n2 or n not in (3,5,7,9,11):
        return None, None
    mats = df.values.reshape(-1, n, n)
    return mats, n

def load_model_from_bytes_cached(b):
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        tmp.write(b)
        tmp_path = tmp.name
    return load_model(tmp_path, compile=False)

def kernels_to_matrices(K):
    if K.ndim != 4:
        return None, None, None
    if K.shape[0] <= 11 and K.shape[1] <= 11:
        h, w, in_ch, out_ch = K.shape
        mats = np.transpose(K, (2, 3, 0, 1))
    else:
        out_ch, in_ch, h, w = K.shape
        mats = np.transpose(K, (1, 0, 2, 3))
    mats = mats.reshape(-1, h, w)
    return mats, h, w

def recondition_kernel(F, C):
    C = max(float(C), 1.0)
    U, s, Vh = np.linalg.svd(F, full_matrices=False)
    if s.size == 0:
        return F.copy(), np.inf, np.inf, s.copy(), s.copy()
    sigma_max = float(s[0])
    sigma_min = float(s[-1])
    cond_before = np.inf if sigma_min == 0.0 else sigma_max / sigma_min
    if cond_before <= C or sigma_max == 0.0:
        return F.copy(), cond_before, cond_before, s.copy(), s.copy()
    floor_val = sigma_max / C
    s_new = s.copy()
    n = len(s_new)
    j = n - 1
    while j >= 0 and s_new[j] < floor_val:
        s_new[j] = floor_val
        j -= 1
    for i in range(j + 1, n):
        s_new[i] = 0.5 * (s_new[i - 1] + s_new[i])
    F_rec = (U * s_new) @ Vh
    sigma_min_after = float(s_new[-1])
    cond_after = np.inf if sigma_min_after == 0.0 else float(s_new[0]) / sigma_min_after
    return F_rec, cond_before, cond_after, s.copy(), s_new
