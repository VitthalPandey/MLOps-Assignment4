# src/quantize.py

import numpy as np
from joblib import dump
import src.utils as utils

def quantize_array(arr, min_val=None, max_val=None):
    """
    Quantize a NumPy array to uint8 using min-max scaling.
    Handles edge case where all values are the same.
    """
    if min_val is None:
        min_val = arr.min()
    if max_val is None:
        max_val = arr.max()

    if max_val == min_val:
        # Constant value → use 127 (midpoint of uint8)
        q_arr = np.full_like(arr, fill_value=127, dtype=np.uint8)
    else:
        scale = 255 / (max_val - min_val)
        q_arr = ((arr - min_val) * scale).round().astype(np.uint8)

    return q_arr, min_val, max_val

def main():
    # Load trained model
    model = utils.load_model("model.joblib")

    coef = model.coef_
    intercept = model.intercept_

    # Save raw unquantized weights
    dump((coef, intercept), "unquant_params.joblib")
    print("✅ Saved unquantized parameters to unquant_params.joblib")

    # Quantize coefficients and intercept
    q_coef, coef_min, coef_max = quantize_array(coef)
    q_intercept, int_min, int_max = quantize_array(np.array([intercept]))

    # Store everything needed for dequantization
    quant_info = {
        "q_coef": q_coef,
        "coef_min": coef_min,
        "coef_max": coef_max,
        "q_intercept": q_intercept[0],  # convert back from array
        "intercept_min": int_min,
        "intercept_max": int_max
    }

    dump(quant_info, "quant_params.joblib")
    print("✅ Saved quantized parameters to quant_params.joblib")

if __name__ == "__main__":
    main()
