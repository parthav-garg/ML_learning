
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


if __name__ == "__main__":
    B, C_in, L_in = 4, 3, 8
    K = 3                # kernel size
    C_out = 5            # number of output channels
    L_out = L_in - K + 1

    # Input and kernel
    x = np.random.randn(B, C_in, L_in)           # shape: (B, C_in, L_in)
    kernel = np.random.randn(C_out, C_in, K)     # shape: (C_out, C_in, K)

    # Step 1: Sliding windows over last dimension
    # shape: (B, C_in, L_out, K)
    windows = sliding_window_view(x, window_shape=K, axis=2)
    print(windows.shape)
    # Step 2: reshape windows to (B, L_out, C_in * K)
    X_unrolled = windows.transpose(0, 2, 1, 3).reshape(B, L_out, C_in * K)

    # Step 3: reshape kernel to (C_out, C_in * K)
    K_flat = kernel.reshape(C_out, C_in * K)

    # Step 4: Matrix multiply: (B, L_out, C_in*K) @ (C_in*K, C_out)^T → (B, L_out, C_out)
    output = X_unrolled @ K_flat.T

    # Step 5: transpose to (B, C_out, L_out)
    output = output.transpose(0, 2, 1)

    print("Output shape:", output.shape)  # (4, 5, 6)
    print(output)