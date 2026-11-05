
import numpy as np

def simple_conv2d(input_matrix: np.ndarray, kernel: np.ndarray, padding: int, stride: int):

    input_height, input_width = input_matrix.shape
    kernel_height, kernel_width = kernel.shape

    # 1. Apply zero padding
    padded_matrix = np.pad(
        input_matrix,
        ((padding, padding), (padding, padding)),
        mode='constant'
    )

    # 2. Calculate output dimensions
    output_height = (input_height + 2 * padding - kernel_height) // stride + 1
    output_width = (input_width + 2 * padding - kernel_width) // stride + 1

    # 3. Initialize output matrix
    output_matrix = np.zeros((output_height, output_width))

    # 4. Perform convolution
    for i in range(output_height):
        for j in range(output_width):

            # Get current region
            start_i = i * stride
            start_j = j * stride

            region = padded_matrix[
                start_i:start_i + kernel_height,
                start_j:start_j + kernel_width
            ]

            # Element-wise multiplication and summation
            output_matrix[i, j] = np.sum(region * kernel)

    return output_matrix
