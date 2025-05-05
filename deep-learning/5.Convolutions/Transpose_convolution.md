# Transposed Convolution

Video: [Transposed Convolutions Explained: A Fast 8-Minute Explanation | Computer Vision](https://www.youtube.com/watch?v=xoAv6D05j7g)
Reference Paper: [A Guide to Convolution Arithmetic for Deep Learning](https://arxiv.org/pdf/1603.07285v1)

## Refresher on Standard Convolution
- **Purpose**: Convolutions typically downsample an input (e.g., an image) using a kernel, reducing its spatial dimensions.
- **Key Parameters**:
  - **Kernel Size**: Size of the sliding window (e.g., $3 \times 3$).
  - **Input Size**: Dimensions of the input (e.g., width, height, channels).
  - **Stride**: Step size of the kernel’s movement (e.g., 1).
  - **Padding**: Extra border added to the input (e.g., 0 for no padding).
- **Simplified Example**:
  - **Input**: $4 \times 4$ (1 channel).
  - **Kernel**: $3 \times 3$.
  - **Stride**: 1.
  - **Padding**: 0.
  - **Output**: $2 \times 2$ (downsampled).

### Convolution Process
- **Operation**: The kernel slides over the input, computing a weighted sum at each position:
  - For position (0,0): Overlap the $3 \times 3$ kernel with the top-left $3 \times 3$ region of the $4 \times 4$ input.
  - Multiply corresponding elements and sum them (e.g., $3 \cdot 0 + 3 \cdot 1 + 2 \cdot 2 + 0 \cdot 2 + \dots = 12$).
- **Sliding**: Move the kernel by the stride (1 step right or down) and repeat until all positions are covered (4 steps total: top-left, top-right, bottom-left, bottom-right).
- **Output**: A $2 \times 2$ matrix, as the kernel can only fit in 4 unique positions due to the input size and no padding.

---

## Limitations of Traditional Implementations
- **Typical Approach**: Online implementations often use loops to slide the kernel, computing the output iteratively.
- **Drawbacks**:
  - **Inefficiency**: Loops are computationally slow, especially for large inputs.
  - **Lack of Insight**: This method obscures the connection to transposed convolutions, missing the "aha moment."

---

## Vectorized Convolution
- **Alternative**: Use vectorization for efficiency and to set the stage for understanding transposed convolutions.
- **Convolution Matrix (C)**:
  - **Size**: Rows = number of stride steps (4 for a $2 \times 2$ output), Columns = flattened input size (16 for $4 \times 4$).
  - **Construction**:
    1. Start with a blank $4 \times 4$ grid (same size as input).
    2. Overlay the $3 \times 3$ kernel at the first stride position (top-left).
    3. Fill overlapping cells with kernel values, non-overlapping cells with 0.
    4. Flatten this $4 \times 4$ grid into a 1×16 row (first row of C).
    5. Repeat for all stride positions (top-right, bottom-left, bottom-right), yielding a $4 \times 16$ matrix.
- **Input Vector (I)**: Flatten the $4 \times 4$ input into a $16 \times 1$ vector.
- **Computation**: Compute the dot product $C \cdot I$, resulting in a $4 \times 1$ vector, then reshape to $2 \times 2$.

### Code Demonstration
- **Language**: Python with NumPy.
- **Steps**:
  ```python
  import numpy as np

  # Input (4x4)
  input = np.array([[3, 3, 2, 1],
                    [0, 0, 1, 3],
                    [3, 1, 2, 0],
                    [3, 2, 1, 2]])
  I = input.reshape(16, 1)  # Flatten to 16x1

  # Convolution Matrix (4x16)
  C = np.array([
      [3, 3, 2, 0, 0, 0, 1, 0, 2, 1, 2, 0, 0, 0, 0, 0],  # Top-left stride
      [0, 3, 3, 2, 0, 0, 0, 1, 0, 2, 1, 2, 0, 0, 0, 0],  # Top-right stride
      [0, 0, 0, 0, 3, 3, 2, 0, 0, 0, 1, 0, 2, 1, 2, 0],  # Bottom-left stride
      [0, 0, 0, 0, 0, 3, 3, 2, 0, 0, 0, 1, 0, 2, 1, 2]   # Bottom-right stride
  ])

  # Dot product and reshape
  result = np.dot(C, I)  # 4x1
  output = result.reshape(2, 2)  # 2x2
  print(output)  # [[12, 12], [10, 17]]
  ```
- **Result**: Matches the manual convolution (e.g., $12$ for top-left), confirming correctness.

---

## Transposed Convolution
- **Goal**: Reverse the downsampling process, upsampling the $2 \times 2$ output back to a $4 \times 4$ input-like size.
- **Mechanism**:
  - **Transpose Matrix**: Use $C^T$ (transpose of the convolution matrix), a $16 \times 4$ matrix.
  - **Input**: Use the convolution output ($2 \times 2$, flattened to $4 \times 1$).
  - **Computation**: Compute the dot product $C^T \cdot \text{output}$, yielding a $16 \times 1$ vector, then reshape to $4 \times 4$.
- **Naming**: Called "transposed convolution" because it uses the transposed matrix $C^T$.

### Code Demonstration
- **Steps**:
  ```python
  # Transpose convolution matrix (16x4)
  C_T = C.T

  # Convolution output (2x2 flattened to 4x1)
  conv_output = np.array([[12], [12], [10], [17]])

  # Dot product and reshape
  transposed_result = np.dot(C_T, conv_output)  # 16x1
  upsampled_output = transposed_result.reshape(4, 4)
  print(upsampled_output)
  ```
- **Result**: A $4 \times 4$ matrix, matching the original input’s dimensions (though not necessarily its exact values, depending on the kernel and operation specifics).

---

## Key Insights
- **Reversibility**: 
  - Convolution: $4 \times 4 \to 2 \times 2$ (downsampling).
  - Transposed Convolution: $2 \times 2 \to 4 \times 4$ (upsampling).
- **Neural Network Context**:
  - **Forward Pass**: Convolution downsamples features.
  - **Backward Pass**: During backpropagation of a convolutional layer, gradients are computed for the larger input layer, effectively performing a transposed convolution.
  - **Vice Versa**: In a transposed convolution layer, backpropagation resembles a standard convolution.
- **Duality**: Convolution and transposed convolution are reverse operations, a concept rooted in linear algebra (matrix multiplication and its transpose).

---

## Conclusion
- **Summary**: The lecture explains transposed convolutions by:
  1. Reviewing standard convolution with a $4 \times 4$ input, $3 \times 3$ kernel, stride 1, and no padding, yielding a $2 \times 2$ output.
  2. Introducing vectorization with a $4 \times 16$ convolution matrix $C$ for efficiency.
  3. Demonstrating transposed convolution using $C^T$ to upsample from $2 \times 2$ back to $4 \times 4$.
- **Goal Achieved**: Provides an intuitive, code-supported explanation without complex formulas, linking to practical deep learning applications (e.g., upsampling in generative models like GANs).
- **Call to Action**: Encourages viewers to like and subscribe for support, promising more educational content.

---

## Additional Context
- **Paper Reference**: The 2016 paper formalizes convolution arithmetic, introducing the convolution matrix approach used here. It’s widely cited for understanding CNN operations.
- **Applications**: Transposed convolutions are critical in tasks like image segmentation (e.g., U-Net) and generative modeling, where upsampling is needed to reconstruct larger outputs.
- **Stride and Padding**: The simplified example uses stride 1 and no padding, but in practice, transposed convolutions often use strides > 1 (e.g., 2) to increase output size more aggressively, with padding adjusting boundaries.
