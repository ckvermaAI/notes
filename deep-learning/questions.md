# Deep Learning: 50 Questions for Senior/Staff AI/ML Engineers

Questions drawn directly from the lecture notes. Covers optimization, gradients, normalization, residuals, convolutions, attention, positional embeddings, and regularization.

---

## Optimization & Gradient Descent

**Q1.** Full-batch gradient descent computes the exact gradient of the loss over the entire dataset on every update, yet practitioners almost universally prefer mini-batch SGD. Explain the variance-convergence trade-off precisely: what does the variance of the mini-batch gradient estimate equal as a function of batch size $B$, and why does this not make $B = \text{dataset size}$ the obvious winner in practice?

**A1.** Full-batch GD will have low variance and gradients will point to local maxima, while mini-batch SGD will have higher variance and gradients may or may not point to local optimum. But, the SGD will be light compute and the variance will keep the SGD out from local minimas. We can add first and second momentum to SGD to further increase the convergence speed of SGD. Hence, with lower compute on every iteration we can optimize the weights of our neural networks very fast.

**Q2.** The momentum update rule is $\mathbf{v}_t = \mu \mathbf{v}_{t-1} + (1 - \mu) \nabla_\theta l$ and $\theta \leftarrow \theta - \epsilon \mathbf{v}_t$. Mini-batches reduce variance *spatially* (across data points within one step), while momentum reduces variance *temporally* (across successive steps). Describe a loss landscape geometry where each technique helps in a way the other cannot, and explain what it means for them to be *complementary* rather than redundant.



**Q3.** When training a two-layer network on MNIST, a learning rate that works well for layer 1 can cause instability in layer 2 and vice versa. This forces vanilla SGD to use the worst-case learning rate across all layers. Precisely what property of adaptive optimizers (e.g., RMSProp, Adam) solves this problem, and what is the mathematical mechanism that makes per-parameter scaling possible?

**Q4.** Adam v0 initializes both $m$ and $v$ to zero. Explain why this causes a specific failure mode in the early iterations, what the ratio $m/\sqrt{v}$ converges to on the first step, and how the bias-correction terms in Adam v1 fix this. Why does the fix matter practically for very large models?

**Q5.** AdamW moves weight decay out of the gradient and into the weight update directly: $\theta \leftarrow \theta - \epsilon \cdot (m_t / (\sqrt{v_t} + \epsilon_{\text{small}}) + \lambda \theta)$. Adam with L2 regularization added to the loss achieves what is conceptually the same goal, yet these are *not equivalent*. Explain why they diverge mathematically and what AdamW's formulation corrects.

**Q6.** The Lion optimizer takes the *sign* of a momentum blend rather than the magnitude-scaled gradient that Adam uses. What are the memory implications (weights, gradients, optimizer states), and under what circumstances does Lion's sensitivity to batch size become a practical disadvantage compared to AdamW?

**Q7.** The linear scaling rule says: if you increase batch size by factor $k$, you can increase learning rate by $k$. Explain the theoretical justification (via gradient variance), the regime where it breaks down, and why a warm-up phase is commonly added when scaling to large batches.

**Q8.** Describe the gradient-to-weight ratio as a diagnostic tool. What does a high ratio imply about training dynamics, and why does a learning rate of 1 on even a simple linear network cause periodic loss spikes rather than monotonic divergence?

---

## Vanishing & Exploding Gradients

**Q9.** In a chain of $n$ scalar linear layers with weight $w$, the gradient arriving at layer $k$ from the output is proportional to $w^k$. Derive the condition on $|w|$ that separates vanishing from exploding gradients, and explain why $w = 1$ is theoretically ideal but practically unachievable with random initialization.

**Q10.** A network with vanishing gradients can exhibit a characteristic loss curve: an initial drop followed by a plateau with minor fluctuations. Describe the specific diagnostic procedure using a *zero learning rate* baseline to distinguish genuine learning from loss fluctuations caused purely by batch variance. Why is this test more reliable than simply plotting gradient norms?

**Q11.** Xavier and He initialization aim to keep the product of weight matrix norms $\prod_{i=1}^n \|W_i\|$ close to 1. Why is this the right objective, and how do these initializations achieve it for networks using (a) tanh/sigmoid and (b) ReLU activations? What different variance formulas do they use?

**Q12.** Gradient clipping caps gradient magnitudes during backpropagation. Under what conditions is this a better remedy for exploding gradients than simply reducing the learning rate? In what type of architecture was the exploding gradient problem historically most severe before modern solutions?

---

## Normalization

**Q13.** Batch normalization computes statistics over dimensions $(B, H, W)$ for each channel $c$: $\mu_c = \frac{1}{BHW}\sum_{b,h,w} x_{bchw}$. Explain why BatchNorm is problematic with (a) small batch sizes, (b) distributed training across multiple GPUs, and (c) autoregressive inference at test time. What specific problem occurs at inference, and how is it resolved?

**Q14.** Layer normalization normalizes across all channels for each individual example: $\mu = \frac{1}{C}\sum_c x_c$. BatchNorm and LayerNorm differ in which dimensions they average over. Draw the normalization "axes" for a 4D tensor $(B, C, H, W)$ for BatchNorm, LayerNorm, and GroupNorm, and explain why LayerNorm is the default choice for transformers.

**Q15.** Normalization layers typically add learnable scale ($\gamma$) and bias ($\beta$) parameters after the normalization step: $y = \gamma \hat{x} + \beta$. Why are these parameters necessary, specifically in the context of placing BatchNorm before a ReLU activation?

**Q16.** The notes state that normalization works by keeping the eigenvalues of the network's Jacobian near one. Connect this mathematical insight to the intuition that normalization prevents activations from saturating at the extremes of an activation function. What does saturation imply for the gradient flow?

**Q17.** A practitioner argues: "I should use BatchNorm for my convolutional image classification model and LayerNorm for my transformer-based language model." Construct a principled defense or refutation of each half of this claim, referencing the specific statistics each norm computes.

---

## Residual Connections

**Q18.** A 56-layer network empirically underperforms a 20-layer network, even on the *training set* — not just on test. This rules out overfitting as the cause. What is the actual cause, and why can't this be fixed by simply making the deeper network "learn the identity function" in its extra layers?

**Q19.** The gradient of the loss with respect to the input of a residual block is $\frac{\partial L}{\partial x} = \frac{\partial L}{\partial f(x)} \cdot (1 + \frac{\partial (Wx+b)}{\partial x})$. Explain specifically what the "+1" term does for gradient flow compared to a non-residual layer, and why this prevents vanishing gradients even in very deep networks.

**Q20.** Residual connections require input and output dimensions to match for the skip addition. When dimensions change (e.g., channel doubling at a strided layer), a $1 \times 1$ convolution is applied to the residual path. The notes state this "slightly weakens gradient flow" and should be used sparingly (4-5 times per network). Explain why the $1 \times 1$ projection weakens gradient flow compared to a direct identity shortcut.

**Q21.** Stochastic depth randomly drops entire residual blocks during training, and performance remains strong. What does this robustness reveal about how residual networks distribute their function across layers? How does this relate to the theoretical claim that wide residual networks can approximate invertible functions?

**Q22.** Normalization alone (without residuals) enables training up to ~20-30 layers; residuals alone (without normalization) help but have practical limits; combined, they enable networks with hundreds or thousands of layers. Describe the distinct failure mode each technique addresses, and explain why they are complementary rather than solving the same problem.

---

## Activation Functions

**Q23.** The dying ReLU problem occurs when a neuron's pre-activation is persistently negative, driving its gradient to zero and halting learning. Describe two *distinct* mechanisms that can push a neuron into this dead state, and explain why Leaky ReLU and ELU address the problem differently at a gradient level.

**Q24.** GELU is defined as $\text{GELU}(x) = x \cdot \Phi(x)$ where $\Phi$ is the standard normal CDF. It has a slight dip below zero for small positive $x$, unlike ReLU which is strictly non-negative. What property does this non-monotonicity give GELU that ReLU lacks, and why has it become the activation of choice in transformer architectures?

**Q25.** Sigmoid and tanh were the dominant activation functions before ~2010. Explain precisely *why* they cause vanishing gradients for large inputs that ReLU does not, using the derivative expressions of each. Why does this become catastrophically worse as network depth increases?

---

## Convolutions

**Q26.** A $1024 \times 1024 \times 3$ image fed into a fully connected layer with 4000 outputs requires ~13 billion parameters. A single $3 \times 3$ convolution with 3 input channels and 3 output channels requires 81 parameters. Identify the two structural assumptions convolutions make about natural images that justify this compression, and explain why these assumptions hold for images but would be inappropriate for, say, a tabular dataset.

**Q27.** The receptive field of a standard $3 \times 3$ convolutional stack grows linearly: $2n+1$ for $n$ layers. Striding (stride 2) grows it exponentially. Derive the receptive field after 3 layers of $3 \times 3$ conv with stride pattern [2, 2, 1], and explain why this exponential growth comes at a specific cost that matters for dense prediction tasks.

**Q28.** Dilated (atrous) convolution with dilation factor $d$ inserts $d-1$ zeros between kernel elements, expanding the receptive field without reducing output resolution. Compare dilation to striding across three dimensions: (a) receptive field growth, (b) output resolution, and (c) computational cost. Under what task requirement is dilation clearly preferred over striding?

**Q29.** Transposed convolution (up-convolution) is conceptually the reverse of strided convolution: it inserts zeros between input elements before applying a standard conv. The notes warn it is incorrectly called "deconvolution." What does deconvolution mean in signal processing, and why is this name wrong? Also describe the checkerboard artifact that transposed convolutions can produce and how it arises.

**Q30.** A U-Net architecture uses strided convolutions in the encoder (downsampling path) and transposed convolutions in the decoder (upsampling path), often with skip connections between encoder and decoder levels. Explain why the skip connections are necessary even though the decoder already has upsampling, specifically in terms of what information is preserved vs. lost through the bottleneck.

---

## Attention Mechanism

**Q31.** Basic self-attention without linear projections ($Q = K = V = X$) has a fundamental constraint when using cosine distance: an element always attends more strongly to itself than to any other element. Explain the mathematical reason for this constraint and why it limits the expressiveness of the attention operator.

**Q32.** Scaled dot-product attention divides the attention scores by $\sqrt{C}$ before the softmax: $A = \text{softmax}(QK^T / \sqrt{C})$. Explain what problem occurs in high dimensions without this scaling, specifically in terms of the distribution of dot products and what it does to the softmax gradients.

**Q33.** Self-attention has computational complexity $O(N^2 \cdot C)$ for a sequence of length $N$ and dimension $C$. Explain why this becomes a bottleneck for long sequences. What is the nature of the $N^2$ term — what operation generates an $N \times N$ matrix — and what are the two main families of approaches used to approximate or avoid this quadratic cost?

**Q34.** In cross-attention, queries $Q$ come from one sequence and keys/values $K, V$ come from another. The output has shape $M \times C$ where $M$ is the number of queries and $N$ is the number of keys/values, and $M \neq N$ is allowed. Give a concrete example from computer vision or multimodal learning where this asymmetry ($M \neq N$) is not just allowed but architecturally essential.

**Q35.** Tokenization (subword units) is described as a "middle ground" between character-level and word-level splitting. Explain the trade-off: what goes wrong at the character level (sequence too long) and what goes wrong at the word level (vocabulary too large), and why subword tokenization resolves both simultaneously?

---

## Multi-Head Attention

**Q36.** Multi-head attention with $h$ heads is claimed to be mathematically as expressive as a 2D convolution with kernel size $\sqrt{h} \times \sqrt{h}$. Sketch the argument: how can individual attention heads be arranged to replicate the function of a convolutional kernel? What do the weight matrices $W_V^i$ and $W_O$ correspond to in this analogy?

**Q37.** Multi-head attention uses two sets of linear layers: pre-attention projections $W_Q^i, W_K^i, W_V^i$ (per head) and a post-attention output projection $W_O$. Since attention is linear in values $V$, $W_V^i$ and $W_O$ could theoretically be merged into one. The notes give two practical reasons they are kept separate — state these reasons and explain the efficiency argument in terms of per-head dimensionality.

**Q38.** A single-head attention mechanism must represent all relationships within a sequence through one shared attention matrix, so it can only focus on one location or produce an average. Multi-head attention addresses this with parallel heads. What is the analogy to convolutional neural networks — specifically, what is the equivalent property CNNs achieve through multiple output channels?

---

## Positional Embeddings

**Q39.** Attention is permutation-invariant: permuting all of $Q$, $K$, $V$ together leaves the output unchanged. This is sometimes *desirable* (give a concrete example) and sometimes a fatal flaw (give a different concrete example). What structural property of the input determines whether permutation invariance is a feature or a bug?

**Q40.** Sinusoidal positional embeddings use $PE(n, 2i) = \sin(n / 10000^{2i/c})$ and $PE(n, 2i+1) = \cos(n / 10000^{2i/c})$. The frequencies sweep from high (small $i$) to low (large $i$). Explain why this multi-frequency design is superior to simply using the position integer $n$ as a scalar feature, and how the network can use low- and high-frequency components together to localize any position efficiently.

**Q41.** Learnable positional embeddings assign a trained vector to each position index. They can outperform sinusoidal embeddings when training sequence length is fixed, but have a hard failure mode. Describe this failure mode precisely and explain why sinusoidal embeddings do not share it.

**Q42.** Relative positional embeddings encode the *difference* $m - n$ between query position $m$ and key position $n$, rather than their absolute values. Explain two advantages this gives: (a) better generalization to unseen sequence lengths, and (b) an analogy to a structural property of CNNs. What is that CNN property called?

**Q43.** RoPE (Rotary Positional Embedding) applies a rotation matrix $R_m$ to queries and $R_n$ to keys *before* the dot product. The key insight is that $R_m^T R_n = R_{m-n}$. Trace through the math to show that the attention score $q_m^T k_n$ becomes a function of only the *relative* position $m - n$, even though $R_m$ and $R_n$ encode *absolute* positions. Why does this property make RoPE superior to additive sinusoidal embeddings for long-context language models?

**Q44.** Positional embeddings are also used in Neural Radiance Fields (NeRF) to map 3D coordinates $(x, y, z)$ to occupancy and color. In this context, the purpose is opposite to the transformer use case: rather than helping the model distinguish positions, the embeddings help the model represent *high-frequency* spatial detail. Explain why a plain MLP would fail to learn high-frequency functions of coordinates without these embeddings, referencing the spectral bias of neural networks.

---

## Overfitting & Regularization

**Q45.** The notes explain overfitting through a high-dimensional geometry argument: in high-dimensional spaces, training, validation, and test points occupy distinct regions, making it easy to find separating hyperplanes. Connect this to the bias-variance trade-off. Does more depth (more parameters) always increase the risk of overfitting, or does the double-descent phenomenon complicate this picture?

**Q46.** Dropout randomly zeros a fraction $\alpha$ of activations during training and scales the remaining activations by $1/(1-\alpha)$. The notes note that dropout placed before a standard convolutional layer is ineffective. Explain precisely why: what property of convolutional receptive fields undermines the regularization effect, and where should dropout be placed in a CNN to be effective?

**Q47.** The notes make a counterintuitive claim about weight decay: it does *not* significantly prevent overfitting, but is still valuable. Explain what weight decay actually does mechanistically (in terms of gradient magnitudes and weight explosions), and why the traditional narrative ("L2 penalty reduces model complexity → less overfitting") is misleading for modern deep networks.

**Q48.** Ensembling $M$ models and averaging their predictions gives 1-3% accuracy gains in practice. The notes justify this via Jensen's inequality. State the relevant inequality, explain what convexity assumption is being made, and describe the source of diversity between ensemble members (since they are trained on the same data).

**Q49.** The test set becomes "contaminated" if used repeatedly for model selection. Explain the statistical mechanism: why does repeated evaluation on the same test set cause the reported performance to become an overestimate of true generalization? What is the correct protocol, and how does it relate to the multiple comparisons problem in statistics?

**Q50.** Data augmentation during training can, in principle, produce *better validation accuracy than training accuracy* — a form of "negative overfitting" the notes mention. Describe the mechanism: what about aggressive augmentation makes the training task harder than the validation task, and in what real-world scenario might you deliberately engineer this imbalance?

---

*Answer these at your own pace. We'll review them together afterwards.*
