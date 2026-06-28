# Advances in Deep Learning — Quiz

Topics covered: Introduction (§1), Advanced Training (§2), LLMs (§4)

---

# Part 1 — Introduction

## Deep Network Structure & Training (§1.1, §1.2)

**Q1.**
A deep network is described as a "large, differentiable function." Why is differentiability the essential property for training, and what breaks if even one component in the computation graph is non-differentiable?

**Q2.**
Compare linear layers and non-linear layers in terms of (a) parameter count, (b) computational cost, and (c) their respective roles in a deep network. Why can't you build a useful deep network with only linear layers?

**Q3.**
Transformer blocks and convolutional blocks are the two dominant building blocks in deep networks. What is the fundamental difference in how they process their inputs, and what does this mean for the types of data each is suited to?

**Q4.**
Categorical inputs need special handling before they can enter a deep network. Explain the difference between one-hot encoding and learned embeddings, and why embeddings are strictly more expressive for most tasks.

**Q5.**
The Adam optimizer stores extra state per parameter. Specifically: what two quantities does it track, what are they used for, and why does this mean a model with N parameters requires roughly 3N floating-point values during training?

**Q6.**
L1 and L2 regression losses both measure prediction error. When would you prefer L1 over L2 and why? What property of L1 makes it more robust in practice?

**Q7.**
Cross-entropy loss for multi-class classification is derived from maximum likelihood estimation under a categorical distribution. Trace through why minimizing cross-entropy is equivalent to maximizing the log-likelihood of the correct class labels.

## Modern GPU Architecture (§1.3)

**Q8.**
The H100 GPU has ~130 SMs each with 128 FP32 cores, yet the dominant bottleneck in deep learning workloads is memory bandwidth, not compute. Explain why, using the relationship between FLOPs/byte and typical matrix multiplication arithmetic intensity.

**Q9.**
The GPU memory hierarchy goes: register → L1/shared memory (SRAM, ~200 KB per SM) → L2 cache → global HBM memory (80 GB). How does this hierarchy explain why algorithms like FlashAttention achieve better real-world performance than a naive attention implementation, even if both have the same FLOP count?

**Q10.**
A warp executes 32 threads in lockstep (SIMT). What happens to throughput when different threads in a warp take different branches in an if-else statement? What is this problem called, and how do deep learning kernels typically avoid it?

---

# Part 2 — Advanced Training

## Memory & Precision (§2.1, §2.2)

**Q11.**
For an 8B parameter model trained with Adam in FP32, estimate the total memory required for weights, gradients, first momentum, and second momentum. Now apply mixed precision (BF16 for gradients and first momentum, FP32 for weights and second momentum). What is the memory reduction, and why is the second momentum kept in FP32?

**Q12.**
FP16 and BF16 both use 16 bits, but BF16 is preferred for training. What is the key structural difference between them, what failure mode does this prevent in BF16, and why does FP16 typically require gradient scaling while BF16 does not?

**Q13.**
`torch.autocast` silently keeps certain operations (e.g., layer normalization) in FP32 even inside a BF16 context. Why? What would go wrong if normalization ran in BF16?

## Distributed Training (§2.3, §2.4)

**Q14.**
Data Parallelism 1.0 uses a centralized model server; Data Parallelism 1.1 uses all-reduce. What are the two bottlenecks that the server introduces, and how does all-reduce (specifically ring-allreduce) eliminate them?

**Q15.**
Pipeline parallelism splits the model by layers across GPUs. What is a "pipeline bubble," what causes it, and how does micro-batching reduce its impact? Why does memory scale linearly with the number of GPUs in this scheme?

**Q16.**
ZeRO has three stages (ZeRO-1, ZeRO-2, ZeRO-3). For each stage, state precisely what is partitioned, what collective operation is needed to reconstruct it during forward/backward, and the approximate memory reduction per stage.

**Q17.**
FSDP is described as a more efficient implementation of ZeRO-3. What specific engineering improvement does FSDP make over naive ZeRO-3 to reduce communication overhead, and why does this matter at scale (thousands of nodes)?

**Q18.**
In distributed training with all-reduce, why can't you simply use gradient accumulation across micro-batches the same way you would on a single GPU? What synchronization constraint does this impose?

## LoRA and QLoRA (§2.5, §2.7)

**Q19.**
LoRA represents a weight update as $\Delta W = A \cdot B$ where $A \in \mathbb{R}^{n \times r}$, $B \in \mathbb{R}^{r \times m}$, $r \ll n, m$. Why is $B$ initialized to zeros and $A$ to small random values rather than the other way around? What would happen if $B$ were also initialized randomly?

**Q20.**
LoRA is applied to linear and attention layers but rarely to convolutional layers. What is the practical reason for this, and in a transformer with query, key, value, and output projections, which of these typically benefit most from LoRA adaptation?

**Q21.**
QLoRA's memory formula is $0.5N + 16M$, where $N$ is the number of base model parameters and $M$ is LoRA parameters. Explain what each term represents, why the base model can be 0.5 bytes/parameter, and why LoRA adapters must remain in higher precision.

## Quantization (§2.6)

**Q22.**
Scale quantization maps weights to integers in $[-T, T]$; affine quantization uses the actual min/max. When are they equivalent? In what distribution of weights does affine quantization give significantly better precision, and why?

**Q23.**
A single large outlier weight destroys the precision of integer quantization for all other weights in a tensor. Explain why this happens mechanically (in terms of bin spacing), and how blockwise quantization solves it. What is the memory overhead of blockwise quantization?

**Q24.**
Empirically, Llama 3.1 shows no perplexity difference between FP16 and 6-bit quantization, minor degradation at 4-bit, and severe degradation at 3-bit. What does this suggest about the information density of large language models, and why is quantization mostly restricted to inference rather than training?

## Activation Checkpointing & Flash Attention (§2.9, §2.10)

**Q25.**
Activation checkpointing stores only a subset of intermediate activations and recomputes the rest during the backward pass. (a) What is the memory-compute tradeoff in the naive approach (store everything) vs. the no-storage approach (recompute everything)? (b) How does checkpointing at block boundaries achieve approximately $2\times$ the forward compute cost while reducing activation memory to $O(D^{1/2})$ or $O(D^{1/3})$?

**Q26.**
Standard attention computes the full $N \times N$ attention score matrix, storing it in global GPU memory (HBM). FlashAttention computes the same mathematical result but uses only $O(N)$ HBM memory. What is the key algorithmic idea that enables this? How does FlashAttention handle the backward pass without storing the softmax outputs?

**Q27.**
FlashAttention II and III achieve 70% and 75% of theoretical GPU peak performance respectively. What specific hardware features do FA-II and FA-III target (A100 vs. H100), and why can't `torch.compile` automatically discover these optimizations?

---

# Part 3 — LLMs (§4)

## Architecture & Tokenization (§4.1)

**Q28.**
Byte Pair Encoding (BPE) is the dominant tokenization method in modern LLMs. Describe the training algorithm precisely. Then give two concrete failure modes that BPE creates for the model at inference time (not for humans reading the tokens — for the model's ability to reason).

**Q29.**
Decoder-only models (GPT, Llama) have largely replaced encoder-decoder models for most tasks. What specific advantage does the decoder-only architecture have for training efficiency (hint: teacher forcing), and what capability does it sacrifice by using causal rather than bidirectional attention?

**Q30.**
The original Transformer uses sinusoidal positional encodings; modern LLMs like Llama use RoPE (Rotary Position Embeddings). Without needing the RoPE formula, explain what property of positional encoding is essential for LLMs to generalize to sequences longer than those seen during training, and why absolute sinusoidal encodings fail here.

**Q31.**
Grouped Query Attention (GQA) is used in Llama, and Multi-Query Attention (MQA) in some other models. Explain the difference between Multi-Head, Grouped-Query, and Multi-Query Attention in terms of how K and V are shared. What is the specific downstream benefit for inference, and what quality might be sacrificed?

## Text Generation (§4.2)

**Q32.**
Greedy decoding and beam search both attempt to find the maximum-probability sequence, yet they often produce worse text than sampling-based methods. Why does maximizing likelihood not correlate with generating "good" text? Provide a concrete failure mode of each method.

**Q33.**
Top-$p$ (nucleus) sampling is preferred over top-$k$ sampling in most production systems. Explain the adaptive property that makes top-$p$ superior: specifically, what does top-$p$ do differently when the model is confident (peaked distribution) vs. uncertain (flat distribution) that top-$k$ cannot do?

**Q34.**
Temperature scales the logits before softmax: $p_t(v) \propto \exp(z_t(v)/T)$. What are the exact effects of $T \to 0$, $T = 1$, and $T \to \infty$ on the output distribution? If a user wants "more creative" outputs for a storytelling application, should they increase or decrease temperature, and what is the risk?

## Instruction Tuning, RLHF, DPO (§4.3, §4.4, §4.5)

**Q35.**
Instruction tuning (SFT) on as few as 13,000 prompt-response pairs (InstructGPT) substantially changes model behavior. Why can a small supervised fine-tuning dataset have such a large effect on a 175B parameter model, given that pre-training used trillions of tokens?

**Q36.**
The RLHF reward model is trained using the Bradley-Terry pairwise preference model. Write out the loss function, identify each term, and explain why pairwise ranking (better/worse) is used instead of absolute score annotation. What architectural choice keeps the reward model smaller than the policy model, and why?

**Q37.**
PPO for RLHF requires four models simultaneously: the reference model, the generator (policy), the critic, and the reward model. What is the role of each? Why is the KL-divergence term $-\beta D_{KL}[P(y|x) \| P_{ref}(y|x)]$ included in the PPO objective, and what happens empirically when it is removed?

**Q38.**
DPO eliminates the reward model and RL component entirely. Derive the key insight: starting from the RLHF RL objective, what is its closed-form solution, and how does substituting this into the Bradley-Terry loss cause $\log Z(x)$ to cancel? What does the final DPO loss do to the token probabilities of $y^+$ vs. $y^-$?

**Q39.**
DPO has three known empirical weaknesses compared to RLHF. Name them and explain the mechanism behind each. In particular, why does DPO tend to produce longer sequences than RLHF, and what data limitation prevents DPO from scaling to larger training sets the way RLHF can?

## KV Cache & Inference Efficiency (§4.9, §4.10)

**Q40.**
Without KV caching, autoregressive inference has $O(N^3 L)$ runtime (generating $N$ tokens through an $L$-layer model). Trace through exactly why: what operation is $O(N^2 L)$ per token, and why must it be repeated $N$ times?

**Q41.**
KV caching reduces inference runtime from $O(N^3 L)$ to $O(N^2 L)$ but increases peak memory from $O(N)$ to $O(NL)$. Explain the memory cost in concrete terms: for a Llama 70B model with 80 layers, 8192-token context, and 8 KV heads of dimension 128, approximately how many bytes does the KV cache require per request (assume BF16)?

**Q42.**
Speculative decoding uses a small draft model to generate $k$ candidate tokens, then verifies them with the large model in a single forward pass. The acceptance criterion is: accept if $P_T(\text{token}) \geq P_Q(\text{token})$, otherwise sample with probability $P_T/P_Q$. Why does this procedure produce outputs with the exact same distribution as running the large model alone (i.e., why is it unbiased)?

**Q43.**
Speculative decoding achieves 3–4x speedup in memory-bound settings, but the gain largely disappears when KV caching is enabled. Explain precisely why KV caching eliminates the speedup from a compute/memory access perspective. In what specific scenario does speculative decoding still help even with KV caching?

**Q44.**
Medusa adds auxiliary prediction heads to the large model itself (rather than using a separate small draft model) and uses tree attention to verify multiple guesses simultaneously. What is tree attention, how does it allow sharing computation across different candidate sequences, and what does PyTorch's Flex Attention provide that makes this practical?

## LLM Training & Infrastructure (§4.7, §4.8, §4.11)

**Q45.**
LLM pre-training uses the next-token prediction objective with teacher forcing. What is teacher forcing, how does it enable parallelization across the entire sequence in a single forward pass, and what is the discrepancy between training and inference that this introduces (exposure bias)?

**Q46.**
Sequence parallelism is needed when the sequence length $N$ becomes so large that activations during training don't fit on a single GPU. At what point does sequence length become the binding memory constraint rather than model size? Briefly describe how sequence parallelism distributes activations across devices while keeping attention computation correct.

**Q47.**
The distinction between "open," "open-weight," and "open-source" LLMs is blurred in practice. Define all three precisely. What are the three components a model needs to be released for a third party to fully reproduce a training run, and why do most "open-source" LLMs fail this definition?

## Alignment, Limits & RL (§4.17, §4.18)

**Q48.**
LLMs are known to "hallucinate" — generate confident-sounding but factually incorrect content. Give two distinct mechanistic explanations for why hallucination occurs (one related to training objective, one related to the generation process), and describe one practical mitigation for each.

**Q49.**
Reinforcement learning applied to LLMs (e.g., for reasoning, as in DeepSeek-R1 or GRPO) differs fundamentally from RLHF. In RLHF, the reward comes from a learned reward model; in reasoning RL, rewards come from verifiable outcomes (correct/incorrect answer). What does this verifiability property enable that RLHF cannot guarantee? What is the key challenge in applying outcome-based RL to tasks where correctness is not easily verified?

**Q50.**
RLHF and DPO both require a reference model $P_{ref}$ and include a KL-divergence penalty to the reference. What is the role of the reference model — why not just optimize the reward without the KL constraint? What failure mode (colloquially called "reward hacking") does the KL penalty prevent, and how does the coefficient $\beta$ let you trade off alignment quality against diversity?
