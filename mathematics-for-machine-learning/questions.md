# Mathematics for ML — Part 1 Quiz

Topics covered: Linear Algebra, Analytic Geometry, Matrix Decomposition, Vector Calculus, Continuous Optimization

---

**Q1.**
The rank-nullity theorem states dim(ker(A)) + dim(im(A)) = dim(V).
For A in R^(3x5) with rank 2, what is the dimension of the null space? What does this tell you about the solution set of Ax = b when b is in the column space of A?

**A1.**
dim(ker(A)) = 5 - rank(A) = 5 - 2 = 3, by rank-nullity (domain dimension is n=5, not m=3).

When b is in the column space of A, a solution exists. Since the null space is 3-dimensional, the solution set is an affine subspace: one particular solution x_p plus all vectors in ker(A) — infinitely many solutions of the form x = x_p + sum(lambda_i * n_i) where n_i span the 3D null space.

---

**Q2.**
You have two matrices A and B representing the same linear map under different bases. What property do they share, and what is the algebraic relationship between them? What invariants are preserved?

**A2.**
They are similar matrices: B = S^{-1}AS for some invertible S. This represents the same linear map expressed in a different basis — S is the change-of-basis matrix.

Preserved invariants: eigenvalues, characteristic polynomial, determinant, trace, rank. These are intrinsic to the linear map, not the coordinate representation.

---

**Q3.**
A is a 4x4 matrix with characteristic polynomial p(lambda) = (lambda - 3)^2 * (lambda - 1)^2.
(a) What are the possible geometric multiplicities for lambda = 3?
(b) Under what condition is A diagonalizable?
(c) If A is not diagonalizable, what is it called?

**A3.**
(a) Geometric multiplicity of lambda=3 can be 1 or 2 — it must be at least 1 (every eigenvalue has at least one eigenvector) and at most its algebraic multiplicity of 2.

(b) A is diagonalizable iff geometric multiplicity equals algebraic multiplicity for every eigenvalue — so both lambda=3 and lambda=1 must each have geometric multiplicity 2, giving 4 independent eigenvectors total.

(c) If not diagonalizable, A is called defective.

---

**Q4.**
Explain why the projection matrix P = B(B^T B)^{-1} B^T satisfies P^2 = P. What breaks down if B does not have full column rank, and how would you fix it?

**A4.**
Expand P^2 directly:

    P^2 = B(B^T B)^{-1} B^T * B(B^T B)^{-1} B^T
        = B(B^T B)^{-1} (B^T B)(B^T B)^{-1} B^T
        = B(B^T B)^{-1} B^T
        = P

The middle (B^T B)(B^T B)^{-1} collapses to I, which is the key step.

If B lacks full column rank, B^T B is singular — not invertible, so the normal equations have no unique solution. Fix: use the SVD-based Moore-Penrose pseudo-inverse B^+ = V Sigma^+ U^T, which handles rank deficiency gracefully.

---

**Q5.**
You want to sample from a multivariate Gaussian N(mu, Sigma). Walk through the role of Cholesky decomposition in doing this efficiently. Why does the Cholesky factor exist for a valid covariance matrix?

**A5.**
A valid covariance matrix Sigma is always SPD — symmetric by definition, positive definite because variance in any direction must be positive. SPD is precisely the condition guaranteeing Cholesky exists uniquely: Sigma = LL^T with L lower triangular and positive diagonal.

Sampling procedure:
1. Compute L via Cholesky on Sigma (O(n^3), done once)
2. Draw z ~ N(0, I) — n independent standard normals, trivial to sample
3. Set x = mu + Lz

Then x ~ N(mu, L * I * L^T) = N(mu, LL^T) = N(mu, Sigma).

Efficiency: Cholesky costs O(n^3) once, then each sample costs only O(n^2) for the matrix-vector product Lz — far cheaper than repeated eigendecompositions.

---

**Q6.**
For f: R^n -> R, the gradient is defined as a row vector. Why does the gradient descent update use its transpose? What is the Jacobian for f(x) = Ax, and what are its dimensions?

**A6.**
The gradient descent update x_{t+1} = x_t - gamma * grad(f)^T uses the transpose because x is a column vector — the update must be a column vector to be added to x. The gradient as a row vector must be transposed to match dimensions.

For f(x) = Ax where A is m×n (mapping R^n -> R^m), the Jacobian is J = A with dimensions R^(m×n) — m rows, n columns. The domain is R^n and codomain is R^m.

Note: the notation "f -> n : m" has the direction flipped — it is domain R^n to codomain R^m, so m rows and n columns.

---

**Q7.**
The Hessian at a critical point is indefinite. What does this tell you about the point? Why is this situation particularly relevant in training deep neural networks?

**A7.**
An indefinite Hessian at a critical point means it is a saddle point — the function curves upward in some directions and downward in others. It is neither a local minimum nor a local maximum.

This matters deeply for deep networks: in high-dimensional loss landscapes, saddle points vastly outnumber local minima. Near a saddle, gradients are near zero in some directions, so vanilla gradient descent slows dramatically or stalls. This is a core motivation for momentum and adaptive optimizers like Adam, which can move through flat/saddle regions more effectively.

---

**Q8.**
State the KKT complementary slackness condition and explain its geometric meaning. In the context of SVMs, what does it imply about non-support vectors?

**A8.**
**Complementary slackness:** For each inequality constraint $g_i(\boldsymbol{x}) \leq 0$ with multiplier $\lambda_i \geq 0$:
$$\lambda_i^* \cdot g_i(\boldsymbol{x}^*) = 0$$
Either the constraint is active ($g_i = 0$, binding) or its multiplier is zero (no shadow price) — they cannot both be nonzero simultaneously.

**Geometric meaning:** An inactive constraint (the feasible region has slack at $\boldsymbol{x}^*$) contributes nothing to the optimum — the solution would be the same if that constraint were removed. Its shadow price $\lambda_i = 0$ reflects this. An active constraint is exactly tight at the optimum, and its positive $\lambda_i$ encodes how much the objective would improve if the constraint were relaxed.

**In SVMs:** The soft-margin complementary slackness conditions are $\alpha_n(y_n(\langle \boldsymbol{w}, \boldsymbol{x}_n \rangle + b) - 1 + \xi_n) = 0$ and $\gamma_n \xi_n = 0$ (where $\gamma_n = C - \alpha_n$). For a non-support vector ($\alpha_n = 0$): the first condition is satisfied for any margin value, which implies $y_n f(\boldsymbol{x}_n) \geq 1$ strictly — the point is outside the margin with slack. Its dual variable is zero, so it makes no contribution to $\boldsymbol{w}^* = \sum_n \alpha_n y_n \boldsymbol{x}_n$ and does not influence the decision boundary at all.

---

**Q9.**
SVD gives A = U Sigma V^T. The Eckart-Young theorem says the best rank-k approximation is obtained by truncating to the top k singular values.
(a) What is the approximation error in the 2-norm?
(b) Why can't you get a better rank-k approximation by using a different set of k singular vectors?

**A9.**
(a) The approximation error is sigma_{k+1} — the (k+1)-th singular value, i.e. the largest singular value not included in the truncation.

(b) Singular values capture the magnitude of the transformation in each direction. The Eckart-Young theorem proves that no rank-k matrix of any form can achieve a smaller 2-norm error — it is not just intuition but a formal optimality result. The top k singular vectors capture the directions of greatest "energy" in A, and any other rank-k basis would leave more residual.

---

**Q10.**
A function f is convex. You run gradient descent and find a point where the gradient is zero.
(a) What can you conclude about that point?
(b) Now suppose the Hessian at that point is PSD but not PD (i.e., it has a zero eigenvalue). Does your conclusion change?

**A10.**
(a) It is a global minimum. This is the central property of convex functions — any zero-gradient point is a global minimum, and every local minimum is global. (A local maximum would require a zero gradient with a negative definite Hessian, which violates convexity.)

(b) The conclusion does not change — it is still a global minimum. However, with a zero eigenvalue in the Hessian there is a flat direction, meaning it may not be a strict global minimum. There could be a whole subspace of minimizers rather than a unique point.

---

# Part 2 — Applied ML (Senior/Staff Level)

Topics covered: When Models Meet Data, Linear Regression, Dimensionality Reduction (PCA), Density Estimation (GMMs), Support Vector Machines

---

## When Models Meet Data / Linear Regression

**Q11.**
You train a high-degree polynomial regression model and it fits the training data perfectly but performs poorly on the test set. What is happening and what are three distinct ways to fix it?

**A11.**
The model is overfitting: with polynomial degree $M \geq N$, the design matrix $\boldsymbol{\Phi}$ loses full column rank and the model interpolates the training data exactly, including noise. The empirical risk is zero but true risk is high — high variance, near-zero bias.

Three distinct fixes:
1. **Reduce model complexity** — lower polynomial degree $M$, selected via cross-validation on a held-out validation set.
2. **Regularize** — add L2 (ridge) or L1 (LASSO) penalty. Ridge adds $\lambda\boldsymbol{I}$ to $\boldsymbol{\Phi}^T\boldsymbol{\Phi}$, making it always invertible and shrinking weights; LASSO additionally sparsifies them.
3. **Increase training data** — with more samples, the model cannot fit all noise degrees of freedom; model complexity should scale with $N$.

**Q12.**
What is the difference between MLE and MAP estimation in practical terms? If you increase the strength of L2 regularization in ridge regression, what are you implicitly saying about your prior on the weights?

**A12.**
MLE finds the $\boldsymbol{\theta}$ that maximizes $p(\mathcal{D} \mid \boldsymbol{\theta})$ — pure data fit, no prior. MAP maximizes $p(\mathcal{D} \mid \boldsymbol{\theta})\,p(\boldsymbol{\theta})$, i.e. it adds $-\log p(\boldsymbol{\theta})$ to the NLL. In practice MLE tends to overfit with small $N$; MAP regularizes by pulling the solution toward the prior.

For ridge regression, the L2 penalty $\lambda\|\boldsymbol{\theta}\|^2$ corresponds exactly to a Gaussian prior $p(\boldsymbol{\theta}) = \mathcal{N}(\mathbf{0}, \frac{1}{2\lambda}\boldsymbol{I})$. Increasing $\lambda$ is equivalent to decreasing the prior variance $\tau^2 = \frac{1}{2\lambda}$ — you are saying the weights should be concentrated tightly around zero, placing high prior probability on small-magnitude weights. A very large $\lambda$ drives all weights to zero regardless of the data; a very small $\lambda$ recovers MLE.

**Q13.**
In Bayesian linear regression, the predictive variance has two components. What are they, what do they represent, and why does one shrink with more data while the other does not?

**A13.**
The predictive variance at test point $\boldsymbol{x}_*$ decomposes as:
$$\text{Var}[y_*] = \underbrace{\phi(\boldsymbol{x}_*)^T \boldsymbol{S}_N \phi(\boldsymbol{x}_*)}_{\text{epistemic}} + \underbrace{\sigma^2}_{\text{aleatoric}}$$

**Epistemic (model) uncertainty**: $\phi^T\boldsymbol{S}_N\phi$ — uncertainty about the parameter $\boldsymbol{\theta}$. The posterior covariance $\boldsymbol{S}_N = (\boldsymbol{S}_0^{-1} + \sigma^{-2}\boldsymbol{\Phi}^T\boldsymbol{\Phi})^{-1}$ shrinks as more data arrives (each point adds $\sigma^{-2}\phi_n\phi_n^T$ to the precision matrix), so this term decreases with $N$. It also grows in regions far from training data, which is the correct inductive bias.

**Aleatoric (observation) noise**: $\sigma^2$ — irreducible noise in the data-generating process itself. Even with infinite data, every new observation has noise $\epsilon \sim \mathcal{N}(0, \sigma^2)$. No amount of additional training data can reduce this term because it reflects the stochasticity of nature, not our ignorance about the model.


**Q14.**
Your colleague says "we should always normalize features before training." Do you agree? When would you not normalize, and what goes wrong if you skip it when you should have done it?

**A14.**
Mostly agree, but "always" is too strong.

**Why you usually should:** Without normalization, features with large magnitudes dominate the loss gradient, making gradient descent take tiny steps in small-scale directions and huge steps in large-scale directions. This makes the loss surface elongated and ill-conditioned (high $\kappa(\boldsymbol{\Phi}^T\boldsymbol{\Phi})$), slowing convergence dramatically. Regularization also becomes scale-dependent: $\lambda\|\boldsymbol{\theta}\|^2$ treats a weight on a feature measured in km differently than one measured in mm.

**When you should not:**
- Features with the same units and a meaningful common scale (e.g., pixel intensities 0–255 in an image — standardizing per-channel destroys the relative relationship between pixels).
- Tree-based models (XGBoost, random forests) are scale-invariant by construction — normalization has no effect.
- When the scale itself carries information (e.g., raw counts in a Poisson model where larger counts are genuinely more informative).

**The critical rule:** Always fit the normalization statistics (mean, std) on the training set only. Applying training-set statistics to the test set ensures no information leakage.

---

## PCA / Dimensionality Reduction

**Q15.**
You run PCA on a dataset and keep enough components to explain 95% of variance. A junior engineer asks: "what did we throw away?" How do you answer, and when is throwing it away a problem?

**A15.**
You threw away the eigenvectors of the sample covariance $\boldsymbol{S}$ corresponding to the smallest 5% of total variance — the directions in which the data varies the least. The reconstruction error equals exactly $\sum_{j=M+1}^D \lambda_j$ (Eckart-Young), so by construction the discarded components are the ones contributing least to data variation.

**When discarding them is a problem:**
- **The label lives in the low-variance directions.** A classic failure mode: if $y$ is correlated with a low-variance feature (e.g., a rare but discriminative signal), PCA will discard it. PCA is unsupervised — it does not know about the labels. Supervised alternatives (LDA, PLS) avoid this.
- **Anomaly detection.** Anomalies often manifest in the low-variance directions (they are unusual precisely because they deviate from the main structure). Throwing away those components blinds you to them.
- **The "5% is noise" assumption is wrong.** In some domains (e.g., biology, finance) low-variance components capture genuine structure, not noise.

**Q16.**
Why must you fit PCA (compute the mean and principal components) only on the training set and not on the full dataset? What goes wrong if you don't?

**A16.**
PCA on the full dataset causes **data leakage**: the principal components and mean are computed using test examples. This means the transformation is influenced by the test set, breaking the independence assumption that makes test performance an unbiased estimate of generalization.

Concretely: the test set is supposed to simulate unseen data at deployment time. At deployment, you will not have access to future data when computing the projection — you project new points using the axes learned from historical data. If your evaluation uses axes computed on the test set, you are evaluating a procedure that cannot exist in production.

The correct pipeline: `fit` on train, `transform` on train and test using the training-fit parameters (mean $\bar{\boldsymbol{x}}_\text{train}$, components $\boldsymbol{B}_\text{train}$). In sklearn: `pca.fit(X_train)`, then `pca.transform(X_train)` and `pca.transform(X_test)`.

**Q17.**
When would you prefer PCA over a nonlinear dimensionality reduction method like a deep autoencoder, and when would you not?

**A17.**
**Prefer PCA when:**
- **Interpretability is required.** PCA components are linear combinations of original features — each PC has a direct coefficient for every input dimension. Autoencoder latent codes are opaque transformations.
- **Data is small or training budget is limited.** PCA has a closed-form solution ($O(ND^2)$ or $O(N^3)$ via SVD); autoencoders require iterative training with hyperparameter tuning.
- **The data is approximately linear/Gaussian.** PCA is the optimal linear dimensionality reduction under MSE. If the data's intrinsic manifold is linear (or nearly so), autoencoders will not improve much.
- **Reproducibility and stability matter.** PCA is deterministic given the data; autoencoders have random initialization and can give different solutions across runs.
- **Downstream use is a linear model.** If the latent representation feeds into linear regression or an SVM, there is no benefit to nonlinear encoding.

**Prefer a deep autoencoder when:**
- The data lies on a curved, nonlinear manifold (e.g., images — pixel space is not a linear manifold of the visual world).
- You need a generative model (VAE extends the autoencoder to a full generative model with a structured latent space).
- Reconstruction quality matters at low dimensionality (nonlinear encoders can compress much more aggressively before reconstruction quality degrades).

---

## GMMs / Density Estimation

**Q18.**
Explain the E-step and M-step of EM for a GMM in plain terms. Why is EM guaranteed to not decrease the likelihood at each step, and what is its main weakness?

**A18.**
**E-step (Expectation):** Given current parameters $\boldsymbol{\theta}^{(t)} = \{\pi_k, \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k\}$, compute the responsibility $r_{nk} = p(z_k = 1 \mid \boldsymbol{x}_n, \boldsymbol{\theta}^{(t)})$ for every data point and component — the posterior probability that point $n$ came from component $k$. This is a straightforward application of Bayes' rule. In plain terms: "given what we currently believe about the components, how much does each component claim each point?"

**M-step (Maximization):** Update the parameters by maximizing the expected complete-data log-likelihood $\mathcal{Q}(\boldsymbol{\theta}) = \sum_{n,k} r_{nk} \log[\pi_k \mathcal{N}(\boldsymbol{x}_n \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)]$. The closed-form updates are responsibility-weighted means and covariances: each component's new mean is the weighted average of points it claims, and the new mixing weight is its average responsibility. In plain terms: "given how much each component claims each point, fit each component to its claimed share of the data."

**Why guaranteed non-decrease:** EM is coordinate ascent on the ELBO $= \mathcal{L}(\boldsymbol{\theta}) - \text{KL}[q(\boldsymbol{z}) \| p(\boldsymbol{z} \mid \boldsymbol{x}, \boldsymbol{\theta})]$. The E-step tightens the ELBO to $\mathcal{L}$ by setting $q = p(\boldsymbol{z} \mid \boldsymbol{x}, \boldsymbol{\theta}^{(t)})$ (zeroing the KL). The M-step then maximizes the ELBO over $\boldsymbol{\theta}$, which can only increase it. Since the ELBO lower-bounds $\mathcal{L}$ and equals $\mathcal{L}$ at the E-step, the marginal log-likelihood is non-decreasing across iterations. A decrease indicates a numerical bug.

**Main weakness:** EM converges to a *local* optimum of the likelihood. The objective is non-convex (due to the log-sum structure), so the solution depends on initialization. Multiple random restarts with K-means++ seeding are the standard mitigation; there is no guarantee of finding the global optimum.

**Q19.**
K-means and GMMs are closely related. What is the precise difference algorithmically, and when would you prefer one over the other?

**A19.**
K-means is the hard-assignment limit of EM for GMMs with isotropic, equal-variance components. The precise algorithmic differences:

| | K-means | GMM (EM) |
|---|---|---|
| Assignments | Hard: $r_{nk} \in \{0,1\}$ (each point assigned to exactly one cluster) | Soft: $r_{nk} \in [0,1]$ (fractional membership) |
| Distance metric | L2 distance to centroid (implicit $\boldsymbol{\Sigma}_k = \sigma^2\boldsymbol{I}$, equal) | Mahalanobis distance under per-cluster $\boldsymbol{\Sigma}_k$ |
| Output | Cluster labels | Responsibilities + full generative model |
| Objective | Sum of squared distances to assigned centroids | Log-marginal-likelihood |

**Prefer K-means when:** clusters are roughly spherical and equal-size, you need scalability ($O(NKD)$ per iteration, no matrix inversions), or you only need cluster assignments (not probabilities or a generative model).

**Prefer GMMs when:** clusters have different shapes, orientations, or densities; you need soft assignments (e.g., overlapping clusters); you need a generative model for sampling or anomaly detection (need $p(\boldsymbol{x})$); or you want principled model selection via BIC/AIC to choose $K$.

**Q20.**
A GMM component collapses — its covariance shrinks to near zero and it fits a single data point perfectly, driving the likelihood to infinity. Why does this happen and how do you prevent it?

**A20.**
**Why it happens:** The MLE objective for a GMM is unbounded above. A component with mean $\boldsymbol{\mu}_k = \boldsymbol{x}_n$ (collapsed onto a single data point) has $\mathcal{N}(\boldsymbol{x}_n \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) \propto |\boldsymbol{\Sigma}_k|^{-1/2}$, which diverges as $|\boldsymbol{\Sigma}_k| \to 0$. So the MLE is technically $+\infty$, achieved by a degenerate solution. This is a pathology of MLE for mixture models — the likelihood surface has singularities at degenerate configurations.

**How to prevent it:**

1. **Regularize the covariance matrix:** Add a small $\epsilon\boldsymbol{I}$ to $\boldsymbol{\Sigma}_k$ after every M-step (e.g., `reg_covar=1e-6` in sklearn). This enforces a minimum eigenvalue, preventing $|\boldsymbol{\Sigma}_k| \to 0$.

2. **Bayesian/MAP estimation:** Place a conjugate prior on $\boldsymbol{\Sigma}_k$ — an inverse-Wishart prior with sufficient degrees of freedom sets a floor on covariance. This converts MLE into MAP; the posterior mode has a bounded covariance by construction.

3. **Detect and reinitialize collapsed components:** Monitor $N_k = \sum_n r_{nk}$; if it falls below a threshold (e.g., $N_k < 1$), reinitialize that component randomly.

4. **Tied or diagonal covariance:** Using `covariance_type='diag'` or `'tied'` reduces the degrees of freedom and makes collapse harder to achieve.

---

## SVMs

**Q21.**
What does the margin in an SVM represent geometrically, and why does maximizing it improve generalization? What is the role of C in the soft-margin SVM?

**A21.**
**Geometric meaning:** The margin is the width of the largest "slab" that can be placed between the two classes — specifically, the distance between the two support hyperplanes $\langle \boldsymbol{w}, \boldsymbol{x} \rangle + b = \pm 1$, which equals $2/\|\boldsymbol{w}\|$. Equivalently, it is twice the distance from the decision boundary to the nearest training point on either side.

**Why maximizing margin improves generalization:** The VC-theoretic bound on test error scales as $O(R^2/(\gamma^2 N))$ where $R$ is the radius of the data and $\gamma = 1/\|\boldsymbol{w}\|$ is the margin. Larger margin → smaller VC dimension → tighter generalization bound, independent of the input dimension $D$. Geometrically, a wider margin means the classifier is robust to small perturbations of training points — a test point must move further to cross the boundary.

**Role of C:** $C$ is the regularization hyperparameter controlling the margin-versus-error tradeoff.
- Large $C$: the penalty for margin violations $\xi_n > 0$ is high, so the solver prefers hard margin (narrow, tight margin), tolerating no misclassifications. Low bias, high variance.
- Small $C$: violations are cheap, so the solver accepts misclassifications in exchange for a wider margin. High bias, low variance.
$C = \infty$ recovers the hard-margin SVM (infeasible if data is not separable). Tune $C$ on a log scale via cross-validation.

**Q22.**
SVMs only depend on support vectors at prediction time. What are support vectors, and what does a non-support vector's alpha value tell you about the KKT conditions?

**A22.**
**Support vectors** are the training examples with $\alpha_n > 0$ in the dual solution. They are the points that "support" the margin — they lie on or within the margin boundaries. The decision function is $f(\boldsymbol{x}_*) = \sum_n \alpha_n y_n k(\boldsymbol{x}_n, \boldsymbol{x}_*) + b$, which depends on the data only through these points.

There are two types:
- $0 < \alpha_n < C$: the example lies exactly on the margin boundary ($\xi_n = 0$, constraint active).
- $\alpha_n = C$: the example is inside or beyond the margin ($\xi_n > 0$, margin constraint violated).

**For a non-support vector ($\alpha_n = 0$):** By complementary slackness, $\alpha_n(y_n f(\boldsymbol{x}_n) - 1 + \xi_n) = 0$ must hold. With $\alpha_n = 0$, this is satisfied regardless of the constraint value — the constraint $y_n f(\boldsymbol{x}_n) \geq 1$ must be strictly satisfied ($y_n f(\boldsymbol{x}_n) > 1$, slack $\xi_n = 0$). In words: non-support vectors are correctly classified with margin strictly greater than 1. They are irrelevant to the decision boundary — moving them would not change $\boldsymbol{w}^* = \sum_n \alpha_n y_n \boldsymbol{x}_n$ at all.

**Q23.**
Explain the kernel trick in one or two sentences. What property must a kernel function satisfy to be valid, and why does the RBF kernel correspond to an infinite-dimensional feature space?

**A23.**
The kernel trick replaces every inner product $\langle \boldsymbol{x}_i, \boldsymbol{x}_j \rangle$ in the dual objective and prediction function with $k(\boldsymbol{x}_i, \boldsymbol{x}_j) = \langle \phi(\boldsymbol{x}_i), \phi(\boldsymbol{x}_j) \rangle_\mathcal{H}$, implicitly mapping to a (possibly infinite-dimensional) feature space $\mathcal{H}$ without explicitly computing $\phi(\boldsymbol{x})$. This allows the SVM to find a linear separator in $\mathcal{H}$ that corresponds to a nonlinear boundary in the original input space, at cost $O(D)$ per kernel evaluation rather than $O(\dim \mathcal{H})$.

**Validity condition:** $k$ is a valid (Mercer) kernel iff the $N \times N$ Gram matrix $K_{ij} = k(\boldsymbol{x}_i, \boldsymbol{x}_j)$ is symmetric positive semidefinite for every finite input set. This ensures $k$ corresponds to a genuine inner product in some Hilbert space.

**Why RBF is infinite-dimensional:** Apply the Taylor expansion to $\exp(-\|\boldsymbol{x} - \boldsymbol{x}'\|^2 / 2\ell^2) = \exp(-\|\boldsymbol{x}\|^2/2\ell^2)\exp(\boldsymbol{x}^T\boldsymbol{x}'/\ell^2)\exp(-\|\boldsymbol{x}'\|^2/2\ell^2)$. Expanding $\exp(\boldsymbol{x}^T\boldsymbol{x}'/\ell^2) = \sum_{n=0}^\infty \frac{(\boldsymbol{x}^T\boldsymbol{x}')^n}{\ell^{2n} n!}$ gives a sum of polynomial kernels of every degree — the implicit feature map $\phi$ contains all monomials of all degrees, which is countably infinite-dimensional. The RKHS of the RBF kernel consists of infinitely smooth functions.

**Q24.**
In what practical scenarios would you still reach for an SVM over a neural network in 2025?

**A24.**
1. **Small datasets** ($N < 10^4$): Neural networks are data-hungry; SVMs with RBF kernels often outperform them in the low-data regime where there is insufficient data to learn useful representations. The SVM's generalization bound depends on margin, not $N$ directly.

2. **Tabular/structured data with no obvious feature hierarchy:** Deep networks excel at learning hierarchical representations from raw inputs (pixels, tokens). For flat tabular data, SVMs and gradient boosted trees are competitive with much less hyperparameter sensitivity.

3. **When theoretical guarantees matter:** Margin-based generalization bounds are tight and interpretable. In safety-critical applications, the ability to bound test error analytically is valuable.

4. **Non-vectorial inputs with a good kernel:** If your inputs are graphs, strings, sequences, trees, or sets, you can define problem-specific kernels (graph kernels, string kernels, tree kernels) and immediately get a well-founded classifier. Designing a neural architecture for these structures requires far more engineering.

5. **Compute-constrained inference:** With a small number of support vectors, SVM prediction is a handful of kernel evaluations — extremely fast and memory-efficient compared to a neural network forward pass.

6. **Baseline and interpretability:** SVMs with linear kernels give directly interpretable weight vectors and are a strong, well-understood baseline for any new problem.
