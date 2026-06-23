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
Skipped — Part 2 topic (covered in Chapter 12 / SVM section).

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

**Q12.**
What is the difference between MLE and MAP estimation in practical terms? If you increase the strength of L2 regularization in ridge regression, what are you implicitly saying about your prior on the weights?

**Q13.**
In Bayesian linear regression, the predictive variance has two components. What are they, what do they represent, and why does one shrink with more data while the other does not?

**Q14.**
Your colleague says "we should always normalize features before training." Do you agree? When would you not normalize, and what goes wrong if you skip it when you should have done it?

---

## PCA / Dimensionality Reduction

**Q15.**
You run PCA on a dataset and keep enough components to explain 95% of variance. A junior engineer asks: "what did we throw away?" How do you answer, and when is throwing it away a problem?

**Q16.**
Why must you fit PCA (compute the mean and principal components) only on the training set and not on the full dataset? What goes wrong if you don't?

**Q17.**
When would you prefer PCA over a nonlinear dimensionality reduction method like a deep autoencoder, and when would you not?

---

## GMMs / Density Estimation

**Q18.**
Explain the E-step and M-step of EM for a GMM in plain terms. Why is EM guaranteed to not decrease the likelihood at each step, and what is its main weakness?

**Q19.**
K-means and GMMs are closely related. What is the precise difference algorithmically, and when would you prefer one over the other?

**Q20.**
A GMM component collapses — its covariance shrinks to near zero and it fits a single data point perfectly, driving the likelihood to infinity. Why does this happen and how do you prevent it?

---

## SVMs

**Q21.**
What does the margin in an SVM represent geometrically, and why does maximizing it improve generalization? What is the role of C in the soft-margin SVM?

**Q22.**
SVMs only depend on support vectors at prediction time. What are support vectors, and what does a non-support vector's alpha value tell you about the KKT conditions?

**Q23.**
Explain the kernel trick in one or two sentences. What property must a kernel function satisfy to be valid, and why does the RBF kernel correspond to an infinite-dimensional feature space?

**Q24.**
In what practical scenarios would you still reach for an SVM over a neural network in 2025?
