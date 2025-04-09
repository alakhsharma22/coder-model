# UNCODE: Structured Code Synthesis with Reward-Driven Fine-Tuning

UNCODE is a rigorous framework for reasoning-based generative code synthesis, combining large language models with an execution-grounded reward signal. Built upon parameter-efficient fine-tuning via Low-Rank Adaptation (LoRA), the system enables efficient generation of C++ solutions for competitive programming problems, under functional and resource constraints.

---

## 1. Introduction

UNCODE integrates formal reasoning and execution-based feedback to create functionally correct and resource-efficient code. Unlike traditional supervised approaches, it incorporates a sampling-based inference pipeline that evaluates generated code through actual execution, enabling training signals rooted in semantic correctness.

---

## 2. Fine-Tuning via LoRA

Let M_base denote the frozen base transformer model. LoRA introduces a low-rank perturbation:

$$
M_\theta(x) \;=\; M_{\mathrm{base}}(x) \;+\; A_\theta\,x
$$

where $\(A_\theta\)$ is a low‑rank matrix of shape $\(r \times d\)$ with $\(r \ll d\)$, i.e.

$$
A_\theta \in \mathbb{R}^{r \times d}, \quad r \ll d.
$$

Given a dataset 

$$
D \;=\; \{(P_i, C_i)\}_{i=1}^N,
$$

the loss function is

$$
L(\theta) \;=\; \frac{1}{N} \sum_{i=1}^N \mathrm{CE}\bigl(M_\theta(P_i),\,C_i\bigr),
$$

---

## 3. Inference as Sampling

Given a problem prompt $\(P\)$, the model samples a set of candidate solutions:

$$
\Omega_P = \{C^{(1)}, C^{(2)}, \dots, C^{(k)}\}, \quad
C^{(i)} \sim \mathrm{Sample}(M_\theta, P).
$$

These candidates are then evaluated via a reward function grounded in execution.


---

## 4. Execution-Based Reward Function

Let $\(T = \{(x_j, y_j)\}_{j=1}^n\)$ be a test suite. The reward function is defined as

$$
\begin{aligned}
R(C, T) &= \frac{1}{n}\sum_{j=1}^n \mathbf{1}\bigl(\mathrm{Exec}(C, x_j) = y_j\bigr)\\
        &\quad - \lambda_{\mathrm{err}}\,\mathrm{err\_flag}
          - \lambda_{\mathrm{tle}}\,\mathrm{tle\_flag}
          - \lambda_{\mathrm{mle}}\,\mathrm{mle\_flag}.
\end{aligned}
$$


Where:
- $\(\mathrm{Exec}(C, x_j)\)$ runs the compiled code $\(C\)$ on input $\(x_j\)$.
- $\(\mathrm{err\_flag}\)$, $\(\mathrm{tle\_flag}\)$, and $\(\mathrm{mle\_flag}\)$ are indicator variables for compile errors, time limit exceeded, and memory limit exceeded.
- $\(\lambda_{\dots}\)$ are penalty weights.

The optimal solution is:

$$
C^* \;=\; \arg\max_{C \in \Omega_P} R(C, T).
$$

---

## 5. Learning via Sampled Pairwise Loss

To improve the model, we select the best and worst candidates from Omega_P based on reward. The loss is then computed as:

$$
L_{\mathrm{PairWise}}
\=\
\mathrm{Loss}\!\left(C_{\mathrm{worst}}\right)
\-\
\mathrm{Loss}\!\left(C_{\mathrm{best}}\right).
$$

This encourages the model to prefer high-reward solutions by maximizing the loss difference.

---

## 6. Theoretical Properties

### 6.1 Correctness Guarantee

If $\(R(C, T) = 1\)$ and $\(T\)$ spans the full domain, then $\(C\)$ is functionally correct under all test cases and resource constraints.

---

### 6.2 Resource-Constrained Solution Space

Define the valid solution space:

$$
C_{t,m} = \{\,C \in \mathcal{C} \mid \mathrm{ExecTime}(C)\le t,\;\mathrm{MemUsage}(C)\le m\}.
$$


The output $\(C^*\)$ is guaranteed to lie in $\(C_{t,m}\)$ and $\(C_{\mathrm{valid}}\)$, ensuring feasibility.

### 6.3 Lipschitz Continuity

If $\(M_{\mathrm{base}}\)$ is Lipschitz continuous with constant $\(L_0\)$, and the LoRA adapter $\(A_\theta\)$ satisfies $\(\|A_\theta\|\le L_A\)$, then

$$
\|M_\theta(x_1) - M_\theta(x_2)\|
\;\le\;
\bigl(L_0 + L_A\bigr)\,\|x_1 - x_2\|.
$$

Ensuring robustness to small prompt variations.

### 6.4 Sampling Convergence

Given a probability $$\(p > 0\)$$ of sampling a perfect candidate:


$$
\lim_{k \to \infty}
\Pr\!\left(\max_{1 \le i \le k} R\bigl(C^{(i)}, T\bigr) = 1\right)
= 1.
$$

---

## 7. Evaluation Results

On the problem **"Maximum Vending Machine Profit"**, the system achieved:
- Perfect reward: R = 1.0
- All tests passed, no errors or resource violations
- Output matched ground truth exactly

---

## 8. Implementation Stack

- PyTorch + Transformers
- LoRA (Parameter-Efficient Fine-Tuning)
- Unsloth for fast inference
- Execution evaluation via GCC + Python `subprocess`

---

## 9. Future Directions

- Integration of formal verification systems
- Efficient sampling via importance-weighted schemes
- Docker-based sandboxing for secure execution
- Extension to multi-language code synthesis

UNCODE presents a unified pipeline that fuses sampling, reasoning, and execution to build code-generation systems grounded in mathematical rigor and real-world correctness.
