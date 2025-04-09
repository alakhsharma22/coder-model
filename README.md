# UNCODE: Structured Code Synthesis with Reward-Driven Fine-Tuning

UNCODE is a rigorous framework for reasoning-based generative code synthesis, combining large language models with an execution-grounded reward signal. Built upon parameter-efficient fine-tuning via Low-Rank Adaptation (LoRA), the system enables efficient generation of C++ solutions for competitive programming problems, under functional and resource constraints.

---

## 1. Introduction

UNCODE integrates formal reasoning and execution-based feedback to create functionally correct and resource-efficient code. Unlike traditional supervised approaches, it incorporates a sampling-based inference pipeline that evaluates generated code through actual execution, enabling training signals rooted in semantic correctness.

---

## 2. Fine-Tuning via LoRA

Let M_base denote the frozen base transformer model. LoRA introduces a low-rank perturbation:

M_theta(x) = M_base(x) + A_theta * x

Where `A_theta` is a low-rank matrix of shape (r × d), with `r << d`.

Given a dataset D = {(P_i, C_i)} for i = 1 to N, the loss function is:

L(θ) = (1 / N) * sum_{i=1}^{N} CE(M_theta(P_i), C_i)

---

## 3. Inference as Sampling

Given a problem prompt \( P \), the model samples a set of candidate solutions:

Omega_P = {C^(1), C^(2), ..., C^(k)}, where each C^(i) ~ Sample(M_theta, P)

These candidates are evaluated through a reward function grounded in execution.

---

## 4. Execution-Based Reward Function

Let T = {(x_j, y_j)} for j = 1 to n be a test suite. The reward function is defined as:

R(C, T) = (1/n) * sum_{j=1}^{n} [Exec(C, x_j) == y_j] - lambda_err * err_flag - lambda_tle * tle_flag - lambda_mle * mle_flag

Where:
- `Exec(C, x_j)` runs the compiled code C on input x_j.
- `err_flag`, `tle_flag`, and `mle_flag` are indicator variables for compile errors, time limit exceeded, and memory limit exceeded.
- `lambda_...` are penalty weights.

The optimal solution is:

C* = argmax_{C in Omega_P} R(C, T)

---

## 5. Learning via Sampled Pairwise Loss

To improve the model, we select the best and worst candidates from Omega_P based on reward. The loss is then computed as:

L_PairWise = Loss(C_worst) - Loss(C_best)

This encourages the model to prefer high-reward solutions by maximizing the loss difference.

---

## 6. Theoretical Properties

### 6.1 Correctness Guarantee

If `R(C, T) = 1` and T spans the full domain, then C is functionally correct under all test cases and resource constraints.

---

### 6.2 Resource-Constrained Solution Space

Define the valid solution space:

C_{t, m} = {C in C | ExecTime(C) <= t and MemUsage(C) <= m}

The output C* is guaranteed to lie in C_{t, m} and C_valid, ensuring feasibility.

### 6.3 Lipschitz Continuity

If `M_base` is Lipschitz continuous with constant L0, and the LoRA adapter `A_theta` has norm <= LA, then:

||M_theta(x1) - M_theta(x2)|| <= (L0 + LA) * ||x1 - x2||

Ensuring robustness to small prompt variations.

### 6.4 Sampling Convergence

Given p > 0 probability of sampling a perfect candidate:

As k -> infinity, P(max_{i} R(C^(i), T) = 1) -> 1

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
