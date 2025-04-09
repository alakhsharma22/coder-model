# UNCODE: Structured Code Synthesis with Reward-Driven Fine-Tuning

UNCODE is a rigorous framework for reasoning-based generative code synthesis, combining large language models with an execution-grounded reward signal. Built upon parameter-efficient fine-tuning via Low-Rank Adaptation (LoRA), the system enables efficient generation of C++ solutions for competitive programming problems, under functional and resource constraints.

---

## 1. Introduction

UNCODE integrates formal reasoning and execution-based feedback to create functionally correct and resource-efficient code. Unlike traditional supervised approaches, it incorporates a sampling-based inference pipeline that evaluates generated code through actual execution, enabling training signals rooted in semantic correctness.

---

## 2. Fine-Tuning via LoRA

Let \( M_{\text{base}} \) denote the frozen base transformer model. LoRA introduces a low-rank perturbation:

\[
M_{\theta}(x) = M_{\text{base}}(x) + A_{\theta}x, \quad A_{\theta} \in \mathbb{R}^{r \times d}, \quad r \ll d
\]

Given a dataset \( \mathcal{D} = \{(P_i, C_i)\}_{i=1}^N \), we optimize the cross-entropy loss:

\[
\mathcal{L}(\theta) = \frac{1}{N} \sum_{i=1}^N \text{CE}(M_{\theta}(P_i), C_i)
\]

---

## 3. Inference as Sampling

Given a problem prompt \( P \), the model samples a set of candidate solutions:

\[
\Omega_P = \{C^{(1)}, \dots, C^{(k)}\}, \quad C^{(i)} \sim \text{Sample}(M_{\theta}, P)
\]

These candidates are evaluated through a reward function grounded in execution.

---

## 4. Execution-Based Reward Function

Let \( T = \{(x_j, y_j)\}_{j=1}^n \) be a test suite. The reward function \( R : C \times T \to \mathbb{R} \) is defined as:

\[
R(C, T) = \frac{1}{n} \sum_{j=1}^n \delta(\text{Exec}(C, x_j) = y_j) - \lambda_{\text{err}}1_{\text{err}} - \lambda_{\text{tle}}1_{\text{tle}} - \lambda_{\text{mle}}1_{\text{mle}}
\]

Where:
- \( \delta \) is the correctness indicator.
- \( \lambda \)'s are penalty terms for errors (compilation, time, memory).

The optimal candidate is selected via:

\[
C^* = \arg\max_{C \in \Omega_P} R(C, T)
\]

---

## 5. Learning via Sampled Pairwise Loss

To improve the model, we select the best and worst candidates from \( \Omega_P \) based on reward. The loss is then computed as:

\[
\mathcal{L}_{\text{GRPO}} = \mathcal{L}(C_{\text{worst}}) - \mathcal{L}(C_{\text{best}})
\]

This encourages the model to move towards high-reward outputs.

---

## 6. Theoretical Properties

### 6.1 Correctness Guarantee

If \( R(C, T) = 1 \) and \( T \) spans the full domain, then \( C \) correctly implements the desired function under resource constraints.

### 6.2 Resource-Constrained Solution Space

Define:

\[
C_{t,m} = \{ C \in \mathcal{C} \mid \text{ExecTime}(C) \leq t, \text{MemUsage}(C) \leq m \}
\]

The output \( C^* \) is guaranteed to lie in \( C_{t,m} \cap C_{\text{valid}} \), ensuring feasibility.

### 6.3 Lipschitz Continuity

If \( M_{\text{base}} \) is Lipschitz with constant \( L_0 \), and \( \|A_\theta\| \leq L_A \), then:

\[
\|M_\theta(x_1) - M_\theta(x_2)\| \leq (L_0 + L_A)\|x_1 - x_2\|
\]

Ensuring robustness to small prompt variations.

### 6.4 Sampling Convergence

Given \( p > 0 \) probability of sampling a perfect candidate:

\[
\lim_{k \to \infty} \mathbb{P}(\max_i R(C^{(i)}, T) = 1) = 1
\]

---

## 7. Evaluation Results

On the problem **"Maximum Vending Machine Profit"**, the system achieved:
- Perfect reward: \( R = 1.0 \)
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

---

UNCODE presents a unified pipeline that fuses sampling, reasoning, and execution to build code-generation systems grounded in mathematical rigor and real-world correctness.# UNCODE: Structured Code Synthesis with Reward-Driven Fine-Tuning

UNCODE is a rigorous framework for reasoning-based generative code synthesis, combining large language models with an execution-grounded reward signal. Built upon parameter-efficient fine-tuning via Low-Rank Adaptation (LoRA), the system enables efficient generation of C++ solutions for competitive programming problems, under functional and resource constraints.

---

## 1. Introduction

UNCODE integrates formal reasoning and execution-based feedback to create functionally correct and resource-efficient code. Unlike traditional supervised approaches, it incorporates a sampling-based inference pipeline that evaluates generated code through actual execution, enabling training signals rooted in semantic correctness.

---

## 2. Fine-Tuning via LoRA

Let \( M_{\text{base}} \) denote the frozen base transformer model. LoRA introduces a low-rank perturbation:

\[
M_{\theta}(x) = M_{\text{base}}(x) + A_{\theta}x, \quad A_{\theta} \in \mathbb{R}^{r \times d}, \quad r \ll d
\]

Given a dataset \( \mathcal{D} = \{(P_i, C_i)\}_{i=1}^N \), we optimize the cross-entropy loss:

\[
\mathcal{L}(\theta) = \frac{1}{N} \sum_{i=1}^N \text{CE}(M_{\theta}(P_i), C_i)
\]

---

## 3. Inference as Sampling

Given a problem prompt \( P \), the model samples a set of candidate solutions:

\[
\Omega_P = \{C^{(1)}, \dots, C^{(k)}\}, \quad C^{(i)} \sim \text{Sample}(M_{\theta}, P)
\]

These candidates are evaluated through a reward function grounded in execution.

---

## 4. Execution-Based Reward Function

Let \( T = \{(x_j, y_j)\}_{j=1}^n \) be a test suite. The reward function \( R : C \times T \to \mathbb{R} \) is defined as:

\[
R(C, T) = \frac{1}{n} \sum_{j=1}^n \delta(\text{Exec}(C, x_j) = y_j) - \lambda_{\text{err}}1_{\text{err}} - \lambda_{\text{tle}}1_{\text{tle}} - \lambda_{\text{mle}}1_{\text{mle}}
\]

Where:
- \( \delta \) is the correctness indicator.
- \( \lambda \)'s are penalty terms for errors (compilation, time, memory).

The optimal candidate is selected via:

\[
C^* = \arg\max_{C \in \Omega_P} R(C, T)
\]

---

## 5. Learning via Sampled Pairwise Loss

To improve the model, we select the best and worst candidates from \( \Omega_P \) based on reward. The loss is then computed as:

\[
\mathcal{L}_{\text{GRPO}} = \mathcal{L}(C_{\text{worst}}) - \mathcal{L}(C_{\text{best}})
\]

This encourages the model to move towards high-reward outputs.

---

## 6. Theoretical Properties

### 6.1 Correctness Guarantee

If \( R(C, T) = 1 \) and \( T \) spans the full domain, then \( C \) correctly implements the desired function under resource constraints.

### 6.2 Resource-Constrained Solution Space

Define:

\[
C_{t,m} = \{ C \in \mathcal{C} \mid \text{ExecTime}(C) \leq t, \text{MemUsage}(C) \leq m \}
\]

The output \( C^* \) is guaranteed to lie in \( C_{t,m} \cap C_{\text{valid}} \), ensuring feasibility.

### 6.3 Lipschitz Continuity

If \( M_{\text{base}} \) is Lipschitz with constant \( L_0 \), and \( \|A_\theta\| \leq L_A \), then:

\[
\|M_\theta(x_1) - M_\theta(x_2)\| \leq (L_0 + L_A)\|x_1 - x_2\|
\]

Ensuring robustness to small prompt variations.

### 6.4 Sampling Convergence

Given \( p > 0 \) probability of sampling a perfect candidate:

\[
\lim_{k \to \infty} \mathbb{P}(\max_i R(C^{(i)}, T) = 1) = 1
\]

---

## 7. Evaluation Results

On the problem **"Maximum Vending Machine Profit"**, the system achieved:
- Perfect reward: \( R = 1.0 \)
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

---

UNCODE presents a unified pipeline that fuses sampling, reasoning, and execution to build code-generation systems grounded in mathematical rigor and real-world correctness.
