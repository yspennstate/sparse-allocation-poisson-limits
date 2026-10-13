# Poisson limits for large boxes in sparse random allocations

Companion material for the paper of the same title (Y. Shmalo, October 2026). [Read the paper](paper/main.pdf).

Throw m balls into n boxes so that an allocation (y_1, ..., y_n) has probability proportional to
w_{y_1} w_{y_2} ... w_{y_n}, and let ξ_1, ..., ξ_n be independent with P(ξ_i = k) proportional to w_k τ^k, where τ is
chosen so that E ξ_i = m/n. This is the balls-in-boxes model of S. Janson, *Simply generated trees, conditioned
Galton–Watson trees, random allocations and condensation*, Probab. Surveys 9 (2012). Theorem 19.7 of the survey shows
that when m/n → λ with 0 < λ < ν, the numbers of boxes with many balls are asymptotically Poisson and the largest boxes
are close in distribution to the largest of ξ_1, ..., ξ_n. Problem 19.9 of the survey (18.9 in the arXiv version) asks
whether the same holds when m/n → 0. The paper answers the question. Let a be the least k ≥ 1 with w_k > 0.

- Theorem 1.1. If every k with w_k > 0 is a multiple of a, for instance if w_1 > 0, the conclusions of Theorem 19.7
  hold if and only if m → ∞. The j largest boxes are then close in total variation to the j largest of the ξ_i, for
  every fixed j.
- Proposition 1.2. The conclusions fail whenever m is bounded. If some k with w_k > 0 is not a multiple of a, and b is
  the least such k, they also fail along sequences with m → ∞ and m/n → 0, for instance m = b + a⌊n^{(b−a)/(2b)}⌋.
- Theorem 1.3. When b is coprime to a, let c_n = n P(ξ_1 = b). The conclusions hold if and only if m → ∞ and every
  subsequence has a further subsequence along which either c_n → ∞, or c_n → 0 and a divides m. When c_n → c ∈ (0, ∞)
  and m ≡ ℓ (mod a), the number of boxes with at least b balls converges to a Poisson(c) variable X conditioned on
  X ≡ b′ℓ (mod a), where bb′ ≡ 1 (mod a).
- Example 1.4. For w_1 = 0 and w_k = 1 for k ≠ 1, if m^3/(8n) → c^2 with 0 < c < ∞, then P(Y_(1) ≤ 2) → 1/cosh c
  along even m and P(Y_(1) ≤ 2) = 0 for odd m, while P(ξ_(1) ≤ 2) → e^{−c} for both parities.

All of these come from a comparison of the boxes whose sizes are neither 0 nor a. Given which ξ_i are neither 0 nor a,
and their values, the number of the other ξ_i that equal a is binomial, and Bayes' formula gives the law of the
corresponding boxes of the allocation exactly: it is their law under the ξ_i, reweighted by a binomial point
probability (the identity (3.2)). With a local limit theorem for the binomial distribution with explicit constants
(Lemma 2.1), Theorem 1.5 shows that when m → ∞ and m/n → 0 these boxes are close in total variation to the
corresponding ξ_i conditioned on ξ_1 + ... + ξ_n ≡ m (mod a), as long as that event has probability bounded away
from zero.

## Contents

- `paper/` the LaTeX source and the compiled PDF.
- `code/checks.py` the computations of Section 6: Lemma 2.1 on 88 binomial laws; the identity (3.2) in exact rational
  arithmetic for three weight sequences, with a negative control that must fail; for w_k = 1, the exact total
  variation distance between the reduced vectors and the local limit ratio (Table 1), with a control at m/n = 0.1,
  and the Kolmogorov distance between Y_(1) and ξ_(1) for m = √n and for m = 1; and for the weights of Example 1.4,
  the law of the largest box when n = m^2 and n = m^4, the law of the number of boxes with at least 3 balls when
  n = m^3/8 against the conditioned and the plain Poisson laws (Table 2), and the local limit formula of Theorem 1.5
  with a = 2. It uses only the Python standard library. `code/checks.out` is its output.

The proofs have not been formalized.

## Running the checks

    python code/checks.py

Python 3.8 or later. The script runs in a few seconds and ends with `ALL CHECKS PASS`.
