"""Computations for "Poisson limits for large boxes in sparse random allocations".

Standard library only; Python 3.8 or later.  Run:  python checks.py

A. Lemma 2.1 (local limit theorem for the binomial distribution) on a grid of (N, q).
B. The identity (3.2) for the record, exactly in rational arithmetic, for two small weight sequences
   (one with w_1 > 0 and one with w_1 = 0), with a negative control.
C. Weights w_k = 1 for all k: the exact total variation distance between the record of Y and the record of the
   i.i.d. sequence (Theorem 1.5 with a = 1), computed through the sufficient statistic (|I|, K), and the local
   limit formula of Theorem 1.5 for P(S_n = m).
D. Weights w_k = 1 for all k: the exact Kolmogorov distance between Y_(1) and the largest of n i.i.d. copies of
   xi^(n), for m = sqrt(n) (Theorem 1.1) and for m = 1 (Proposition 1.2).
E. Weights w_0 = 1, w_1 = 0, w_k = 1 for k >= 2 (Example 1.4): exact laws of Y_(1) and of N_[3,inf) in the
   regimes n = m^2, n = m^3/8 and n = m^4, and the local limit formula of Theorem 1.5 with a = 2.
"""
import math
import sys
from fractions import Fraction
from itertools import product
from math import exp, log, log1p, sqrt, pi, cosh, sinh, lgamma

FAILED = []


def check(cond, msg):
    print(("  ok    " if cond else "  FAIL  ") + msg)
    if not cond:
        FAILED.append(msg)
    sys.stdout.flush()


def lC(a, b):
    if a < 0 or b < 0 or b > a:
        return -math.inf
    return lgamma(a + 1) - lgamma(b + 1) - lgamma(a - b + 1)


def comb(a, b):
    if a < 0 or b < 0 or b > a:
        return 0
    return math.comb(a, b)


# ---------------------------------------------------------------------------------------------------------------
def eps(s):
    return 1 / s + 4 * s ** (-1 / 3) * exp(-s ** (2 / 3) / 5)


def part_A():
    print("A. Lemma 2.1. B ~ Bin(N, q), s^2 = Nq(1 - q).")
    print("   (a) s max_j P(B = j) <= sqrt(pi/8) = %.6f" % sqrt(pi / 8))
    print("   (b) if s >= 1: max_j |sqrt(2 pi) s P(B = j) - exp(-(j - Nq)^2/(2 s^2))| <= eps(s) = 1/s + 4 s^(-1/3) exp(-s^(2/3)/5)")
    print("   %7s %7s %9s %12s %12s %12s" % ("N", "q", "s", "s max P", "error (b)", "eps(s)"))
    worst_a = 0.0
    worst_b = 0.0
    nrows = 0
    for N in [1, 2, 5, 10, 30, 100, 300, 1000, 3000, 10000, 30000]:
        for q in [1e-4, 1e-3, 0.01, 0.05, 0.2, 0.5, 0.8, 0.99]:
            s = sqrt(N * q * (1 - q))
            mx = 0.0
            err = 0.0
            for j in range(N + 1):
                P = exp(lC(N, j) + j * log(q) + (N - j) * log1p(-q))
                mx = max(mx, P)
                if s >= 1:
                    err = max(err, abs(sqrt(2 * pi) * s * P - exp(-(j - N * q) ** 2 / (2 * s * s))))
            worst_a = max(worst_a, s * mx / sqrt(pi / 8))
            if s >= 1:
                worst_b = max(worst_b, err / eps(s))
            nrows += 1
            if q in (0.01, 0.5) and N >= 30:
                tail = "%12.3e %12.3e" % (err, eps(s)) if s >= 1 else "%12s %12s" % ("-", "-")
                print("   %7d %7.2f %9.3f %12.6f %s" % (N, q, s, s * mx, tail))
    check(worst_a <= 1.0, "(a) on %d pairs (N, q): largest ratio s max P / sqrt(pi/8) = %.4f" % (nrows, worst_a))
    check(worst_b <= 1.0, "(b) on the pairs with s >= 1: largest ratio error / eps(s) = %.4f" % worst_b)


# ---------------------------------------------------------------------------------------------------------------
def identity_exact(w, m, n, tau, wrong_q=False):
    """Exact check of P(R_Y = x) = P(R = x) g(x) / E g(R) and E g(R) = P(S_n = m), V = {0, a}."""
    a = min(k for k in range(1, len(w)) if w[k] > 0)
    Phi = sum(Fraction(wk) * tau ** k for k, wk in enumerate(w))
    pxi = [Fraction(wk) * tau ** k / Phi for k, wk in enumerate(w)]
    V = (0, a)
    Z = 0
    lawY = {}
    for y in product(range(len(w)), repeat=n):
        if sum(y) != m:
            continue
        wy = 1
        for v in y:
            wy *= w[v]
        if wy == 0:
            continue
        Z += wy
        x = tuple(0 if v in V else v for v in y)
        lawY[x] = lawY.get(x, 0) + wy
    pV = pxi[0] + pxi[a]
    q = pxi[a] if wrong_q else pxi[a] / pV

    def g(x):
        free = n - sum(1 for v in x if v)
        rest = m - sum(x)
        if rest < 0 or rest % a:
            return Fraction(0)
        j = rest // a
        if j > free:
            return Fraction(0)
        return comb(free, j) * q ** j * (1 - q) ** (free - j)

    vals = [0] + [k for k in range(len(w)) if w[k] > 0 and k not in V]
    Eg = Fraction(0)
    lawR = {}
    for x in product(vals, repeat=n):
        pr = Fraction(1)
        for v in x:
            pr *= pV if v == 0 else pxi[v]
        val = pr * g(x)
        if val:
            lawR[x] = val
            Eg += val
    PS = Z * tau ** m / Phi ** n
    same = set(lawR) == set(lawY) and all(lawR[x] / Eg == Fraction(lawY[x], Z) for x in lawY)
    return a, len(lawY), Eg == PS, same


def part_B():
    print("B. The identity (3.2), exactly: P(R_Y = x) = P(R = x) g(x) / E g(R), and E g(R) = P(S_n = m).")
    for w, m, n, tau in [([2, 1, 0, 3, 1, 0, 2], 7, 4, Fraction(2, 5)), ([1, 0, 2, 1, 3, 0, 1, 2], 7, 4, Fraction(1, 3)),
                         ([1, 0, 2, 1, 3, 0, 1, 2], 8, 4, Fraction(3, 7))]:
        a, nx, ok_sum, ok_law = identity_exact(w, m, n, tau)
        check(ok_sum and ok_law, "w = %s, m = %d, n = %d, tau = %s (a = %d): %d record values, both identities exact"
              % (w, m, n, tau, a, nx))
    a, nx, ok_sum, ok_law = identity_exact([1, 0, 2, 1, 3, 0, 1, 2], 7, 4, Fraction(1, 3), wrong_q=True)
    check(not ok_sum and not ok_law, "negative control: q = P(xi = a) in place of P(xi = a | xi in V) breaks both identities")


# ---------------------------------------------------------------------------------------------------------------
def record_tv_uniform(m, n):
    """w_k = 1: xi^(n) is geometric, P(xi = k) = (1 - t) t^k with t = m/(m + n).  Returns lower and upper bounds for
    d_TV(R_Y, R) (they differ only by the neglected mass) and the included mass of the law of R_Y."""
    t = m / (m + n)
    p = t * t                       # P(xi >= 2)
    q = t / (1 + t)                 # P(xi = 1 | xi <= 1)
    lPS = lC(m + n - 1, n - 1) + n * log1p(-t) + m * log(t)       # log P(S_n = m)
    mean_i = n * p
    imax = min(n, int(mean_i + 12 * sqrt(mean_i + 1) + 20))
    tv_in = 0.0
    mass = 0.0
    massY = 0.0
    for i in range(imax + 1):
        lpi = lC(n, i) + i * log(p) + (n - i) * log1p(-p)
        if lpi < -80:
            continue
        if i == 0:
            cells = [(0, 0.0)]
        else:
            cells = [(2 * i + u, lC(u + i - 1, i - 1) + i * log1p(-t) + u * log(t)) for u in range(m - 2 * i + 1)]
        for k, lk in cells:
            lp = lpi + lk
            if lp < -80:
                continue
            P = exp(lp)
            lg = lC(n - i, m - k) + (m - k) * log(q) + (n - i - (m - k)) * log1p(-q)
            L = exp(lg - lPS) if lg > -math.inf else 0.0
            mass += P
            massY += P * L
            tv_in += P * abs(L - 1)
    return 0.5 * tv_in, 0.5 * (tv_in + abs(1 - mass) + abs(1 - massY)), massY


def part_C():
    print("C. w_k = 1: exact d_TV between the record of Y (boxes with at least 2 balls) and that of xi^(n), and the ratio")
    print("   sqrt(2 pi m) P(S_n = m), which tends to 1 by Theorem 1.5 (here a = 1 and P(E_m) = 1).")
    print("   %6s %8s %9s %12s %12s %10s %14s %10s" % ("m", "n", "m/n", "d_TV lower", "d_TV upper", "d_TV n/m", "mass of R_Y",
                                                       "LLT ratio"))
    rows = []
    for (m, n) in [(20, 400), (50, 2500), (100, 10000), (200, 40000), (400, 160000), (800, 640000),
                   (100, 1000), (200, 2000), (400, 4000), (800, 8000)]:
        lo, up, massY = record_tv_uniform(m, n)
        t = m / (m + n)
        llt = sqrt(2 * pi * m) * exp(lC(m + n - 1, n - 1) + n * log1p(-t) + m * log(t))
        rows.append((m, n, lo, up, massY, llt))
        print("   %6d %8d %9.5f %12.6f %12.6f %10.4f %14.12f %10.6f" % (m, n, m / n, lo, up, lo * n / m, massY, llt))
        sys.stdout.flush()
    lr = [abs(r[5] - 1) for r in rows[:6]]
    check(all(lr[i + 1] < lr[i] for i in range(5)) and lr[-1] < 0.001,
          "n = m^2: sqrt(2 pi m) P(S_n = m) tends to 1 (%.6f at m = 800)" % rows[5][5])
    check(all(abs(r[4] - 1) < 1e-8 for r in rows), "the included cells carry the whole law of R_Y up to rounding (1e-8)")
    # The gap between the bounds is P(K(R) > m) (records of xi^(n) that Y cannot have) plus the cells below e^-80.
    check(all(r[3] - r[2] < 1e-7 for r in rows), "the lower and upper bounds for d_TV agree to 1e-7")
    sq = [r[2] for r in rows[:6]]
    check(all(sq[i + 1] < sq[i] for i in range(5)) and sq[-1] < 0.002, "n = m^2: d_TV decreases to %.5f" % sq[-1])
    fixed = [r[2] for r in rows[6:]]
    check(min(fixed) > 0.08, "control, m/n = 0.1 fixed: d_TV stays above 0.08 (min %.5f)" % min(fixed))


# ---------------------------------------------------------------------------------------------------------------
def uniform_dK_max(m, n):
    """w_k = 1: d_K(Y_(1), max of n iid geometric(t)), with P(Y_(1) <= h) by inclusion-exclusion (exact integers)."""
    t = m / (m + n)
    Ztot = comb(m + n - 1, n - 1)
    best = 0.0
    for h in range(0, m + 2):
        s = 0
        for j in range(0, m // (h + 1) + 1):
            term = comb(n, j) * comb(m - j * (h + 1) + n - 1, n - 1)
            s += -term if j % 2 else term
        pY = s / Ztot
        pX = exp(n * log1p(-t ** (h + 1)))
        best = max(best, abs(pY - pX))
    return best


def part_D():
    print("D. w_k = 1: exact d_K(Y_(1), xi_(1)), where xi_(1) is the largest of n iid copies of xi^(n).")
    vals = []
    for n in [100, 900, 10000, 90000]:
        m = int(round(sqrt(n)))
        d = uniform_dK_max(m, n)
        vals.append(d)
        print("   m = sqrt(n) = %4d, n = %6d:  d_K = %.6f" % (m, n, d))
        sys.stdout.flush()
    check(all(vals[i + 1] < vals[i] for i in range(3)), "m = sqrt(n): d_K decreases")
    for n in [10, 100, 10000, 1000000]:
        d = uniform_dK_max(1, n)
        target = exp(n * log1p(-1 / (n + 1)))
        print("   m = 1, n = %7d:  d_K = %.6f, P(xi_(1) = 0) = %.6f, e^-1 = %.6f" % (n, d, target, exp(-1)))
        check(abs(d - target) < 1e-12, "m = 1, n = %d: d_K = P(xi_(1) = 0), which tends to e^-1" % n)


# ---------------------------------------------------------------------------------------------------------------
def parity_tau(m, n):
    Phi = lambda t: 1 + t * t / (1 - t)
    Psi = lambda t: t * (2 * t - t * t) / (1 - t) ** 2 / Phi(t)
    lo, hi = 0.0, 1.0 - 1e-12
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if Psi(mid) < m / n:
            lo = mid
        else:
            hi = mid
    tau = 0.5 * (lo + hi)
    return tau, Phi(tau)


def parity_Z(m, n):
    return sum(comb(n, j) * comb(m - j - 1, j - 1) for j in range(1, m // 2 + 1))


def parity_Zx(m, n, x):
    """Number of allocations with every box in {0} u [2, x]."""
    if x >= m:
        return parity_Z(m, n)
    tot = 0
    for j in range(1, m // 2 + 1):
        c = 0
        for i in range(0, j + 1):
            r = m - j - i * (x - 1) - 1
            if r < j - 1:
                break
            term = comb(j, i) * comb(r, j - 1)
            c += -term if i % 2 else term
        tot += comb(n, j) * c
    return tot


def parity_counts_big(m, n, k):
    """Number of allocations with exactly k boxes holding at least 3 balls (the others hold 0 or 2)."""
    if k == 0:
        return comb(n, m // 2) if m % 2 == 0 else 0
    tot = 0
    r = 0
    while m - 2 * r - 2 * k - 1 >= k - 1:
        tot += comb(n - k, r) * comb(m - 2 * r - 2 * k - 1, k - 1)
        r += 1
    return comb(n, k) * tot


def parity_dK(m, n):
    tau, Phi = parity_tau(m, n)
    Z = parity_Z(m, n)

    def tail(x):                 # P(xi > x)
        if x <= 0:
            return tau * tau / ((1 - tau) * Phi)
        return tau ** (x + 1) / ((1 - tau) * Phi)

    best = 0.0
    for x in range(0, m + 2):
        pY = 0.0 if x < 2 else parity_Zx(m, n, x) / Z
        pX = exp(n * log1p(-tail(x)))
        best = max(best, abs(pY - pX))
    c_n = n * tau ** 3 / Phi
    return best, c_n


def cond_poisson(c, parity, kmax):
    w = [exp(-c) * c ** k / math.factorial(k) if k % 2 == parity else 0.0 for k in range(kmax + 1)]
    s = sum(w)
    return [v / s for v in w]


def part_E():
    print("E. w = (1, 0, 1, 1, 1, ...), so a = 2, b = 3 and c_n = n P(xi^(n) = 3) ~ (m^3/(8n))^(1/2).")
    print("   E1. n = m^2 (c_n -> infinity): d_K(Y_(1), xi_(1)) -> 0 for both parities of m.")
    res = {}
    for m in [20, 21, 40, 41, 80, 81, 160, 161]:
        d, c_n = parity_dK(m, m * m)
        res[m] = d
        print("       m = %3d, n = %6d, c_n = %6.3f:  d_K = %.5f" % (m, m * m, c_n, d))
        sys.stdout.flush()
    check(res[40] < res[20] and res[80] < res[40] and res[160] < res[80], "n = m^2, m even: d_K decreases")
    check(res[41] < res[21] and res[81] < res[41] and res[161] < res[81], "n = m^2, m odd: d_K decreases")

    print("   E2. n = m^3/8 (c_n -> 1): the law of N = N_[3,inf)(Y) against X ~ Po(c_n) conditioned on X = m mod 2.")
    print("       %4s %8s %7s %3s %11s %16s %13s %13s" % ("m", "n", "c_n", "k", "P(N = k)", "P(X = k | X = m)", "dTV(cond Po)",
                                                         "dTV(Po)"))
    tv_cond = {0: [], 1: []}
    tv_po = {0: [], 1: []}
    for m in [20, 21, 40, 41, 80, 81, 160, 161]:
        n = int(round(m ** 3 / 8))
        tau, Phi = parity_tau(m, n)
        c_n = n * tau ** 3 / Phi
        Z = parity_Z(m, n)
        kmax = 30
        law = [parity_counts_big(m, n, k) / Z for k in range(kmax + 1)]
        missing = 1 - sum(law)
        cp = cond_poisson(c_n, m % 2, kmax)
        po = [exp(-c_n) * c_n ** k / math.factorial(k) for k in range(kmax + 1)]
        d1 = 0.5 * (sum(abs(law[k] - cp[k]) for k in range(kmax + 1)) + abs(missing))
        d2 = 0.5 * (sum(abs(law[k] - po[k]) for k in range(kmax + 1)) + abs(missing) + abs(1 - sum(po)))
        tv_cond[m % 2].append(d1)
        tv_po[m % 2].append(d2)
        k = m % 2
        print("       %4d %8d %7.4f %3d %11.6f %16.6f %13.6f %13.6f" % (m, n, c_n, k, law[k], cp[k], d1, d2))
        sys.stdout.flush()
    for par in (0, 1):
        tc = tv_cond[par]
        check(all(tc[i + 1] < tc[i] for i in range(len(tc) - 1)) and tc[-1] < 0.02,
              "n = m^3/8, m %s: d_TV(N_[3,inf)(Y), Po(c_n) given the parity) decreases to %.5f" % ("even" if par == 0 else "odd", tc[-1]))
        check(min(tv_po[par]) > 0.2, "control, m %s: d_TV(N_[3,inf)(Y), Po(c_n)) stays above 0.2" % ("even" if par == 0 else "odd"))
    print("       limits for c = 1: P(X = 0 | X even) = 1/cosh 1 = %.6f, P(X = 1 | X odd) = 1/sinh 1 = %.6f;"
          % (1 / cosh(1), 1 / sinh(1)))
    print("       on the xi side P(xi_(1) <= 2) -> e^-1 = %.6f for both parities" % exp(-1))

    print("   E3. n = m^4 (c_n -> 0): d_K(Y_(1), xi_(1)) -> 0 for m even and -> 1 for m odd.")
    ev = []
    od = []
    for m in [10, 11, 20, 21, 40, 41, 80, 81]:
        d, c_n = parity_dK(m, m ** 4)
        (ev if m % 2 == 0 else od).append(d)
        print("       m = %3d, n = %9d, c_n = %.4f:  d_K = %.5f" % (m, m ** 4, c_n, d))
        sys.stdout.flush()
    check(all(ev[i + 1] < ev[i] for i in range(3)), "n = m^4, m even: d_K decreases")
    check(all(od[i + 1] > od[i] for i in range(3)) and od[-1] > 0.9, "n = m^4, m odd: d_K increases, above 0.9 at m = 81")

    print("   E4. Theorem 1.5 with a = 2: r = sqrt(pi m) P(S_n = m) / P(E_m), where E_m = {S_n = m mod 2}, tends to 1.")
    print("       %4s %9s %10s %10s %10s" % ("m", "n", "tau", "P(E_m)", "r"))

    def ratio(m, n):
        tau, Phi = parity_tau(m, n)
        lP = log(parity_Z(m, n)) + m * log(tau) - n * log(Phi)
        Phi_minus = 1 + tau * tau / (1 + tau)
        PE = 0.5 * (1 + (-1) ** m * exp(n * log(Phi_minus / Phi)))
        r = exp(lP) * sqrt(pi * m) / PE
        print("       %4d %9d %10.6f %10.6f %10.6f" % (m, n, tau, PE, r))
        sys.stdout.flush()
        return abs(r - 1)

    def falls(v):
        return all(v[i + 1] < v[i] for i in range(len(v) - 1))

    print("       n = m^2:")
    e2 = {0: [], 1: []}
    for m in [20, 21, 40, 41, 80, 81, 160, 161, 320, 321, 640, 641]:
        e2[m % 2].append((m, ratio(m, m * m)))
    for par in (0, 1):
        v = [d for (m, d) in e2[par] if m >= 80]
        check(falls(v) and v[-1] < 0.011, "n = m^2, m %s: |r - 1| decreases from m = 80 on, %.6f at m = %d"
              % ("even" if par == 0 else "odd", v[-1], e2[par][-1][0]))
    print("       n = m^3/8:")
    e3 = {0: [], 1: []}
    for m in [20, 21, 40, 41, 80, 81, 160, 161, 320, 321]:
        e3[m % 2].append(ratio(m, int(round(m ** 3 / 8))))
    for par in (0, 1):
        v = e3[par]
        check(falls(v) and v[-1] < 0.006, "n = m^3/8, m %s: |r - 1| decreases, %.6f at the largest m"
              % ("even" if par == 0 else "odd", v[-1]))
    print("       n = m^4, m even (for m odd P(E_m) -> 0 and the theorem does not apply):")
    v = [ratio(m, m ** 4) for m in [10, 20, 40, 80]]
    check(falls(v) and v[-1] < 0.003, "n = m^4, m even: |r - 1| decreases, %.6f at m = 80" % v[-1])


if __name__ == "__main__":
    part_A()
    part_B()
    part_C()
    part_D()
    part_E()
    print()
    if FAILED:
        print("%d CHECK(S) FAILED" % len(FAILED))
        sys.exit(1)
    print("ALL CHECKS PASS")
