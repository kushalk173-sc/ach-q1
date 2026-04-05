# Admissible Classes for the 2-Factor Filtering Theorem

## Model

Let the latent process be

- `theta_t in {-1,+1}` with `P(theta_t != theta_{t-1}) = eps`
- `z_t in {-1,+1}` with `P(z_t != z_{t-1}) = rho`
- `0 < eps << rho < 1`

and observations

- `y_t = a * theta_t + b * z_t + eta_t`
- `eta_t ~ N(0, sigma^2)`

The online decoder produces estimates from observation history:

- `(theta_hat_t, z_hat_t) = d_t(X_t)`

where `X_t` is the internal state of the code.

We evaluate either

- weighted filtering loss
  `L = limsup_{T -> inf} (1 / T) sum_t [alpha * 1{theta_hat_t != theta_t} + beta * 1{z_hat_t != z_t}]`

or a soft version such as total MSE / Bayes risk / negative mutual information.

## Design Goal

We want a competitor class `C1(R)` that means "one characteristic timescale" in a nontrivial way:

- strong enough to be a real rival
- weak enough to forbid hidden dual-timescale behavior inside a nominally single state
- budget-matched against a dual-summary class `C2(R)`

The main loophole to avoid is a broadband escape hatch: a supposedly single-timescale encoder that internally implements both a slow and fast channel through a rich state update.

## Recommended `C1(R)`: One-Pole Predictive State Class

Define `C1(R)` as the class of online encoders with state `X_t in R^m` satisfying:

- state budget: `m <= R`
- one-step recursion:
  `X_t = A X_{t-1} + phi(y_t)`
- decoder:
  `(theta_hat_t, z_hat_t) = d(X_t)`

with the following restrictions.

### 1. Single characteristic timescale

All eigenvalues of `A` must lie in a narrow annulus around one radius `lambda in (0,1)`:

- `spec(A) subset {re^{i omega} : r in [lambda - delta, lambda + delta]}`

where `delta` is fixed and small relative to `lambda`.

Interpretation:

- the memory half-life of every internal mode is of order `1 / (1 - lambda)`
- there is no separated pair of slow and fast decay constants inside `X_t`

This is the cleanest spectral way to say "single-timescale."

### 2. No hidden multiresolution input bank

The input map `phi(y_t)` is instantaneous:

- it may be nonlinear
- it may be vector-valued
- but it cannot depend on lagged observations except through `X_{t-1}`

This rules out cheating via a hand-built filter bank over the raw observations.

### 3. No block decomposition into separated radii

Even if `A` is not diagonalizable, its Jordan blocks must all be associated with eigenvalues inside the same narrow annulus above. In particular, `A` may not be written, after similarity transform, as two large blocks with radii near `1 - c_1 eps` and `1 - c_2 rho` for well-separated constants `c_1`, `c_2`.

This is the explicit anti-broadband clause.

### 4. Equal resource budget

The budget is the state dimension:

- `cost(X) = m`
- admissibility requires `m <= R`

If desired, this can be strengthened to

- `cost(X) = m + kappa * bits(d) + gamma * bits(phi)`

but dimension-only is the simplest first theorem target.

## Matching `C2(R)`: Dual-Pole Split-State Class

Define `C2(R)` as the class of pairs `(X_t^s, X_t^f)` with

- `X_t^s in R^{m_s}`, `X_t^f in R^{m_f}`
- `m_s + m_f <= R`

and updates

- `X_t^s = A_s X_{t-1}^s + phi_s(y_t)`
- `X_t^f = A_f X_{t-1}^f + phi_f(y_t)`
- `(theta_hat_t, z_hat_t) = d(X_t^s, X_t^f)`

with spectral constraints

- `spec(A_s)` concentrated near radius `lambda_s`
- `spec(A_f)` concentrated near radius `lambda_f`
- `lambda_s > lambda_f`
- the induced memory scales are matched to the latent rates, e.g.
  `1 - lambda_s ~ eps`, `1 - lambda_f ~ rho`

This class is allowed to place part of the budget on slow memory and part on fast memory.

## The Scoped Theorem Target

With these classes, the first claim to try is:

> There exist parameters `(eps, rho, sigma, a, b, R, alpha, beta)` with `eps << rho` such that
> `inf_{X in C1(R)} L(X) > inf_{(X^s, X^f) in C2(R)} L(X^s, X^f)`.

This is not universal over all imaginable single-state machines. It is a theorem about equal-budget one-pole versus two-pole online codes.

## Why This Definition Is Useful

It matches the intuition behind the experiment:

- one pole near the slow scale preserves context but smears events
- one pole near the fast scale tracks events but destabilizes context
- a compromise pole underserves both
- two poles can allocate memory where each latent lives

And it avoids triviality:

- `C1(R)` is still expressive because `phi` and `d` can be nonlinear
- the rival is not reduced to a scalar exponential smoother
- the theorem burden is concentrated exactly where it should be: one spectral band versus two

## If You Want an Information-Theoretic Version

Replace the classification loss with a rate-distortion style objective:

- `I(X_t ; theta_t, z_t)` at fixed state budget `dim(X_t) <= R`

Then the target becomes

> There exists a separated-timescale regime in which the best one-band state `X_t` carries strictly less relevant information about `(theta_t, z_t)` than the best split-band pair `(X_t^s, X_t^f)` under the same total dimension budget.

## Alternate Backup Definition

If the spectral annulus feels too linear-algebra heavy, use a scalar summary version:

- `C1(R)`: all encoders whose internal state is a function of `R` features, each updated with the same forgetting factor `tau`
- `C2(R)`: same, except features are partitioned into two groups with factors `tau_s` and `tau_f`

That version is weaker mathematically, but much easier to analyze first and much closer to the current Python experiment.

## Practical Recommendation

For the first paper-quality result, prove the theorem for the backup class first:

- same-`tau` feature family for `C1(R)`
- split-`tau` feature family for `C2(R)`

Then present the annulus-based class above as the natural stronger conjecture. That gives you:

- a clean computational result now
- a nontrivial formal target next
- a clear explanation of what "single-timescale" means without allowing hidden multiscale escape hatches
