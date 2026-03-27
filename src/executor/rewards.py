"""Reward functions for the trading environment.

The Differential Sharpe Ratio (DSR) provides an online, incremental
estimate of the Sharpe ratio at each timestep. It was introduced by
Moody & Saffell (2001) and is well-suited for RL because it gives
meaningful per-step rewards rather than only episode-level metrics.

DSR formula:
    D_t = (B_{t-1} * ΔA_t - 0.5 * A_{t-1} * ΔB_t) / (B_{t-1} - A_{t-1}^2)^{3/2}

Where:
    A_t = A_{t-1} + η * (R_t - A_{t-1})          [EMA of returns]
    B_t = B_{t-1} + η * (R_t^2 - B_{t-1})        [EMA of squared returns]
    η = adaptation rate (e.g., 0.01)
    R_t = portfolio return at time t
"""

import numpy as np


class DifferentialSharpeReward:
    """Computes the Differential Sharpe Ratio at each timestep.

    This provides a per-step reward that approximates the marginal
    contribution of the current return to the overall Sharpe ratio.
    """

    def __init__(self, eta: float = 0.01, clip_value: float = 3.0):
        """
        Args:
            eta: Adaptation rate for EMA estimates. Smaller values
                 give smoother but slower-adapting estimates.
                 Typical range: 0.001 to 0.05.
            clip_value: Clip DSR output to [-clip_value, clip_value].
                        Prevents the denominator (B - A²)^{3/2} → 0 feedback
                        loop that causes mode collapse when the policy learns
                        a constant-action strategy (zero variance → tiny
                        denominator → runaway large DSR → reinforces same
                        action). Default 3.0 ≈ 2× the Sharpe of an excellent
                        daily-return strategy.
        """
        self.eta = eta
        self.clip_value = clip_value
        self.reset()

    def reset(self) -> None:
        """Reset EMA state for a new episode."""
        self.A = 0.0  # EMA of returns
        self.B = 0.0  # EMA of squared returns
        self._step = 0

    def compute(self, portfolio_return: float) -> float:
        """Compute the Differential Sharpe Ratio for this timestep.

        Args:
            portfolio_return: The portfolio return R_t at this step.

        Returns:
            The DSR reward value.
        """
        self._step += 1

        # For the first few steps, use simple reward to bootstrap
        if self._step < 3:
            self.A = self.A + self.eta * (portfolio_return - self.A)
            self.B = self.B + self.eta * (portfolio_return**2 - self.B)
            return portfolio_return

        # Compute deltas
        delta_A = portfolio_return - self.A
        delta_B = portfolio_return**2 - self.B

        # Denominator: (B - A^2)^{3/2}
        variance_est = self.B - self.A**2
        if variance_est <= 1e-12:
            # Avoid division by zero; fall back to simple return
            dsr = portfolio_return
        else:
            denominator = variance_est**1.5
            dsr = (self.B * delta_A - 0.5 * self.A * delta_B) / denominator

        # Update EMAs
        self.A = self.A + self.eta * delta_A
        self.B = self.B + self.eta * delta_B

        # Clip to prevent runaway reinforcement of degenerate constant-action
        # policies where low variance makes the denominator near-zero.
        return float(np.clip(dsr, -self.clip_value, self.clip_value))


class LogReturnReward:
    """Simple log-return reward — stateless, strongly scaled.

    Computes ``r_t = log(1 + portfolio_return_t)`` at each timestep.

    Unlike the Differential Sharpe Ratio, this reward is:
    - **Stateless:** no EMA state, no warm-up phase, no variance
      denominator that can collapse to zero.
    - **Strongly scaled:** daily AAPL returns are O(1e-3 to 1e-2),
      producing rewards of the same magnitude — 100–1000× larger than
      typical DSR values.  This gives PPO a meaningful gradient signal
      even on small datasets.
    - **Monotonically aligned** with economic performance: positive
      returns always yield positive rewards, and the log transform
      naturally penalises large losses more than it rewards equivalent
      gains (asymmetric risk sensitivity).

    The log transform is preferred over raw returns because it is
    additive across time (``sum(log(1+r_t)) = log(cumulative_return)``)
    and better-behaved for the value function approximator.
    """

    def __init__(self) -> None:
        pass

    def reset(self) -> None:
        """No-op: LogReturnReward is stateless."""
        pass

    def compute(self, portfolio_return: float) -> float:
        """Compute log-return reward for this timestep.

        Args:
            portfolio_return: The portfolio return R_t at this step
                              (after transaction costs).

        Returns:
            ``log(1 + R_t)``.  For typical daily returns |R_t| < 0.1
            this is very close to R_t itself, but better-behaved for
            compounding and value estimation.
        """
        return float(np.log1p(portfolio_return))
