"""
Backtest Optimizer — Walk-forward optimization, Optuna, Monte Carlo overfit detection.
"""

from __future__ import annotations

import copy
import random
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from loguru import logger

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from src.backtest.engine import BacktestEngine


class WalkForwardOptimizer:
    """
    Walk-forward optimization with overfit detection.

    Process:
    1. Divide data into in-sample (IS) and out-of-sample (OOS) windows
    2. Optimize parameters on IS
    3. Test on OOS
    4. Slide forward and repeat
    5. Aggregate OOS results for true performance estimate
    """

    def __init__(
        self,
        is_window_days: int = 252,
        oos_window_days: int = 63,
        step_days: int = 63,
        n_monte_carlo: int = 1000,
    ):
        self.is_window = is_window_days
        self.oos_window = oos_window_days
        self.step_days = step_days
        self.n_monte_carlo = n_monte_carlo

    def walk_forward(
        self,
        data: dict[str, pd.DataFrame],
        param_space: dict[str, tuple],
        objective_fn: Callable,
        signal_generator_factory: Callable,
        initial_capital: float = 500000,
    ) -> dict[str, Any]:
        """
        Run walk-forward optimization.

        Args:
            data: {symbol: DataFrame with OHLCV + features}
            param_space: {param_name: (min, max)} for optimization
            objective_fn: Callable that returns metric to maximize
            signal_generator_factory: Creates signal generator from params
            initial_capital: Starting capital

        Returns:
            Walk-forward results with aggregated OOS performance
        """
        # Get date range
        all_dates = sorted(set(
            d for df in data.values()
            for d in pd.to_datetime(df["datetime"]).dt.date
        ))

        total_days = len(all_dates)
        min_required = self.is_window + self.oos_window
        if total_days < min_required:
            logger.error(
                f"Insufficient data: {total_days} days < {min_required} required"
            )
            return {"error": "insufficient data"}

        # Generate walk-forward windows
        windows = []
        start_idx = 0
        while start_idx + self.is_window + self.oos_window <= total_days:
            is_start = all_dates[start_idx]
            is_end = all_dates[start_idx + self.is_window - 1]
            oos_start = all_dates[start_idx + self.is_window]
            oos_end_idx = min(
                start_idx + self.is_window + self.oos_window - 1,
                total_days - 1,
            )
            oos_end = all_dates[oos_end_idx]

            windows.append({
                "is_start": is_start.isoformat(),
                "is_end": is_end.isoformat(),
                "oos_start": oos_start.isoformat(),
                "oos_end": oos_end.isoformat(),
            })

            start_idx += self.step_days

        logger.info(f"Walk-forward: {len(windows)} windows")

        # Run each window
        is_results = []
        oos_results = []

        for i, window in enumerate(windows):
            logger.info(
                f"Window {i+1}/{len(windows)}: "
                f"IS={window['is_start']} to {window['is_end']}, "
                f"OOS={window['oos_start']} to {window['oos_end']}"
            )

            # Optimize on in-sample
            best_params = self._optimize_window(
                data, param_space, objective_fn, signal_generator_factory,
                window["is_start"], window["is_end"], initial_capital,
            )

            # Test on out-of-sample
            signal_gen = signal_generator_factory(best_params)
            engine = BacktestEngine(initial_capital=initial_capital)
            oos_result = engine.run(
                data, signal_gen,
                start_date=window["oos_start"],
                end_date=window["oos_end"],
            )

            is_results.append({"window": i + 1, "params": best_params})
            oos_results.append({
                "window": i + 1,
                "params": best_params,
                **oos_result.get("overview", {}),
                **oos_result.get("returns", {}),
            })

        # Aggregate OOS
        oos_df = pd.DataFrame(oos_results)
        aggregated = {
            "n_windows": len(windows),
            "avg_return": round(oos_df.get("total_return_pct", pd.Series([0])).mean(), 2),
            "avg_sharpe": round(oos_df.get("sharpe_ratio", pd.Series([0])).mean(), 3),
            "windows": windows,
            "oos_results": oos_results,
        }

        return aggregated

    def _optimize_window(
        self,
        data: dict[str, pd.DataFrame],
        param_space: dict[str, tuple],
        objective_fn: Callable,
        signal_generator_factory: Callable,
        start_date: str,
        end_date: str,
        initial_capital: float,
    ) -> dict[str, Any]:
        """Optimize parameters for a single IS window."""
        if OPTUNA_AVAILABLE:
            return self._optuna_optimize(
                data, param_space, objective_fn, signal_generator_factory,
                start_date, end_date, initial_capital,
            )
        return self._grid_optimize(
            data, param_space, signal_generator_factory,
            start_date, end_date, initial_capital,
        )

    def _optuna_optimize(
        self,
        data: dict[str, pd.DataFrame],
        param_space: dict[str, tuple],
        objective_fn: Callable,
        signal_generator_factory: Callable,
        start_date: str,
        end_date: str,
        initial_capital: float,
        n_trials: int = 50,
    ) -> dict[str, Any]:
        """Optimize using Optuna."""
        def objective(trial):
            params = {}
            for name, (low, high) in param_space.items():
                if isinstance(low, float):
                    params[name] = trial.suggest_float(name, low, high)
                elif isinstance(low, int):
                    params[name] = trial.suggest_int(name, low, high)

            signal_gen = signal_generator_factory(params)
            engine = BacktestEngine(initial_capital=initial_capital)
            result = engine.run(data, signal_gen, start_date, end_date)
            return objective_fn(result)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        return study.best_params

    def _grid_optimize(
        self,
        data: dict[str, pd.DataFrame],
        param_space: dict[str, tuple],
        signal_generator_factory: Callable,
        start_date: str,
        end_date: str,
        initial_capital: float,
    ) -> dict[str, Any]:
        """Simple grid search fallback."""
        best_score = -float("inf")
        best_params = {}

        # Generate grid points (5 per param)
        grids = {}
        for name, (low, high) in param_space.items():
            if isinstance(low, float):
                grids[name] = np.linspace(low, high, 5).tolist()
            else:
                step = max(1, (high - low) // 4)
                grids[name] = list(range(low, high + 1, step))

        # Evaluate random subset to keep runtime reasonable
        all_combos = 1
        for vals in grids.values():
            all_combos *= len(vals)

        max_evals = min(all_combos, 50)

        for _ in range(max_evals):
            params = {
                name: random.choice(vals) for name, vals in grids.items()
            }

            signal_gen = signal_generator_factory(params)
            engine = BacktestEngine(initial_capital=initial_capital)
            result = engine.run(data, signal_gen, start_date, end_date)

            sharpe = result.get("returns", {}).get("sharpe_ratio", 0)
            if sharpe > best_score:
                best_score = sharpe
                best_params = params

        return best_params

    def monte_carlo_test(
        self,
        trades: list[dict],
        n_simulations: int = 1000,
        initial_capital: float = 500000,
    ) -> dict[str, Any]:
        """
        Monte Carlo simulation to test robustness.

        Randomly shuffles trade order to see if results are robust
        or dependent on specific sequencing.
        """
        if not trades:
            return {"error": "no trades"}

        pnls = [t.get("pnl", t.pnl if hasattr(t, "pnl") else 0) for t in trades]

        final_equities = []
        max_drawdowns = []
        sharpe_ratios = []

        for _ in range(n_simulations):
            shuffled = random.sample(pnls, len(pnls))
            equity = initial_capital
            peak = equity
            max_dd = 0
            returns = []

            for pnl in shuffled:
                prev = equity
                equity += pnl
                ret = (equity - prev) / prev if prev > 0 else 0
                returns.append(ret)
                peak = max(peak, equity)
                dd = (peak - equity) / peak * 100
                max_dd = max(max_dd, dd)

            final_equities.append(equity)
            max_drawdowns.append(max_dd)

            if np.std(returns) > 0:
                sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
                sharpe_ratios.append(sharpe)

        return {
            "n_simulations": n_simulations,
            "n_trades": len(pnls),
            "original_final_equity": initial_capital + sum(pnls),
            "mc_median_equity": round(float(np.median(final_equities)), 2),
            "mc_5th_pct_equity": round(float(np.percentile(final_equities, 5)), 2),
            "mc_95th_pct_equity": round(float(np.percentile(final_equities, 95)), 2),
            "mc_median_drawdown": round(float(np.median(max_drawdowns)), 2),
            "mc_95th_pct_drawdown": round(float(np.percentile(max_drawdowns, 95)), 2),
            "mc_median_sharpe": round(float(np.median(sharpe_ratios)), 3) if sharpe_ratios else 0,
            "probability_profit": round(
                sum(1 for e in final_equities if e > initial_capital) / n_simulations * 100, 1
            ),
            "overfit_score": self._overfit_score(pnls, final_equities, initial_capital),
        }

    def _overfit_score(
        self,
        pnls: list[float],
        mc_equities: list[float],
        initial_capital: float,
    ) -> float:
        """
        Overfit detection score.

        If original sequence performance is much better than random
        shuffles, the strategy may be overfit to the specific sequence.

        Score 0-1: higher = more likely overfit.
        """
        original_return = sum(pnls) / initial_capital
        mc_returns = [(e - initial_capital) / initial_capital for e in mc_equities]

        # What percentile is the original result in MC distribution?
        better_count = sum(1 for r in mc_returns if r >= original_return)
        percentile = better_count / len(mc_returns)

        # If original is in top 5% of random shuffles → likely overfit
        if percentile < 0.05:
            return round(1 - percentile, 3)
        return round(max(0, 0.5 - percentile), 3)
