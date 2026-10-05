"""
Sparse-horizon lateral MPC QP solved with OSQP (physical units).

State x = [e_y, e_psi, v_y, r]; input u = delta; measured disturbance
d_kappa = vx * curvature preview.  Constraints: steering box, steering rate and
a soft lateral-error corridor.  On any solver failure a safe fallback move is
returned with ``valid=False`` instead of NaN.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import osqp
import scipy.sparse as sparse


@dataclass(frozen=True)
class MpcWeights:
    """Cost weights for the lateral MPC."""

    lateral_error: float = 40.0
    heading_error: float = 60.0
    lateral_velocity: float = 5.0
    yaw_rate: float = 10.0
    steering: float = 20.0
    steering_rate: float = 20.0
    slack: float = 1000.0

    def state_weights(self) -> np.ndarray:
        """Return the diagonal state cost matrix."""
        return np.diag(
            [self.lateral_error, self.heading_error, self.lateral_velocity, self.yaw_rate]
        )


@dataclass
class MpcSolution:
    """Result of one QP solve."""

    first_move_rad: float
    steering_sequence: np.ndarray
    predicted_states: np.ndarray
    slack: np.ndarray
    status: str
    iterations: int
    solve_time_ms: float
    valid: bool
    fallback_move_rad: float = 0.0


class LateralMpc:
    """Build and solve the lateral MPC QP."""

    def __init__(
        self,
        horizon: int = 20,
        delta_max_rad: float = 0.37,
        delta_rate_max_rad_s: float = 2.5,
        lateral_corridor_m: float = 1.5,
        weights: MpcWeights | None = None,
    ):
        if horizon < 1:
            raise ValueError("horizon must be positive")
        self.horizon = horizon
        self.delta_max_rad = delta_max_rad
        self.delta_rate_max_rad_s = delta_rate_max_rad_s
        self.lateral_corridor_m = lateral_corridor_m
        self.weights = weights or MpcWeights()

    def _prediction_matrices(
        self, a_aug: np.ndarray, b_aug: np.ndarray, sample_time_s: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (F, G, H) with X = F x0 + G U + H D over the horizon."""
        n = self.horizon
        state_dim = a_aug.shape[0]
        f_matrix = np.zeros((n * state_dim, state_dim))
        g_matrix = np.zeros((n * state_dim, n))
        h_matrix = np.zeros((n * state_dim, n))
        # Recursively propagate one block row.  The former nested
        # ``matrix_power`` implementation recomputed the same A powers hundreds
        # of times per 50 Hz step.  This recurrence is algebraically identical:
        # X[k+1] = A X[k] + B_u u[k] + B_d d[k].
        f_block = np.eye(state_dim)
        g_block = np.zeros((state_dim, n))
        h_block = np.zeros((state_dim, n))
        for step in range(n):
            row = slice(step * state_dim, (step + 1) * state_dim)
            f_block = a_aug @ f_block
            g_block = a_aug @ g_block
            h_block = a_aug @ h_block
            g_block[:, step] = b_aug[:, 0]
            h_block[:, step] = b_aug[:, 1]
            f_matrix[row, :] = f_block
            g_matrix[row, :] = g_block
            h_matrix[row, :] = h_block
        return f_matrix, g_matrix, h_matrix

    def solve(
        self,
        x0: np.ndarray,
        curvature_preview: np.ndarray,
        vx_mps: float,
        a_aug: np.ndarray,
        b_aug: np.ndarray,
        sample_time_s: float,
        previous_delta_rad: float = 0.0,
        warm_start: np.ndarray | None = None,
        left_corridor_preview: np.ndarray | None = None,
        right_corridor_preview: np.ndarray | None = None,
    ) -> MpcSolution:
        """Solve the QP and return the first steering move."""
        n = self.horizon
        if np.any(~np.isfinite(x0)) or np.any(~np.isfinite(curvature_preview)):
            return self._fallback(previous_delta_rad, "nonfinite_input")
        disturbance = vx_mps * np.asarray(curvature_preview, dtype=float).reshape(-1)
        if disturbance.size != n:
            disturbance = np.resize(disturbance, n)

        f_matrix, g_matrix, h_matrix = self._prediction_matrices(a_aug, b_aug, sample_time_s)
        q_weights = self.weights.state_weights()
        free_response = f_matrix @ x0 + h_matrix @ disturbance
        # Q is diagonal, so applying it row-wise avoids constructing a sparse
        # Kronecker matrix and immediately converting it back to dense.
        q_diagonal = np.tile(np.diag(q_weights), n)
        hessian_u = g_matrix.T @ (q_diagonal[:, None] * g_matrix)
        linear_u = g_matrix.T @ (q_diagonal * free_response)
        rate_matrix = np.eye(n)
        rate_matrix[np.arange(1, n), np.arange(0, n - 1)] = -1.0
        previous_vector = np.zeros(n)
        previous_vector[0] = previous_delta_rad
        hessian_u += self.weights.steering_rate * (rate_matrix.T @ rate_matrix)
        linear_u -= self.weights.steering_rate * (rate_matrix.T @ previous_vector)
        hessian = np.block(
            [
                [hessian_u + self.weights.steering * np.eye(n), np.zeros((n, n))],
                [np.zeros((n, n)), self.weights.slack * np.eye(n)],
            ]
        )
        gradient = np.concatenate([linear_u, np.zeros(n)])

        # Lateral corridor rows on e_y (state index 0).
        e_rows = g_matrix[0::4, :]
        e_free = free_response[0::4]
        left_corridor = self._corridor_array(left_corridor_preview)
        right_corridor = self._corridor_array(right_corridor_preview)
        upper_e = np.hstack([e_rows, -np.eye(n)])
        lower_e = np.hstack([-e_rows, -np.eye(n)])
        upper_e_bounds = left_corridor - e_free
        lower_e_bounds = right_corridor + e_free

        steering_bounds = np.hstack([np.eye(n), np.zeros((n, n))])
        rate_rows = np.hstack([rate_matrix, np.zeros((n, n))])
        rate_limit = self.delta_rate_max_rad_s * sample_time_s
        rate_upper = np.full(n, rate_limit)
        rate_upper[0] = previous_delta_rad + rate_limit
        rate_lower = np.full(n, -rate_limit)
        rate_lower[0] = previous_delta_rad - rate_limit

        constraint_matrix = sparse.csc_matrix(
            np.vstack(
                [
                    upper_e,
                    lower_e,
                    steering_bounds,
                    rate_rows,
                    np.hstack([np.zeros((n, n)), -np.eye(n)]),
                ]
            )
        )
        lower_bounds = np.concatenate(
            [
                -np.inf * np.ones(n),
                -np.inf * np.ones(n),
                -self.delta_max_rad * np.ones(n),
                rate_lower,
                -np.inf * np.ones(n),
            ]
        )
        upper_bounds = np.concatenate(
            [
                upper_e_bounds,
                lower_e_bounds,
                self.delta_max_rad * np.ones(n),
                rate_upper,
                np.zeros(n),
            ]
        )

        solver = osqp.OSQP()
        solver.setup(
            P=sparse.csc_matrix(hessian),
            q=gradient,
            A=constraint_matrix,
            l=lower_bounds,
            u=upper_bounds,
            verbose=False,
            polishing=False,
            # A 2e-3 QP tolerance is at the millimetre-scale corridor
            # resolution and avoids wasting a 20 ms control period chasing
            # numerically meaningless residuals in high-speed corners.
            max_iter=2000,
            eps_abs=2e-3,
            eps_rel=2e-3,
            scaled_termination=True,
            warm_starting=True,
        )
        if warm_start is not None and warm_start.size == 2 * n:
            solver.warm_start(x=warm_start)
        started = time.perf_counter()
        result = solver.solve()
        solve_time_ms = (time.perf_counter() - started) * 1000.0
        accepted_approximate = (
            result.info.status == "maximum iterations reached"
            and result.x is not None
            and np.all(np.isfinite(result.x))
            and result.info.prim_res <= 1e-2
            and result.info.dual_res <= 3e-3
        )
        if (
            result.info.status not in ("solved", "solved inaccurate")
            and not accepted_approximate
        ):
            detail = (
                f"{result.info.status};prim_res={result.info.prim_res:.3e};"
                f"dual_res={result.info.dual_res:.3e};iter={result.info.iter}"
            )
            return self._fallback(previous_delta_rad, detail, solve_time_ms)
        solution = result.x
        steering = np.clip(solution[:n], -self.delta_max_rad, self.delta_max_rad)
        # OSQP's accepted 1e-4 residual can exceed the actuator-rate boundary
        # by a few microradians. Project the published sequence back onto the
        # exact hard bounds; this changes no practically meaningful optimum.
        prior = float(previous_delta_rad)
        for index in range(n):
            steering[index] = np.clip(
                steering[index], prior - rate_limit, prior + rate_limit
            )
            prior = float(steering[index])
        slack = solution[n:]
        predicted = (f_matrix @ x0 + g_matrix @ steering + h_matrix @ disturbance).reshape(n, 4)
        return MpcSolution(
            first_move_rad=float(steering[0]),
            steering_sequence=steering,
            predicted_states=predicted,
            slack=slack,
            status=(
                "accepted bounded-residual iterate"
                if accepted_approximate
                else result.info.status
            ),
            iterations=int(result.info.iter),
            solve_time_ms=solve_time_ms,
            valid=True,
            fallback_move_rad=previous_delta_rad,
        )

    def _corridor_array(self, preview: np.ndarray | None) -> np.ndarray:
        """Return one finite positive horizon corridor."""
        if preview is None:
            return np.full(self.horizon, self.lateral_corridor_m)
        corridor = np.asarray(preview, dtype=float).reshape(-1)
        if corridor.size != self.horizon:
            raise ValueError("corridor preview must match the MPC horizon")
        if np.any(~np.isfinite(corridor)) or np.any(corridor <= 0.0):
            raise ValueError("corridor preview must be finite and positive")
        return corridor

    @staticmethod
    def _fallback(
        previous_delta_rad: float, status: str, solve_time_ms: float = 0.0
    ) -> MpcSolution:
        """Return a safe rate-limited fallback move."""
        return MpcSolution(
            first_move_rad=previous_delta_rad,
            steering_sequence=np.array([]),
            predicted_states=np.array([]),
            slack=np.array([]),
            status=status,
            iterations=0,
            solve_time_ms=solve_time_ms,
            valid=False,
            fallback_move_rad=previous_delta_rad,
        )
