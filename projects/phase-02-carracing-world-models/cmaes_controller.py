"""C（极简线性控制器）与简化版 CMA-ES 优化器。"""

import math

import numpy as np

from memory_mdn import ACTION_DIM, HIDDEN_DIM
from vision_vae import Z_DIM


class LinearController:
    """867 参数（[z_t ; h_t] → a_t）或 99 参数（z_t → a_t）线性控制器。"""

    def __init__(self, use_hidden_state=True):
        self.use_hidden_state = use_hidden_state
        self.in_dim = Z_DIM + HIDDEN_DIM if use_hidden_state else Z_DIM
        self.n_params = ACTION_DIM * self.in_dim + ACTION_DIM
        self.w = np.zeros((ACTION_DIM, self.in_dim), dtype=np.float32)
        self.b = np.zeros(ACTION_DIM, dtype=np.float32)

    def set_params(self, flat_params):
        flat = np.asarray(flat_params, dtype=np.float32)
        assert flat.shape == (self.n_params,)
        split = ACTION_DIM * self.in_dim
        self.w = flat[:split].reshape(ACTION_DIM, self.in_dim)
        self.b = flat[split:]

    def act(self, z, h=None):
        feat = np.concatenate([z, h], axis=0) if self.use_hidden_state else z
        raw = self.w @ feat + self.b
        steer = math.tanh(float(raw[0]))
        gas = 0.5 * (math.tanh(float(raw[1])) + 1.0)
        brake = min(1.0, max(0.0, math.tanh(float(raw[2]))))
        return np.array([steer, gas, brake], dtype=np.float32)


class SimplifiedCMAES:
    """实现均值重组、协方差 rank-1/rank-μ 更新与累积步长控制的 CMA-ES。"""

    def __init__(self, dim, sigma=0.15, pop_size=16, seed=0):
        self.dim = int(dim)
        self.mean = np.zeros(self.dim, dtype=np.float64)
        self.sigma = float(sigma)
        self.pop_size = int(pop_size)
        self.n_elite = self.pop_size // 2
        raw = np.log((self.pop_size + 1) / 2.0) - np.log(np.arange(1, self.n_elite + 1))
        self.weights = raw / raw.sum()
        self.mu_eff = float(1.0 / np.sum(self.weights ** 2))

        n, mu_eff = self.dim, self.mu_eff
        self.c_sigma = (mu_eff + 2.0) / (n + mu_eff + 5.0)
        self.d_sigma = 1.0 + 2.0 * max(0.0, math.sqrt((mu_eff - 1.0) / (n + 1.0)) - 1.0) + self.c_sigma
        self.c_c = (4.0 + mu_eff / n) / (n + 4.0 + 2.0 * mu_eff / n)
        self.c_1 = 2.0 / ((n + 1.3) ** 2 + mu_eff)
        self.c_mu = min(1.0 - self.c_1, 2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((n + 2.0) ** 2 + mu_eff))
        self.chi_n = math.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n))

        self.cov = np.eye(n, dtype=np.float64)
        self.path_c = np.zeros(n, dtype=np.float64)
        self.path_sigma = np.zeros(n, dtype=np.float64)
        self.rng = np.random.default_rng(seed)
        self._update_eigensystem()

    def _update_eigensystem(self):
        self.cov = 0.5 * (self.cov + self.cov.T)
        vals, vecs = np.linalg.eigh(self.cov)
        vals = np.maximum(vals, 1e-20)
        self._B = vecs
        self._D = np.sqrt(vals)
        self._inv_sqrt_C = (vecs * (1.0 / self._D)) @ vecs.T

    def ask(self):
        eps = self.rng.standard_normal((self.pop_size, self.dim))
        return self.mean[None, :] + self.sigma * ((eps * self._D) @ self._B.T)

    def tell(self, candidates, fitness):
        order = np.argsort(fitness)[::-1]
        steps = (candidates - self.mean[None, :]) / max(self.sigma, 1e-20)
        elite_steps = steps[order[: self.n_elite]]
        d_w = self.weights @ elite_steps
        self.mean = self.mean + self.sigma * d_w

        cs = self.c_sigma
        self.path_sigma = (1.0 - cs) * self.path_sigma + math.sqrt(cs * (2.0 - cs) * self.mu_eff) * (
            self._inv_sqrt_C @ d_w
        )
        self.sigma *= math.exp((cs / self.d_sigma) * (np.linalg.norm(self.path_sigma) / self.chi_n - 1.0))

        cc = self.c_c
        self.path_c = (1.0 - cc) * self.path_c + math.sqrt(cc * (2.0 - cc) * self.mu_eff) * d_w
        c_mu_mat = (elite_steps.T * self.weights) @ elite_steps
        self.cov = (1.0 - self.c_1 - self.c_mu) * self.cov + self.c_1 * np.outer(self.path_c, self.path_c) + self.c_mu * c_mu_mat
        self._update_eigensystem()
