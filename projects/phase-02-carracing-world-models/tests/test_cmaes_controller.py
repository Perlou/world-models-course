"""测试 C（LinearController）参数量、动作范围与 CMA-ES 在斜椭圆上的收敛。"""

import numpy as np

from cmaes_controller import LinearController, SimplifiedCMAES


def test_controller_params_and_bounds():
    c_z = LinearController(use_hidden_state=False)
    c_zh = LinearController(use_hidden_state=True)
    assert c_z.n_params == 99 and c_zh.n_params == 867
    a0 = c_zh.act(np.zeros(32, dtype=np.float32), np.zeros(256, dtype=np.float32))
    assert np.allclose(a0, [0.0, 0.5, 0.0], atol=1e-6)

    c_zh.set_params(np.full(867, 10.0))
    a_hi = c_zh.act(np.ones(32, dtype=np.float32), np.ones(256, dtype=np.float32))
    assert -1.0 <= a_hi[0] <= 1.0 and 0.0 <= a_hi[1] <= 1.0 and 0.0 <= a_hi[2] <= 1.0


def test_cmaes_solves_tilted_ellipsoid():
    es = SimplifiedCMAES(dim=2, sigma=1.0, pop_size=12, seed=0)
    target = np.array([2.5, -1.5])
    rot = np.array([[0.8, -0.6], [0.6, 0.8]])
    for _ in range(60):
        cand = es.ask()
        diff = (cand - target) @ rot.T
        fit = -(diff[:, 0] ** 2 + 50.0 * diff[:, 1] ** 2)
        es.tell(cand, fit)
    assert np.linalg.norm(es.mean - target) < 1e-3
