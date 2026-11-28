"""测试 M（MDNRNN）单步与整段序列等价性、MDN 损失与温度采样。"""

import torch

from memory_mdn import MDNRNN, mdn_nll, sample_mdn


def test_step_matches_sequence_forward():
    torch.manual_seed(0)
    rnn = MDNRNN()
    z = torch.randn(2, 6, 32)
    a = torch.randn(2, 6, 3)
    (lp_seq, mu_seq, ls_seq), _ = rnn(z, a)

    state = None
    lps = []
    for t in range(6):
        (lp, _, _), state = rnn.step(z[:, t], a[:, t], state)
        lps.append(lp)
    assert torch.allclose(lp_seq, torch.stack(lps, dim=1), atol=1e-5)


def test_mdn_nll_and_temperature_spread():
    log_pi = torch.zeros(1, 32, 5) - torch.log(torch.tensor(5.0))
    mu = torch.zeros(1, 32, 5)
    log_sigma = torch.zeros(1, 32, 5)
    loss = mdn_nll(log_pi, mu, log_sigma, torch.zeros(1, 32))
    assert loss.item() > 0.0

    g1 = torch.Generator().manual_seed(0)
    g2 = torch.Generator().manual_seed(0)
    low_t = torch.stack([sample_mdn(log_pi, mu, log_sigma, temperature=0.1, generator=g1) for _ in range(64)])
    high_t = torch.stack([sample_mdn(log_pi, mu, log_sigma, temperature=1.5, generator=g2) for _ in range(64)])
    assert high_t.std().item() > 2.5 * low_t.std().item()
