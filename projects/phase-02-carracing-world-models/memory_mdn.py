"""M（记忆模型）：MDN-RNN、混合高斯负对数似然与带温度的梦境采样。"""

import math

import torch
import torch.nn as nn

from vision_vae import Z_DIM

ACTION_DIM = 3
HIDDEN_DIM = 256
N_COMPONENTS = 5
HALF_LOG_2PI = 0.5 * math.log(2.0 * math.pi)


class MDNRNN(nn.Module):
    """LSTM + 对角混合密度输出头。"""

    def __init__(self, z_dim=Z_DIM, action_dim=ACTION_DIM, hidden_dim=HIDDEN_DIM, n_components=N_COMPONENTS):
        super().__init__()
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.n_components = n_components
        self.lstm = nn.LSTM(z_dim + action_dim, hidden_dim, batch_first=True)
        self.mdn_head = nn.Linear(hidden_dim, z_dim * n_components * 3)

    def forward(self, z, a, state=None):
        hidden, state = self.lstm(torch.cat([z, a], dim=-1), state)
        out = self.mdn_head(hidden).view(*hidden.shape[:-1], self.z_dim, self.n_components, 3)
        logits, mu, log_sigma = out.unbind(dim=-1)
        log_sigma = log_sigma.clamp(-5.0, 2.0)
        return (torch.log_softmax(logits, dim=-1), mu, log_sigma), state

    def step(self, z, a, state=None):
        """单步推进：z [B, Z], a [B, A] → (log_pi, mu, log_sigma), new_state。"""
        (log_pi, mu, log_sigma), state = self.forward(z.unsqueeze(1), a.unsqueeze(1), state)
        return (log_pi.squeeze(1), mu.squeeze(1), log_sigma.squeeze(1)), state

    def step_hidden(self, z, a, state=None):
        """仅更新 LSTM 状态，跳过 MDN 线性层。"""
        _, state = self.lstm(torch.cat([z, a], dim=-1).unsqueeze(1), state)
        return state


def mdn_nll(log_pi, mu, log_sigma, target):
    """混合密度负对数似然（使用 logsumexp 保证数值稳定），返回每步每维平均值。"""
    sigma = log_sigma.exp()
    log_comp = -0.5 * ((target.unsqueeze(-1) - mu) / sigma).pow(2) - log_sigma - HALF_LOG_2PI
    log_prob = torch.logsumexp(log_pi + log_comp, dim=-1)
    return -log_prob.mean()


@torch.no_grad()
def sample_mdn(log_pi, mu, log_sigma, temperature=1.0, generator=None):
    """按温度 τ 从混合高斯分布采样下一个 z：混合权重除以 τ，组件标准差乘 √τ。"""
    probs = torch.softmax(log_pi / temperature, dim=-1)
    flat_probs = probs.reshape(-1, probs.shape[-1])
    comp = torch.multinomial(flat_probs, 1, generator=generator).view(*probs.shape[:-1], 1)
    chosen_mu = mu.gather(-1, comp).squeeze(-1)
    chosen_sigma = log_sigma.exp().gather(-1, comp).squeeze(-1) * math.sqrt(temperature)
    noise = torch.randn(chosen_mu.shape, generator=generator, device=mu.device, dtype=mu.dtype)
    return chosen_mu + chosen_sigma * noise
