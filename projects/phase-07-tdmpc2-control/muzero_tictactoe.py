"""Phase 7 附带交付物：无规则知情的 MuZero 学会井字棋（只在根节点屏蔽合法动作，树内全靠学到的隐动力学）。"""

import math
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import seed_everything

WIN_LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]


class TTTBoard:
    def __init__(self, board=None, player=1):
        self.board = np.zeros(9, dtype=np.int8) if board is None else np.array(board, dtype=np.int8, copy=True)
        self.player = int(player)

    def legal_actions(self):
        return [i for i in range(9) if self.board[i] == 0]

    def step(self, a):
        nxt = TTTBoard(self.board, self.player)
        nxt.board[a] = self.player
        nxt.player = -self.player
        return nxt

    def winner(self):
        b = self.board
        for i, j, k in WIN_LINES:
            s = int(b[i]) + int(b[j]) + int(b[k])
            if s == 3:
                return 1
            if s == -3:
                return -1
        return 0 if 0 not in b else None

    def encode(self):
        me = (self.board == self.player).astype(np.float32)
        opp = (self.board == -self.player).astype(np.float32)
        return np.concatenate([me, opp], axis=0)


class MuZeroTTTNet(nn.Module):
    def __init__(self, latent_dim=32, hidden=64):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(18, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Linear(hidden, latent_dim))
        self.dyn = nn.Sequential(nn.Linear(latent_dim + 9, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Linear(hidden, latent_dim))
        self.pred_h = nn.Sequential(nn.Linear(latent_dim, hidden), nn.ReLU())
        self.pred_p = nn.Linear(hidden, 9)
        self.pred_v = nn.Linear(hidden, 1)

    def _scale(self, z):
        mn = z.min(dim=-1, keepdim=True).values
        mx = z.max(dim=-1, keepdim=True).values
        return (z - mn) / (mx - mn).clamp_min(1e-5)

    def initial_inference(self, obs):
        z = self._scale(self.enc(obs))
        h = self.pred_h(z)
        return z, self.pred_p(h), torch.tanh(self.pred_v(h)).squeeze(-1)

    def recurrent_inference(self, z, act_idx):
        a_oh = F.one_hot(act_idx.long(), num_classes=9).to(z.dtype)
        z_nxt = self._scale(self.dyn(torch.cat([z, a_oh], dim=-1)))
        h = self.pred_h(z_nxt)
        return z_nxt, self.pred_p(h), torch.tanh(self.pred_v(h)).squeeze(-1)


class _Node:
    def __init__(self, prior=1.0):
        self.prior = float(prior)
        self.n = 0
        self.v_sum = 0.0
        self.z = None
        self.children = {}

    @property
    def q(self):
        return 0.0 if self.n == 0 else self.v_sum / self.n


@torch.no_grad()
def muzero_ttt_mcts(net, board, n_sims=24, add_noise=False, rng=None):
    rng = np.random.default_rng(0) if rng is None else rng
    z0, p0_l, _ = net.initial_inference(torch.from_numpy(board.encode()[None]))
    legal = board.legal_actions()
    mask = torch.zeros(9, dtype=torch.bool)
    mask[legal] = True
    priors0 = F.softmax(p0_l[0].masked_fill(~mask, -1e9), dim=-1).numpy()
    if add_noise:
        noise = rng.dirichlet([0.8] * len(legal))
        for a, nv in zip(legal, noise):
            priors0[a] = 0.75 * priors0[a] + 0.25 * float(nv)

    root = _Node()
    root.z = z0
    for a in legal:
        root.children[a] = _Node(prior=priors0[a])

    for _ in range(n_sims):
        nd = root
        path = [nd]
        acts = []
        while len(nd.children) > 0:
            n_tot = max(1, sum(c.n for c in nd.children.values()))
            best_sc, best_a, best_c = -1e9, 0, None
            for a, ch in nd.children.items():
                sc = -ch.q + 1.25 * ch.prior * math.sqrt(n_tot) / (1 + ch.n)
                if sc > best_sc:
                    best_sc, best_a, best_c = sc, a, ch
            acts.append(best_a)
            nd = best_c
            path.append(nd)
        z_new, p_l, v_new = net.recurrent_inference(path[-2].z, torch.tensor([acts[-1]]))
        nd.z = z_new
        prs = F.softmax(p_l[0], dim=-1).numpy()
        for a in range(9):
            nd.children[a] = _Node(prior=prs[a])
        v = float(v_new.item())
        for node in reversed(path):
            node.n += 1
            node.v_sum += v
            v = -v

    counts = np.zeros(9, dtype=np.float32)
    for a, ch in root.children.items():
        counts[a] = ch.n
    return counts / max(1.0, counts.sum())


def train_muzero_tictactoe(n_games=120, n_sims=24, unroll_k=3, seed=0):
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    net = MuZeroTTTNet(latent_dim=32, hidden=64)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
    replay = deque(maxlen=250)

    for g in range(n_games):
        net.eval()
        b = TTTBoard()
        hist = []
        while b.winner() is None:
            pi = muzero_ttt_mcts(net, b, n_sims=n_sims, add_noise=True, rng=rng)
            a = int(rng.choice(9, p=pi)) if len(hist) < 4 else int(np.argmax(pi))
            hist.append((b.encode(), a, pi, b.player))
            b = b.step(a)
        w = b.winner()
        replay.append([(o, a, pi, float(w * pl)) for (o, a, pi, pl) in hist])

        if len(replay) >= 10:
            net.train()
            for _ in range(8):
                B = 32
                o0_b, a_b, pi_b, v_b, pm_b = [], [], [], [], []
                for _ in range(B):
                    game = replay[int(rng.integers(0, len(replay)))]
                    L = len(game)
                    s = int(rng.integers(0, L))
                    o0_b.append(game[s][0])
                    acts = np.zeros(unroll_k, dtype=np.int64)
                    pis = np.zeros((unroll_k + 1, 9), dtype=np.float32)
                    vs = np.zeros(unroll_k + 1, dtype=np.float32)
                    pms = np.zeros(unroll_k + 1, dtype=np.float32)
                    for k in range(unroll_k + 1):
                        if s + k < L:
                            pis[k] = game[s + k][2]
                            vs[k] = game[s + k][3]
                            pms[k] = 1.0
                        if k > 0 and s + k - 1 < L:
                            acts[k - 1] = game[s + k - 1][1]
                    a_b.append(acts)
                    pi_b.append(pis)
                    v_b.append(vs)
                    pm_b.append(pms)

                o0_t = torch.from_numpy(np.stack(o0_b))
                a_t = torch.from_numpy(np.stack(a_b))
                pi_t = torch.from_numpy(np.stack(pi_b))
                v_t = torch.from_numpy(np.stack(v_b))
                pm_t = torch.from_numpy(np.stack(pm_b))

                z, p_l, v_p = net.initial_inference(o0_t)
                loss = (-(pi_t[:, 0] * F.log_softmax(p_l, -1)).sum(-1) * pm_t[:, 0]).mean() + F.mse_loss(v_p, v_t[:, 0])
                for k in range(1, unroll_k + 1):
                    z = z * 0.5 + z.detach() * 0.5
                    z, p_l, v_p = net.recurrent_inference(z, a_t[:, k - 1])
                    loss = loss + ((-(pi_t[:, k] * F.log_softmax(p_l, -1)).sum(-1) * pm_t[:, k]).mean() + F.mse_loss(v_p, v_t[:, k])) / unroll_k
                opt.zero_grad()
                loss.backward()
                opt.step()

    # 评估对随机对手胜/和/负
    net.eval()
    wins, draws, losses = 0, 0, 0
    n_eval = 40
    for gi in range(n_eval):
        b = TTTBoard()
        me = 1 if gi % 2 == 0 else -1
        while b.winner() is None:
            if b.player == me:
                pi = muzero_ttt_mcts(net, b, n_sims=n_sims, add_noise=False, rng=rng)
                a = int(np.argmax(pi))
            else:
                a = int(rng.choice(b.legal_actions()))
            b = b.step(a)
        w = b.winner()
        if w == me:
            wins += 1
        elif w == 0:
            draws += 1
        else:
            losses += 1
    return net, dict(win_rate=wins / n_eval, draw_rate=draws / n_eval, loss_rate=losses / n_eval)
