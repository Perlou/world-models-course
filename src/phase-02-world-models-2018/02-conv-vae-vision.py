"""
V：用卷积 VAE 把画面压成隐向量
==============================

学习目标：
    1. 实现论文里的卷积 VAE，逐层核对形状，手算的参数量和代码数出来的一致
    2. 写对 VAE 的损失：重建项对像素求和，KL 项对 32 维求和
    3. 亲手制造一次后验坍塌，看清"重建项求和还是求平均"差在哪
    4. 读重建图：VAE 保留了什么，丢了什么
    5. 在隐空间里插值、逐维扰动，建立对 z 的直觉
    6. 把整个数据集编码成 z，供后面的 M（MDN-RNN）使用

核心概念：
    - 变分自编码器（VAE）：编码器给出 q(z|o) = N(μ, σ²)，解码器从 z 重建画面
    - 重参数化（reparameterization）：z = μ + σ·ε，让采样这一步可以反向传播
    - 后验坍塌（posterior collapse）：编码分布退化成先验，z 里不含任何信息
    - 隐空间（latent space）：z 所在的 32 维空间

前置知识：
    - deep-learning-course 的 VAE 一章（ELBO、重参数化）
    - 01-from-states-to-pixels.py（数据集）

对应概念文档：
    - CONCEPT.md 第 2 节「V：用卷积 VAE 把画面压成隐向量」

预计运行时间：约 6 分钟（MPS）
"""

import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import torch
import torch.nn as nn

from utils import get_device, is_quick, output_dir, save_figure, seed_everything, setup_matplotlib

plt = setup_matplotlib()
OUT = output_dir("phase-02")
SEED = 0
Z_DIM = 32
DEVICE = get_device()
SUFFIX = "-quick" if is_quick() else ""
DATASET_PATH = OUT / f"carracing-random{SUFFIX}.npz"
VAE_PATH = OUT / f"vae{SUFFIX}.pt"
LATENTS_PATH = OUT / f"latents{SUFFIX}.npz"


# ==================== 第一部分：V 要做什么 ====================


def introduction():
    print("=" * 60)
    print("第一部分：V 要做什么")
    print("=" * 60)
    print("""
上一节采到的每一帧是 64×64×3 = 12288 个数。V 的任务是把它压成 32 个数，
而且这 32 个数要能大致还原出原来的画面。

    o [3, 64, 64]                                         ô [3, 64, 64]
        │                                                     ▲
        ▼                                                     │
    ┌────────┐   μ, log σ²   ┌──────────────┐   z [32]   ┌────────┐
    │ 编码器  │─────────────▶│ z = μ + σ·ε   │──────────▶│ 解码器  │
    │ 4 层卷积 │              │ ε ~ N(0, I)   │            │ 4 层反卷积│
    └────────┘               └──────────────┘            └────────┘

    损失 = 重建损失（ô 和 o 像不像）+ KL 损失（q(z|o) 离标准高斯 N(0, I) 有多远）

为什么用 VAE 而不是普通自编码器：KL 项把所有画面的 z 都拉向同一个标准高斯，
隐空间因此是连续、平滑的——相近的 z 解码出相近的画面。后面 M 预测出的 z 总会有偏差，
隐空间平滑，偏一点的 z 才能解码出合理的画面。
    """)


def ensure_artifact(path, lesson):
    """跨讲义共享的产物不存在时，先运行生成它的那一节讲义（继承当前的 WM_QUICK 设置）。"""
    if path.exists():
        return
    print(f"⚠️  没找到 {path.name}，先运行 {lesson} 生成它……")
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), lesson)
    subprocess.run([sys.executable, script], check=True, stdout=subprocess.DEVNULL)


def load_frames():
    """读取 01 节采集的数据集，返回训练帧、验证帧（uint8，已放到设备上）和完整数据集。"""
    ensure_artifact(DATASET_PATH, "01-from-states-to-pixels.py")
    with np.load(DATASET_PATH) as f:
        data = {k: f[k] for k in f.files}
    n_episodes = len(data["lengths"])
    n_val = max(1, n_episodes // 10)                                  # 最后 10% 的回合当验证集

    def gather(episodes):
        frames = np.concatenate([data["frames"][e, : data["lengths"][e]] for e in episodes])
        return torch.from_numpy(frames).to(DEVICE)                    # [N, 64, 64, 3] uint8

    train = gather(range(n_episodes - n_val))
    val = gather(range(n_episodes - n_val, n_episodes))
    return train, val, data


def to_obs(frames_u8):
    """[B, 64, 64, 3] uint8 → [B, 3, 64, 64] float32，取值 [0, 1]。"""
    return frames_u8.permute(0, 3, 1, 2).float() / 255.0


# ==================== 第二部分：网络结构 ====================


class ConvVAE(nn.Module):
    """论文附录的卷积 VAE。"""

    def __init__(self, z_dim=Z_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2), nn.ReLU(),                 # [B, 32, 31, 31]
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),                # [B, 64, 14, 14]
            nn.Conv2d(64, 128, 4, stride=2), nn.ReLU(),               # [B, 128, 6, 6]
            nn.Conv2d(128, 256, 4, stride=2), nn.ReLU(),              # [B, 256, 2, 2]
            nn.Flatten(),                                             # [B, 1024]
        )
        self.fc_mu = nn.Linear(1024, z_dim)
        self.fc_logvar = nn.Linear(1024, z_dim)
        self.fc_decode = nn.Linear(z_dim, 1024)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 128, 5, stride=2), nn.ReLU(),    # [B, 128, 5, 5]
            nn.ConvTranspose2d(128, 64, 5, stride=2), nn.ReLU(),      # [B, 64, 13, 13]
            nn.ConvTranspose2d(64, 32, 6, stride=2), nn.ReLU(),       # [B, 32, 30, 30]
            nn.ConvTranspose2d(32, 3, 6, stride=2), nn.Sigmoid(),     # [B, 3, 64, 64]
        )

    def encode(self, obs):
        """obs [B, 3, 64, 64] → mu [B, 32]，logvar [B, 32]。"""
        hidden = self.encoder(obs)
        return self.fc_mu(hidden), self.fc_logvar(hidden)

    def reparameterize(self, mu, logvar):
        """z = μ + σ·ε。随机性全在 ε 里，梯度可以穿过 μ 和 σ 传回编码器。"""
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def decode(self, z):
        """z [B, 32] → recon [B, 3, 64, 64]，取值 [0, 1]。"""
        return self.decoder(self.fc_decode(z).view(-1, 1024, 1, 1))

    def forward(self, obs):
        mu, logvar = self.encode(obs)
        return self.decode(self.reparameterize(mu, logvar)), mu, logvar


def vae_loss(obs, recon, mu, logvar):
    """返回（重建损失，KL 损失）。两项都是"对每个样本求和，再对批次求平均"。

    重建损失：Σ_像素 (o - ô)²                     对 12288 个值求和
    KL 损失：  0.5 · Σ_维 (μ² + σ² - 1 - log σ²)   对 32 维求和
    """
    recon_loss = ((obs - recon) ** 2).sum(dim=(1, 2, 3)).mean()
    kl_loss = (0.5 * (mu**2 + logvar.exp() - 1 - logvar)).sum(dim=1).mean()
    return recon_loss, kl_loss


def check_architecture():
    print("=" * 60)
    print("第二部分：网络结构")
    print("=" * 60)

    vae = ConvVAE()
    x = torch.zeros(2, 3, 64, 64)
    print("\n编码器逐层的输出形状（批大小 2）：")
    h = x
    for layer in vae.encoder:
        h = layer(h)
        if not isinstance(layer, nn.ReLU):
            print(f"    {layer.__class__.__name__:<16} → {list(h.shape)}")
    mu, logvar = vae.encode(x)
    print(f"    fc_mu / fc_logvar → {list(mu.shape)}")
    print("\n解码器逐层的输出形状：")
    h = vae.fc_decode(mu).view(-1, 1024, 1, 1)
    print(f"    fc_decode        → {list(h.shape)}")
    for layer in vae.decoder:
        h = layer(h)
        if isinstance(layer, nn.ConvTranspose2d):
            print(f"    ConvTranspose2d  → {list(h.shape)}")

    n_params = sum(p.numel() for p in vae.parameters())
    deconv1 = sum(p.numel() for p in vae.decoder[0].parameters())
    print(f"\n参数量：{n_params:,}（论文参数量表里 VAE 那一行是 4,348,547）")
    print(f"其中解码器第一个转置卷积占 {deconv1:,}，即 {deconv1 / n_params * 100:.0f}%。")
    assert n_params == 4_348_547

    # 手算例子：KL 的闭式解
    mu1, sigma1 = 0.5, 0.8
    kl_hand = 0.5 * (mu1**2 + sigma1**2 - 1 - np.log(sigma1**2))
    _, kl_code = vae_loss(torch.zeros(1, 3, 64, 64), torch.zeros(1, 3, 64, 64),
                          torch.tensor([[mu1]]), torch.tensor([[np.log(sigma1**2)]], dtype=torch.float32))
    print(f"""
手算核对 KL：一维，μ = {mu1}，σ = {sigma1}
    0.5 × (μ² + σ² - 1 - ln σ²) = 0.5 × ({mu1**2} + {sigma1**2:.2f} - 1 - ({np.log(sigma1**2):.4f})) = {kl_hand:.4f}
    vae_loss 算出来的 KL：{kl_code.item():.4f}
    """)
    assert abs(kl_hand - kl_code.item()) < 1e-5


# ==================== 第三部分：训练 ====================


@torch.no_grad()
def evaluate(vae, frames, batch_size=512):
    """在一批帧上算平均的重建损失和 KL（用 μ 解码，不采样）。"""
    recon_sum = kl_sum = 0.0
    for i in range(0, len(frames), batch_size):
        obs = to_obs(frames[i:i + batch_size])
        mu, logvar = vae.encode(obs)
        recon_loss, kl_loss = vae_loss(obs, vae.decode(mu), mu, logvar)
        recon_sum += recon_loss.item() * len(obs)
        kl_sum += kl_loss.item() * len(obs)
    return recon_sum / len(frames), kl_sum / len(frames)


def train_vae(train, val, epochs, recon_scale=1.0, batch_size=128, lr=1e-3, label="VAE"):
    """训练 VAE。recon_scale 是重建项的权重：1 就是"对像素求和"，1/12288 相当于"对像素求平均"。"""
    vae = ConvVAE().to(DEVICE)
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    history = []
    start = time.time()
    for epoch in range(epochs):
        perm = torch.randperm(len(train), generator=generator).to(DEVICE)
        for i in range(0, len(perm) - batch_size + 1, batch_size):
            obs = to_obs(train[perm[i:i + batch_size]])               # [B, 3, 64, 64]
            recon, mu, logvar = vae(obs)
            recon_loss, kl_loss = vae_loss(obs, recon, mu, logvar)
            loss = recon_scale * recon_loss + kl_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        val_recon, val_kl = evaluate(vae, val)
        history.append((recon_loss.item(), kl_loss.item(), val_recon, val_kl))
        print(f"    [{label}] 第 {epoch + 1:>2} 轮  训练：重建 {recon_loss.item():7.2f}  KL {kl_loss.item():6.2f}"
              f"   验证：重建 {val_recon:7.2f}  KL {val_kl:6.2f}   已用时 {time.time() - start:5.0f} 秒")
    return vae, history, time.time() - start


def train_main_vae(train, val, epochs):
    print("=" * 60)
    print("第三部分：训练 V")
    print("=" * 60)
    print(f"""
训练集 {len(train)} 帧，验证集 {len(val)} 帧（最后 10% 的回合），设备 {DEVICE}。
画面打散成单帧来训练：V 不需要动作，也不需要时间顺序。
下面打印的两项损失都是"每个样本的平均值"：重建损失是 12288 个像素值的平方误差之和，KL 是 32 维之和。
    """)
    vae, history, elapsed = train_vae(train, val, epochs)
    val_recon, val_kl = history[-1][2], history[-1][3]
    print(f"""
训练用时 {elapsed:.0f} 秒。验证集：重建损失 {val_recon:.2f}，KL {val_kl:.2f}。
换算成直观的量：
    每个像素值的均方误差 = {val_recon:.2f} / 12288 = {val_recon / 12288:.5f}，即平均每个像素值差 {np.sqrt(val_recon / 12288) * 255:.1f} 个灰度级（满刻度 255）
    KL = {val_kl:.1f} nat，平摊到 32 维是每维 {val_kl / 32:.2f} nat
    """)
    torch.save(vae.state_dict(), VAE_PATH)
    print(f"💾 VAE 已保存：{VAE_PATH.relative_to(OUT.parents[1])}")
    return vae, history


# ==================== 第四部分：读重建图 ====================


@torch.no_grad()
def reconstruction_report(vae, val):
    print("\n" + "=" * 60)
    print("第四部分：读重建图——保留了什么，丢了什么")
    print("=" * 60)

    idx = torch.linspace(0, len(val) - 1, 8).long().to(DEVICE)
    obs = to_obs(val[idx])
    mu, _ = vae.encode(obs)
    recon = vae.decode(mu)
    error = (obs - recon).abs().mean(dim=1)                           # [8, 64, 64]

    fig, axes = plt.subplots(3, 8, figsize=(14, 5.6))
    for col in range(8):
        axes[0, col].imshow(obs[col].permute(1, 2, 0).cpu().numpy())
        axes[1, col].imshow(recon[col].permute(1, 2, 0).cpu().numpy())
        axes[2, col].imshow(error[col].cpu().numpy(), cmap="magma", vmin=0, vmax=0.3)
    for ax, title in zip(axes[:, 0], ["原图", "重建（用 μ 解码）", "误差"]):
        ax.set_ylabel(title, fontsize=11)
    for ax in axes.ravel():
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("VAE 重建：32 个数还原 12288 个数", fontsize=12)
    save_figure(fig, OUT / "02-reconstructions.png")

    # 误差集中在哪：把像素按"原图里是不是赛道（灰色）"分开统计
    obs_all = to_obs(val[:: max(1, len(val) // 2000)])
    mu_all, logvar_all = vae.encode(obs_all)
    err_all = ((obs_all - vae.decode(mu_all)) ** 2).mean(dim=1)       # [N, 64, 64]
    r, g, b = obs_all[:, 0], obs_all[:, 1], obs_all[:, 2]
    is_road = ((r - g).abs() < 0.06) & ((g - b).abs() < 0.06) & (r > 0.3) & (r < 0.5)   # 灰色：三个通道接近
    road_share = is_road.float().mean().item()
    err_road = err_all[is_road].mean().item() if is_road.any() else float("nan")
    err_other = err_all[~is_road].mean().item()
    print(f"""
在 {len(obs_all)} 张验证帧上，按"原图里这个像素是不是赛道（灰色）"分开统计均方误差：

    赛道像素（占全部像素的 {road_share * 100:.1f}%）：{err_road:.5f}
    其余像素（草地、车）            ：{err_other:.5f}
    """)
    if err_road > 1.2 * err_other:
        print(f"赛道像素的误差是其余像素的 {err_road / err_other:.1f} 倍。草地是大片的纯色，好重建；"
              "赛道的边缘、弯道的形状这些细节被抹平了。\n而对开车来说，恰恰是赛道的位置和形状最重要。"
              "损失按像素计票，面积小的东西票少——这是这套方法最根本的局限（CONCEPT.md 第 9 节）。")
    else:
        print("这次训练里赛道像素的误差并不比其余像素高多少。重建的模糊主要体现在边缘和小物体上，看上面的图。")

    # 每一维的 KL：哪些维度在用
    kl_per_dim = (0.5 * (mu_all**2 + logvar_all.exp() - 1 - logvar_all)).mean(dim=0).cpu().numpy()   # [32]
    active = int((kl_per_dim > 0.1).sum())
    order = np.argsort(-kl_per_dim)
    print(f"""
每一维的 KL（在验证帧上取平均）：32 维里有 {active} 维大于 0.1 nat，算"在用"。
最大的 4 维：{", ".join(f"第 {d} 维 {kl_per_dim[d]:.2f}" for d in order[:4])}
最小的 4 维：{", ".join(f"第 {d} 维 {kl_per_dim[d]:.3f}" for d in order[-4:])}
KL 接近 0 的维度，编码分布和先验几乎一样，解码器不看它们。VAE 自己决定了用多少维。
    """)
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.bar(range(Z_DIM), kl_per_dim[order], color="tab:blue")
    ax.axhline(0.1, color="red", ls="--", lw=1, label="0.1 nat")
    ax.set_xlabel("隐变量的维度（按 KL 从大到小排列）")
    ax.set_ylabel("平均 KL（nat）")
    ax.set_title(f"每一维的 KL：{active} / {Z_DIM} 维在用")
    ax.legend()
    save_figure(fig, OUT / "02-kl-per-dim.png")
    return order, kl_per_dim


# ==================== 第五部分：后验坍塌 ====================


@torch.no_grad()
def posterior_collapse_report(good, bad, val):
    obs = to_obs(val[torch.linspace(0, len(val) - 1, 6).long().to(DEVICE)])
    fig, axes = plt.subplots(3, 6, figsize=(11, 5.6))
    for col in range(6):
        axes[0, col].imshow(obs[col].permute(1, 2, 0).cpu().numpy())
        axes[1, col].imshow(good.decode(good.encode(obs)[0])[col].permute(1, 2, 0).cpu().numpy())
        axes[2, col].imshow(bad.decode(bad.encode(obs)[0])[col].permute(1, 2, 0).cpu().numpy())
    for ax, title in zip(axes[:, 0], ["原图", "重建项对像素求和", "重建项对像素求平均"]):
        ax.set_ylabel(title, fontsize=10)
    for ax in axes.ravel():
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("后验坍塌：重建项权重太小，所有输入都重建成同一张平均画面", fontsize=12)
    save_figure(fig, OUT / "02-posterior-collapse.png")
    # 不同输入的重建之间差多少：坍塌时接近 0
    spread = lambda vae: vae.decode(vae.encode(obs)[0]).std(dim=0).mean().item()
    return spread(good), spread(bad)


def posterior_collapse(vae, train, val, epochs):
    print("=" * 60)
    print("第五部分：亲手制造一次后验坍塌")
    print("=" * 60)
    print("""
只改一处：把重建项从"对像素求和"换成"对像素求平均"（代码里很容易顺手写成 F.mse_loss）。
这相当于把重建项的权重缩小 12288 倍，KL 项的相对权重放大 12288 倍。
    """)
    subset = train[: min(len(train), 20_000)]
    bad, _, _ = train_vae(subset, val, epochs, recon_scale=1.0 / 12288, label="求平均")
    good_recon, good_kl = evaluate(vae, val)
    bad_recon, bad_kl = evaluate(bad, val)
    good_spread, bad_spread = posterior_collapse_report(vae, bad, val)
    print(f"""
                      重建损失（像素求和）   KL（nat）   不同输入的重建之间的差异（像素标准差）
    重建项对像素求和     {good_recon:10.2f}        {good_kl:7.2f}      {good_spread:.4f}
    重建项对像素求平均   {bad_recon:10.2f}        {bad_kl:7.2f}      {bad_spread:.4f}
    """)
    if bad_kl < 0.05 * good_kl:
        print(f"""KL 掉到了 {bad_kl:.2f}：编码分布几乎就是先验 N(0, I)，z 里没有关于输入的信息。
解码器拿不到信息，只能对任何输入都输出同一张"平均画面"（看 02-posterior-collapse.png 的第三行）。
这就是后验坍塌。排查方法：把重建损失和 KL 分开打印，KL 接近 0 就是它。
    """)
    else:
        print(f"这次实验里 KL 降到了 {bad_kl:.2f}，比正常训练的 {good_kl:.2f} 小，但没有完全坍塌到 0。")


# ==================== 第六部分：隐空间 ====================


@torch.no_grad()
def interpolate_latents(vae, z_a, z_b, steps):
    """在两个隐向量之间线性插值并解码。返回 [steps, 3, 64, 64]。"""
    alphas = torch.linspace(0, 1, steps, device=z_a.device).view(-1, 1)
    return vae.decode((1 - alphas) * z_a + alphas * z_b)


@torch.no_grad()
def traverse_latent_dim(vae, z, dim, values):
    """只改 z 的第 dim 维，其余维不动。返回 [len(values), 3, 64, 64]。"""
    zs = z.repeat(len(values), 1)
    zs[:, dim] = torch.as_tensor(values, device=z.device, dtype=z.dtype)
    return vae.decode(zs)


@torch.no_grad()
def latent_space_tour(vae, val, order):
    print("=" * 60)
    print("第六部分：在隐空间里走一走")
    print("=" * 60)

    obs = to_obs(val[:: max(1, len(val) // 1500)])
    mu, _ = vae.encode(obs)                                           # [N, 32]
    # 挑两张差别大的画面：隐空间里离得最远的一对（在前 300 张里找）
    dist = torch.cdist(mu[:300].cpu(), mu[:300].cpu())
    a, b = np.unravel_index(dist.argmax().item(), dist.shape)
    steps = 8
    frames = interpolate_latents(vae, mu[a:a + 1], mu[b:b + 1], steps)
    pixel_mix = torch.stack([(1 - t) * obs[a] + t * obs[b] for t in torch.linspace(0, 1, steps)])

    fig, axes = plt.subplots(2, steps, figsize=(14, 3.9))
    for col in range(steps):
        axes[0, col].imshow(frames[col].permute(1, 2, 0).cpu().numpy())
        axes[1, col].imshow(pixel_mix[col].permute(1, 2, 0).cpu().numpy())
        axes[0, col].set_title(f"α = {col / (steps - 1):.2f}", fontsize=9)
    axes[0, 0].set_ylabel("隐空间插值", fontsize=10)
    axes[1, 0].set_ylabel("像素直接混合", fontsize=10)
    for ax in axes.ravel():
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("两帧之间插值：在 z 上插值 vs 在像素上直接混合", fontsize=12)
    save_figure(fig, OUT / "02-latent-interpolation.png")

    values = np.linspace(-2.5, 2.5, 7)
    dims = order[:4]
    z0 = mu.mean(dim=0, keepdim=True)                                 # 从"平均画面"出发
    fig, axes = plt.subplots(4, 7, figsize=(11, 6.6))
    sensitivity = []
    for row, dim in enumerate(dims):
        frames = traverse_latent_dim(vae, z0, int(dim), values)
        sensitivity.append((frames[-1] - frames[0]).abs().mean().item())
        for col in range(7):
            axes[row, col].imshow(frames[col].permute(1, 2, 0).cpu().numpy())
            if row == 0:
                axes[row, col].set_title(f"z = {values[col]:+.1f}", fontsize=9)
        axes[row, 0].set_ylabel(f"第 {dim} 维", fontsize=10)
    for ax in axes.ravel():
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("逐维扰动：只改一维，其余维保持在平均值", fontsize=12)
    save_figure(fig, OUT / "02-latent-traversal.png")

    # 对照：扰动一个 KL 接近 0 的维度，画面应该几乎不变
    dead = int(order[-1])
    dead_frames = traverse_latent_dim(vae, z0, dead, values)
    dead_change = (dead_frames[-1] - dead_frames[0]).abs().mean().item()
    print(f"""
插值：从验证集里挑了隐空间距离最远的两帧，在它们的 z 之间走 {steps} 步（02-latent-interpolation.png）。
    上排在 z 上插值，每一步解码出来的都是一张"像样的"赛道画面；
    下排在像素上直接混合，中间是两张图叠在一起的重影。

逐维扰动：对 KL 最大的 4 个维度，把该维从 -2.5 扫到 +2.5（02-latent-traversal.png）。
    画面从一端到另一端的平均像素变化：{", ".join(f"第 {d} 维 {s:.3f}" for d, s in zip(dims, sensitivity))}
    对照：KL 最小的第 {dead} 维做同样的扰动，变化只有 {dead_change:.4f}——解码器不看它。

看图时注意：一个维度通常同时改变好几件事（赛道的弯曲方向、赛道在画面里的位置、草地色块的排布），
不是"一维对应一个干净的含义"。VAE 的损失里没有任何一项要求各维解耦。
    """)


# ==================== 第七部分：把数据集编码成 z ====================


@torch.no_grad()
def encode_dataset(vae, data):
    print("=" * 60)
    print("第七部分：把整个数据集编码成 z")
    print("=" * 60)

    n_episodes, horizon = data["frames"].shape[:2]
    mu = np.zeros((n_episodes, horizon, Z_DIM), dtype=np.float32)
    logvar = np.zeros((n_episodes, horizon, Z_DIM), dtype=np.float32)
    start = time.time()
    for e in range(n_episodes):
        obs = to_obs(torch.from_numpy(data["frames"][e]).to(DEVICE))  # [T, 3, 64, 64]
        m, lv = vae.encode(obs)
        mu[e], logvar[e] = m.cpu().numpy(), lv.cpu().numpy()
    np.savez(LATENTS_PATH, mu=mu, logvar=logvar, actions=data["actions"], rewards=data["rewards"],
             dones=data["dones"], speeds=data["speeds"], lengths=data["lengths"])
    raw_mb = data["frames"].nbytes / 1e6
    latent_mb = (mu.nbytes + logvar.nbytes) / 1e6
    print(f"""
{n_episodes} 个回合 × {horizon} 帧，编码用时 {time.time() - start:.1f} 秒。
存盘：{LATENTS_PATH.relative_to(OUT.parents[1])}

    mu      {list(mu.shape)}  每一帧的 μ
    logvar  {list(logvar.shape)}  每一帧的 log σ²
    （连同动作、奖励、结束标志、车速一起存，M 的训练不再需要原始画面）

画面占 {raw_mb:.0f} MB，μ 和 log σ² 加起来 {latent_mb:.0f} MB，是原来的 1/{raw_mb / latent_mb:.0f}。
存 μ 和 log σ² 而不是采样好的 z：训练 M 时每个批次重新采样 z = μ + σ·ε，
相当于数据增强，让 M 见过 z 的抖动。

z 的整体统计（全部帧的 μ）：均值 {mu.mean():+.3f}，标准差 {mu.std():.3f}；σ 的平均值 {np.exp(0.5 * logvar).mean():.3f}。
    """)


# ==================== 第八部分：练习 ====================


def exercises():
    print("=" * 60)
    print("练习与思考")
    print("=" * 60)
    print("""
练习 1：把 Z_DIM 改成 8 和 128 各训练一次，比较验证集的重建损失和"在用"的维度数。
        维度加大之后，重建一定变好吗？KL 呢？

练习 2：给 train_vae 的损失加一个系数 β：loss = 重建 + β × KL。分别取 β = 0.1、1、10，
        观察重建损失、KL、在用的维度数怎么变。

练习 3：手算。一维情况下，q = N(1, 1)、p = N(0, 1)，KL 是多少？q = N(0, 0.1²) 呢？
        哪一种"偏离先验"的代价更大？

思考题 1：解码器最后一层是 sigmoid，重建损失用平方误差。如果换成逐像素的二元交叉熵，
          对应的是对解码分布做了什么不同的假设？

思考题 2：V 训练时完全没见过动作和时间顺序。这会给后面的 M 带来什么麻烦？
          （提示：CONCEPT.md 第 1.4 节"好预测"这一条）

参考答案：
    练习 1：8 维时容量不够，重建明显变差。128 维时重建通常只略好一点，多出来的维度大多 KL 接近 0：
            KL 项对每个"在用"的维度收费，多用一维带来的重建收益抵不上这笔费用时，VAE 就不用它。
    练习 2：β 越大，KL 越小、在用的维度越少、重建越模糊；β 很大时后验坍塌（和第五部分是同一件事）。
            β 越小越接近普通自编码器：重建更清晰，但隐空间不再平滑，M 预测的 z 稍有偏差就解码出乱码。
    练习 3：KL = 0.5 × (μ² + σ² - 1 - ln σ²)。
            N(1, 1)：0.5 × (1 + 1 - 1 - 0) = 0.5 nat。
            N(0, 0.1²)：0.5 × (0 + 0.01 - 1 - ln 0.01) = 0.5 × (0.01 - 1 + 4.605) = 1.81 nat。
            把方差缩到很小（编码得很"确定"）比把均值移开一个标准差更贵。
    思考题 1：平方误差对应"每个像素是方差固定的高斯"；二元交叉熵对应"每个像素值是伯努利分布的参数"。
              后者对接近 0 或 1 的像素惩罚更重，重建往往更锐利一些，但像素值并不真的是二值的，这只是一个好用的代理。
    思考题 2：z 只为"重建单帧"而优化，不保证随时间的变化有规律。画面里对重建贡献大但和动力学无关的东西
              （草地色块的排布）占据了 z 的大部分容量，M 得花力气去预测这些。RSSM（Phase 3）把表示和动力学一起训练，
              正是为了解决这个问题。
    """)


def main():
    seed_everything(SEED)
    epochs, collapse_epochs = (1, 1) if is_quick() else (10, 3)

    introduction()
    check_architecture()
    train, val, data = load_frames()
    vae, _ = train_main_vae(train, val, epochs)
    order, _ = reconstruction_report(vae, val)
    posterior_collapse(vae, train, val, collapse_epochs)
    latent_space_tour(vae, val, order)
    encode_dataset(vae, data)
    exercises()

    print("=" * 60)
    print("本节小结")
    print("=" * 60)
    print("""
    1. V 是一个卷积 VAE：编码器把 [3, 64, 64] 压成 μ、log σ²（各 32 维），解码器从 z 还原画面。
    2. 损失两项都是"对每个样本求和、对批次求平均"。重建项写成对像素求平均，会导致后验坍塌。
    3. 重建图要读它丢了什么：大片的草地重建得好，赛道边缘这类细节被抹平。
    4. KL 项让隐空间平滑：在 z 上插值，中间每一步都是像样的画面。
    5. z 的各维不是解耦的，也不含任何时间信息（车速）。时间信息要靠 M 从 z 的序列里补。

下一节：先离开 CarRacing，在一维玩具问题上学一个新工具——混合密度网络（MDN）。
M 要预测"下一个 z 的分布"，而这个分布可能有多个峰。
    """)
    if is_quick():
        print("（当前是冒烟模式 WM_QUICK=1：只训练了 1 轮，上面的数字没有参考价值。）")


if __name__ == "__main__":
    main()
