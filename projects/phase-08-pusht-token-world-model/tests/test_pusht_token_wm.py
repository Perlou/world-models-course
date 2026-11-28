import numpy as np
import torch

from pusht_token_wm import PushTTokenTransformerWM, PushTTokenizer, inspect_pusht_metadata, plan_pusht_token_cem


def test_pusht_metadata():
    meta = inspect_pusht_metadata()
    assert meta["num_episodes"] == 206
    assert meta["num_frames"] == 25650
    assert meta["image_shape"] == [96, 96, 3]


def test_tokenizer_and_wm_shapes():
    tok = PushTTokenizer(levels=(5, 5, 5))
    x = torch.rand(2, 3, 64, 64)
    rec, idx = tok(x)
    assert rec.shape == (2, 3, 64, 64) and idx.shape == (2, 16)

    wm = PushTTokenTransformerWM(vocab_size=125, tokens_per_frame=16, act_dim=2, d_model=64, n_layers=2, n_heads=4)
    toks_seq = torch.randint(0, 125, (2, 5, 16))
    acts = torch.randn(2, 5, 2)
    rews = torch.rand(2, 5)
    loss, ce, rw = wm.loss(toks_seq, acts, rews)
    assert np.isfinite(loss.item()) and ce >= 0 and rw >= 0

    pt, pr = wm.imagine_rollout(toks_seq[:, 0], acts[:, :3])
    assert pt.shape == (2, 3, 16) and pr.shape == (2, 3)


def test_plan_pusht_token_cem_smoke():
    tok = PushTTokenizer(levels=(5, 5, 5)).eval()
    wm = PushTTokenTransformerWM(vocab_size=125, tokens_per_frame=16, act_dim=2, d_model=32, n_layers=1, n_heads=2).eval()
    dummy_img = np.zeros((96, 96, 3), dtype=np.uint8)
    act, ms = plan_pusht_token_cem(tok, wm, dummy_img, horizon=2, n_candidates=8, n_iters=2)
    assert act.shape == (2,) and np.all((act >= 0.0) & (act <= 512.0)) and ms >= 0.0
