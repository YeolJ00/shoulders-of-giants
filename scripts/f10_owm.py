"""F10 -- does OUR MYRIAD reproduce the paper's OWM number? (setup check, not physics)

Every MYRIAD result here (f5-f9) rests on our loading: DINOv3's gated config rebuilt by
hand, 408 keys renamed for transformers 5, no torch.compile. A strict load checks shapes,
not numerics. Their own benchmark script is the arbiter: run it UNCHANGED on their released
OWM-95 data with our model, and compare to the paper (arXiv 2604.09527, Table 1):

    MYRIAD on OWM, "minADE" best-of-5 = 0.029

THE PAPER'S "minADE" SQUARES THE DISTANCE: min_k 1/(KT) sum ||x_hat - x||_2^2 (Sec. 4.2).
It is therefore their script's top_k_MSE, in the script's [-1,1] coordinates -- NOT the
usual (non-squared) ADE. Reading it as a plain distance made a faithful setup look 1.8x
worse (0.053 vs 0.029) and cost a day of suspecting the loader. The non-squared values are
still printed, labelled as such.

Their protocol, not ours: one conditioning step (the true first-step motion of every
annotated track), the video's own fps, fp32, ensemble 5, scored over 32 frames.

Missing deps are kept out of the validated cosmos env: pandas is vendored in
models/myriad/_deps; hydra/omegaconf (imported by qual.py, unused on this path) are stubbed;
mediapy.read_video is replaced by imageio with the same fps metadata.

    C scripts/f10_owm.py [--limit N] [--bf16]
"""
import argparse
import json
import sys
import time
import types
from pathlib import Path

import numpy as np

REPO = Path("/home/nas5/jooyeolyun/repos/simulation-assestization")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "models/myriad"))
sys.path.append(str(REPO / "models/myriad/_deps"))     # appended: env packages win
DATA = REPO / "models/myriad/owm"


def _stubs():
    for nm in ("hydra", "omegaconf", "matplotlib", "matplotlib.pyplot"):
        mod = types.ModuleType(nm)      # imported by qual.py for its plotting; unused here
        mod.DictConfig = dict
        sys.modules[nm] = mod

    class _Video(np.ndarray):
        pass

    def read_video(path):
        import imageio.v2 as imageio
        rd = imageio.get_reader(str(path), "ffmpeg")
        fps = float(rd.get_meta_data()["fps"])
        fr = np.stack([f[..., :3] for f in rd]); rd.close()
        v = fr.view(_Video)
        v.metadata = types.SimpleNamespace(fps=fps)
        return v

    mp = types.ModuleType("mediapy")
    mp.read_video = read_video
    sys.modules["mediapy"] = mp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--bf16", action="store_true")
    a = ap.parse_args()
    _stubs()
    import torch
    import myriad_eval.openset_prediction as O
    import f5_myriad as M

    torch.set_grad_enabled(False)
    model = M.load_model("cuda:0").to(O.MYRIAD_DTYPE)
    ade = []
    orig = O.evaluate_for_idx_generic

    def evaluate(row, annotations, pred_sim, query_splits, cut_video_to_len):
        g, idx = orig(row, annotations, pred_sim, query_splits, cut_video_to_len)
        s0 = row["start_frame"] - 1
        p = np.split(pred_sim[:, :, s0:s0 + cut_video_to_len], query_splits, axis=-2)[-1][0]
        gt = annotations["annotated"]["tracks"][s0:s0 + cut_video_to_len]
        d = np.linalg.norm(p - gt[None], axis=-1).mean(axis=(1, 2))      # per sample, [-1,1]
        ade.append({"minADE_pm1": float(d.min()), "minADE_01": float(d.min() / 2),
                    "meanADE_pm1": float(d.mean()), "topk_MSE": g["top_k_MSE"]})
        print(f"  video {len(ade):3d}: best-of-5 mean sq. dist {g['top_k_MSE']:.5f} (paper metric)"
              f"   plain ADE {d.min()/2:.4f} [0,1]   {time.time()-t0:6.0f} s", flush=True)
        return g, idx

    O.evaluate_for_idx_generic = evaluate
    if a.limit:
        import pandas as pd
        _rj = pd.read_json
        O.pd.read_json = lambda p, *x, **k: _rj(p, *x, **k).iloc[:a.limit]
    out = REPO / "outputs/owm_eval"
    t0 = time.time()
    O.benchmark(data_root=DATA, dataset_name="owm", model=model, ensemble_size=5,
                get_annotation_fn_kwargs=dict(num_bg=10, num_random=1024, jitter_scale=3.5),
                pred_fn_kwargs=dict(traj_kind="annotated", dt_scaling_factor=30, bf16=a.bf16),
                out_path=out, cut_video_to_len=32)
    m = {k: float(np.mean([x[k] for x in ade])) for k in ade[0]}
    m["n"] = len(ade); m["bf16"] = a.bf16; m["paper_best5"] = 0.029
    print(f"\nOWM, {len(ade)} videos: best-of-5 mean SQUARED distance, [-1,1] = "
          f"{m['topk_MSE']:.4f}   (paper Table 1: 0.029)\n"
          f"  (plain, non-squared ADE for reference: {m['minADE_01']:.4f} in [0,1])")
    Path(f"docs/results/f10_owm{'_bf16' if a.bf16 else ''}{'_n%d' % a.limit if a.limit else ''}"
         f".json").write_text(json.dumps({"summary": m, "per_video": ade}, indent=1))


if __name__ == "__main__":
    main()
