"""F11 -- is MYRIAD's image encoder computed the same under transformers 5.14 and 4.56?

f10 scored 0.053 on OWM against the paper's 0.029. MYRIAD was saved with transformers 4.56
(DINOv3 config, key names); we run 5.14, which needed 408 keys renamed. A strict load checks
shapes, not what the code computes with them. Run the SAME weights through both versions
on the SAME images and compare, stage by stage.

    C scripts/f11_dino_versions.py dump --tag tf514
    PYTHONPATH=models/myriad/_tf456 C scripts/f11_dino_versions.py dump --tag tf456
    C scripts/f11_dino_versions.py compare
"""
import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/nas5/jooyeolyun/repos/simulation-assestization")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "models/myriad"))
OUT = REPO / "outputs/f11"


def images():
    """A real OWM frame, one of our renders, and MYRIAD's own demo image, square, [-1,1]."""
    import imageio.v2 as imageio
    from PIL import Image
    ims = {}
    rd = imageio.get_reader(str(sorted((REPO / "models/myriad/owm/video").glob("*.mp4"))[0]),
                            "ffmpeg")
    ims["owm0"] = next(iter(rd))[..., :3]; rd.close()
    ims["render"] = imageio.imread(REPO / "outputs/scene/f7_box/sim_s0@a/f0000.png")[..., :3]
    ims["ball_roll"] = np.asarray(Image.open(REPO / "models/myriad/qual_examples/ball_roll.jpg")
                                  .convert("RGB"))
    out = {}
    for k, im in ims.items():
        h, w = im.shape[:2]; s = min(h, w)
        sq = im[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
        out[k] = sq.astype(np.float32) / 127.5 - 1.0
    return out


def do_dump(args):
    import torch
    import transformers
    import f5_myriad as M
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    print("transformers", transformers.__version__, transformers.__file__)
    m = M.load_model("cuda:0")
    emb = m.image_embedder
    res = {}
    hooks = {}

    # capture DINOv3's own last_hidden_state and the embedder's final output
    dino = emb.model.model if hasattr(emb.model, "model") else emb.model
    print("dino module:", type(dino).__name__)
    for k, im in images().items():
        x = torch.tensor(im.transpose(2, 0, 1)[None], device="cuda:0")
        with torch.no_grad():
            d = m.embed_image(x)
            res[f"{k}_feat"] = d["feat"].float().cpu().numpy()
            # raw DINOv3 forward on the embedder's own preprocessing
            from myriad.dinov3 import better_resize
            import torchvision.transforms.functional as TF
            xi = better_resize(x, dino_size(emb))
            xi = TF.normalize((xi + 1) / 2, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            hs = dino(pixel_values=xi, output_hidden_states=True)
            res[f"{k}_last"] = hs.last_hidden_state.float().cpu().numpy()
            res[f"{k}_emb0"] = hs.hidden_states[0].float().cpu().numpy() \
                if hs.hidden_states is not None else np.zeros(1)
            res[f"{k}_h1"] = hs.hidden_states[1].float().cpu().numpy() \
                if hs.hidden_states is not None and len(hs.hidden_states) > 1 else np.zeros(1)
        print(f"  {k}: feat {res[f'{k}_feat'].shape}  last {res[f'{k}_last'].shape}")
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"{args.tag}.npz", **res)
    print("wrote", OUT / f"{args.tag}.npz")
    return 0


def dino_size(emb):
    """The square size the embedder resizes to (DinoFeatureExtractor.image_size)."""
    return emb.image_size


def do_compare(args):
    a = np.load(OUT / "tf514.npz"); b = np.load(OUT / "tf456.npz")
    print(f"{'array':<22}{'shape':>22}{'max|a-b|':>12}{'rel':>10}{'cos':>9}")
    for k in a.files:
        x, y = a[k], b[k]
        if x.shape != y.shape:
            print(f"{k:<22}{str(x.shape)+' vs '+str(y.shape):>22}  SHAPE MISMATCH"); continue
        d = np.abs(x - y).max()
        rel = np.linalg.norm(x - y) / max(np.linalg.norm(y), 1e-12)
        cs = float((x.ravel() @ y.ravel()) / max(np.linalg.norm(x) * np.linalg.norm(y), 1e-12))
        print(f"{k:<22}{str(x.shape):>22}{d:>12.2e}{rel:>10.2e}{cs:>9.5f}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("dump"); s.set_defaults(f=do_dump); s.add_argument("--tag", required=True)
    s = sub.add_parser("compare"); s.set_defaults(f=do_compare)
    a = ap.parse_args()
    sys.exit(a.f(a))
