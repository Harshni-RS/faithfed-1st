"""
Compare the OLD insertion/deletion setup with the CORRECTED one on a real fundus image.
Run from the project root, for example:

    python scripts/check_faithfulness.py --image data/aptos/train_images/002c21358ce6.png

Reads artifacts/global_model.pth by default. Prints, for the same image and heatmap:
  - old setup  : weak 11x11 blur + softmax probability, no normalisation
  - new setup  : strong Gaussian blur + log-odds, normalised (0 = baseline, 1 = original)
and a random-pixel-order control. A faithful Grad-CAM should score clearly above the
random control (faithfulness_vs_random > 0).
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference import RetinalDiagnosticEngine
from src.faithfulness import compute_insertion_deletion_auc


def show(tag, r):
    print(f"\n[{tag}]  metric={r['metric']}  substrate={r['substrate']}  degenerate={r['degenerate']}")
    print(f"  insertion AUC {r['insertion_auc']:.4f} | deletion AUC {r['deletion_auc']:.4f} | "
          f"faithfulness {r['faithfulness_score']:.4f}")
    if "random_faithfulness" in r:
        print(f"  random-order control {r['random_faithfulness']:.4f} | "
              f"faithfulness - random = {r['faithfulness_vs_random']:+.4f}")
    print(f"  probability: original {r['original_prob']:.4f} -> baseline image {r['baseline_prob']:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--model", default="artifacts/global_model.pth")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--n_random", type=int, default=5)
    a = ap.parse_args()

    eng = RetinalDiagnosticEngine(model_path=a.model, img_size=128)
    rep = eng.diagnose(a.image, compute_faithfulness=False)
    print(f"Prediction: {rep['grade_name']}  (confidence {rep['confidence']:.4f})")

    # Rebuild the exact tensor the engine used
    raw = eng.load_image(a.image)
    from src.data_preprocessing import ben_graham_normalize
    norm = ben_graham_normalize(raw, img_size=eng.img_size, sigma=eng.ben_graham_sigma)
    x = eng.transform(norm).unsqueeze(0).to(eng.device)
    heat = rep["heatmap"]
    tgt = rep["predicted_grade"]

    old = compute_insertion_deletion_auc(eng.model, x, heat, tgt, num_steps=a.steps, device=eng.device,
                                         substrate="legacy", metric="prob", normalize=False)
    new = compute_insertion_deletion_auc(eng.model, x, heat, tgt, num_steps=a.steps, device=eng.device,
                                         substrate="blur", metric="logit", normalize=True,
                                         n_random=a.n_random)
    show("OLD", old)
    show("NEW", new)
    if new["degenerate"]:
        print("\nNOTE: the model is not more confident on the real image than on the blurred baseline,")
        print("so insertion/deletion cannot be interpreted for this image (the model/training is the issue).")


if __name__ == "__main__":
    main()
