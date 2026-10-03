# facetrust — face biometrics metrics under morphing and watermarking attacks

Codebase for the demonstrator of the tutorial *"Biometrics under attack: On provenance, trust and threat robustness"*. It implements the **biometric identification/verification** and **image quality** metrics described in the paper above and in the slides, to evaluate attacks (morphing) and defenses (watermarking — SSLWatermarking, RoSTEALS) along the pipeline:

```
capture -> processing -> enrollment -> recognition (decision)
```

**What this package does NOT do:** it does not generate watermarks, it does not decode SSLWatermarking/RoSTEALS messages, and it does not generate morphs. It assumes these images (clean, watermarked and morphed) already exist, and focuses on turning them into comparable metrics, tables and plots.

## Installation

```bash
pip install -e .                       # core: verification/identification/
                                       # morphing/image_quality/pipeline/plotting
pip install -e ".[arcface]"            # + ready-to-use ArcFaceEmbedder (insightface)
pip install -e ".[lpips]"              # + LPIPS perceptual metric
pip install -e ".[arcface,lpips]"      # combine as many extras as you like
pip install -e ".[all]"                # everything, including ssl-watermarking and rosteals
```

Without any extras the package already works: just provide your own embeddings through `CallableEmbedder`, and use PSNR/SSIM only instead of LPIPS.

`requirements.txt` is still available as an alternative for those who prefer `pip install -r requirements.txt` over installing the package itself.

### About the `ssl-watermarking` and `rosteals` extras

Neither SSLWatermarking (`facebookresearch/ssl_watermarking`) nor RoSTEALS (`TuBui/RoSteALS`) is published on PyPI — both are research repositories distributed as "clone the repo + `pip install -r requirements.txt` + download the weights manually", with no `setup.py`/`pyproject.toml` of their own. In other words, **`pip install -e ".[rosteals]"` will never install RoSTEALS itself** — these two extras only pre-install the heavy runtime dependencies that both repositories require (`torch` etc.), so the environment is ready for you to:

1. clone each repository manually (`git clone ...`) and download the weights following each one's instructions;
2. point each repo's `embed`/`extract` functions to `facetrust.watermark.SSLWatermarkAdapter` / `RoSTEALSAdapter`.

The versions pinned in these two extras are a best-effort starting point — check each repository's official `requirements.txt` before pinning versions in production, since they evolve independently of this package.

## Structure

```
facetrust/
  embeddings.py     - face embedding extraction (ArcFace or custom)
  image_quality.py  - PSNR, SSIM, LPIPS (optional)
  verification.py   - similarity, genuine/impostor, FAR/FRR/ROC/DET/EER/TAR@FAR, embedding drift
  identification.py - CMC and rank-N accuracy (1:N scenario)
  morphing.py       - MMPMR / FMMPMR / ProdAvg-MMPMR / RMMR (morphing attack potential; MMPMR, ProdAvg and RMMR checked against the reference implementation by Scherhag et al., https://github.com/dasec/mvr)
  watermark.py      - BER, detection rate, adapters for SSLWatermarking and RoSTEALS
  transforms.py     - JPEG, resize, blur, crop, rotation (robustness)
  plotting.py       - distributions, ROC/DET, alpha sweep, MMPMR curve
  pipeline.py       - orchestrates all of the above over a dataset of samples

  adapters/
    ssl_watermarking.py - actual integration with facebookresearch/ssl_watermarking
                          (calls their encode/decode, does not reimplement them)

examples/
  alpha_sweep_demo.py       - equivalent to the "starter loop" in the slides
  demo_full_pipeline.py     - full demo reading a data folder
  ssl_watermarking_demo.py  - runs the actual SSLWatermarking over data/clean/

tests/
  test_metrics_synthetic.py             - sanity tests with synthetic data
  test_ssl_watermarking_adapter_glue.py - tests for the SSL adapter glue layer (no torch)
```

## Quick start

```python
from facetrust.embeddings import CallableEmbedder
from facetrust.pipeline import Sample, EvaluationDataset, PipelineEvaluator

# Replace with ArcFaceEmbedder() or your own extractor.
embedder = CallableEmbedder(my_embedding_function)

dataset = EvaluationDataset()
dataset.add(Sample("alice_clean", clean_image, subject_id="alice", condition="clean"))
dataset.add(Sample("alice_wm_0.5", watermarked_image, subject_id="alice",
                    condition="watermarked", alpha=0.5))
dataset.add(Sample("morph_ab_stylegan2", morphed_image, subject_id="alice+bob",
                    condition="morphed", contributors=["alice", "bob"],
                    morph_algorithm="stylegan2"))

dataset.compute_embeddings(embedder)
evaluator = PipelineEvaluator(dataset)

print(evaluator.verification_report("clean"))          # EER, TAR@FAR
print(evaluator.image_quality_matrix())                 # PSNR/SSIM clean vs. other conditions
print(evaluator.embedding_drift_report("watermarked"))  # drift per alpha
print(evaluator.morph_attack_report())                  # MMPMR/FMMPMR/RMMR per target FAR
print(evaluator.morph_attack_report(group_by_algorithm=True))  # same, broken down by morph_algorithm
print(evaluator.robustness_table(embedder))             # JPEG/resize/blur/crop/rotation table
```

For the full workflow reading images from disk, see `examples/demo_full_pipeline.py` (the expected folder structure is documented in the file header), and `examples/alpha_sweep_demo.py` for the minimal watermark-strength sweep example.

## Connecting SSLWatermarking and RoSTEALS

`facetrust.watermark` does not reimplement these methods — it standardizes the results they already produce. For **SSLWatermarking**, there is a ready-made integration in `facetrust.adapters.ssl_watermarking` (it calls the actual `encode.watermark_0bit`/`decode.decode_0bit` from the cloned repository and does not reimplement the embedding math):

```python
from facetrust.adapters.ssl_watermarking import load_ssl_watermarking, make_single_image_adapter_fns
from facetrust.watermark import SSLWatermarkAdapter

ctx = load_ssl_watermarking(
    repo_path="/path/to/ssl_watermarking",   # git clone facebookresearch/ssl_watermarking
    model_path="/path/to/dino_r50_plus.pth", # pretrained backbone (see their README)
    normlayer_path="/path/to/out2048_yfcc_orig.pth",
    target_fpr=1e-6,
)
embed_fn, extract_fn = make_single_image_adapter_fns(ctx, mode="0bit")
ssl_wm = SSLWatermarkAdapter(embed_fn, extract_fn)
```

Requires `pip install -e ".[ssl-watermarking]"` (torch/torchvision/augly) + their repository's specific dependencies + the pretrained weights downloaded manually (see `facetrust/adapters/ssl_watermarking.py` for the full step-by-step guide, including a direct link to the weights). ssl_watermarking license: **CC-BY-NC** (non-commercial use). To process an entire dataset at once (more efficient than `SSLWatermarkAdapter` image by image), use `embed_0bit`/`decode_0bit` directly — see `examples/ssl_watermarking_demo.py`. **Performance warning:** each image goes through an iterative optimization (not a single forward pass) — run this offline, before the live tutorial session, not during the hands-on.

For **RoSTEALS**, there is no ready-made integration yet (the structure of `ssl_watermarking.py` can serve as a template if you want to build one):

```python
from facetrust.watermark import SSLWatermarkAdapter, RoSTEALSAdapter

rosteals = RoSTEALSAdapter(embed_fn=my_rosteals_embed, extract_fn=my_rosteals_decode)
```

From there, `bit_error_rate()` (RoSTEALS, payload-based) and `detection_metrics()` (SSLWatermarking, detection-based) standardize the results so they can be fed into `PipelineEvaluator.robustness_table(..., watermark_detector=...)`.

## Comparing multiple morphing algorithms

A morph has no single reference "subject" (it blends two or more contributors), so `Sample.subject_id` for `condition="morphed"` is just a convenience label (suggestion: `"+".join(contributors)`, e.g. `"alice+bob"`) — it is **not** used to look up a single "clean" reference, because no such reference exists.

What identifies WHICH algorithm generated the morph is `Sample.morph_algorithm` (e.g. `"stylegan2"`, `"stylegan3"`, `"diffae"`, `"opencv"`). This is what allows evaluating several algorithms side by side on the SAME pair of contributors:

```python
dataset.add(Sample("m1", img_stylegan2, subject_id="alice+bob",
                    condition="morphed", contributors=["alice", "bob"],
                    morph_algorithm="stylegan2"))
dataset.add(Sample("m2", img_stylegan3, subject_id="alice+bob",
                    condition="morphed", contributors=["alice", "bob"],
                    morph_algorithm="stylegan3"))
dataset.add(Sample("m3", img_diffae, subject_id="alice+bob",
                    condition="morphed", contributors=["alice", "bob"],
                    morph_algorithm="diffae"))

# Image quality per algorithm (1 row per morph x contributor):
iq = evaluator.image_quality_matrix()
iq[iq["condition"] == "morphed"].groupby("morph_algorithm")[["psnr", "ssim"]].mean()

# MMPMR/FMMPMR/RMMR per algorithm, over the same FAR sweep:
evaluator.morph_attack_report(group_by_algorithm=True)
```

When loading from disk with `examples/demo_full_pipeline.py`, the algorithm is the subfolder in `data/morphed/<algorithm>/<contributor_a>+<contributor_b>.png` (see the script header for the full convention, including the optional suffix for multiple morphs of the same pair by the same algorithm).

## Paper/slides → code mapping

| Concept in the paper/slides                           | Where it is in the code                                |
|-------------------------------------------------------|--------------------------------------------------------|
| Cosine similarity, genuine/impostor                   | `verification.cosine_similarity`, `build_score_pairs`  |
| FAR, FRR, threshold                                   | `verification.far_frr`, `threshold_at_far`             |
| ROC, DET, EER, TAR@FAR                                | `verification.roc_curve_data`, `det_curve_data`, `eer`, `tar_at_far` |
| Embedding drift (`1 - cos(f(I), f(I_w))`)             | `verification.embedding_drift`                         |
| Four core comparison scenarios (Clean/WM × Clean/WM)  | `verification.cross_condition_scores`, `pipeline.PipelineEvaluator.four_scenario_report` |
| CMC / rank-N (identification scenario)                | `identification.cmc_curve`, `rank_n_accuracy`          |
| MMPMR / FMMPMR / ProdAvg-MMPMR / RMMR                 | `morphing.mmpmr`, `fmmpmr`, `prodavg_mmpmr`, `rmmr`, `mmpmr_at_far`, `map_curve` |
| PSNR, SSIM, (LPIPS)                                   | `image_quality.psnr`, `ssim`, `lpips_distance`         |
| Detection rate, BER, payload                          | `watermark.detection_metrics`, `bit_error_rate`        |
| Robustness to JPEG/resize/blur/crop/rotation          | `transforms.DEFAULT_TRANSFORMS`, `pipeline.PipelineEvaluator.robustness_table` |
| Joint evaluation matrix                               | `pipeline.PipelineEvaluator.joint_evaluation_summary`  |

## Tests

```bash
PYTHONPATH=. python -m pytest tests/ -v
# or, without pytest installed, run the test_* functions directly:
PYTHONPATH=. python -c "import tests.test_metrics_synthetic as t; [getattr(t, n)() for n in dir(t) if n.startswith('test_')]; print('OK')"
```

The tests use only synthetic data (no real face or watermarking models), serving as a check that the metric logic is correct before running on your real data.
