import argparse
import csv
import json
import os
import sys
import types
from pathlib import Path
from statistics import mean, stdev

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
FULL_REFERENCE_METRICS = {"psnr", "ssim", "lpips", "dists"}
NO_REFERENCE_METRICS = {"niqe", "musiq", "maniqa", "clipiqa"}
DEFAULT_METRICS = ["psnr", "ssim", "niqe", "lpips", "dists", "musiq", "maniqa", "clipiqa"]


def install_torchvision_functional_tensor_shim():
    """
    BasicSR imports torchvision.transforms.functional_tensor, which was removed in newer torchvision versions.
    The only symbol BasicSR needs there is rgb_to_grayscale, now available from torchvision.transforms.functional.
    """
    module_name = "torchvision.transforms.functional_tensor"
    if module_name in sys.modules:
        return

    try:
        from torchvision.transforms.functional import rgb_to_grayscale
    except ImportError:
        return

    functional_tensor = types.ModuleType(module_name)
    functional_tensor.rgb_to_grayscale = rgb_to_grayscale
    sys.modules[module_name] = functional_tensor


def import_basicsr_metrics():
    try:
        install_torchvision_functional_tensor_shim()
        from basicsr.metrics.niqe import calculate_niqe
        from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim
    except ImportError as exc:
        raise SystemExit(
            "BasicSR is required for PSNR/SSIM/NIQE. Install dependencies with:\n"
            "  pip install -r requirements.txt\n"
            f"Original import error: {exc}"
        ) from exc
    return calculate_psnr, calculate_ssim, calculate_niqe


def import_pyiqa():
    try:
        import torch
        import pyiqa
    except ImportError as exc:
        raise SystemExit(
            "PyIQA and PyTorch are required for LPIPS/DISTS/MUSIQ/MANIQA/CLIPIQA. Install dependencies with:\n"
            "  pip install -r requirements.txt\n"
            f"Original import error: {exc}"
        ) from exc
    return torch, pyiqa


def collect_images(path):
    path = Path(path)
    if path.is_file():
        if path.suffix.lower() not in IMG_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {path}")
        return [path]

    image_paths = []
    for root, _, files in os.walk(path):
        for filename in files:
            image_path = Path(root) / filename
            if image_path.suffix.lower() in IMG_EXTENSIONS:
                image_paths.append(image_path)
    return sorted(image_paths)


def normalize_key(path):
    return path.as_posix().lower()


def build_gt_index(gt_dir):
    gt_dir = Path(gt_dir)
    gt_paths = collect_images(gt_dir)
    by_relative_stem = {}
    by_name_stem = {}
    duplicate_name_stems = set()

    for gt_path in gt_paths:
        relative_stem = gt_path.relative_to(gt_dir).with_suffix("")
        by_relative_stem[normalize_key(relative_stem)] = gt_path

        name_stem = gt_path.stem.lower()
        if name_stem in by_name_stem:
            duplicate_name_stems.add(name_stem)
        else:
            by_name_stem[name_stem] = gt_path

    for name_stem in duplicate_name_stems:
        by_name_stem.pop(name_stem, None)

    return gt_paths, by_relative_stem, by_name_stem, duplicate_name_stems


def match_gt_path(restored_path, restored_root, gt_dir, gt_by_relative_stem, gt_by_name_stem):
    restored_path = Path(restored_path)
    restored_root = Path(restored_root)
    relative_stem = restored_path.relative_to(restored_root).with_suffix("")

    gt_path = gt_by_relative_stem.get(normalize_key(relative_stem))
    if gt_path is not None:
        return gt_path

    return gt_by_name_stem.get(restored_path.stem.lower())


def count_name_matches(restored_paths, restored_root, gt_dir, gt_by_relative_stem, gt_by_name_stem):
    count = 0
    for restored_path in restored_paths:
        if match_gt_path(restored_path, restored_root, gt_dir, gt_by_relative_stem, gt_by_name_stem) is not None:
            count += 1
    return count


def select_gt_match_mode(args, restored_paths, gt_paths, restored_root, gt_by_relative_stem, gt_by_name_stem):
    if args.gt_match == "order":
        if len(restored_paths) > len(gt_paths):
            raise SystemExit(
                f"Order matching requires at least as many GT images as restored images: "
                f"{len(restored_paths)} restored vs {len(gt_paths)} GT."
            )
        return "order"

    name_match_count = count_name_matches(
        restored_paths,
        restored_root,
        args.gt_dir,
        gt_by_relative_stem,
        gt_by_name_stem,
    )

    if args.gt_match == "name":
        return "name"

    if name_match_count == len(restored_paths):
        return "name"

    if len(restored_paths) <= len(gt_paths):
        print(
            "GT names do not fully match restored names "
            f"({name_match_count}/{len(restored_paths)} by name). "
            "Using sorted-order GT matching."
        )
        return "order"

    return "name"


def read_cv2_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.integer):
            max_value = np.iinfo(image.dtype).max
            image = image.astype(np.float32) / max_value * 255.0
        else:
            image = np.clip(image, 0, 1) * 255.0
        image = image.round().astype(np.uint8)
    return image


def read_pil_rgb(path):
    return Image.open(path).convert("RGB")


def pil_to_tensor(image, torch, device):
    array = np.asarray(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)


def resize_like_restored(gt_cv2, gt_pil, restored_cv2, restored_pil, resize_mode):
    if gt_cv2.shape[:2] == restored_cv2.shape[:2]:
        return gt_cv2, gt_pil, restored_cv2, restored_pil

    if resize_mode == "none":
        raise ValueError(
            "Image size mismatch: "
            f"restored={restored_cv2.shape[1]}x{restored_cv2.shape[0]}, "
            f"gt={gt_cv2.shape[1]}x{gt_cv2.shape[0]}. "
            "Use --resize restored_to_gt or --resize gt_to_restored if this is expected."
        )

    if resize_mode == "restored_to_gt":
        width, height = gt_pil.size
        restored_pil = restored_pil.resize((width, height), Image.BICUBIC)
        restored_cv2 = cv2.resize(restored_cv2, (width, height), interpolation=cv2.INTER_CUBIC)
    elif resize_mode == "gt_to_restored":
        width, height = restored_pil.size
        gt_pil = gt_pil.resize((width, height), Image.BICUBIC)
        gt_cv2 = cv2.resize(gt_cv2, (width, height), interpolation=cv2.INTER_CUBIC)
    else:
        raise ValueError(f"Unknown resize mode: {resize_mode}")

    return gt_cv2, gt_pil, restored_cv2, restored_pil


def create_pyiqa_metrics(metric_names, device):
    if not metric_names:
        return None, {}

    torch, pyiqa = import_pyiqa()
    models = {}
    for metric_name in metric_names:
        models[metric_name] = pyiqa.create_metric(metric_name, device=device)
    return torch, models


def tensor_to_float(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "mean"):
        value = value.mean()
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def evaluate_image(
    restored_path,
    gt_path,
    args,
    basicsr_metrics,
    pyiqa_torch,
    pyiqa_models,
):
    calculate_psnr, calculate_ssim, calculate_niqe = basicsr_metrics
    row = {"image": str(restored_path)}

    restored_cv2 = read_cv2_image(restored_path)
    restored_pil = read_pil_rgb(restored_path)

    gt_cv2 = None
    gt_pil = None
    if gt_path is not None:
        gt_cv2 = read_cv2_image(gt_path)
        gt_pil = read_pil_rgb(gt_path)
        gt_cv2, gt_pil, restored_cv2, restored_pil = resize_like_restored(
            gt_cv2,
            gt_pil,
            restored_cv2,
            restored_pil,
            args.resize,
        )
        row["gt"] = str(gt_path)

    if "psnr" in args.metrics and gt_cv2 is not None:
        row["psnr"] = calculate_psnr(
            restored_cv2,
            gt_cv2,
            crop_border=args.crop_border,
            input_order="HWC",
            test_y_channel=args.test_y_channel,
        )

    if "ssim" in args.metrics and gt_cv2 is not None:
        row["ssim"] = calculate_ssim(
            restored_cv2,
            gt_cv2,
            crop_border=args.crop_border,
            input_order="HWC",
            test_y_channel=args.test_y_channel,
        )

    if "niqe" in args.metrics:
        row["niqe"] = calculate_niqe(
            restored_cv2,
            crop_border=args.crop_border,
            input_order="HWC",
            convert_to=args.niqe_convert_to,
        )

    if pyiqa_models:
        restored_tensor = pil_to_tensor(restored_pil, pyiqa_torch, args.device)
        gt_tensor = pil_to_tensor(gt_pil, pyiqa_torch, args.device) if gt_pil is not None else None

        with pyiqa_torch.no_grad():
            for metric_name, metric_model in pyiqa_models.items():
                if metric_name in FULL_REFERENCE_METRICS:
                    if gt_tensor is None:
                        continue
                    score = metric_model(restored_tensor, gt_tensor)
                else:
                    score = metric_model(restored_tensor)
                row[metric_name] = tensor_to_float(score)

    return row


def summarize(rows, metric_names):
    summary = {}
    for metric_name in metric_names:
        values = [row[metric_name] for row in rows if metric_name in row and row[metric_name] is not None]
        if not values:
            continue
        summary[metric_name] = {
            "mean": float(mean(values)),
            "std": float(stdev(values)) if len(values) > 1 else 0.0,
            "count": len(values),
        }
    return summary


def write_csv(path, rows, metric_names):
    fieldnames = ["image", "gt"] + metric_names
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_summary(summary):
    if not summary:
        print("No metrics were computed.")
        return

    print("\nMetric summary")
    print("-" * 56)
    print(f"{'metric':<12} {'mean':>14} {'std':>14} {'count':>8}")
    for metric_name, values in summary.items():
        print(
            f"{metric_name:<12} "
            f"{values['mean']:>14.6f} "
            f"{values['std']:>14.6f} "
            f"{values['count']:>8}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate deblurring outputs with BasicSR and PyIQA metrics.")
    parser.add_argument("--restored_dir", type=str, required=True, help="Directory or image file with restored outputs.")
    parser.add_argument("--gt_dir", type=str, default="", help="Directory or image file with sharp ground truth images.")
    parser.add_argument("--output_dir", type=str, default="", help="Directory for metrics.csv and metrics_summary.json.")
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS, choices=DEFAULT_METRICS)
    parser.add_argument("--crop_border", type=int, default=0)
    parser.add_argument("--test_y_channel", action="store_true", help="Evaluate PSNR/SSIM on Y channel.")
    parser.add_argument("--niqe_convert_to", type=str, default="y", choices=["y", "gray"])
    parser.add_argument("--resize", type=str, default="none", choices=["none", "restored_to_gt", "gt_to_restored"])
    parser.add_argument("--device", type=str, default="auto", help="auto, cuda, cuda:0, or cpu for PyIQA metrics.")
    parser.add_argument("--gt_match", type=str, default="auto", choices=["auto", "name", "order"])
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--skip_missing_gt", action="store_true")
    parser.add_argument("--disable_cudnn", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.metrics = [metric.lower() for metric in args.metrics]

    if args.disable_cudnn:
        try:
            import torch

            torch.backends.cudnn.enabled = False
            print("cuDNN is disabled for metric evaluation.")
        except ImportError:
            pass

    has_gt = bool(args.gt_dir)
    requested_fr_metrics = [metric for metric in args.metrics if metric in FULL_REFERENCE_METRICS]
    requested_pyiqa_metrics = [
        metric
        for metric in args.metrics
        if metric in {"lpips", "dists", "musiq", "maniqa", "clipiqa"}
    ]

    if requested_fr_metrics and not has_gt:
        print(
            "Ground truth is not set. Skipping full-reference metrics: "
            + ", ".join(requested_fr_metrics)
        )

    basicsr_metrics = import_basicsr_metrics()

    pyiqa_torch = None
    pyiqa_models = {}
    if requested_pyiqa_metrics:
        if not has_gt:
            requested_pyiqa_metrics = [
                metric for metric in requested_pyiqa_metrics if metric not in FULL_REFERENCE_METRICS
            ]
        if requested_pyiqa_metrics:
            torch, _ = import_pyiqa()
            if args.device == "auto":
                args.device = "cuda" if torch.cuda.is_available() else "cpu"
            pyiqa_torch, pyiqa_models = create_pyiqa_metrics(requested_pyiqa_metrics, args.device)
            print(f"PyIQA device: {args.device}")
        elif args.device == "auto":
            args.device = "cpu"
    elif args.device == "auto":
        args.device = "cpu"

    restored_root = Path(args.restored_dir)
    restored_match_root = restored_root if restored_root.is_dir() else restored_root.parent
    restored_paths = collect_images(restored_root)
    if args.max_images > 0:
        restored_paths = restored_paths[: args.max_images]
    if not restored_paths:
        raise SystemExit(f"No images found under {args.restored_dir}")

    gt_by_relative_stem = {}
    gt_by_name_stem = {}
    gt_by_order = {}
    gt_match_mode = args.gt_match
    gt_paths = []
    duplicate_name_stems = set()
    if has_gt:
        gt_path = Path(args.gt_dir)
        if gt_path.is_file():
            gt_by_name_stem[gt_path.stem.lower()] = gt_path
        else:
            gt_paths, gt_by_relative_stem, gt_by_name_stem, duplicate_name_stems = build_gt_index(gt_path)
            if duplicate_name_stems:
                print(
                    "Duplicate GT filename stems are ignored for fallback name matching: "
                    + ", ".join(sorted(duplicate_name_stems)[:10])
                )
            gt_match_mode = select_gt_match_mode(
                args,
                restored_paths,
                gt_paths,
                restored_match_root,
                gt_by_relative_stem,
                gt_by_name_stem,
            )
            if gt_match_mode == "order":
                gt_by_order = dict(zip(restored_paths, gt_paths))
                print(f"GT match mode: order ({len(gt_by_order)} pairs)")
            else:
                print("GT match mode: name")

    rows = []
    missing_gt = []
    for restored_path in tqdm(restored_paths, desc="Evaluating"):
        gt_path = None
        if has_gt:
            if Path(args.gt_dir).is_file():
                gt_path = Path(args.gt_dir)
            else:
                if gt_match_mode == "order":
                    gt_path = gt_by_order.get(restored_path)
                else:
                    gt_path = match_gt_path(
                        restored_path,
                        restored_match_root,
                        args.gt_dir,
                        gt_by_relative_stem,
                        gt_by_name_stem,
                    )
            if gt_path is None:
                missing_gt.append(str(restored_path))
                if args.skip_missing_gt:
                    continue
                raise SystemExit(f"Missing ground truth for {restored_path}")

        row = evaluate_image(
            restored_path,
            gt_path,
            args,
            basicsr_metrics,
            pyiqa_torch,
            pyiqa_models,
        )
        rows.append(row)

    if missing_gt:
        print(f"Skipped {len(missing_gt)} images without ground truth.")

    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif restored_root.is_file():
        output_dir = restored_root.parent / "metrics"
    else:
        output_dir = restored_root / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "metrics.csv"
    summary_path = output_dir / "metrics_summary.json"
    summary = summarize(rows, args.metrics)

    write_csv(csv_path, rows, args.metrics)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print_summary(summary)
    print(f"\nPer-image metrics: {csv_path}")
    print(f"Summary metrics:   {summary_path}")


if __name__ == "__main__":
    main()
