import argparse
from functools import partial
import math
import multiprocessing as mp
from pathlib import Path
import random
from typing import Dict, List
import cv2
import numpy as np
from _common import (
    bound_xyxy,
    cxcywh_to_xyxy,
    get_image_paths,
    image_path_to_label_path,
    input_path_to_output_path,
    read_image,
    read_label,
    write_image,
    write_label,
    xyxy_to_abs_xyxy,
    xyxy_to_cxcywh,
)


def remove_overlapped_label(
    paste_label: np.ndarray,
    target_label: np.ndarray,
    occlusion_threshold: float,
):
    xyxy_paste_label = cxcywh_to_xyxy(paste_label)
    xyxy_target_label = cxcywh_to_xyxy(target_label)
    tn = xyxy_target_label.shape[0]
    px1, py1, px2, py2 = (
        np.repeat(xyxy_paste_label[:, 1, None], tn, -1),  # (pn, tn)
        np.repeat(xyxy_paste_label[:, 2, None], tn, -1),  # (pn, tn)
        np.repeat(xyxy_paste_label[:, 3, None], tn, -1),  # (pn, tn)
        np.repeat(xyxy_paste_label[:, 4, None], tn, -1),  # (pn, tn)
    )
    tx1, ty1, tx2, ty2 = (
        xyxy_target_label[None, :, 1],  # (1, tn)
        xyxy_target_label[None, :, 2],  # (1, tn)
        xyxy_target_label[None, :, 3],  # (1, tn)
        xyxy_target_label[None, :, 4],  # (1, tn)
    )
    t_area = (tx2 - tx1) * (ty2 - ty1)  # (1, tn)
    ix1 = np.maximum(px1, tx1)  # (pn, tn)
    iy1 = np.maximum(py1, py1)  # (pn, tn)
    ix2 = np.minimum(px2, tx2)  # (pn, tn)
    iy2 = np.minimum(py2, py2)  # (pn, tn)
    i_area = np.maximum(ix2 - ix1, 0) * np.maximum(iy2 - iy1, 0)  # (pn, tn)
    occlusion = i_area / t_area  # (pn, tn)
    keep_indices = np.all(occlusion <= occlusion_threshold, 1)  # (pn)
    return paste_label[keep_indices]

def random_affine(
    image: np.ndarray,
    label: np.ndarray,
    shear_x_range: List[float],
    shear_y_range: List[float],
):
    sh, sw, _ = image.shape
    # Translate origin to image center
    to_center_matrix = np.eye(3)
    to_center_matrix[0, 2] = -sw / 2  # x translation (pixels)
    to_center_matrix[1, 2] = -sh / 2  # y translation (pixels)
    # Shear
    shear_matrix = np.eye(3)
    shear_matrix[0, 1] = math.tan(random.uniform(*shear_x_range))  # x shear (rad)
    shear_matrix[1, 0] = math.tan(random.uniform(*shear_y_range))  # y shear (rad)
    # Translate origin to top-left corner
    from_center_matrix = np.eye(3)
    from_center_matrix[0, 2] = sw / 2  # x translation (pixels)
    from_center_matrix[1, 2] = sh / 2  # y translation (pixels)
    affine_matrix = from_center_matrix @ shear_matrix @ to_center_matrix
    image = cv2.warpAffine(
        image, affine_matrix[:2], dsize=(sw, sh), borderValue=(0, 0, 0, 0)
    )
    # Transform label
    n = len(label)
    if n:
        label = label.copy()
        xyxy_label = cxcywh_to_xyxy(label)
        xyxy_label[:, [1, 3]] *= sw
        xyxy_label[:, [2, 4]] *= sh
        affine_corners = np.ones((n * 4, 3))
        affine_corners[:, :2] = xyxy_label[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(
            n * 4, 2
        )  # x1y1, x2y2, x1y2, x2y1
        affine_corners = affine_corners @ affine_matrix.T  # transform
        corners = affine_corners[:, :2].reshape(n, 8)
        corners_x = corners[:, [0, 2, 4, 6]]
        corners_y = corners[:, [1, 3, 5, 7]]
        boxes = np.stack(
            (corners_x.min(1), corners_y.min(1), corners_x.max(1), corners_y.max(1)), 1
        )
        xyxy_label[:, 1:5] = boxes
        xyxy_label[:, [1, 3]] /= sw
        xyxy_label[:, [2, 4]] /= sh
        label = xyxy_to_cxcywh(xyxy_label)
    return image, label


def random_hsv_adjust(
    image: np.ndarray,
    *,
    hue_gain_range: List[float],
    sat_gain_range: List[float],
    val_gain_range: List[float],
):
    bgr_image = image[..., :3]
    hue_r = random.uniform(*hue_gain_range)
    sat_r = random.uniform(*sat_gain_range)
    val_r = random.uniform(*val_gain_range)
    hue, sat, val = cv2.split(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV))

    x = np.arange(0, 256)
    hue_lut = ((x * hue_r) % 180).astype(np.uint8)
    sat_lut = np.clip(x * sat_r, 0, 255).astype(np.uint8)
    val_lut = np.clip(x * val_r, 0, 255).astype(np.uint8)

    hsv_image = cv2.merge(
        (cv2.LUT(hue, hue_lut), cv2.LUT(sat, sat_lut), cv2.LUT(val, val_lut))
    )
    return np.concatenate(
        (cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR), image[..., -1:]), -1
    )
    return image


def random_lr_flip(image: np.ndarray, label: np.ndarray, p: float = 0.5):
    if random.random() > p:
        image = np.flip(image, axis=1)
        label = label.copy()
        label[:, 1] = 1 - label[:, 1]
        pass
    return image, label


def random_augment(
    image: np.ndarray,
    label: np.ndarray,
    *,
    hue_gain_range: List[float],
    sat_gain_range: List[float],
    val_gain_range: List[float],
    shear_x_range: List[float],
    shear_y_range: List[float],
):
    image, label = random_lr_flip(image, label)
    image, label = random_affine(
        image,
        label,
        shear_x_range=shear_x_range,
        shear_y_range=shear_y_range,
    )
    image = random_hsv_adjust(
        image,
        hue_gain_range=hue_gain_range,
        sat_gain_range=sat_gain_range,
        val_gain_range=val_gain_range,
    )
    return image, label


def compute_alpha(image: np.ndarray, alpha_threshold: int, alpha_blur_kernel_size: int):
    _, _, c = image.shape
    if c == 4:
        return image
    # if no alpha, compute alpha by thresholding followed by mean blurring
    mask: np.ndarray = image.mean(-1) > alpha_threshold  # (h, w)
    alpha = mask.astype(np.uint8) * 255
    alpha: np.ndarray = (
        cv2.blur(alpha, (alpha_blur_kernel_size, alpha_blur_kernel_size)) * mask
    )
    return np.concatenate((image, alpha[..., None]), -1)


def paste_objects(
    source_image: np.ndarray,  # (sH, sW, 3)
    source_label: np.ndarray,  # (sN, 5)
    target_image: np.ndarray,  # (tH, tW, 3)
    target_label: np.ndarray,  # (tN, 5)
    *,
    min_box_size: int,
    occlusion_threshold: float,
    global_alpha_range: List[float],
    h_range: List[float],
    offset_x_range: List[float],
    offset_y_range: List[float],
    aspect_ratio_gain_range: List[float],
):
    sh, sw, _ = source_image.shape
    th, tw, _ = target_image.shape
    paste_label = get_paste_label(
        source_label,
        sh,
        tw,
        h_range=h_range,
        offset_x_range=offset_x_range,
        offset_y_range=offset_y_range,
        aspect_ratio_gain_range=aspect_ratio_gain_range,
    )
    paste_label = remove_overlapped_label(
        paste_label, target_label, occlusion_threshold
    )
    xyxy_source_label = bound_xyxy(cxcywh_to_xyxy(source_label))
    xyxy_paste_label = bound_xyxy(cxcywh_to_xyxy(paste_label))
    abs_xyxy_source_label = xyxy_to_abs_xyxy(xyxy_source_label, ih=sh, iw=sw)
    abs_xyxy_paste_label = xyxy_to_abs_xyxy(xyxy_paste_label, ih=th, iw=tw)
    paste_image = target_image.copy()
    global_alpha = random.uniform(*global_alpha_range)
    for s_l, p_l in zip(abs_xyxy_source_label, abs_xyxy_paste_label):
        sx1, sy1, sx2, sy2 = s_l[1:]
        px1, py1, px2, py2 = p_l[1:]
        pw = px2 - px1
        ph = py2 - py1
        if pw < min_box_size or ph < min_box_size:
            continue
        s_crop = source_image[sy1:sy2, sx1:sx2]
        t_crop = target_image[py1:py2, px1:px2]
        p_crop = paste_image[py1:py2, px1:px2]
        s_crop: np.ndarray = cv2.resize(s_crop, (pw, ph), interpolation=cv2.INTER_AREA)
        # Assume target has no alpha, i.e. alpha = 255
        blend_ratio = global_alpha * s_crop[..., -1:] / 255
        s_blend = (
            ((1 - blend_ratio) * t_crop + blend_ratio * s_crop[..., :3])
            .round()
            .astype(np.uint8)
        )
        p_crop[...] = s_blend
    paste_label = np.concatenate((target_label, paste_label))
    return paste_image, paste_label


def task(
    target_image_path: Path,
    seed: int,
    *,
    output_dir: Path,
    target_dir: Path,
    source_image_paths: List[Path],
    paste_prob: float,
    max_sources_per_target: int,
    min_box_size: int,
    occlusion_threshold: float,
    alpha_threshold: int,
    alpha_blur_kernel_size: int,
    global_alpha_range: List[float],
    h_range: List[float],
    offset_x_range: List[float],
    offset_y_range: List[float],
    aspect_ratio_gain_range: List[float],
    hue_gain_range: List[float],
    sat_gain_range: List[float],
    val_gain_range: List[float],
    shear_x_range: List[float],
    shear_y_range: List[float],
):
    print(f"Processing {target_image_path}")
    random.seed(seed)
    target_image = read_image(target_image_path)
    target_label_path = image_path_to_label_path(target_image_path)
    target_label = read_label(target_label_path)
    sources_per_target = int(paste_prob > random.random()) * random.randint(
        1, max_sources_per_target
    )
    chosen_source_image_paths = random.choices(source_image_paths, k=sources_per_target)
    output_image = target_image
    output_label = target_label
    for source_image_path in chosen_source_image_paths:
        source_image = read_image(
            source_image_path, cv2.IMREAD_UNCHANGED
        )  # keep alpha channel if exists
        source_image = compute_alpha(
            source_image,
            alpha_threshold,
            alpha_blur_kernel_size,
        )
        source_label_path = image_path_to_label_path(source_image_path)
        source_label = read_label(source_label_path)
        source_image, source_label = random_augment(
            source_image,
            source_label,
            hue_gain_range=hue_gain_range,
            sat_gain_range=sat_gain_range,
            val_gain_range=val_gain_range,
            shear_x_range=shear_x_range,
            shear_y_range=shear_y_range,
        )
        output_image, output_label = paste_objects(
            source_image,
            source_label,
            output_image,
            output_label,
            min_box_size=min_box_size,
            occlusion_threshold=occlusion_threshold,
            global_alpha_range=global_alpha_range,
            h_range=h_range,
            offset_x_range=offset_x_range,
            offset_y_range=offset_y_range,
            aspect_ratio_gain_range=aspect_ratio_gain_range,
        )
    output_image_path = input_path_to_output_path(
        target_image_path, target_dir, output_dir
    )
    output_label_path = input_path_to_output_path(
        target_label_path, target_dir, output_dir
    )
    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    write_image(output_image_path, output_image)
    write_label(output_label_path, output_label)


def main():
    args = parse_args()
    source_dir: Path = args.source_dir
    target_dir: Path = args.target_dir
    output_dir: Path = args.output_dir
    target_image_paths = get_image_paths(target_dir)
    source_image_paths = list(get_image_paths(source_dir))
    seed: int = args.seed
    partial_task = partial(
        task,
        target_dir=target_dir,
        output_dir=output_dir,
        source_image_paths=source_image_paths,
        paste_prob=args.paste_prob,
        max_sources_per_target=args.max_sources_per_target,
        min_box_size=args.min_box_size,
        occlusion_threshold=args.occlusion_threshold,
        alpha_threshold=args.alpha_threshold,
        alpha_blur_kernel_size=args.alpha_blur_kernel_size,
        global_alpha_range=args.global_alpha_range,
        h_range=args.h_range,
        offset_x_range=args.offset_x_range,
        offset_y_range=args.offset_y_range,
        aspect_ratio_gain_range=args.aspect_ratio_gain_range,
        hue_gain_range=args.hue_gain_range,
        sat_gain_range=args.sat_gain_range,
        val_gain_range=args.val_gain_range,
        shear_x_range=args.shear_x_range,
        shear_y_range=args.shear_y_range,
    )
    if args.single_process:
        # single process version
        for i, target_image_path in enumerate(target_image_paths):
            partial_task(
                target_image_path,
                seed + i,
            )
    else:
        # multiple processes version
        pool = mp.Pool()
        job = pool.starmap_async(
            partial_task,
            (
                (
                    target_image_path,
                    seed + i,
                )
                for i, target_image_path in enumerate(target_image_paths)
            ),
        )
        job.get()
        pool.close()
        pool.join()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--source-dir",
        type=Path,
        help="Root dir of source images and yolo annotations.",
        required=True,
    )
    parser.add_argument(
        "-t",
        "--target-dir",
        type=Path,
        help="Root dir of target images and yolo annotations.",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        help="Root dir of output images and yolo annotations",
        required=True,
    )
    parser.add_argument(
        "--paste-prob",
        type=float,
        default=0.6,
        help="Paste probability.",
    )
    parser.add_argument(
        "--max-sources-per-target",
        type=int,
        default=3,
        help="Maximum number of sources to be sampled per target.",
    )
    parser.add_argument(
        "--min-box-size",
        type=int,
        default=6,
        help="Minimum box size to keep. Size is measured after resizing and croping.",
    )
    parser.add_argument(
        "--occlusion-threshold",
        type=float,
        default=0.1,
        help="If a pasting box overlaps an existing box for more than the threshold, skip it.",
    )
    parser.add_argument(
        "--alpha-threshold",
        type=int,
        default=2,
        help="Threshold value to create alpha channel. It is only used when source images have no alpha channel.",
    )
    parser.add_argument(
        "--alpha-blur-kernel-size",
        type=int,
        default=5,
        help="Kernel size of mean blur applying to alpha channel. It is only used when source images have no alpha channel.",
    )
    parser.add_argument(
        "--global-alpha-range",
        nargs=2,
        type=float,
        default=[0.8, 0.95],
        help="Range of random global alpha applying to source images.",
    )
    parser.add_argument(
        "--h-range",
        nargs=2,
        type=float,
        default=[0.05, 0.3],
        help="Range of random h the source images scaling to.",
    )
    parser.add_argument(
        "--offset-x-range",
        nargs=2,
        type=float,
        default=[-0.3, 0.3],
        help="Range of random offset x of source images. Relative to the paste regions.",
    )
    parser.add_argument(
        "--offset-y-range",
        nargs=2,
        type=float,
        default=[-0.3, 0.1],
        help="Range of random offset y of source images. Relative to the paste regions.",
    )
    parser.add_argument(
        "--aspect-ratio-gain-range",
        nargs=2,
        type=float,
        default=[0.9, 1.1],
        help="Range of random aspect ratio gain of source images.",
    )
    parser.add_argument(
        "--hue-gain-range",
        nargs=2,
        type=float,
        default=[0.8, 1.2],
        help="Range of random hue gain of source images.",
    )
    parser.add_argument(
        "--sat-gain-range",
        nargs=2,
        type=float,
        default=[0.7, 1.3],
        help="Range of random sat gain of source images.",
    )
    parser.add_argument(
        "--val-gain-range",
        nargs=2,
        type=float,
        default=[0.6, 1.4],
        help="Range of random val gain of source images.",
    )
    parser.add_argument(
        "--shear-x-range",
        nargs=2,
        type=float,
        default=[-0.17453, 0.17453],  # ~10 deg in rad
        help="Range of random shear x of source images.",
    )
    parser.add_argument(
        "--shear-y-range",
        nargs=2,
        type=float,
        default=[-0.17453, 0.17453],  # ~10 deg in rad
        help="Range of random shear y of source images.",
    )
    parser.add_argument("--seed", type=int, default=777, help="Random seed.")
    parser.add_argument(
        "--single-process",
        action="store_true",
        default=False,
        help="Run with single process. Mainly for debugging.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    print("Start")
    main()
    print("Done")
