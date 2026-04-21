"""
yolo_temporal_study.py — Three temporal fusion experiments using only YOLO.

No trained models. No split-dependent evaluation. Uses nuScenes GT boxes
(available at every keyframe) and raw camera images only.

Experiments
-----------
1  K-FRAME RECALL
   How does detection recall improve as you give YOLO more frames of context?
   Sweep K = 1..7 keyframes (causal window t-K+1..t). An object is "detected"
   if it appears in at least one frame (union) or the majority of frames (vote).
   Shows the value of temporal context for detection coverage.

2  KEYFRAME INTERPOLATION
   nuScenes cameras run at 12Hz but keyframes are at 2Hz (5 intermediate frames
   between each pair). Given YOLO detections at keyframe t and t+1, interpolate
   box positions for the 5 intermediate unlabelled frames. Compare interpolated
   vs carry-forward vs actual YOLO on those real (but unannotated) images.
   Measures how well linear track interpolation approximates reality.

3  FUTURE-ASSISTED DETECTION RECOVERY
   For each triplet (t-1, t, t+1): find GT objects at t that YOLO misses.
   Among those, count how many are detected at both t-1 and t+1. Reconstruct
   their position at t by linear interpolation between the flanking detections.
   Measures how often temporal consensus recovers missed detections and how
   accurately the interpolated position matches ground truth.

Usage
-----
    python scripts/yolo_temporal_study.py --exp all
    python scripts/yolo_temporal_study.py --exp 1
    python scripts/yolo_temporal_study.py --exp 2,3

Output
------
    temporal_analysis/temporal_study/
        exp1_kframe_recall.json
        exp2_interpolation.json
        exp3_recovery.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from nuscenes.nuscenes import NuScenes

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from data.nuscenes_loader import get_sensor_transforms, load_gt_boxes
from data.temporal_loader import get_triplet_tokens, load_raw_image

OUT_DIR  = Path('temporal_analysis/temporal_study')
CAM      = 'CAM_FRONT'
CONF     = 0.25
IOU_THRESH = 0.5   # for matching predictions to GT

# nuScenes dynamic categories that YOLO can plausibly detect
DYNAMIC_NAMES = {
    'car', 'truck', 'bus', 'motorcycle', 'bicycle', 'pedestrian',
    'construction_vehicle', 'trailer',
}


# ---------------------------------------------------------------------------
# Shared geometry helpers
# ---------------------------------------------------------------------------

def _box_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU between two (x1,y1,x2,y2) boxes."""
    xi1, yi1 = max(a[0], b[0]), max(a[1], b[1])
    xi2, yi2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
    if inter == 0:
        return 0.0
    ua = (a[2]-a[0])*(a[3]-a[1])
    ub = (b[2]-b[0])*(b[3]-b[1])
    return inter / (ua + ub - inter)


def _project_ego_to_image(
    xyz_ego: np.ndarray,
    intrinsic: np.ndarray,
    cam2ego: np.ndarray,
) -> Optional[Tuple[float, float]]:
    """Project a 3D ego-frame point to (u, v) image coordinates."""
    cam2ego_inv = np.linalg.inv(cam2ego)
    xyz_cam = cam2ego_inv[:3, :3] @ xyz_ego + cam2ego_inv[:3, 3]
    if xyz_cam[2] < 0.5:
        return None
    uv_h = intrinsic @ xyz_cam
    return float(uv_h[0] / uv_h[2]), float(uv_h[1] / uv_h[2])


def _load_yolo_dets(yolo, img: np.ndarray) -> List[Dict]:
    """Run YOLO; return list of {'xyxy': np.ndarray, 'conf', 'cls'}."""
    dets = []
    for r in yolo.predict(img, conf=CONF, verbose=False):
        if r.boxes is None:
            continue
        for xyxy, conf, cls in zip(
            r.boxes.xyxy.cpu().numpy(),
            r.boxes.conf.cpu().numpy(),
            r.boxes.cls.cpu().numpy().astype(int),
        ):
            dets.append({'xyxy': xyxy.astype(float), 'conf': float(conf), 'cls': int(cls)})
    return dets


def _recall_at_t(
    gt_boxes: List[Dict],
    yolo_dets: List[Dict],
    intrinsic: np.ndarray,
    cam2ego: np.ndarray,
    img_w: int,
    img_h: int,
) -> Tuple[int, int]:
    """
    Return (n_detected, n_total) GT dynamic objects visible from this camera.

    A GT object is considered detected if any YOLO box contains its projected
    centre point. Class-agnostic — we only care whether something is detected.
    """
    total, detected = 0, 0
    for gt in gt_boxes:
        if gt['name'] not in DYNAMIC_NAMES:
            continue
        xyz = np.array([gt['x'], gt['y'], gt['z']])
        uv = _project_ego_to_image(xyz, intrinsic, cam2ego)
        if uv is None:
            continue
        u, v = uv
        if not (0 <= u < img_w and 0 <= v < img_h):
            continue   # not visible from this camera
        total += 1
        for det in yolo_dets:
            x1, y1, x2, y2 = det['xyxy']
            if x1 <= u <= x2 and y1 <= v <= y2:
                detected += 1
                break
    return detected, total


def _interpolate_box(box_a: np.ndarray, box_b: np.ndarray, alpha: float) -> np.ndarray:
    """Linear interpolation: alpha=0 → box_a, alpha=1 → box_b."""
    return (1 - alpha) * box_a + alpha * box_b


def _match_dets(
    preds: List[Dict], refs: List[Dict], iou_thresh: float = IOU_THRESH
) -> Tuple[int, int, int]:
    """Return (TP, FP, FN) between two detection lists using greedy IoU matching."""
    if not refs:
        return 0, len(preds), 0
    if not preds:
        return 0, 0, len(refs)

    matched_pred, matched_ref = set(), set()
    scores = np.zeros((len(preds), len(refs)))
    for i, p in enumerate(preds):
        for j, r in enumerate(refs):
            scores[i, j] = _box_iou(p['xyxy'], r['xyxy'])

    order = np.dstack(np.unravel_index(np.argsort(-scores, axis=None), scores.shape))[0]
    for i, j in order:
        if scores[i, j] < iou_thresh:
            break
        if i in matched_pred or j in matched_ref:
            continue
        matched_pred.add(i)
        matched_ref.add(j)

    tp = len(matched_pred)
    return tp, len(preds) - tp, len(refs) - tp


# ---------------------------------------------------------------------------
# Helper: get keyframe chain (causal window of length K ending at token)
# ---------------------------------------------------------------------------

def _get_causal_keyframes(nusc: NuScenes, sample_token: str, K: int) -> List[str]:
    """Return up to K keyframe tokens ending at sample_token (oldest first)."""
    chain = [sample_token]
    cur = nusc.get('sample', sample_token)
    while len(chain) < K and cur.get('prev'):
        prev = nusc.get('sample', cur['prev'])
        if prev['scene_token'] != nusc.get('sample', sample_token)['scene_token']:
            break
        chain.insert(0, prev['token'])
        cur = prev
    return chain


# ---------------------------------------------------------------------------
# Helper: get intermediate sample_data frames between two keyframes
# ---------------------------------------------------------------------------

def _get_intermediate_frames(nusc: NuScenes, sample_token: str, cam: str) -> List[Dict]:
    """
    Return the sample_data records between keyframe sample_token and its next
    keyframe for the given camera. Excludes both endpoint keyframes.
    """
    sample = nusc.get('sample', sample_token)
    sd_token = sample['data'][cam]
    sd = nusc.get('sample_data', sd_token)

    intermediates = []
    cur_token = sd.get('next', '')
    while cur_token:
        cur_sd = nusc.get('sample_data', cur_token)
        if cur_sd['is_key_frame']:
            break
        intermediates.append(cur_sd)
        cur_token = cur_sd.get('next', '')
    return intermediates


def _load_raw_from_sd(nusc: NuScenes, sd: Dict) -> np.ndarray:
    img_path = Path(nusc.dataroot) / sd['filename']
    return np.array(Image.open(img_path).convert('RGB'))


# ---------------------------------------------------------------------------
# Experiment 1 — K-frame recall
# ---------------------------------------------------------------------------

def run_exp1(nusc: NuScenes, yolo, max_samples: Optional[int]) -> Dict:
    """
    Sweep K = 1..7 keyframes. Measure recall of GT dynamic objects at t
    using union of detections across the K-frame causal window.
    Also measure majority-vote recall.
    """
    print("\n=== Experiment 1: K-frame recall ===")

    K_values = [1, 2, 3, 5, 7]
    results_per_K = {K: {'detected_union': 0, 'detected_vote': 0, 'total': 0}
                     for K in K_values}

    all_tokens = [s['token'] for s in nusc.sample]
    if max_samples:
        all_tokens = all_tokens[:max_samples]

    sample = nusc.get('sample', all_tokens[0])
    sd = nusc.get('sample_data', sample['data'][CAM])
    intrinsic, cam2ego = get_sensor_transforms(nusc, sd['token'])
    dummy_img = _load_raw_from_sd(nusc, sd)
    IMG_H, IMG_W = dummy_img.shape[:2]

    for idx, token in enumerate(all_tokens):
        gt_boxes = load_gt_boxes(nusc, token)
        if not any(g['name'] in DYNAMIC_NAMES for g in gt_boxes):
            continue

        sd_token = nusc.get('sample', token)['data'][CAM]
        intrinsic, cam2ego = get_sensor_transforms(nusc, sd_token)

        # Gather per-frame detections for the max window
        chain = _get_causal_keyframes(nusc, token, max(K_values))
        frame_dets = []
        for frame_tok in chain:
            img = load_raw_image(nusc, frame_tok, CAM)
            frame_dets.append(_load_yolo_dets(yolo, img))

        _, n_total = _recall_at_t(gt_boxes, [], intrinsic, cam2ego, IMG_W, IMG_H)
        if n_total == 0:
            continue

        for K in K_values:
            window_dets = frame_dets[-K:]   # most recent K frames
            majority_thresh = (K + 1) // 2

            # Union: detected in any frame
            all_union = [d for dets in window_dets for d in dets]
            det_union, _ = _recall_at_t(gt_boxes, all_union, intrinsic, cam2ego, IMG_W, IMG_H)

            # Majority vote: detected in >= majority_thresh frames
            # Find GT objects covered by majority
            vote_detected = 0
            for gt in gt_boxes:
                if gt['name'] not in DYNAMIC_NAMES:
                    continue
                xyz = np.array([gt['x'], gt['y'], gt['z']])
                uv = _project_ego_to_image(xyz, intrinsic, cam2ego)
                if uv is None:
                    continue
                u, v = uv
                if not (0 <= u < IMG_W and 0 <= v < IMG_H):
                    continue
                frame_hits = sum(
                    1 for dets in window_dets
                    for det in dets
                    if det['xyxy'][0] <= u <= det['xyxy'][2]
                    and det['xyxy'][1] <= v <= det['xyxy'][3]
                )
                if frame_hits >= majority_thresh:
                    vote_detected += 1

            results_per_K[K]['detected_union'] += det_union
            results_per_K[K]['detected_vote']  += vote_detected
            results_per_K[K]['total']          += n_total

        if (idx + 1) % 50 == 0:
            print(f"  [{idx+1}/{len(all_tokens)}]", end='  ')
            for K in K_values:
                r = results_per_K[K]
                if r['total'] > 0:
                    print(f"K={K}: {r['detected_union']/r['total']:.3f}", end='  ')
            print()

    # Compile summary
    summary = {}
    print(f"\n  {'K':>4}  {'Recall(union)':>14}  {'Recall(vote)':>13}")
    print('  ' + '-' * 35)
    for K in K_values:
        r = results_per_K[K]
        if r['total'] == 0:
            continue
        rec_union = r['detected_union'] / r['total']
        rec_vote  = r['detected_vote']  / r['total']
        summary[K] = {'recall_union': rec_union, 'recall_vote': rec_vote,
                      'n_gt': r['total']}
        print(f"  {K:>4}  {rec_union:>14.4f}  {rec_vote:>13.4f}")

    return summary


# ---------------------------------------------------------------------------
# Experiment 2 — Keyframe interpolation
# ---------------------------------------------------------------------------

def run_exp2(nusc: NuScenes, yolo, max_samples: Optional[int]) -> Dict:
    """
    For each pair of consecutive keyframes, interpolate YOLO track positions
    to the 5 intermediate 12Hz frames. Compare against actual YOLO on those
    frames using IoU-based TP/FP/FN.

    Baselines:
      carry  — use keyframe-t detections unchanged (no interpolation)
      interp — linearly interpolate between keyframe-t and keyframe-t+1
    """
    print("\n=== Experiment 2: Keyframe interpolation ===")

    stats = {
        'carry':  {'tp': 0, 'fp': 0, 'fn': 0},
        'interp': {'tp': 0, 'fp': 0, 'fn': 0},
        'n_intermediate': 0,
    }

    all_tokens = [s['token'] for s in nusc.sample]
    if max_samples:
        all_tokens = all_tokens[:max_samples]

    processed = 0
    for idx, token in enumerate(all_tokens):
        sample = nusc.get('sample', token)
        if not sample.get('next'):
            continue
        next_sample = nusc.get('sample', sample['next'])
        if next_sample['scene_token'] != sample['scene_token']:
            continue

        intermediates = _get_intermediate_frames(nusc, token, CAM)
        if not intermediates:
            continue

        # YOLO on both keyframes
        img_t      = load_raw_image(nusc, token, CAM)
        img_t1     = load_raw_image(nusc, sample['next'], CAM)
        dets_t     = _load_yolo_dets(yolo, img_t)
        dets_t1    = _load_yolo_dets(yolo, img_t1)

        # Match detections between keyframes for interpolation
        n_inter = len(intermediates)
        for k, sd in enumerate(intermediates):
            alpha = (k + 1) / (n_inter + 1)   # fraction of way from t to t+1

            # Carry: use keyframe-t detections
            carry_dets = [{'xyxy': d['xyxy'].copy(), 'conf': d['conf'], 'cls': d['cls']}
                          for d in dets_t]

            # Interp: match t→t+1 pairs, interpolate; unmatched from t at alpha=0
            interp_dets = []
            matched_t1 = set()
            for det_a in dets_t:
                best_iou, best_j = 0.0, -1
                for j, det_b in enumerate(dets_t1):
                    if j in matched_t1:
                        continue
                    iou = _box_iou(det_a['xyxy'], det_b['xyxy'])
                    if iou > best_iou:
                        best_iou, best_j = iou, j
                if best_iou >= 0.3 and best_j >= 0:
                    matched_t1.add(best_j)
                    interp_box = _interpolate_box(det_a['xyxy'], dets_t1[best_j]['xyxy'], alpha)
                    interp_dets.append({'xyxy': interp_box, 'conf': det_a['conf'],
                                        'cls': det_a['cls']})
                else:
                    # Unmatched: carry forward (position won't change much)
                    interp_dets.append({'xyxy': det_a['xyxy'].copy(),
                                        'conf': det_a['conf'], 'cls': det_a['cls']})

            # Ground truth for this intermediate frame: actual YOLO on the real image
            img_inter  = _load_raw_from_sd(nusc, sd)
            actual_dets = _load_yolo_dets(yolo, img_inter)

            # Evaluate
            for method, pred_dets in [('carry', carry_dets), ('interp', interp_dets)]:
                tp, fp, fn = _match_dets(pred_dets, actual_dets)
                stats[method]['tp'] += tp
                stats[method]['fp'] += fp
                stats[method]['fn'] += fn
            stats['n_intermediate'] += 1

        processed += 1
        if processed % 20 == 0:
            print(f"  [{processed} keyframe pairs]  "
                  f"interp recall={stats['interp']['tp']/(stats['interp']['tp']+stats['interp']['fn']+1e-6):.3f}  "
                  f"carry recall={stats['carry']['tp']/(stats['carry']['tp']+stats['carry']['fn']+1e-6):.3f}")

        if max_samples and processed >= max_samples:
            break

    # Compile summary
    summary = {}
    print(f"\n  {'Method':<8}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}")
    print('  ' + '-' * 38)
    for method in ('carry', 'interp'):
        s = stats[method]
        prec = s['tp'] / (s['tp'] + s['fp'] + 1e-6)
        rec  = s['tp'] / (s['tp'] + s['fn'] + 1e-6)
        f1   = 2 * prec * rec / (prec + rec + 1e-6)
        summary[method] = {'precision': prec, 'recall': rec, 'f1': f1}
        print(f"  {method:<8}  {prec:>10.4f}  {rec:>8.4f}  {f1:>8.4f}")
    print(f"  Total intermediate frames evaluated: {stats['n_intermediate']}")

    return summary


# ---------------------------------------------------------------------------
# Experiment 3 — Future-assisted detection recovery
# ---------------------------------------------------------------------------

def run_exp3(nusc: NuScenes, yolo, max_samples: Optional[int]) -> Dict:
    """
    For each triplet (t-1, t, t+1):
      - Find GT dynamic objects at t that YOLO misses at t
      - Among those, check if they are detected at t-1 AND t+1
      - If so, interpolate position at t; measure error vs GT

    Reports:
      miss_rate         — fraction of GT objects missed by single-frame YOLO at t
      recovery_rate     — fraction of misses recoverable via flanking detections
      recovery_ade_px   — average pixel displacement of recovered box vs GT centre
    """
    print("\n=== Experiment 3: Future-assisted recovery ===")

    total_gt       = 0
    total_missed   = 0
    total_flanked  = 0   # missed at t but detected at t-1 AND t+1
    total_ade_sum  = 0.0
    total_ade_n    = 0

    all_tokens = [s['token'] for s in nusc.sample]
    if max_samples:
        all_tokens = all_tokens[:max_samples]

    for idx, token in enumerate(all_tokens):
        triplet = get_triplet_tokens(nusc, token)
        if triplet is None:
            continue
        tok_tm1, tok_t, tok_tp1 = triplet

        gt_boxes = load_gt_boxes(nusc, tok_t)
        gt_dynamic = [g for g in gt_boxes if g['name'] in DYNAMIC_NAMES]
        if not gt_dynamic:
            continue

        sd_token = nusc.get('sample', tok_t)['data'][CAM]
        intrinsic, cam2ego = get_sensor_transforms(nusc, sd_token)
        dummy_img = load_raw_image(nusc, tok_t, CAM)
        IMG_H, IMG_W = dummy_img.shape[:2]

        dets_tm1 = _load_yolo_dets(yolo, load_raw_image(nusc, tok_tm1, CAM))
        dets_t   = _load_yolo_dets(yolo, dummy_img)
        dets_tp1 = _load_yolo_dets(yolo, load_raw_image(nusc, tok_tp1, CAM))

        for gt in gt_dynamic:
            xyz = np.array([gt['x'], gt['y'], gt['z']])
            uv = _project_ego_to_image(xyz, intrinsic, cam2ego)
            if uv is None:
                continue
            u_gt, v_gt = uv
            if not (0 <= u_gt < IMG_W and 0 <= v_gt < IMG_H):
                continue

            total_gt += 1

            # Is it detected at t?
            det_at_t = any(
                d['xyxy'][0] <= u_gt <= d['xyxy'][2] and
                d['xyxy'][1] <= v_gt <= d['xyxy'][3]
                for d in dets_t
            )
            if det_at_t:
                continue

            total_missed += 1

            # Is it detected at t-1?
            det_tm1 = next((
                d for d in dets_tm1
                if d['xyxy'][0] <= u_gt <= d['xyxy'][2] and
                   d['xyxy'][1] <= v_gt <= d['xyxy'][3]
            ), None)

            # Is it detected at t+1?
            det_tp1 = next((
                d for d in dets_tp1
                if d['xyxy'][0] <= u_gt <= d['xyxy'][2] and
                   d['xyxy'][1] <= v_gt <= d['xyxy'][3]
            ), None)

            if det_tm1 is None or det_tp1 is None:
                continue

            # Flanked — can reconstruct
            total_flanked += 1

            # Interpolated centre at t (alpha=0.5)
            cx_tm1 = (det_tm1['xyxy'][0] + det_tm1['xyxy'][2]) / 2
            cy_tm1 = (det_tm1['xyxy'][1] + det_tm1['xyxy'][3]) / 2
            cx_tp1 = (det_tp1['xyxy'][0] + det_tp1['xyxy'][2]) / 2
            cy_tp1 = (det_tp1['xyxy'][1] + det_tp1['xyxy'][3]) / 2
            cx_interp = (cx_tm1 + cx_tp1) / 2
            cy_interp = (cy_tm1 + cy_tp1) / 2

            ade = float(np.hypot(cx_interp - u_gt, cy_interp - v_gt))
            total_ade_sum += ade
            total_ade_n   += 1

        if (idx + 1) % 50 == 0:
            print(f"  [{idx+1}/{len(all_tokens)}]  "
                  f"miss_rate={total_missed/(total_gt+1e-6):.3f}  "
                  f"recovery={total_flanked/(total_missed+1e-6):.3f}")

    miss_rate     = total_missed  / (total_gt     + 1e-6)
    recovery_rate = total_flanked / (total_missed + 1e-6)
    ade           = total_ade_sum / (total_ade_n  + 1e-6)

    summary = {
        'n_gt':           total_gt,
        'n_missed':       total_missed,
        'n_recovered':    total_flanked,
        'miss_rate':      miss_rate,
        'recovery_rate':  recovery_rate,
        'recovery_ade_px': ade,
    }

    print(f"\n  GT objects visible in CAM_FRONT : {total_gt}")
    print(f"  Missed by single-frame YOLO     : {total_missed}  ({miss_rate:.1%})")
    print(f"  Recoverable via t-1 + t+1       : {total_flanked}  ({recovery_rate:.1%} of misses)")
    print(f"  Interpolation ADE (pixels)       : {ade:.1f}")

    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp',        default='all',
                        help='Which experiments to run: 1, 2, 3, or all (comma-separated)')
    parser.add_argument('--yolo-model', default='yolo11x.pt')
    parser.add_argument('--version',    default='v1.0-mini')
    parser.add_argument('--dataroot',   default='./data/nuscenes')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Limit samples per experiment (for quick runs)')
    args = parser.parse_args()

    exps = set()
    if args.exp == 'all':
        exps = {1, 2, 3}
    else:
        exps = {int(e) for e in args.exp.split(',')}

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading nuScenes {args.version}...")
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    print(f"Loading YOLO: {args.yolo_model}...")
    from ultralytics import YOLO
    yolo = YOLO(args.yolo_model)

    if 1 in exps:
        summary = run_exp1(nusc, yolo, args.max_samples)
        with open(OUT_DIR / 'exp1_kframe_recall.json', 'w') as f:
            json.dump(summary, f, indent=2)

    if 2 in exps:
        summary = run_exp2(nusc, yolo, args.max_samples)
        with open(OUT_DIR / 'exp2_interpolation.json', 'w') as f:
            json.dump(summary, f, indent=2)

    if 3 in exps:
        summary = run_exp3(nusc, yolo, args.max_samples)
        with open(OUT_DIR / 'exp3_recovery.json', 'w') as f:
            json.dump(summary, f, indent=2)

    print(f"\nAll results saved to {OUT_DIR}/")


if __name__ == '__main__':
    main()
