"""
yolo_track_predict.py — Idea 1: YOLO tracking + constant-velocity t+1 prediction.

Pipeline (per camera frame-triplet):
  1. Run YOLO on t-1, t, t+1 to detect dynamic agents.
  2. IoU-match detections between t-1 and t to form tracks.
  3. Extrapolate each track forward by one step (constant velocity):
       box_pred_{t+1} = box_t + (box_t - box_{t-1})
  4. Evaluate predictions against actual t+1 detections.

Metrics:
  - ADE  (Average Displacement Error)   — centroid L2 distance, predicted vs. actual
  - IoU  — median IoU of matched prediction/actual pairs
  - Prec/Rec at IoU ≥ 0.5              — detection-level accuracy
  - Zero-velocity baseline              — box_t used as the t+1 prediction

Usage:
    python scripts/yolo_track_predict.py [--version v1.0-mini] [--cam CAM_FRONT]
                                         [--model yolo11n.pt] [--conf 0.25]
                                         [--vis] [--max-triplets 200]

Outputs (in temporal_analysis/yolo_prediction/):
    results.json          per-triplet metrics
    summary.json          aggregated mean/std
    vis_<n>.png           visualisation panels (if --vis)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data.temporal_loader import iter_triplets, load_raw_image
from nuscenes.nuscenes import NuScenes

# ---------------------------------------------------------------------------
# YOLO dynamic-class filter (COCO ids → nuScenes equivalents)
# ---------------------------------------------------------------------------
DYNAMIC_COCO_IDS = {
    0:  'pedestrian',    # person
    1:  'bicycle',       # bicycle
    2:  'car',           # car
    3:  'motorcycle',    # motorcycle
    5:  'bus',           # bus
    7:  'truck',         # truck
}

# Colours for visualisation
_COLOURS = {
    'actual_prev':  (100, 149, 237),   # cornflower blue — t-1 detections
    'actual_cur':   (30,  144, 255),   # dodger blue     — t   detections
    'actual_next':  (0,   200,  80),   # green           — t+1 actual detections
    'pred':         (255,  60,  60),   # red             — t+1 CV prediction (tracked)
    'untracked':    (120, 120, 120),   # grey            — t+1 carried forward (no velocity)
    'arrow':        (255, 100, 100),   # light red       — velocity arrow on t panel
}


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def box_centroid(box: np.ndarray) -> np.ndarray:
    """Return (cx, cy) from (x1, y1, x2, y2)."""
    return np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0])


def box_iou(b1: np.ndarray, b2: np.ndarray) -> float:
    """Compute IoU between two boxes in (x1, y1, x2, y2) format."""
    xi1 = max(b1[0], b2[0])
    yi1 = max(b1[1], b2[1])
    xi2 = min(b1[2], b2[2])
    yi2 = min(b1[3], b2[3])
    inter = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
    if inter == 0.0:
        return 0.0
    a1    = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2    = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def match_boxes(
    dets_a: List[Dict],
    dets_b: List[Dict],
    iou_thresh: float = 0.30,
) -> List[Tuple[int, int]]:
    """
    Greedy IoU matching between two detection lists.

    Matches are class-constrained (same class only) and sorted by descending
    IoU. Each detection participates in at most one match.

    Returns a list of (idx_in_a, idx_in_b) index pairs.
    """
    if not dets_a or not dets_b:
        return []

    # Build score matrix (same class only)
    scores = np.zeros((len(dets_a), len(dets_b)))
    for i, da in enumerate(dets_a):
        for j, db in enumerate(dets_b):
            if da['cls'] == db['cls']:
                scores[i, j] = box_iou(da['xyxy'], db['xyxy'])

    matched_a, matched_b = set(), set()
    pairs = []

    # Descending IoU order
    order = np.dstack(np.unravel_index(np.argsort(-scores, axis=None), scores.shape))[0]
    for i, j in order:
        if scores[i, j] < iou_thresh:
            break
        if i in matched_a or j in matched_b:
            continue
        pairs.append((int(i), int(j)))
        matched_a.add(i)
        matched_b.add(j)

    return pairs


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def run_yolo(model, img: np.ndarray, conf: float) -> List[Dict]:
    """
    Run YOLO on a raw uint8 RGB numpy image.

    Returns a list of detection dicts:
        {'xyxy': np.ndarray [4], 'conf': float, 'cls': int, 'label': str}
    Only DYNAMIC_COCO_IDS classes are kept.
    """
    results = model.predict(img, conf=conf, verbose=False)
    dets = []
    for r in results:
        if r.boxes is None:
            continue
        boxes = r.boxes
        for xyxy, conf_val, cls_id in zip(
            boxes.xyxy.cpu().numpy(),
            boxes.conf.cpu().numpy(),
            boxes.cls.cpu().numpy().astype(int),
        ):
            if cls_id not in DYNAMIC_COCO_IDS:
                continue
            dets.append({
                'xyxy':  xyxy.astype(float),
                'conf':  float(conf_val),
                'cls':   cls_id,
                'label': DYNAMIC_COCO_IDS[cls_id],
            })
    return dets


def predict_boxes(
    dets_tm1: List[Dict],
    dets_t:   List[Dict],
    matches:  List[Tuple[int, int]],
) -> Tuple[List[Dict], List[Dict]]:
    """
    Produce t+1 predictions for the constant-velocity and zero-velocity models.

    Constant velocity: box_{t+1} = box_t + (box_t - box_{t-1})
    Zero velocity:     box_{t+1} = box_t  (no movement)

    Unmatched detections at t are kept in both models as zero-velocity.

    Returns (cv_predictions, zv_predictions) — parallel lists.
    """
    matched_t = {j for _, j in matches}
    cv_preds, zv_preds = [], []

    # Matched tracks → constant-velocity extrapolation
    for i, j in matches:
        delta = dets_t[j]['xyxy'] - dets_tm1[i]['xyxy']
        cv_box = np.clip(dets_t[j]['xyxy'] + delta, 0, 1e9)
        zv_box = dets_t[j]['xyxy'].copy()
        meta   = {'cls': dets_t[j]['cls'], 'label': dets_t[j]['label'], 'tracked': True}
        cv_preds.append({**meta, 'xyxy': cv_box})
        zv_preds.append({**meta, 'xyxy': zv_box})

    # Unmatched at t → zero-velocity for both models
    for j, det in enumerate(dets_t):
        if j not in matched_t:
            meta = {'cls': det['cls'], 'label': det['label'], 'tracked': False}
            cv_preds.append({**meta, 'xyxy': det['xyxy'].copy()})
            zv_preds.append({**meta, 'xyxy': det['xyxy'].copy()})

    return cv_preds, zv_preds


def evaluate(
    preds:    List[Dict],
    actuals:  List[Dict],
    iou_thresh: float = 0.5,
) -> Dict:
    """
    Match predictions against actual detections and compute:
      - ade         : mean centroid L2 distance for matched pairs
      - median_iou  : median IoU for matched pairs
      - precision   : TP / (TP + FP)
      - recall      : TP / (TP + FN)
      - n_pred      : number of predictions
      - n_actual    : number of actual detections
    """
    if not preds and not actuals:
        return {'ade': 0.0, 'median_iou': 1.0, 'precision': 1.0, 'recall': 1.0,
                'n_pred': 0, 'n_actual': 0}

    if not preds:
        return {'ade': None, 'median_iou': 0.0, 'precision': 0.0, 'recall': 0.0,
                'n_pred': 0, 'n_actual': len(actuals)}

    if not actuals:
        return {'ade': None, 'median_iou': 0.0, 'precision': 0.0, 'recall': 1.0,
                'n_pred': len(preds), 'n_actual': 0}

    pairs = match_boxes(preds, actuals, iou_thresh=iou_thresh)
    tp = len(pairs)

    ious, ades = [], []
    for i, j in pairs:
        iou = box_iou(preds[i]['xyxy'], actuals[j]['xyxy'])
        ade = float(np.linalg.norm(
            box_centroid(preds[i]['xyxy']) - box_centroid(actuals[j]['xyxy'])
        ))
        ious.append(iou)
        ades.append(ade)

    precision = tp / len(preds)   if preds   else 0.0
    recall    = tp / len(actuals) if actuals else 0.0

    return {
        'ade':         float(np.mean(ades)) if ades else None,
        'median_iou':  float(np.median(ious)) if ious else 0.0,
        'precision':   precision,
        'recall':      recall,
        'n_pred':      len(preds),
        'n_actual':    len(actuals),
    }


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _draw_boxes(draw: ImageDraw.ImageDraw, dets: List[Dict], colour: Tuple, label_prefix: str = ''):
    for det in dets:
        x1, y1, x2, y2 = det['xyxy'].astype(int)
        draw.rectangle([x1, y1, x2, y2], outline=colour, width=3)
        tag = f"{label_prefix}{det['label']}"
        draw.text((x1 + 2, max(0, y1 - 14)), tag, fill=colour)


def visualise_triplet(
    imgs:      Tuple[np.ndarray, np.ndarray, np.ndarray],
    dets:      Tuple[List[Dict], List[Dict], List[Dict]],
    matches:   List[Tuple[int, int]],
    cv_preds:  List[Dict],
    zv_preds:  List[Dict],   # kept in signature for API compat; not drawn
    save_path: Path,
    triplet_idx: int,
):
    """
    Save a 3-panel visualisation: t-1 | t | t+1.

    t-1 panel: blue boxes — YOLO detections
    t   panel: blue boxes + red arrows pointing to each tracked object's
               predicted t+1 centroid position
    t+1 panel:
        green  — actual YOLO detections at t+1
        red    — CV prediction (tracked objects only; box displaced by velocity)
        grey   — objects from t with no velocity estimate (untracked, carried forward)

    ZV (zero-velocity) baseline is excluded from the visualisation to reduce
    clutter; it is captured in metrics only.
    """
    img_tm1, img_t, img_tp1 = [Image.fromarray(i) for i in imgs]
    det_tm1, det_t, det_tp1 = dets

    d0 = ImageDraw.Draw(img_tm1)
    d1 = ImageDraw.Draw(img_t)
    d2 = ImageDraw.Draw(img_tp1)

    # t-1 and t: plain detections
    _draw_boxes(d0, det_tm1, _COLOURS['actual_prev'])
    _draw_boxes(d1, det_t,   _COLOURS['actual_cur'])

    # t+1: actual detections drawn first (bottom layer)
    _draw_boxes(d2, det_tp1, _COLOURS['actual_next'])

    # t+1: CV predictions — tracked (red) and untracked (grey) separately
    tracked   = [p for p in cv_preds if p['tracked']]
    untracked = [p for p in cv_preds if not p['tracked']]
    _draw_boxes(d2, untracked, _COLOURS['untracked'], label_prefix='? ')
    _draw_boxes(d2, tracked,   _COLOURS['pred'],      label_prefix='CV ')

    # t panel: velocity arrows for tracked objects
    for i, j in matches:
        c_tm1  = box_centroid(det_tm1[i]['xyxy'])
        c_t    = box_centroid(det_t[j]['xyxy'])
        c_pred = (c_t + (c_t - c_tm1)).astype(int)
        # Arrow: current centroid → predicted centroid
        d1.line([tuple(c_t), tuple(c_pred)], fill=_COLOURS['arrow'], width=3)
        # Arrowhead dot
        r = 4
        d1.ellipse([c_pred[0]-r, c_pred[1]-r, c_pred[0]+r, c_pred[1]+r],
                   fill=_COLOURS['arrow'])

    # Legend on t+1 panel
    legend_items = [
        ('Actual t+1',         _COLOURS['actual_next']),
        ('CV pred (tracked)',   _COLOURS['pred']),
        ('No velocity (grey)',  _COLOURS['untracked']),
    ]
    for k, (txt, col) in enumerate(legend_items):
        d2.rectangle([8, 8 + k * 20, 18, 18 + k * 20], fill=col)
        d2.text((22, 8 + k * 20), txt, fill=(255, 255, 255))

    # Stitch panels
    total_w = img_tm1.width + img_t.width + img_tp1.width
    max_h   = max(img_tm1.height, img_t.height, img_tp1.height)
    canvas  = Image.new('RGB', (total_w, max_h + 24), color=(30, 30, 30))
    canvas.paste(img_tm1, (0, 24))
    canvas.paste(img_t,   (img_tm1.width, 24))
    canvas.paste(img_tp1, (img_tm1.width + img_t.width, 24))

    dc = ImageDraw.Draw(canvas)
    dc.text((img_tm1.width // 2 - 10,  4), 't−1', fill=(200, 200, 200))
    dc.text((img_tm1.width + img_t.width // 2 - 5, 4), 't', fill=(200, 200, 200))
    dc.text((img_tm1.width + img_t.width + img_tp1.width // 2 - 20, 4),
            't+1 predictions', fill=(200, 200, 200))

    n_tracked   = len(tracked)
    n_untracked = len(untracked)
    n_actual    = len(det_tp1)
    dc.text((img_tm1.width + img_t.width + 8, max_h + 24 - 16),
            f'tracked={n_tracked}  untracked={n_untracked}  actual={n_actual}',
            fill=(160, 160, 160))

    save_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(save_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='YOLO constant-velocity t+1 prediction')
    parser.add_argument('--version',      default='v1.0-mini',  help='nuScenes version')
    parser.add_argument('--dataroot',     default='./data/nuscenes')
    parser.add_argument('--cam',          default='CAM_FRONT',  help='Camera channel to use')
    parser.add_argument('--model',        default='yolo11n.pt', help='YOLO model weights')
    parser.add_argument('--conf',         type=float, default=0.25, help='YOLO confidence threshold')
    parser.add_argument('--iou-match',    type=float, default=0.30, help='IoU threshold for frame-to-frame matching')
    parser.add_argument('--iou-eval',     type=float, default=0.50, help='IoU threshold for prediction evaluation')
    parser.add_argument('--vis',          action='store_true',   help='Save visualisation panels')
    parser.add_argument('--vis-every',    type=int,   default=10, help='Save vis every N triplets')
    parser.add_argument('--max-triplets', type=int,   default=None, help='Limit number of triplets (for quick runs)')
    args = parser.parse_args()

    out_dir = Path('temporal_analysis/yolo_prediction')
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load nuScenes
    print(f"Loading nuScenes {args.version} ...")
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    # Load YOLO
    from ultralytics import YOLO
    print(f"Loading YOLO model: {args.model} ...")
    yolo = YOLO(args.model)

    results     = []
    cv_ades_all = []
    zv_ades_all = []

    triplets = list(iter_triplets(nusc))
    if args.max_triplets:
        triplets = triplets[:args.max_triplets]

    print(f"Processing {len(triplets)} triplets on {args.cam} ...")

    for idx, (tok_tm1, tok_t, tok_tp1) in enumerate(triplets):
        img_tm1 = load_raw_image(nusc, tok_tm1, args.cam)
        img_t   = load_raw_image(nusc, tok_t,   args.cam)
        img_tp1 = load_raw_image(nusc, tok_tp1, args.cam)

        det_tm1 = run_yolo(yolo, img_tm1, args.conf)
        det_t   = run_yolo(yolo, img_t,   args.conf)
        det_tp1 = run_yolo(yolo, img_tp1, args.conf)

        matches             = match_boxes(det_tm1, det_t, iou_thresh=args.iou_match)
        cv_preds, zv_preds  = predict_boxes(det_tm1, det_t, matches)

        cv_metrics = evaluate(cv_preds, det_tp1, iou_thresh=args.iou_eval)
        zv_metrics = evaluate(zv_preds, det_tp1, iou_thresh=args.iou_eval)

        if cv_metrics['ade'] is not None:
            cv_ades_all.append(cv_metrics['ade'])
        if zv_metrics['ade'] is not None:
            zv_ades_all.append(zv_metrics['ade'])

        record = {
            'idx':        idx,
            'tok_tm1':    tok_tm1,
            'tok_t':      tok_t,
            'tok_tp1':    tok_tp1,
            'n_dets_tm1': len(det_tm1),
            'n_dets_t':   len(det_t),
            'n_dets_tp1': len(det_tp1),
            'n_matches':  len(matches),
            'cv':         cv_metrics,
            'zv':         zv_metrics,
        }
        results.append(record)

        if (idx + 1) % 20 == 0:
            cv_mean = np.mean(cv_ades_all) if cv_ades_all else float('nan')
            zv_mean = np.mean(zv_ades_all) if zv_ades_all else float('nan')
            print(f"  [{idx+1}/{len(triplets)}]  "
                  f"cv_ADE={cv_mean:.1f}px  zv_ADE={zv_mean:.1f}px  "
                  f"tracks/frame={np.mean([r['n_matches'] for r in results[-20:]]):.1f}")

        if args.vis and idx % args.vis_every == 0:
            save_path = out_dir / f'vis_{idx:04d}.png'
            visualise_triplet(
                imgs     = (img_tm1, img_t, img_tp1),
                dets     = (det_tm1, det_t, det_tp1),
                matches  = matches,
                cv_preds = cv_preds,
                zv_preds = zv_preds,
                save_path = save_path,
                triplet_idx = idx,
            )

    # -----------------------------------------------------------------------
    # Aggregate & save
    # -----------------------------------------------------------------------
    def _agg(vals):
        if not vals:
            return {'mean': None, 'std': None, 'median': None}
        return {
            'mean':   float(np.mean(vals)),
            'std':    float(np.std(vals)),
            'median': float(np.median(vals)),
        }

    def _metric_agg(key, sub_key):
        vals = [r[key][sub_key] for r in results if r[key][sub_key] is not None]
        return _agg(vals)

    summary = {
        'n_triplets':    len(results),
        'camera':        args.cam,
        'model':         args.model,
        'conf':          args.conf,
        'constant_velocity': {
            'ade':         _metric_agg('cv', 'ade'),
            'median_iou':  _metric_agg('cv', 'median_iou'),
            'precision':   _metric_agg('cv', 'precision'),
            'recall':      _metric_agg('cv', 'recall'),
        },
        'zero_velocity': {
            'ade':         _metric_agg('zv', 'ade'),
            'median_iou':  _metric_agg('zv', 'median_iou'),
            'precision':   _metric_agg('zv', 'precision'),
            'recall':      _metric_agg('zv', 'recall'),
        },
        'tracking': {
            'mean_matches_per_frame': float(np.mean([r['n_matches'] for r in results])),
            'mean_dets_t':            float(np.mean([r['n_dets_t']  for r in results])),
        },
    }

    with open(out_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Print summary table
    print('\n' + '=' * 58)
    print(f"  Results  ({args.cam}, {args.model}, conf={args.conf})")
    print('=' * 58)
    print(f"  Triplets processed : {len(results)}")
    print(f"  Mean dets/frame    : {summary['tracking']['mean_dets_t']:.1f}")
    print(f"  Mean tracks/frame  : {summary['tracking']['mean_matches_per_frame']:.1f}")
    print('-' * 58)
    print(f"  {'Metric':<22}  {'Const-Vel':>10}  {'Zero-Vel':>10}")
    print(f"  {'------':<22}  {'----------':>10}  {'--------':>10}")
    for label, key in [('ADE (px)', 'ade'), ('Median IoU', 'median_iou'),
                       ('Precision@0.5', 'precision'), ('Recall@0.5', 'recall')]:
        cv_val = summary['constant_velocity'][key]['mean']
        zv_val = summary['zero_velocity'][key]['mean']
        cv_str = f'{cv_val:.3f}' if cv_val is not None else '   N/A'
        zv_str = f'{zv_val:.3f}' if zv_val is not None else '   N/A'
        print(f"  {label:<22}  {cv_str:>10}  {zv_str:>10}")
    print('=' * 58)
    print(f"\nSaved to {out_dir}/")


if __name__ == '__main__':
    main()