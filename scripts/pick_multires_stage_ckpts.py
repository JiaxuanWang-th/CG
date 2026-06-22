#!/usr/bin/env python3
"""Pick segment-best + late-best ckpts for VM/CVM (epoch) or SPCF (iter)."""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple


def ckpt_for_epoch(ckpt_dir: Path, epoch: int) -> Optional[str]:
    for pat in (
        f"checkpoint_best_cd*_epoch{epoch}.pkl",
        f"checkpoint_{epoch}.pkl",
    ):
        hits = sorted(glob.glob(str(ckpt_dir / pat)))
        if hits:
            return hits[0]
    return None


def ckpt_for_iter(ckpt_dir: Path, it: int) -> Optional[str]:
    for pat in (
        f"checkpoint_best_cd*_iter{it}.pkl",
        f"checkpoint_iter{it}.pkl",
    ):
        hits = sorted(glob.glob(str(ckpt_dir / pat)))
        if hits:
            return hits[0]
    return None


def pick_epoch(log: Path, ckpt_dir: Path, seg_lo: int, seg_hi: int, late_lo: int, late_hi: int):
    text = log.read_text(errors="replace")
    rows = []
    for m in re.finditer(r"\[Epoch\s+(\d+)\][^\n]*val_CD=([\d.]+)", text):
        ep, cd = int(m.group(1)), float(m.group(2))
        rows.append((ep, cd))

    def best_in(lo, hi):
        c = [(ep, cd) for ep, cd in rows if lo <= ep <= hi]
        if not c:
            return None
        ep, cd = min(c, key=lambda x: (x[1], -x[0]))
        return ep, cd, ckpt_for_epoch(ckpt_dir, ep)

    seg = best_in(seg_lo, seg_hi)
    late = best_in(late_lo, late_hi)
    global_best = min(rows, key=lambda x: (x[1], -x[0])) if rows else None
    gb = None
    if global_best:
        ep, cd = global_best
        gb = (ep, cd, ckpt_for_epoch(ckpt_dir, ep))

    # near end periodic
    near = None
    for ep in range(seg_hi, seg_lo - 1, -1):
        p = ckpt_for_epoch(ckpt_dir, ep)
        if p:
            near = (ep, None, p)
            break

    return seg, late, gb, near


def pick_iter(log: Path, ckpt_dir: Path, seg_lo: int, seg_hi: int, late_lo: int, late_hi: int):
    text = log.read_text(errors="replace")
    rows = []
    for m in re.finditer(r"\[Val\] Iter\s+(\d+)\s+\|\s+Chamfer\s+([\d.]+)", text):
        it, cd = int(m.group(1)), float(m.group(2))
        rows.append((it, cd))

    def best_in(lo, hi):
        c = [(it, cd) for it, cd in rows if lo <= it <= hi]
        if not c:
            return None
        it, cd = min(c, key=lambda x: (x[1], -x[0]))
        return it, cd, ckpt_for_iter(ckpt_dir, it)

    seg = best_in(seg_lo, seg_hi)
    late = best_in(late_lo, late_hi)
    gb = min(rows, key=lambda x: (x[1], -x[0])) if rows else None
    gbr = None
    if gb:
        it, cd = gb
        gbr = (it, cd, ckpt_for_iter(ckpt_dir, it))

    near = None
    for it in range(seg_hi, max(seg_lo, seg_hi - 20000) - 1, -5000):
        p = ckpt_for_iter(ckpt_dir, it)
        if p:
            near = (it, None, p)
            break

    return seg, late, gbr, near


def emit_pack_list(seg, late, gb, near):
    out: List[Tuple[str, str]] = []
    seen = set()

    def add(label, item):
        if item is None:
            return
        idx, cd, path = item
        if path is None or not os.path.isfile(path):
            print(f"WARN missing ckpt {label} idx={idx}", file=sys.stderr)
            return
        key = os.path.abspath(path)
        if key in seen:
            return
        seen.add(key)
        cd_s = f"{cd:.6f}" if cd is not None else "n/a"
        out.append((label, path))
        print(f"PICK {label}\t{idx}\t{cd_s}\t{path}")

    add("seg_best", seg)
    if late and (seg is None or late[0] != seg[0]):
        add("late_best", late)
    elif near and (seg is None or near[0] != seg[0]):
        add("near_end", near)
    elif near:
        add("near_end", near)

    for label, path in out:
        print(f"PACK\t{label}\t{path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["epoch", "iter"], required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--seg-lo", type=int, required=True)
    ap.add_argument("--seg-hi", type=int, required=True)
    ap.add_argument("--late-lo", type=int, required=True)
    ap.add_argument("--late-hi", type=int, required=True)
    args = ap.parse_args()

    log = Path(args.log)
    ckpt_dir = Path(args.ckpt_dir)
    if args.mode == "epoch":
        emit_pack_list(*pick_epoch(log, ckpt_dir, args.seg_lo, args.seg_hi, args.late_lo, args.late_hi))
    else:
        emit_pack_list(*pick_iter(log, ckpt_dir, args.seg_lo, args.seg_hi, args.late_lo, args.late_hi))


if __name__ == "__main__":
    main()
