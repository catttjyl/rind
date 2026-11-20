#!/usr/bin/env python3
"""Index and verify existing shards; never builds or changes cache contents."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rind.fast_data import build_or_open_single_tile_shard
from rind.fullday_data import FullDayTileReader, resolve_week_indices
from rind.split_contract import SplitContract, compute_file_sha256

p = argparse.ArgumentParser()
p.add_argument('--cache-dir', type=Path, required=True)
p.add_argument('--split-file', type=Path, required=True)
p.add_argument('--manifest-file', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--temporal-protocol', default='annual_52week')
p.add_argument('--receiver-mode', choices=['all_valid','sampled'], default='all_valid')
p.add_argument('--max-receivers', type=int, default=256)
p.add_argument('--max-candidates', type=int, default=32)
p.add_argument('--num-azimuth-bins', type=int, choices=[0,24], default=0)
p.add_argument('--bbox-margin-m', type=float, default=4000.)
p.add_argument('--allow-synthetic-split', action='store_true')
p.add_argument('--fixture-only', action='store_true', help='permit partial synthetic fixture index only')
a=p.parse_args()
c=SplitContract(split_path=a.split_file,manifest_path=a.manifest_file,enforce_pinned_split_sha256=not a.allow_synthetic_split)
if a.fixture_only and not c.is_synthetic_fixture: raise ValueError('fixture-only requires synthetic manifest')
expected=set(c.get_fit_tiles('fit_67') if not a.fixture_only else [])
if not a.fixture_only:
    expected.update(c.get_calibration_tiles()); expected.update(c.get_eval_tiles('val_spatial_clean', allow_empty=True))
    if len(expected)!=93 or len(c.get_fit_tiles('fit_67'))!=67 or len(c.get_calibration_tiles())!=12 or len(c.get_eval_tiles('val_spatial_clean',allow_empty=True))!=14: raise RuntimeError('Snapshot 67/12/14 mismatch')
root=a.cache_dir.resolve(strict=True); tiles={}
for prov in sorted(root.glob('*/core_*/SHARD_PROVENANCE.json')):
    if prov.is_symlink() or prov.parent.is_symlink() or prov.parent.parent.is_symlink(): raise RuntimeError('symlink rejected')
    doc=json.loads(prov.read_text()); tid=doc['tile_id']
    if not a.fixture_only and tid not in expected: continue
    if tid in tiles: raise RuntimeError(f'duplicate tile {tid}; isolate a single configuration')
    tiles[tid]={'path':prov.parent.relative_to(root).as_posix(),'provenance_sha256':compute_file_sha256(prov)}
if a.fixture_only:
    expected=set(tiles)
    if not expected: raise RuntimeError('empty fixture index')
if set(tiles)!=expected: raise RuntimeError(f'shard coverage differs: missing={sorted(expected-set(tiles))}, extra={sorted(set(tiles)-expected)}')
index={'schema':'rind.portable_index.v1','split_sha256':c.split_sha256,'manifest_sha256':c.manifest_sha256,'tiles':tiles}
r=FullDayTileReader(c); r.portable_index=index
for tid in sorted(tiles):
    build_or_open_single_tile_shard(root,r,tid,week_indices=resolve_week_indices(a.temporal_protocol),receiver_mode=a.receiver_mode,max_receivers=a.max_receivers,max_candidates=a.max_candidates,num_azimuth_bins=a.num_azimuth_bins,bbox_margin_m=a.bbox_margin_m)
if a.output.resolve().is_relative_to(root): raise RuntimeError('index output must be outside read-only cache')
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps(index,sort_keys=True,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'tiles':len(tiles),'index':str(a.output),'sha256':compute_file_sha256(a.output)}))
