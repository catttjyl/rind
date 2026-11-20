"""Experimental scene-scoped int32 triangle references; not a production shard schema."""
import numpy as np
import torch
from .fullday_data import select_azimuth_sector_candidates
from .fast_data import build_canonical37_torch, compute_canonical_azimuth_bins_torch


def build_index_bank(receivers, triangles, *, opaque=None, source_ids=None, k=8, bins=24, horizon=4000.):
    """Index is row in immutable tile-scene table; source IDs are not narrowed."""
    tri = np.asarray(triangles)
    if tri.dtype != np.float32:
        raise ValueError('scene triangles must be float32, matching native scene loader')
    if not np.isfinite(tri).all() or not np.isfinite(receivers).all():
        raise ValueError('non-finite geometry')
    idx, op, mask, sid = select_azimuth_sector_candidates(
        receivers, tri, scene_opaque=opaque, scene_source_triangle_ids=source_ids,
        max_candidates=k, num_azimuth_bins=bins, horizon_m=horizon,
        return_source_ids=True, return_triangle_indices=True)
    return idx, op, mask, sid


def _restore_relative_chunk(indices, triangles, receivers):
    valid = indices >= 0
    verts = triangles[indices.clamp(min=0).long()].to(torch.float64)
    rec = receivers.to(torch.float64).reshape((len(receivers),) + (1,) * (indices.ndim - 1) + (1, 3))
    diff = (verts - rec).to(torch.float32)
    return torch.where(valid[..., None, None], diff, 0.0)


def restore_relative(indices, triangles, receivers, *, max_chunk_bytes=32 * 1024 * 1024):
    """Restore the native float64 subtraction then float32 cast, padding exactly zero.

    Indices shape (N,...), triangles (M,3,3), receivers (N,3).
    Streams over receiver chunks so float64 intermediates stay bounded by ``max_chunk_bytes``.
    """
    if indices.dtype != torch.int32:
        raise ValueError('indices must be int32')
    if triangles.dtype != torch.float32 or receivers.dtype != torch.float32:
        raise ValueError('triangles and receivers must be float32')
    if indices.ndim < 1 or indices.shape[0] != receivers.shape[0]:
        raise ValueError('receiver dimension mismatch')
    if torch.any(indices < -1) or torch.any(indices >= len(triangles)):
        raise ValueError('invalid triangle index')
    shape = (*indices.shape, 3, 3)
    if len(triangles) == 0 or len(receivers) == 0:
        return torch.zeros(shape, device=indices.device, dtype=torch.float32)

    row_bytes = max(1, indices[0].numel()) * 9 * 8
    chunk_n = max(1, min(len(receivers), int(max_chunk_bytes) // row_bytes))
    if chunk_n >= len(receivers) or triangles.requires_grad or receivers.requires_grad:
        if chunk_n >= len(receivers):
            return _restore_relative_chunk(indices, triangles, receivers)
        return torch.cat(
            [_restore_relative_chunk(indices[s:e], triangles, receivers[s:e])
             for s in range(0, len(receivers), chunk_n)
             for e in (min(len(receivers), s + chunk_n),)],
            dim=0,
        )

    out = torch.empty(shape, device=indices.device, dtype=torch.float32)
    for s in range(0, len(receivers), chunk_n):
        e = min(len(receivers), s + chunk_n)
        out[s:e] = _restore_relative_chunk(indices[s:e], triangles, receivers[s:e])
    return out


def materialize_indexed(batch):
    """Select active sectors first, gather only T*N*K, then construct native canonical37."""
    idx, tri, rec, sun = (batch[k] for k in ('az_triangle_index', 'scene_triangles_xyz', 'receivers_xyz', 'sun_dirs'))
    if idx.ndim != 3 or sun.ndim != 2:
        raise ValueError('prototype expects unbatched N,A,K and T,3')
    bins = batch.get('azimuth_bin_idx')
    bins = compute_canonical_azimuth_bins_torch(sun, idx.shape[1]) if bins is None else bins.long() % idx.shape[1]
    chosen = idx[:, bins, :]
    rel = restore_relative(chosen, tri, rec).permute(1, 0, 2, 3, 4)
    mask = (chosen >= 0).permute(1, 0, 2)

    if batch.get('az_opaque') is not None:
        az_op = batch['az_opaque'].to(device=chosen.device, dtype=torch.bool)
        opaque = az_op[:, bins, :].permute(1, 0, 2) & mask
    elif batch.get('scene_opaque') is not None:
        if len(tri) == 0:
            opaque = mask
        else:
            scene_op = batch['scene_opaque'].to(device=chosen.device, dtype=torch.bool)
            opaque = torch.where(chosen >= 0, scene_op[chosen.clamp(min=0).long()], False).permute(1, 0, 2)
    else:
        opaque = mask

    if batch.get('az_source_ids') is not None:
        az_sid = batch['az_source_ids'].to(device=chosen.device, dtype=torch.int64)
        candidate_source_ids = az_sid[:, bins, :].permute(1, 0, 2)
    elif batch.get('scene_source_ids') is not None:
        if len(tri) == 0:
            candidate_source_ids = torch.zeros_like(mask, dtype=torch.int64)
        else:
            scene_sid = batch['scene_source_ids'].to(device=chosen.device, dtype=torch.int64)
            candidate_source_ids = torch.where(chosen >= 0, scene_sid[chosen.clamp(min=0).long()], 0).permute(1, 0, 2)
    else:
        candidate_source_ids = torch.where(chosen >= 0, chosen.to(torch.int64), 0).permute(1, 0, 2)

    tokens = build_canonical37_torch(rel, opaque, mask, sun)
    return dict(
        batch,
        candidate_tokens=tokens,
        candidate_mask=mask,
        candidate_source_ids=candidate_source_ids,
    )


def write_index_bundle(directory, receivers, triangles, source_ids, *, k=8):
    """Write a new scene table and reference bank; no native shard/index compatibility implied."""
    import hashlib, json
    from pathlib import Path
    d=Path(directory); d.mkdir(parents=True,exist_ok=False)
    sid=np.asarray(source_ids,dtype=np.uint64)
    if sid.shape != (len(triangles),) or len(np.unique(sid)) != len(sid):
        raise ValueError('source IDs must be unique and aligned to the scene table')
    idx,_,_,_=build_index_bank(receivers,triangles,source_ids=sid,k=k)
    arrays={'receivers_xyz':np.asarray(receivers,dtype=np.float32),'scene_triangles_xyz':triangles,
            'scene_source_ids':sid,'az_triangle_index':idx}
    meta={'schema':'dual.az24.index_bundle.v1','index_scope':'scene table row; identity is ordered source ID array',
          'k':k,'bins':24,'arrays':{}}
    for name,arr in arrays.items():
        path=d/(name+'.npy'); np.save(path,arr,allow_pickle=False)
        meta['arrays'][name]={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'shape':list(arr.shape),'dtype':str(arr.dtype)}
    (d/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
    return meta


def read_index_bundle(directory):
    """Fail closed on altered or mismatched table, receiver or bank."""
    import hashlib, json
    from pathlib import Path
    d=Path(directory); meta=json.loads((d/'manifest.json').read_text())
    if meta.get('schema')!='dual.az24.index_bundle.v1' or set(meta.get('arrays',{})) != {'receivers_xyz','scene_triangles_xyz','scene_source_ids','az_triangle_index'}:
        raise ValueError('invalid bundle manifest')
    out={}
    for name,spec in meta['arrays'].items():
        path=d/(name+'.npy')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=spec['sha256']: raise ValueError('bundle digest mismatch: '+name)
        arr=np.load(path,allow_pickle=False)
        if list(arr.shape)!=spec['shape'] or str(arr.dtype)!=spec['dtype']: raise ValueError('bundle array mismatch: '+name)
        out[name]=arr
    idx=out['az_triangle_index']; tri=out['scene_triangles_xyz']; rec=out['receivers_xyz']; sid=out['scene_source_ids']
    if idx.dtype!=np.int32 or tri.dtype!=np.float32 or rec.dtype!=np.float32 or sid.dtype!=np.uint64:
        raise ValueError('invalid dtype')
    if tri.ndim!=3 or tri.shape[1:]!=(3,3) or rec.ndim!=2 or rec.shape[1]!=3 or idx.shape!=(len(rec),24,meta['k']) or sid.shape!=(len(tri),):
        raise ValueError('invalid shape')
    if np.any(idx < -1) or np.any(idx >= len(tri)) or len(np.unique(sid))!=len(sid): raise ValueError('invalid references')
    return out


def _sort_and_dedup_bucket(ids_arr, tri_arr):
    if len(ids_arr) == 0:
        return ids_arr, tri_arr
    order = np.argsort(ids_arr, kind='stable')
    sorted_ids = ids_arr[order]
    sorted_tri = tri_arr[order]
    dup = sorted_ids[1:] == sorted_ids[:-1]
    if np.any(dup) and not np.array_equal(sorted_tri[1:][dup], sorted_tri[:-1][dup]):
        raise ValueError('source ID has conflicting geometry')
    first = np.r_[True, ~dup]
    return sorted_ids[first], sorted_tri[first]


def _chunked_searchsorted_int32(unique_ids, sid, chunk_size):
    n = len(sid)
    if n <= chunk_size:
        return np.searchsorted(unique_ids, sid).astype(np.int32)
    out = np.empty((n,), dtype=np.int32)
    for s in range(0, n, chunk_size):
        e = min(n, s + chunk_size)
        out[s:e] = np.searchsorted(unique_ids, sid[s:e]).astype(np.int32)
    return out


def merge_scene_tables(scenes, *, max_chunk_triangles=1_000_000, temp_dir=None):
    """Bounded-memory global dictionary from [(float32 triangles, uint64 IDs), ...].

    Uses quantile-partitioned out-of-core bucketed merge on disk whenever total
    triangles exceed ``max_chunk_triangles``, so arbitrarily large multi-tile
    tables merge without holding all concatenated arrays + argsort index in RAM.
    Stable global IDs are sorted-table int32 row numbers; original uint64 IDs
    remain the stable identity.
    """
    import math
    import tempfile
    from pathlib import Path

    scenes = list(scenes)
    if not scenes:
        raise ValueError('no scenes')
    chunk_cap = max(1, int(max_chunk_triangles))
    total_rows = 0
    samples = []
    for tri, sid in scenes:
        if (
            tri.dtype != np.float32
            or tri.ndim != 3
            or tri.shape[1:] != (3, 3)
            or sid.dtype != np.uint64
            or sid.shape != (len(tri),)
        ):
            raise ValueError('invalid scene table')
        for s in range(0, len(tri), chunk_cap):
            if not np.isfinite(tri[s : min(len(tri), s + chunk_cap)]).all():
                raise ValueError('non-finite triangles')
        total_rows += len(sid)
        if len(sid) > 0:
            step = max(1, len(sid) // max(1, 4096 // max(1, len(scenes))))
            samples.append(np.asarray(sid[::step], dtype=np.uint64))

    if total_rows == 0:
        empty_tri = np.empty((0, 3, 3), dtype=np.float32)
        empty_ids = np.empty((0,), dtype=np.uint64)
        return empty_tri, empty_ids, [np.empty((0,), dtype=np.int32) for _ in scenes]

    num_buckets = min(128, max(1, math.ceil(total_rows / chunk_cap)))
    splitters = np.empty((0,), dtype=np.uint64)
    if num_buckets > 1 and samples:
        sample_arr = np.sort(np.concatenate(samples))
        if len(sample_arr) >= num_buckets:
            q_idx = np.linspace(0, len(sample_arr) - 1, num_buckets + 1, dtype=np.int64)[1:-1]
            splitters = np.unique(sample_arr[q_idx])

    if len(splitters) == 0:
        all_ids = np.concatenate([sid for _, sid in scenes])
        all_tri = np.concatenate([tri for tri, _ in scenes])
        unique_ids, global_tri = _sort_and_dedup_bucket(all_ids, all_tri)
    else:
        actual_buckets = len(splitters) + 1
        with tempfile.TemporaryDirectory(dir=temp_dir) as td:
            td_path = Path(td)
            id_bins = [td_path / f'ids_{b}.bin' for b in range(actual_buckets)]
            tri_bins = [td_path / f'tri_{b}.bin' for b in range(actual_buckets)]

            for tri, sid in scenes:
                n = len(sid)
                for s in range(0, n, chunk_cap):
                    e = min(n, s + chunk_cap)
                    sid_c = np.ascontiguousarray(sid[s:e], dtype=np.uint64)
                    tri_c = np.ascontiguousarray(tri[s:e], dtype=np.float32)
                    b_idx = np.searchsorted(splitters, sid_c, side='right')
                    b_order = np.argsort(b_idx, kind='stable')
                    b_sorted = b_idx[b_order]
                    sid_sorted = sid_c[b_order]
                    tri_sorted = tri_c[b_order]
                    uniq_b, starts, counts = np.unique(
                        b_sorted, return_index=True, return_counts=True
                    )
                    for b, st, cnt in zip(uniq_b.tolist(), starts.tolist(), counts.tolist()):
                        en = st + cnt
                        with open(id_bins[b], 'ab') as fid, open(tri_bins[b], 'ab') as ftri:
                            fid.write(sid_sorted[st:en].tobytes())
                            ftri.write(tri_sorted[st:en].tobytes())

            out_ids_path = td_path / 'out_ids.bin'
            out_tri_path = td_path / 'out_tri.bin'
            total_unique = 0
            with open(out_ids_path, 'wb') as fout_ids, open(out_tri_path, 'wb') as fout_tri:
                for b in range(actual_buckets):
                    if not id_bins[b].exists():
                        continue
                    ids_b = np.fromfile(id_bins[b], dtype=np.uint64)
                    tri_b = np.fromfile(tri_bins[b], dtype=np.float32).reshape(-1, 3, 3)
                    id_bins[b].unlink()
                    tri_bins[b].unlink()
                    u_ids_b, u_tri_b = _sort_and_dedup_bucket(ids_b, tri_b)
                    total_unique += len(u_ids_b)
                    if total_unique > np.iinfo(np.int32).max:
                        raise ValueError('global table exceeds int32 capacity')
                    fout_ids.write(np.ascontiguousarray(u_ids_b, dtype=np.uint64).tobytes())
                    fout_tri.write(np.ascontiguousarray(u_tri_b, dtype=np.float32).tobytes())

            unique_ids = np.fromfile(out_ids_path, dtype=np.uint64)
            global_tri = np.fromfile(out_tri_path, dtype=np.float32).reshape(total_unique, 3, 3)

    if len(unique_ids) > np.iinfo(np.int32).max:
        raise ValueError('global table exceeds int32 capacity')
    mappings = [_chunked_searchsorted_int32(unique_ids, sid, chunk_cap) for _, sid in scenes]
    return global_tri, unique_ids, mappings


def remap_scene_index(indices, local_to_global):
    """Translate local selected rows to the shared global table; -1 stays padding."""
    idx=np.asarray(indices); mapping=np.asarray(local_to_global)
    if idx.dtype!=np.int32 or mapping.dtype!=np.int32 or np.any(idx < -1) or np.any(idx>=len(mapping)):
        raise ValueError('invalid local index mapping')
    out=np.full(idx.shape,-1,dtype=np.int32); valid=idx>=0
    out[valid]=mapping[idx[valid]]
    return out
