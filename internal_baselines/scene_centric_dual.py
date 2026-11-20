"""Scene-centric v3.0 dual model: hierarchical AABB sparse voxel latent bank,
3D ray-cone frustum local query (step <= 0.75 * voxel_size, K_local <= 64),
and smooth log-space front-to-back differentiable transmittance accumulation.
"""
from dataclasses import dataclass
import math
from typing import Dict, Optional, Tuple
import torch
from torch import nn

_VOXEL_OFFSET = 1 << 20
_VOXEL_MASK = (1 << 21) - 1


def _hash_coords(coords: torch.Tensor) -> torch.Tensor:
    cx = (coords[..., 0] + _VOXEL_OFFSET).clamp(0, _VOXEL_MASK)
    cy = (coords[..., 1] + _VOXEL_OFFSET).clamp(0, _VOXEL_MASK)
    cz = (coords[..., 2] + _VOXEL_OFFSET).clamp(0, _VOXEL_MASK)
    return cx | (cy << 21) | (cz << 42)


def _quantize_and_hash(points: torch.Tensor, voxel_size: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize 3D world points into collision-free 63-bit spatial hash keys."""
    coords = torch.floor(points / float(voxel_size)).to(torch.int64)
    return _hash_coords(coords), coords


def _tri_voxel_assignments(
    triangles: torch.Tensor,
    voxel_size: float,
    max_span_per_axis: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Map each triangle to every voxel overlapped by its AABB (clamped per axis)."""
    m_tris = len(triangles)
    dev = triangles.device
    if m_tris == 0:
        return (
            torch.empty((0,), dtype=torch.int64, device=dev),
            torch.empty((0,), dtype=torch.int64, device=dev),
        )
    tri_min = triangles.min(dim=1).values
    tri_max = triangles.max(dim=1).values
    c_min = torch.floor(tri_min / float(voxel_size)).to(torch.int64)
    c_max = torch.floor(tri_max / float(voxel_size)).to(torch.int64)
    c_max = torch.minimum(c_max, c_min + int(max_span_per_axis) - 1)
    span = (c_max - c_min + 1).clamp(min=1)
    n_vox = span[:, 0] * span[:, 1] * span[:, 2]

    if bool(torch.all(n_vox == 1)):
        return torch.arange(m_tris, dtype=torch.int64, device=dev), _hash_coords(c_min)

    tri_idx = torch.repeat_interleave(torch.arange(m_tris, dtype=torch.int64, device=dev), n_vox)
    cum = torch.cumsum(n_vox, dim=0)
    starts = cum - n_vox
    local_flat = torch.arange(len(tri_idx), dtype=torch.int64, device=dev) - starts[tri_idx]
    sy = span[tri_idx, 1]
    sz = span[tri_idx, 2]
    dx = local_flat // (sy * sz)
    rem = local_flat % (sy * sz)
    dy = rem // sz
    dz = rem % sz
    coords = c_min[tri_idx] + torch.stack([dx, dy, dz], dim=-1)
    return tri_idx, _hash_coords(coords)


@dataclass
class SparseVoxelLevel:
    voxel_size: float
    keys: torch.Tensor       # (V,) int64, sorted ascending
    latents: torch.Tensor    # (V, D) float32
    centers: torch.Tensor    # (V, 3) float32
    radii: torch.Tensor      # (V,) float32
    counts: torch.Tensor     # (V,) int64
    edge_tri_idx: torch.Tensor   # (E,) int64 triangle index per (tri, voxel) edge
    edge_keys: torch.Tensor      # (E,) int64 voxel key per (tri, voxel) edge


@dataclass
class SceneLatentBank:
    triangles: torch.Tensor  # (M, 3, 3) float32
    fine: SparseVoxelLevel
    coarse: SparseVoxelLevel
    scene_min: torch.Tensor  # (3,) float32
    scene_max: torch.Tensor  # (3,) float32
    last_reencoded_triangles: int = 0
    last_reencoded_voxels: int = 0


class SceneCentricSharedGeom(nn.Module):
    """v3.0 scene-centric model with AABB sparse voxel bank, 3D ray-cone query, and transmittance."""

    model_type = 'scene_centric_shared_geom'
    version = '3.0'

    def __init__(
        self,
        hidden_dim: int = 32,
        *,
        fine_voxel_size: float = 4.0,
        coarse_voxel_size: float = 16.0,
        near_field_radius: float = 160.0,
        k_local: int = 64,
        cone_base_radius: float = 1.5,
        cone_tan_half_angle: float = 0.05,
        receiver_chunk_size: int = 2048,
    ):
        super().__init__()
        if k_local < 1 or k_local > 64:
            raise ValueError('k_local must be in [1, 64]')
        self.hidden_dim = int(hidden_dim)
        self.fine_voxel_size = float(fine_voxel_size)
        self.coarse_voxel_size = float(coarse_voxel_size)
        self.near_field_radius = float(near_field_radius)
        self.k_local = int(k_local)
        self.cone_base_radius = float(cone_base_radius)
        self.cone_tan_half_angle = float(cone_tan_half_angle)
        self.receiver_chunk_size = int(receiver_chunk_size)

        self.face_encoder = nn.Sequential(
            nn.Linear(9, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.fine_voxel_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.coarse_voxel_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.query = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.opacity_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 5, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.cell = nn.GRUCell(hidden_dim + 6, hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def encode_scene(self, triangles: torch.Tensor) -> torch.Tensor:
        """Encode triangle geometry in world coordinates (independent of receiver count)."""
        return self.face_encoder(triangles.reshape(-1, 9) / 100.0)

    def _aggregate_voxels_from_edges(
        self,
        edge_tri_idx: torch.Tensor,
        edge_keys: torch.Tensor,
        centroids: torch.Tensor,
        tri_radii: torch.Tensor,
        tri_feats: torch.Tensor,
        voxel_size: float,
        voxel_proj: nn.Module,
        all_edge_tri_idx: torch.Tensor,
        all_edge_keys: torch.Tensor,
    ) -> SparseVoxelLevel:
        dev = centroids.device
        if len(edge_keys) == 0:
            return SparseVoxelLevel(
                voxel_size=voxel_size,
                keys=torch.empty((0,), dtype=torch.int64, device=dev),
                latents=torch.empty((0, self.hidden_dim), dtype=tri_feats.dtype, device=dev),
                centers=torch.empty((0, 3), dtype=centroids.dtype, device=dev),
                radii=torch.empty((0,), dtype=centroids.dtype, device=dev),
                counts=torch.empty((0,), dtype=torch.int64, device=dev),
                edge_tri_idx=all_edge_tri_idx,
                edge_keys=all_edge_keys,
            )

        unique_keys, inv_idx = torch.unique(edge_keys, sorted=True, return_inverse=True)
        num_voxels = len(unique_keys)
        counts = torch.bincount(inv_idx, minlength=num_voxels).clamp_min(1)

        e_feats = tri_feats[edge_tri_idx]
        sum_feats = tri_feats.new_zeros((num_voxels, self.hidden_dim)).index_add_(0, inv_idx, e_feats)
        mean_feats = sum_feats / counts.to(tri_feats.dtype).unsqueeze(-1)
        latents = voxel_proj(mean_feats)

        e_ctr = centroids[edge_tri_idx]
        sum_centers = centroids.new_zeros((num_voxels, 3)).index_add_(0, inv_idx, e_ctr)
        centers = sum_centers / counts.to(centroids.dtype).unsqueeze(-1)

        e_rad = tri_radii[edge_tri_idx]
        max_tri_r = torch.zeros((num_voxels,), dtype=tri_radii.dtype, device=dev)
        max_tri_r.scatter_reduce_(0, inv_idx, e_rad, reduce='amax', include_self=False)
        radii = max_tri_r + 0.8660254 * float(voxel_size)

        return SparseVoxelLevel(
            voxel_size=voxel_size,
            keys=unique_keys,
            latents=latents,
            centers=centers,
            radii=radii,
            counts=counts,
            edge_tri_idx=all_edge_tri_idx,
            edge_keys=all_edge_keys,
        )

    def encode_scene_bank(self, triangles: torch.Tensor) -> SceneLatentBank:
        """Encode full scene into a hierarchical (fine + coarse) AABB sparse voxel latent bank."""
        if len(triangles) == 0:
            raise ValueError('empty scene unsupported in this prototype')
        tri_feats = self.encode_scene(triangles)
        centroids = triangles.mean(dim=1)
        tri_radii = torch.linalg.norm(triangles - centroids[:, None, :], dim=-1).max(dim=-1).values

        f_tri_idx, f_keys = _tri_voxel_assignments(triangles, self.fine_voxel_size)
        c_tri_idx, c_keys = _tri_voxel_assignments(triangles, self.coarse_voxel_size)

        fine_lvl = self._aggregate_voxels_from_edges(
            f_tri_idx, f_keys, centroids, tri_radii, tri_feats,
            self.fine_voxel_size, self.fine_voxel_proj, f_tri_idx, f_keys,
        )
        coarse_lvl = self._aggregate_voxels_from_edges(
            c_tri_idx, c_keys, centroids, tri_radii, tri_feats,
            self.coarse_voxel_size, self.coarse_voxel_proj, c_tri_idx, c_keys,
        )
        flat = triangles.reshape(-1, 3)
        return SceneLatentBank(
            triangles=triangles,
            fine=fine_lvl,
            coarse=coarse_lvl,
            scene_min=flat.min(dim=0).values,
            scene_max=flat.max(dim=0).values,
            last_reencoded_triangles=len(triangles),
            last_reencoded_voxels=len(fine_lvl.keys) + len(coarse_lvl.keys),
        )

    @staticmethod
    def _membership_mask(sorted_targets: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
        if len(sorted_targets) == 0 or len(queries) == 0:
            return torch.zeros(queries.shape, dtype=torch.bool, device=queries.device)
        pos = torch.searchsorted(sorted_targets, queries)
        clamped = pos.clamp_max(len(sorted_targets) - 1)
        return (pos < len(sorted_targets)) & (sorted_targets[clamped] == queries)

    def _patch_voxel_level(
        self,
        old_lvl: SparseVoxelLevel,
        affected_keys: torch.Tensor,
        re_tri_idx_local: torch.Tensor,
        re_edge_keys: torch.Tensor,
        sub_centroids: torch.Tensor,
        sub_radii: torch.Tensor,
        sub_feats: torch.Tensor,
        all_new_tri_idx: torch.Tensor,
        all_new_keys: torch.Tensor,
        voxel_proj: nn.Module,
    ) -> Tuple[SparseVoxelLevel, int]:
        keep_mask = ~self._membership_mask(affected_keys, old_lvl.keys)
        kept_keys = old_lvl.keys[keep_mask]
        kept_latents = old_lvl.latents[keep_mask]
        kept_centers = old_lvl.centers[keep_mask]
        kept_radii = old_lvl.radii[keep_mask]
        kept_counts = old_lvl.counts[keep_mask]

        if len(re_edge_keys) > 0:
            re_lvl = self._aggregate_voxels_from_edges(
                re_tri_idx_local, re_edge_keys, sub_centroids, sub_radii, sub_feats,
                old_lvl.voxel_size, voxel_proj, all_new_tri_idx, all_new_keys,
            )
            merged_keys = torch.cat([kept_keys, re_lvl.keys], dim=0)
            order = torch.argsort(merged_keys)
            lvl = SparseVoxelLevel(
                voxel_size=old_lvl.voxel_size,
                keys=merged_keys[order],
                latents=torch.cat([kept_latents, re_lvl.latents], dim=0)[order],
                centers=torch.cat([kept_centers, re_lvl.centers], dim=0)[order],
                radii=torch.cat([kept_radii, re_lvl.radii], dim=0)[order],
                counts=torch.cat([kept_counts, re_lvl.counts], dim=0)[order],
                edge_tri_idx=all_new_tri_idx,
                edge_keys=all_new_keys,
            )
            return lvl, len(re_lvl.keys)

        lvl = SparseVoxelLevel(
            voxel_size=old_lvl.voxel_size,
            keys=kept_keys,
            latents=kept_latents,
            centers=kept_centers,
            radii=kept_radii,
            counts=kept_counts,
            edge_tri_idx=all_new_tri_idx,
            edge_keys=all_new_keys,
        )
        return lvl, 0

    @staticmethod
    def _aabb_overlap_mask(
        triangles: torch.Tensor,
        aabb_min: torch.Tensor,
        aabb_max: torch.Tensor,
    ) -> torch.Tensor:
        tri_min = triangles.min(dim=1).values
        tri_max = triangles.max(dim=1).values
        return torch.all((tri_max >= aabb_min[None, :]) & (tri_min <= aabb_max[None, :]), dim=-1)

    def apply_local_edit(
        self,
        bank: SceneLatentBank,
        edited_triangles: torch.Tensor,
        *,
        changed_mask: Optional[torch.Tensor] = None,
        removed_mask: Optional[torch.Tensor] = None,
        added_mask: Optional[torch.Tensor] = None,
        edit_region_aabb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> SceneLatentBank:
        """True latent-space local edit Z_scene (+) delta_e for modified, added, or removed triangles.

        Re-encodes ONLY triangles residing in voxels affected by the edit and
        replaces only those voxel latents in the bank.
        """
        if len(edited_triangles) == 0:
            raise ValueError('empty scene unsupported in this prototype')

        m_old = len(bank.triangles)
        m_new = len(edited_triangles)
        dev = edited_triangles.device

        if removed_mask is not None or added_mask is not None:
            old_aff = removed_mask if removed_mask is not None else torch.zeros((m_old,), dtype=torch.bool, device=dev)
            if changed_mask is not None:
                old_aff = old_aff | changed_mask
            new_aff = added_mask if added_mask is not None else torch.zeros((m_new,), dtype=torch.bool, device=dev)
        elif edit_region_aabb is not None:
            bmin, bmax = edit_region_aabb
            old_aff = self._aabb_overlap_mask(bank.triangles, bmin.to(dev), bmax.to(dev))
            new_aff = self._aabb_overlap_mask(edited_triangles, bmin.to(dev), bmax.to(dev))
        elif m_new == m_old:
            old_aff = (edited_triangles != bank.triangles).any(dim=(-2, -1)) if changed_mask is None else changed_mask
            new_aff = old_aff
        elif m_new > m_old and torch.equal(edited_triangles[:m_old], bank.triangles):
            old_aff = torch.zeros((m_old,), dtype=torch.bool, device=dev)
            new_aff = torch.zeros((m_new,), dtype=torch.bool, device=dev)
            new_aff[m_old:] = True
        else:
            # Infer bounding box of non-matching triangles via set/row comparison
            old_flat = bank.triangles.reshape(m_old, 9)
            new_flat = edited_triangles.reshape(m_new, 9)
            # Hash each triangle row to 64-bit signature for O(M) diff detection
            weights = torch.arange(1, 10, dtype=torch.int64, device=dev) * 1315423911
            old_sig = (old_flat.view(torch.int32).to(torch.int64) * weights[None, :]).sum(dim=-1)
            new_sig = (new_flat.view(torch.int32).to(torch.int64) * weights[None, :]).sum(dim=-1)
            old_sorted, _ = torch.sort(old_sig)
            new_sorted, _ = torch.sort(new_sig)
            old_aff = ~self._membership_mask(new_sorted, old_sig)
            new_aff = ~self._membership_mask(old_sorted, new_sig)

        if not torch.any(old_aff) and not torch.any(new_aff):
            return SceneLatentBank(
                triangles=edited_triangles,
                fine=bank.fine,
                coarse=bank.coarse,
                scene_min=bank.scene_min,
                scene_max=bank.scene_max,
                last_reencoded_triangles=0,
                last_reencoded_voxels=0,
            )

        new_f_tri, new_f_keys = _tri_voxel_assignments(edited_triangles, self.fine_voxel_size)
        new_c_tri, new_c_keys = _tri_voxel_assignments(edited_triangles, self.coarse_voxel_size)

        old_f_edge_aff = old_aff[bank.fine.edge_tri_idx]
        old_c_edge_aff = old_aff[bank.coarse.edge_tri_idx]
        new_f_edge_aff = new_aff[new_f_tri]
        new_c_edge_aff = new_aff[new_c_tri]

        aff_fine = torch.unique(torch.cat([bank.fine.edge_keys[old_f_edge_aff], new_f_keys[new_f_edge_aff]]))
        aff_coarse = torch.unique(torch.cat([bank.coarse.edge_keys[old_c_edge_aff], new_c_keys[new_c_edge_aff]]))

        f_edge_in_aff = self._membership_mask(aff_fine, new_f_keys)
        c_edge_in_aff = self._membership_mask(aff_coarse, new_c_keys)

        tri_in_aff = torch.zeros((m_new,), dtype=torch.bool, device=dev)
        if torch.any(f_edge_in_aff):
            tri_in_aff[new_f_tri[f_edge_in_aff]] = True
        if torch.any(c_edge_in_aff):
            tri_in_aff[new_c_tri[c_edge_in_aff]] = True

        union_idx = torch.nonzero(tri_in_aff, as_tuple=False).squeeze(-1)
        if len(union_idx) > 0:
            sub_tris = edited_triangles[union_idx]
            sub_feats = self.encode_scene(sub_tris)
            sub_centroids = sub_tris.mean(dim=1)
            sub_radii = torch.linalg.norm(sub_tris - sub_centroids[:, None, :], dim=-1).max(dim=-1).values
            global_to_local = torch.full((m_new,), -1, dtype=torch.int64, device=dev)
            global_to_local[union_idx] = torch.arange(len(union_idx), dtype=torch.int64, device=dev)
            f_re_tri_local = global_to_local[new_f_tri[f_edge_in_aff]]
            c_re_tri_local = global_to_local[new_c_tri[c_edge_in_aff]]
        else:
            sub_feats = edited_triangles.new_zeros((0, self.hidden_dim))
            sub_centroids = edited_triangles.new_zeros((0, 3))
            sub_radii = edited_triangles.new_zeros((0,))
            f_re_tri_local = torch.empty((0,), dtype=torch.int64, device=dev)
            c_re_tri_local = torch.empty((0,), dtype=torch.int64, device=dev)

        fine_lvl, n_re_fine = self._patch_voxel_level(
            bank.fine, aff_fine, f_re_tri_local, new_f_keys[f_edge_in_aff],
            sub_centroids, sub_radii, sub_feats, new_f_tri, new_f_keys, self.fine_voxel_proj,
        )
        coarse_lvl, n_re_coarse = self._patch_voxel_level(
            bank.coarse, aff_coarse, c_re_tri_local, new_c_keys[c_edge_in_aff],
            sub_centroids, sub_radii, sub_feats, new_c_tri, new_c_keys, self.coarse_voxel_proj,
        )
        flat = edited_triangles.reshape(-1, 3)
        return SceneLatentBank(
            triangles=edited_triangles,
            fine=fine_lvl,
            coarse=coarse_lvl,
            scene_min=flat.min(dim=0).values,
            scene_max=flat.max(dim=0).values,
            last_reencoded_triangles=int(len(union_idx)),
            last_reencoded_voxels=int(n_re_fine + n_re_coarse),
        )

    @staticmethod
    def _ray_orthonormal_basis(d_unit: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        up = torch.tensor(
            [0.0, 1.0, 0.0] if abs(float(d_unit[1].item())) < 0.9 else [1.0, 0.0, 0.0],
            dtype=d_unit.dtype,
            device=d_unit.device,
        )
        u_vec = torch.linalg.cross(d_unit, up)
        u_vec = u_vec / torch.linalg.norm(u_vec).clamp_min(1e-12)
        v_vec = torch.linalg.cross(d_unit, u_vec)
        return u_vec, v_vec

    def _merge_dedup_candidates(
        self,
        tok_a: torch.Tensor,
        t_a: torch.Tensor,
        valid_a: torch.Tensor,
        tok_b: torch.Tensor,
        t_b: torch.Tensor,
        valid_b: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Merge two candidate buffers per receiver, deduplicate by token ID, keep top K_local."""
        tok = torch.cat([tok_a, tok_b], dim=1)
        t_val = torch.cat([t_a, t_b], dim=1)
        valid = torch.cat([valid_a, valid_b], dim=1)

        sort_key = torch.where(valid, t_val, torch.full_like(t_val, 1e9))
        order = torch.argsort(sort_key, dim=-1)
        tok_s = torch.gather(tok, 1, order)
        t_s = torch.gather(t_val, 1, order)
        valid_s = torch.gather(valid, 1, order)

        c_len = tok_s.shape[1]
        same_tok = (tok_s[:, :, None] == tok_s[:, None, :]) & (tok_s[:, :, None] >= 0)
        strict_lower = torch.tril(torch.ones((c_len, c_len), dtype=torch.bool, device=tok.device), diagonal=-1)
        is_dup = (same_tok & strict_lower[None, :, :]).any(dim=-1)
        uniq_valid = valid_s & (~is_dup)

        compact_key = torch.where(uniq_valid, t_s, torch.full_like(t_s, 1e9))
        compact_order = torch.argsort(compact_key, dim=-1)[:, : self.k_local]
        return (
            torch.gather(tok_s, 1, compact_order),
            torch.gather(t_s, 1, compact_order),
            torch.gather(uniq_valid, 1, compact_order),
        )

    def _traverse_level_streamed(
        self,
        lvl: SparseVoxelLevel,
        tok_offset: int,
        rec: torch.Tensor,
        d_unit: torch.Tensor,
        u_vec: torch.Tensor,
        v_vec: torch.Tensor,
        t_start: torch.Tensor,
        t_end: torch.Tensor,
        has_span: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Stream 3D cone-frustum voxel probes with step <= 0.75 * voxel_size."""
        n_pts = len(rec)
        dev = rec.device
        buf_tok = torch.full((n_pts, 0), -1, dtype=torch.int64, device=dev)
        buf_t = torch.zeros((n_pts, 0), dtype=rec.dtype, device=dev)
        buf_valid = torch.zeros((n_pts, 0), dtype=torch.bool, device=dev)

        if len(lvl.keys) == 0 or not bool(torch.any(has_span)):
            return buf_tok, buf_t, buf_valid

        span = (t_end - t_start).clamp_min(0.0)
        max_span = float(torch.where(has_span, span, torch.zeros_like(span)).max().item())
        step_size = 0.75 * lvl.voxel_size
        num_steps = max(2, int(math.ceil(max_span / step_size)) + 1)
        u_all = torch.linspace(0.0, 1.0, steps=num_steps, device=dev, dtype=rec.dtype)

        # 13 transverse cone-disk offsets per step: center + 4 half-radius + 8 full-ring (cardinal + diagonal)
        disk = torch.tensor(
            [[0.0, 0.0],
             [0.5, 0.0], [-0.5, 0.0], [0.0, 0.5], [0.0, -0.5],
             [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0],
             [0.7071, 0.7071], [0.7071, -0.7071], [-0.7071, 0.7071], [-0.7071, -0.7071]],
            dtype=rec.dtype,
            device=dev,
        )
        transverse_dir = disk[:, 0:1] * u_vec[None, :] + disk[:, 1:2] * v_vec[None, :]  # (9, 3)

        step_chunk = 32
        straddle_tol = 0.8660254 * lvl.voxel_size
        for s in range(0, num_steps, step_chunk):
            u_c = u_all[s : min(num_steps, s + step_chunk)]
            t_c = t_start[:, None] + span[:, None] * u_c[None, :]  # (N, Sc)
            r_c = self.cone_base_radius + t_c.clamp_min(0.0) * self.cone_tan_half_angle  # (N, Sc)
            center_pts = rec[:, None, :] + t_c[..., None] * d_unit[None, None, :]  # (N, Sc, 3)
            probes = center_pts[:, :, None, :] + r_c[:, :, None, None] * transverse_dir[None, None, :, :]
            probes_flat = probes.reshape(n_pts, -1, 3)

            q_keys, _ = _quantize_and_hash(probes_flat, lvl.voxel_size)
            idx = torch.searchsorted(lvl.keys, q_keys)
            idx_clamped = idx.clamp_max(len(lvl.keys) - 1)
            hit = has_span[:, None] & (idx < len(lvl.keys)) & (lvl.keys[idx_clamped] == q_keys)

            if not bool(torch.any(hit)):
                continue

            c_ctr = lvl.centers[idx_clamped]
            c_rad = lvl.radii[idx_clamped]
            dp = c_ctr - rec[:, None, :]
            t_par = (dp * d_unit[None, None, :]).sum(dim=-1)
            d_perp = torch.linalg.norm(dp - t_par[..., None] * d_unit[None, None, :], dim=-1)
            cone_r = self.cone_base_radius + t_par.clamp_min(0.0) * self.cone_tan_half_angle + c_rad
            in_frustum = hit & (t_par + straddle_tol > 0.0) & (d_perp <= cone_r)

            c_tok = torch.where(in_frustum, idx_clamped + tok_offset, -1)
            buf_tok, buf_t, buf_valid = self._merge_dedup_candidates(
                buf_tok, buf_t, buf_valid, c_tok, t_par, in_frustum
            )

        return buf_tok, buf_t, buf_valid

    def query_ray_cone_frustum(
        self,
        bank: SceneLatentBank,
        rec: torch.Tensor,
        sun_dir: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Query sparse voxel bank along solar ray-cone frustum, capped at K_local <= 64.

        Fine level covers near field (t <= near_field_radius) and coarse level
        covers far field (t > near_field_radius), eliminating double-counting and
        guaranteeing step spacing <= 0.75 * voxel_size.
        """
        n_pts = len(rec)
        dev = rec.device
        d_unit = sun_dir / torch.linalg.norm(sun_dir).clamp_min(1e-12)
        u_vec, v_vec = self._ray_orthonormal_basis(d_unit)

        margin = max(self.fine_voxel_size, self.coarse_voxel_size) + self.cone_base_radius
        bmin = bank.scene_min - margin
        bmax = bank.scene_max + margin
        safe_d = torch.where(d_unit.abs() > 1e-6, d_unit, torch.full_like(d_unit, 1e-6))
        inv_d = 1.0 / safe_d
        t0 = (bmin[None, :] - rec) * inv_d[None, :]
        t1 = (bmax[None, :] - rec) * inv_d[None, :]
        t_enter = torch.minimum(t0, t1).max(dim=-1).values.clamp_min(0.0)
        t_exit = torch.maximum(t0, t1).min(dim=-1).values
        has_box_hit = t_exit >= t_enter

        # Partition fine (t <= R_near) and coarse (t > R_near) to eliminate double-counting
        r_near = rec.new_full((n_pts,), self.near_field_radius)
        f_end = torch.minimum(t_exit, r_near)
        f_has = has_box_hit & (f_end >= t_enter)
        f_tok, f_t, f_valid = self._traverse_level_streamed(
            bank.fine, 0, rec, d_unit, u_vec, v_vec, t_enter, f_end, f_has
        )

        n_fine = len(bank.fine.keys)
        c_start = torch.maximum(t_enter, r_near)
        c_has = has_box_hit & (t_exit > c_start)
        c_tok, c_t, c_valid = self._traverse_level_streamed(
            bank.coarse, n_fine, rec, d_unit, u_vec, v_vec, c_start, t_exit, c_has
        )

        tok, _, valid_mask = self._merge_dedup_candidates(f_tok, f_t, f_valid, c_tok, c_t, c_valid)
        if tok.shape[1] < self.k_local:
            pad_k = self.k_local - tok.shape[1]
            tok = torch.cat([tok, torch.full((n_pts, pad_k), -1, dtype=torch.int64, device=dev)], dim=1)
            valid_mask = torch.cat([valid_mask, torch.zeros((n_pts, pad_k), dtype=torch.bool, device=dev)], dim=1)

        # Gather from unified [fine, coarse] bank
        all_lat = torch.cat([bank.fine.latents, bank.coarse.latents], dim=0)
        all_ctr = torch.cat([bank.fine.centers, bank.coarse.centers], dim=0)
        all_rad = torch.cat([bank.fine.radii, bank.coarse.radii], dim=0)
        all_lvl = torch.cat([
            rec.new_zeros((len(bank.fine.keys),)),
            rec.new_ones((len(bank.coarse.keys),)),
        ], dim=0)

        if len(all_lat) == 0:
            return (
                rec.new_zeros((n_pts, self.k_local, self.hidden_dim)),
                rec.new_zeros((n_pts, self.k_local, 5)),
                torch.zeros((n_pts, self.k_local), dtype=torch.bool, device=dev),
            )

        safe_tok = tok.clamp(min=0, max=len(all_lat) - 1)
        lat_out = all_lat[safe_tok]
        ctr_out = all_ctr[safe_tok]
        rad_out = all_rad[safe_tok]
        lvl_out = all_lvl[safe_tok]

        dp = ctr_out - rec[:, None, :]
        t_out = (dp * d_unit[None, None, :]).sum(dim=-1)
        dp_out = torch.linalg.norm(dp - t_out[..., None] * d_unit[None, None, :], dim=-1)
        cone_out = self.cone_base_radius + t_out.clamp_min(0.0) * self.cone_tan_half_angle + rad_out

        geom = torch.stack(
            [
                t_out / 100.0,
                dp_out / 100.0,
                rad_out / 100.0,
                dp_out / cone_out.clamp_min(1e-3),
                lvl_out,
            ],
            dim=-1,
        )
        geom = torch.where(valid_mask.unsqueeze(-1), geom, 0.0)
        lat_out = torch.where(valid_mask.unsqueeze(-1), lat_out, 0.0)
        return lat_out, geom, valid_mask

    def accumulate_transmittance(
        self,
        cand_latents: torch.Tensor,
        geom_feats: torch.Tensor,
        valid_mask: torch.Tensor,
        q_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Front-to-back differentiable transmittance in log-space with smooth log-odds."""
        k_len = cand_latents.shape[1]
        alpha_in = torch.cat(
            [cand_latents, q_emb[:, None, :].expand(-1, k_len, -1), geom_feats],
            dim=-1,
        )
        alpha = torch.sigmoid(self.opacity_head(alpha_in).squeeze(-1)) * valid_mask.to(cand_latents.dtype)

        eps = 1e-6
        log_one_minus_alpha = torch.log((1.0 - alpha).clamp(min=eps))
        log_t_cum = torch.cumsum(log_one_minus_alpha, dim=-1)
        zeros = torch.zeros_like(log_t_cum[..., :1])
        transmittance = torch.exp(torch.cat([zeros, log_t_cum[..., :-1]], dim=-1))

        log_t_final = log_t_cum[..., -1]
        t_final = torch.exp(log_t_final)
        t_logit = log_t_final - torch.log((1.0 - t_final) + eps)

        weights = transmittance * alpha
        context = (weights.unsqueeze(-1) * self.value(cand_latents)).sum(dim=-2)
        return context, t_final, t_logit

    def forward_from_bank(
        self,
        bank: SceneLatentBank,
        rec: torch.Tensor,
        suns: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Run ray-cone frustum query + transmittance rollout over a pre-encoded SceneLatentBank."""
        n_pts = len(rec)
        t_steps = len(suns)
        if n_pts == 0 or t_steps == 0:
            empty = rec.new_zeros((t_steps, n_pts))
            return {'logits': empty, 'probs': torch.sigmoid(empty), 'transmittance': torch.ones_like(empty)}

        chunk_n = max(1, min(n_pts, self.receiver_chunk_size))
        if chunk_n >= n_pts:
            return self._forward_bank_chunk(bank, rec, suns)

        chunks = [
            self._forward_bank_chunk(bank, rec[s : min(n_pts, s + chunk_n)], suns)
            for s in range(0, n_pts, chunk_n)
        ]
        return {
            'logits': torch.cat([c['logits'] for c in chunks], dim=1),
            'probs': torch.cat([c['probs'] for c in chunks], dim=1),
            'transmittance': torch.cat([c['transmittance'] for c in chunks], dim=1),
        }

    def _forward_bank_chunk(
        self,
        bank: SceneLatentBank,
        rec: torch.Tensor,
        suns: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        state = rec.new_zeros((len(rec), self.hidden_dim))
        logits = []
        t_finals = []
        for sun in suns:
            qin = torch.cat((rec / 100.0, sun.expand_as(rec)), dim=-1)
            q_emb = self.query(qin)
            cand_lat, geom, valid_mask = self.query_ray_cone_frustum(bank, rec, sun)
            context, t_final, t_logit = self.accumulate_transmittance(cand_lat, geom, valid_mask, q_emb)
            state = self.cell(torch.cat((context, qin), dim=-1), state)
            logits.append(t_logit + self.head(state).squeeze(-1))
            t_finals.append(t_final)
        out = torch.stack(logits)
        return {
            'logits': out,
            'probs': torch.sigmoid(out),
            'transmittance': torch.stack(t_finals),
        }

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        use_calibrated_temperature: bool = False,
    ) -> Dict[str, torch.Tensor]:
        rec, suns = batch['receivers_xyz'], batch['sun_dirs']
        bank = batch.get('scene_latent_bank')
        if bank is None:
            bank = self.encode_scene_bank(batch['scene_triangles_xyz'])
        return self.forward_from_bank(bank, rec, suns)
