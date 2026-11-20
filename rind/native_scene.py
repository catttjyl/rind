"""Build edited RIND inputs in the trajectory's native raster and candidate protocol."""
import numpy as np
import torch

from .fast_data import materialize_gpu_tokens_from_azimuth_bank
from .fullday_data import select_azimuth_sector_candidates, extract_receiver_local_features_and_crops, extract_exact_centered_receiver_crops
from .models import compute_tile_global_knn_pack_torch


def _roof_surface(triangles, wx, wz, *, face_indices=None):
    """Maximum interpolated face elevation at each covered raster centre."""
    surface = np.full((len(wz), len(wx)), -np.inf)
    if face_indices is not None:
        winners = np.full(surface.shape,-1,dtype=np.int64)
        weights = np.zeros((*surface.shape,3),dtype=np.float64)
    for i,face in enumerate(triangles):
        x, y, z = face.T
        cols = np.flatnonzero((wx >= x.min()) & (wx <= x.max()))
        rows = np.flatnonzero((wz >= z.min()) & (wz <= z.max()))
        if not len(rows) or not len(cols):
            continue
        det = (z[1]-z[2])*(x[0]-x[2]) + (x[2]-x[1])*(z[0]-z[2])
        if abs(det) < 1e-10:
            continue
        px, pz = np.meshgrid(wx[cols]-x[2], wz[rows]-z[2])
        a = ((z[1]-z[2])*px + (x[2]-x[1])*pz)/det
        b = ((z[2]-z[0])*px + (x[0]-x[2])*pz)/det
        c = 1-a-b
        covered = (a >= -1e-7) & (b >= -1e-7) & (c >= -1e-7)
        ix = np.ix_(rows,cols)
        height = a*y[0]+b*y[1]+c*y[2]
        better = covered & (height > surface[ix])
        if face_indices is not None:
            winners[ix] = np.where(better,face_indices[i],winners[ix])
            weights[ix] = np.where(better[...,None],np.stack((a,b,c),axis=-1),weights[ix])
        surface[ix] = np.where(better,height,surface[ix])
    return (surface,winners,weights) if face_indices is not None else surface


def native_scene_batch(traj, model, triangles, *, receivers=None, face_mask=None, first_k=None,
                       blind_faces=None, exclude_receivers=None, device="cpu", differentiable_triangles=None,
                       crop_size=None):
    """Public geometry only. Preserve native banks at identity, reselect after edits."""
    base = np.asarray(traj.scene_triangles_xyz, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.float64)
    if base.shape != triangles.shape:
        raise ValueError("native scene requires matching source faces in trajectory order")
    rec = traj.receivers_xyz if receivers is None else receivers
    rec = np.asarray(rec, dtype=np.float32)
    include = np.ones(len(base),bool) if face_mask is None else np.asarray(face_mask,bool)
    changed = np.any(triangles != base,axis=(1,2)) | ~include
    fields = np.array(traj.static_fields_8hw,copy=True)
    wx = np.asarray(traj.world_x_m)
    wz = np.asarray(traj.world_z_m)
    if blind_faces is not None:
        # Suppress both archived height fields everywhere. No rasterization residual
        # of a hidden roof may survive between its original and hypothesized heights.
        # Terrain and geometry validity are public; regenerate height from public base anchors.
        ids = np.asarray(traj.scene_source_instance_ids) if traj.scene_source_instance_ids is not None else None
        ground = np.zeros(len(base),bool) if traj.scene_ground_mask is None else traj.scene_ground_mask
        fields[1:3] = 0.
        if ids is None:
            from .inverse_design import assign_building_ids_from_triangles
            ids = assign_building_ids_from_triangles(triangles, ground_mask=ground)
        touches = ((triangles[:,:,0].max(1)>=wx.min()) & (triangles[:,:,0].min(1)<=wx.max()) &
                   (triangles[:,:,2].max(1)>=wz.min()) & (triangles[:,:,2].min(1)<=wz.max()))
        for bid in np.unique(ids[include & ~ground & touches]):
            select = (ids == bid) & include & ~ground
            roof = _roof_surface(triangles[select & touches], wx, wz)
            covered = np.isfinite(roof)
            height = (roof[covered] - triangles[select,:,1].min()) / 100.
            fields[2,covered] = np.maximum(fields[2,covered], height)
            fields[1,covered] = fields[0,covered] + fields[2,covered]
    elif changed.any():
        # Re-evaluate overlapping faces inside the edited footprint for both arms.
        # Clip the raster first, including unchanged higher roofs regardless of ID.
        xlo,xhi = base[changed,:,0].min(),base[changed,:,0].max()
        zlo,zhi = base[changed,:,2].min(),base[changed,:,2].max()
        cols = np.flatnonzero((wx>=xlo)&(wx<=xhi))
        rows = np.flatnonzero((wz>=zlo)&(wz<=zhi))
        if len(rows) and len(cols):
            near = (base[:,:,0].max(1)>=xlo)&(base[:,:,0].min(1)<=xhi)&(base[:,:,2].max(1)>=zlo)&(base[:,:,2].min(1)<=zhi)
            old = _roof_surface(base[near],wx[cols],wz[rows])
            new = _roof_surface(triangles[near & include],wx[cols],wz[rows])
            valid = np.isfinite(old)&np.isfinite(new)
            delta = np.zeros_like(old)
            delta[valid] = (new[valid]-old[valid])/100.
            ix = np.ix_(rows,cols)
            for ch in (1,2):
                patch = fields[ch][ix] + delta
                patch[np.isfinite(old)&~np.isfinite(new)] = 0.
                fields[ch][ix] = patch
    rc = traj.receiver_rc
    known = np.asarray(traj.geometry_known_hw).copy()
    if exclude_receivers is not None:
        hidden = rc[np.asarray(exclude_receivers,bool)]
        known[hidden[:,0],hidden[:,1]] = False
    local = extract_receiver_local_features_and_crops(fields, None, rc, wx, wz, geometry_known_hw=known)["local_features"]
    if crop_size is None:
        crop_size = getattr(model, "crop_size", None)
    crops = (extract_exact_centered_receiver_crops(fields, known, rc, crop_size=int(crop_size))
             if crop_size is not None else None)
    def tensor(v):
        return torch.as_tensor(np.require(v, requirements=["C","W"]), device=device)
    n_bins, k = traj.az_mask.shape[1:]
    bins = traj.azimuth_bin_idx.astype(np.int64)
    ray_count = None
    if getattr(traj, "ray_candidates", False):
        from .ray_candidates import RayCandidateScene
        scene = RayCandidateScene(triangles[include],traj.scene_opaque[include],traj.scene_source_ids[include])
        rel,op,mk,sid,ray_count = scene.query(rec,traj.sun_dirs,traj.bbox_margin_m,k,
                                               traj.receiver_source_triangle_ids)
        sid = sid.view(np.uint64)
        bins = np.arange(len(traj.sun_dirs))
    elif not changed.any() and blind_faces is None and np.array_equal(rec,traj.receivers_xyz):
        rel, op, mk, sid = traj.az_rel_triangles,traj.az_opaque,traj.az_mask,traj.az_source_ids
    else:
        rel,op,mk,sid = select_azimuth_sector_candidates(rec,triangles[include],
            scene_opaque=traj.scene_opaque[include],scene_source_triangle_ids=traj.scene_source_ids[include],
            num_azimuth_bins=n_bins,max_candidates=k,horizon_m=traj.bbox_margin_m,return_source_ids=True)
    rel_t = tensor(rel)
    if differentiable_triangles is not None and len(base):
        face_ids = np.asarray(traj.scene_source_ids)
        order = np.argsort(face_ids)
        indices = order[np.searchsorted(face_ids[order], np.where(mk,sid,face_ids[0]))]
        grad = differentiable_triangles - differentiable_triangles.detach()
        rel_t = rel_t + grad[tensor(indices)].to(rel_t) * tensor(mk)[...,None,None]
    if first_k is not None:
        mk = mk.copy()
        mk[...,int(first_k):] = False
    lf = tensor(local)
    crop = tensor(crops) if crops is not None else None
    field_tensor = tensor(fields)
    if differentiable_triangles is not None and differentiable_triangles.requires_grad:
        # The native raster is the upper envelope. Within a fixed visibility region
        # only the highest interpolated face contributes to the derivative.
        inside = include & (triangles[:,:,0].max(1)>=wx.min()) & (triangles[:,:,0].min(1)<=wx.max())
        inside &= (triangles[:,:,2].max(1)>=wz.min()) & (triangles[:,:,2].min(1)<=wz.max())
        _, winner, bary = _roof_surface(triangles[inside],wx,wz,face_indices=np.flatnonzero(inside))
        r,c = np.nonzero(winner>=0)
        h,w = winner.shape
        raster = field_tensor.new_zeros((h,w))
        if len(r):
            y = differentiable_triangles[tensor(winner[r,c]),:,1]
            dy = ((y-y.detach())*tensor(bary[r,c]).to(y)).sum(-1)/100.
            raster = raster.index_put((tensor(r),tensor(c)),dy.to(raster))
        # Archived raster values anchor the forward pass. Apply the same pixel
        # derivative to top elevation and above-ground height, with terrain fixed.
        ch = field_tensor.new_tensor([0,1,1,0,0,0,0,0])
        field_tensor = field_tensor + ch[:,None,None]*raster
        row,col = tensor(rc).T
        off = torch.arange(-4,5,device=device)
        dr = (row[:,None,None]+off[None,:,None]).clamp(0,h-1)
        dc = (col[:,None,None]+off[None,None,:]).clamp(0,w-1)
        lf = lf + torch.cat((raster[row,col,None]*ch,raster[dr,dc].mean((1,2))[:,None]*ch),dim=-1)
        if crop is not None:
            offsets = torch.arange(crop_size,device=device)-crop_size//2
            dr = row[:,None,None]+offsets[None,:,None]
            dc = col[:,None,None]+offsets[None,None,:]
            in_bounds = (dr>=0)&(dr<h)&(dc>=0)&(dc<w)
            patch = raster[dr.clamp(0,h-1),dc.clamp(0,w-1)]*in_bounds
            crop = crop + torch.cat((ch,ch.new_zeros(1)))[None,:,None,None]*patch[:,None]
    rec_t = tensor(rec)
    obs = tensor(known[rc[:,0],rc[:,1]])
    knn = compute_tile_global_knn_pack_torch(rec_t,obs,lf)
    batch = dict(az_rel_triangles=rel_t,az_mask=tensor(mk),az_opaque=tensor(op),
                 az_source_ids=tensor(sid.astype(np.int64)),azimuth_bin_idx=tensor(bins),
                 receiver_rc=tensor(rc),receiver_obs_mask=obs,receivers_xyz=rec_t,local_features=lf,
                 static_fields_8hw=field_tensor,geometry_known_hw=tensor(known),
                 sun_dirs=tensor(traj.sun_dirs),same_day_adjacent_mask=tensor(traj.same_day_adjacent_mask))
    if ray_count is not None:
        batch["ray_candidate_count"] = tensor(ray_count.T)
    if crop is not None:
        batch.update(receiver_crops=crop, spatial_patch_crops=crop)
    batch.update({"global_"+key:value for key,value in knn.items()})
    return materialize_gpu_tokens_from_azimuth_bank(batch)
