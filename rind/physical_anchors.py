"""Observation-conditional change@k benchmarks on timestamped public-context exports."""
import argparse
import json
from pathlib import Path

import numpy as np


def local_dsm_sunlit(fields, known, receiver_rc, receivers_xyz, suns, x_axis, z_axis):
    """Nearest-cell DSM raymarch up to 100 horizontal meters within the available raster.

    Relative heights cancel the common elevation datum. Uncovered/out-of-map
    cells provide no local blocker, matching the limited local-geometry baseline.
    """
    terrain, roof = fields[0]*100., fields[1]*100.
    surface = np.where(fields[4] > .5, np.maximum(terrain,roof),terrain)
    dx, dz = x_axis[1]-x_axis[0], z_axis[1]-z_axis[0]
    spacing = min(abs(dx),abs(dz))*.5
    distances = np.arange(spacing,100.+1e-10,spacing)
    rows, cols = receiver_rc.T
    bare = fields[4,rows,cols] < .5
    if not bare.any():
        raise ValueError("No bare terrain receiver to establish the public raster elevation datum")
    estimates = receivers_xyz[bare,1] - .05 - terrain[rows[bare],cols[bare]]
    if np.ptp(estimates) > 1e-3:
        raise ValueError("Receiver and raster terrain elevations are not in a common datum")
    origin_y = receivers_xyz[:,1] - np.median(estimates)
    pred = np.ones((len(suns),len(rows)),bool)
    for i,s in enumerate(suns):
        horizontal = np.hypot(s[0],s[2])
        if horizontal == 0:
            continue
        r = rows[:,None] + np.rint(distances*s[2]/(horizontal*dz)).astype(int)
        c = cols[:,None] + np.rint(distances*s[0]/(horizontal*dx)).astype(int)
        inside = (r>=0)&(c>=0)&(r<known.shape[0])&(c<known.shape[1])
        r = r.clip(0,known.shape[0]-1)
        c = c.clip(0,known.shape[1]-1)
        height = origin_y[:,None] + distances*s[1]/horizontal
        pred[i] = ~(inside & known[r,c] & (surface[r,c] >= height)).any(1)
    return pred


def score(table):
    """Rows are (target event, binary change score). Group ties when calculating AP."""
    tp,fp,fn,tn = np.asarray(table,dtype=int).sum(axis=0)
    total = tp+fp+fn+tn
    positives = tp+fn
    precision = tp/max(1,tp+fp)
    ap = (precision*tp + positives/max(1,total)*fn)/positives if positives else None
    return dict(pairs=int(total),tp=int(tp),fp=int(fp),fn=int(fn),tn=int(tn),change_ap=ap,
                f1=float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else None)


def compare(path, lead):
    rows = []
    with np.load(path,allow_pickle=False) as f:
        meta = json.loads(f['metadata'].item())
        memory = {}
        for clip,ident in sorted(meta['identities'].items(),key=lambda x:x[1]['week_index']):
            key = clip+'__'
            sun,rc,truth,valid,ts = (f[key+k] for k in ('sun_dirs','receiver_rc','truth','valid_mask','timestamps_unix_s'))
            dsm = local_dsm_sunlit(f['static_fields_8hw'],f['geometry_known_hw'],rc,f[key+'receivers_xyz'],sun,f['world_x_m'],f['world_z_m'])
            nearest = np.full(truth.shape,-1,dtype=int)
            for n,pos in enumerate(rc):
                past = memory.get(tuple(pos),[])
                if past:
                    angles = np.concatenate([a for a,y in past])
                    labels = np.concatenate([y for a,y in past])
                    nearest[:,n] = labels[np.argmax(sun @ angles.T,axis=1)]
            t = len(ts)-lead
            if t>0:
                ok = np.logical_and.reduce([valid[j:j+t] for j in range(lead+1)])
                cadence = np.logical_and.reduce([np.diff(ts)[j:j+t]==120 for j in range(lead)])
                good_sun = np.logical_and.reduce([sun[j:j+t,1]>=np.sin(np.deg2rad(5.)) for j in range(lead+1)])
                ok &= (cadence & good_sun)[:,None]
                common = ok & (nearest[lead:]>=0)
                y = truth[lead:] != truth[:t]
                for method,forecast,support in (("persistence",truth[:t],ok),("dsm100",dsm[lead:],ok),
                                                ("past_week_nearest_sun",nearest[lead:],common)):
                    for domain,keep in (("available",support),("paired",common)):
                        pred = forecast != truth[:t]
                        counts = [int((keep & a & b).sum()) for a,b in ((pred,y),(pred,~y),(~pred,y),(~pred,~y))]
                        rows.append(dict(tile=meta["tile"],week=ident["week_index"],method=method,domain=domain,counts=counts))
            # Only previously completed weeks can be licensed as an observation history.
            # A second clip of the same week must never enter the nearest-week reference pool.
            for n,pos in enumerate(rc):
                v = valid[:,n]
                if v.any():
                    memory.setdefault(tuple(pos),[]).append((sun[v],truth[v,n]))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("context",type=Path,nargs="+")
    parser.add_argument("--lead",type=int,default=1)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    if args.lead < 1:
        parser.error("--lead must be positive")
    rows = []
    seen = set()
    for path in args.context:
        with np.load(path) as f:
            m = json.loads(f["metadata"].item())
        weeks = [x["week_index"] for x in m["identities"].values()]
        if len(set(weeks))!=len(weeks) or m["tile"] in seen:
            raise ValueError("Supply one export per tile and one clip per week")
        seen.add(m["tile"])
        rows.extend(compare(path,args.lead))
    result = {}
    for method,domain in sorted({(r["method"],r["domain"]) for r in rows}):
        table = np.array([r["counts"] for r in rows if r["method"]==method and r["domain"]==domain])
        point = score(table)
        rng = np.random.default_rng(971)
        vals = [score(table[rng.integers(len(table),size=len(table))])["change_ap"] for _ in range(1000)]
        valid = [v for v in vals if v is not None]
        point.update(tile_week_clusters=len(table),bootstrap_defined=len(valid),
                     change_ap_ci95=np.quantile(valid,[.025,.975]).tolist() if valid else None)
        result[method+"/"+domain] = point
    with args.output.open("x") as out:
        json.dump(dict(lead=args.lead,min_altitude_deg=5,summary=result,tile_weeks=rows),out,indent=2,allow_nan=False)


if __name__ == "__main__":
    main()
