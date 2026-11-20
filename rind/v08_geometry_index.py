"""Source-only conservative ray/AABB candidates. Never computes visibility labels."""
from dataclasses import dataclass
from collections import OrderedDict
import hashlib
import numpy as np

_INDEX_ARRAY_NAMES=('_triangles','_opaque','_refs','_reasons','_low','_high','_links','_order','_face_low','_face_high')


@dataclass(frozen=True)
class Candidates:
    ids: np.ndarray
    available: bool
    reason: str
    visited_nodes: int
    candidate_count_lower_bound: int


def intersections(q, s, low, high, horizon):
    lower=np.zeros(len(low));upper=np.full(len(low),horizon)
    for axis in range(3):
        if s[axis]==0:
            ok=(q[axis]>=low[:,axis])&(q[axis]<=high[:,axis])
            upper[~ok]=-1
        else:
            with np.errstate(over='ignore'):
                a=(low[:,axis]-q[axis])/s[axis];b=(high[:,axis]-q[axis])/s[axis]
            lower=np.maximum(lower,np.minimum(a,b));upper=np.minimum(upper,np.maximum(a,b))
    return lower<=upper


def classify_faces(triangles):
    """Original per-face numerical status, without building an index."""
    t=np.asarray(triangles,dtype=np.float64)
    if t.ndim!=3 or t.shape[1:]!=(3,3):raise ValueError('Original triangle blocks required')
    finite=np.isfinite(t).all((1,2));area=np.full(len(t),np.nan)
    with np.errstate(over='ignore',invalid='ignore'):
        v=t[finite];area[finite]=np.linalg.norm(np.cross(v[:,1]-v[:,0],v[:,2]-v[:,0]),axis=1)
    finite&=np.isfinite(area);reasons=np.full(len(t),3,np.uint8)
    reasons[finite&(area==0)]=1;reasons[finite&(area>0)&(area<=1e-10)]=2;reasons[finite&(area>1e-10)]=0
    return reasons


class ConservativeRayIndex:
    def __init__(self,triangles,opaque,refs,*,leaf_size=32,bounds_margin_m=1e-7):
        t=np.asarray(triangles);o=np.asarray(opaque);r=np.asarray(refs)
        if t.ndim!=3 or t.shape[1:]!=(3,3) or o.shape!=(len(t),) or o.dtype!=bool or r.shape!=(len(t),) or r.dtype!=np.uint64:
            raise ValueError('Original triangles, explicit bool materials and uint64 IDs required')
        if len(np.unique(r))!=len(r) or not 1<=leaf_size<=128 or not np.isfinite(bounds_margin_m) or bounds_margin_m<0:
            raise ValueError('Unique source IDs and conservative finite bound policy required')
        self._triangles=np.array(t,np.float64,copy=True);self._opaque=o.copy();self._refs=r.copy()
        self._reasons=classify_faces(self._triangles)
        self.global_unresolved=bool((self._reasons==3).any())
        ids=np.flatnonzero((self._reasons==0)|(self._reasons==2));bounds=self._triangles[ids]
        low=np.nextafter(bounds.min(1)-bounds_margin_m,-np.inf) if len(ids) else np.empty((0,3))
        high=np.nextafter(bounds.max(1)+bounds_margin_m,np.inf) if len(ids) else np.empty((0,3))
        centres=(low+high)/2;order=np.arange(len(ids));nodes=[];leaves=[]
        def build(indices):
            n=len(nodes);nodes.append(None);lo=low[indices].min(0);hi=high[indices].max(0)
            if len(indices)<=leaf_size:
                start=len(leaves);leaves.extend(ids[indices].tolist());nodes[n]=(lo,hi,-1,-1,start,len(leaves))
            else:
                axis=int(np.ptp(centres[indices],axis=0).argmax());split=np.argsort(centres[indices,axis],kind='stable');mid=len(indices)//2
                left=build(indices[split[:mid]]);right=build(indices[split[mid:]]);nodes[n]=(lo,hi,left,right,-1,-1)
            return n
        self.root=build(order) if len(order) else -1
        self._low=np.array([n[0] for n in nodes]).reshape(-1,3);self._high=np.array([n[1] for n in nodes]).reshape(-1,3)
        self._links=np.array([n[2:] for n in nodes],np.int64).reshape(-1,4);self._order=np.array(leaves,np.int64)
        self._face_low=np.zeros((len(t),3));self._face_high=np.zeros((len(t),3));self._face_low[ids]=low;self._face_high[ids]=high
        self._arrays=(self._triangles,self._opaque,self._refs,self._reasons,self._low,self._high,self._links,self._order,self._face_low,self._face_high)
        for a in self._arrays:a.flags.writeable=False
        self._sealed=None
        self.identity=self._digest()
        self._token_cache=OrderedDict();self._token_bytes=0;self.token_cache_limit=64*1024**2
        self.audit=dict(original_faces=len(t),valid_faces=int((self._reasons==0).sum()),zero_area=int((self._reasons==1).sum()),
                        unresolved_faces=int((self._reasons>=2).sum()),threshold_cross_norm_m2=1e-10,bvh_nodes=len(nodes),
                        bytes=sum(a.nbytes for a in self._arrays),world_complete=False)

    def _digest(self):
        h=hashlib.sha256()
        h.update(str((self.root,self.global_unresolved)).encode())
        for a in self._arrays:
            h.update(a.dtype.str.encode());h.update(str(a.shape).encode())
            if a.nbytes:h.update(memoryview(a).cast('B'))
        return h.hexdigest()

    def assert_intact(self):
        if self._sealed is not None:
            root,unresolved,identity,bindings=self._sealed
            if (self.root!=root or self.global_unresolved!=unresolved or self.identity!=identity
                    or not isinstance(self._arrays,tuple) or len(self._arrays)!=len(bindings)):
                raise ValueError('Bound source/index mutated')
            for current,(name,array,payload,shape,dtype,strides) in zip(self._arrays,bindings):
                if (current is not array or getattr(self,name,None) is not array
                        or array.base is not payload or array.flags.writeable
                        or array.shape!=shape or array.dtype!=dtype or array.strides!=strides):
                    raise ValueError('Bound source/index mutated')
            return
        if any(a.flags.writeable for a in self._arrays) or self._digest()!=self.identity:
            raise ValueError('Bound source/index mutated')

    def seal(self):
        """Verify once and own immutable bytes for repeated world queries.

        Unlike a writeable=False owning ndarray, a bytes-backed array cannot
        have writes re-enabled. Header, buffer, attribute and index bindings
        are still checked on every query boundary. Legacy unsealed consumers
        retain their original full-payload checksum and mutation detection.
        """
        self.assert_intact()
        if self._sealed is not None:return self
        if len(self._arrays)!=len(_INDEX_ARRAY_NAMES) or any(getattr(self,n,None) is not a for n,a in zip(_INDEX_ARRAY_NAMES,self._arrays)):
            raise ValueError('Source/index attribute binding differs before sealing')
        bindings=[]
        for name,array in zip(_INDEX_ARRAY_NAMES,self._arrays):
            payload=array.tobytes(order='C')
            frozen=np.ndarray(array.shape,dtype=array.dtype,buffer=payload)
            bindings.append((name,frozen,payload,frozen.shape,frozen.dtype,frozen.strides))
        for name,array,*_ in bindings:setattr(self,name,array)
        self._arrays=tuple(row[1] for row in bindings)
        # Recheck the exact copied snapshot before certifying immutability.
        # A source change during construction must never receive a new identity.
        if self._digest()!=self.identity:raise ValueError('Source/index changed while sealing')
        self._sealed=(self.root,self.global_unresolved,self.identity,tuple(bindings))
        return self

    def query(self,origin,sun,horizon,*,max_candidates=1024,max_nodes=50000):
        q=np.asarray(origin,float);s=np.asarray(sun,float)
        if q.shape!=(3,) or s.shape!=(3,) or not np.isfinite(q).all() or not np.isfinite(s).all() or s[1]<=0 or abs(np.linalg.norm(s)-1)>1e-3:
            raise ValueError('Finite original origin and upward unit Sun required')
        if not np.isfinite(horizon) or horizon<=0 or not 1<=max_candidates<=65536 or max_nodes<1:
            raise ValueError('Explicit bounded query work required')
        s=s/np.linalg.norm(s);selected=[];stack=[self.root] if self.root>=0 else [];visited=0
        def result(available,reason):
            ids=np.array(sorted(selected),np.int64) if available else np.empty(0,np.int64);ids.flags.writeable=False
            return Candidates(ids,available,reason,visited,len(selected))
        if self.global_unresolved:return result(False,'unresolved_geometry')
        while stack:
            n=stack.pop();visited+=1
            if visited>max_nodes:return result(False,'requires_more_index_work')
            if not intersections(q,s,self._low[n:n+1],self._high[n:n+1],horizon)[0]:continue
            left,right,start,stop=self._links[n]
            if left>=0:stack.extend((int(right),int(left)));continue
            ids=self._order[start:stop];hit=intersections(q,s,self._face_low[ids],self._face_high[ids],horizon);ids=ids[hit]
            if (self._reasons[ids]==2).any():return result(False,'unresolved_geometry')
            selected.extend(ids.tolist())
            if len(selected)>max_candidates:return result(False,'requires_more_candidates')
        return result(True,'loaded_source_candidates')

    def features(self,ids,origin,sun,*,relationship=True,scale_m=1000.):
        ids=np.asarray(ids,np.int64);q=np.asarray(origin,float);s=np.asarray(sun,float);s=s/np.linalg.norm(s)
        t=self._triangles[ids]-q;a=t[:,1]-t[:,0];b=t[:,2]-t[:,0];normal=np.cross(a,b);area=np.linalg.norm(normal,axis=1)
        edges=np.stack((np.linalg.norm(a,axis=1),np.linalg.norm(b,axis=1),np.linalg.norm(t[:,2]-t[:,1],axis=1)),1)
        logscale=np.log1p(scale_m)
        if relationship:
            along=(t*s).sum(-1);radial=np.linalg.norm(t-along[...,None]*s,axis=-1)
            lead=np.concatenate((np.sign(along)*np.log1p(np.abs(along))/logscale,np.log1p(radial)/logscale,np.log1p(edges)/logscale),1)
            incidence=np.abs((normal*s).sum(-1))/np.maximum(area,1e-20)
        else:
            lead=(np.sign(t)*np.log1p(np.abs(t))/logscale).reshape(-1,9);incidence=np.log1p(edges.mean(1))/logscale
        features=np.concatenate((lead,incidence[:,None],self._opaque[ids,None],np.log1p(area)[:,None]/(2*logscale),np.full((len(ids),1),s[1])),1)
        if not np.isfinite(features).all():raise ValueError('Nonfinite source-only relationship features')
        return features.astype(np.float32)

    def query_features(self,origin,sun,horizon,*,max_candidates=1024,max_nodes=50000,relationship=True,scale_m=1000.,include_features=True):
        q=np.asarray(origin,np.float64);s=np.asarray(sun,np.float64)
        key=(self.identity,q.tobytes(),s.tobytes(),float(horizon).hex(),max_candidates,max_nodes,bool(relationship),float(scale_m).hex(),bool(include_features))
        if key in self._token_cache:
            candidate,features,checksum,size=self._token_cache.pop(key)
            if features.flags.writeable or hashlib.sha256(features.tobytes()).hexdigest()!=checksum:raise ValueError('Source-only feature cache mutated')
            self._token_cache[key]=(candidate,features,checksum,size);return candidate,features
        candidate=self.query(q,s,horizon,max_candidates=max_candidates,max_nodes=max_nodes)
        # canonical world tokens do not consume these legacy 13 dimensions.
        # Retain the candidate cache even when their computation is omitted.
        features=self.features(candidate.ids,q,s,relationship=relationship,scale_m=scale_m) if candidate.available and include_features else np.empty((0,13),np.float32)
        features.flags.writeable=False;size=features.nbytes+candidate.ids.nbytes+256
        if size<=self.token_cache_limit:
            while self._token_cache and self._token_bytes+size>self.token_cache_limit:
                _,old=self._token_cache.popitem(last=False);self._token_bytes-=old[-1]
            self._token_cache[key]=(candidate,features,hashlib.sha256(features.tobytes()).hexdigest(),size);self._token_bytes+=size
        return candidate,features
