"""Same regional model, with a separate latent random stream per event interval.

Refreshing a later interval preserves earlier history. All event seeds have
the same independent uniform prior. This is a computational parameterization,
not an added observation or change to the demographic equations.
"""
import numpy as np
from .fast import JitTree,base,needed_survival,site_factor,base_collapsed,positive_poisson_jit,njit


@njit(cache=True)
def event_branch(n_hist,ever_hist,stat_hist,begin,events,seeds,K,wp,wi,ww,gamma,cp,ci,qp,qi,qw,
                 jump=True,max_updates=20000):
    n=n_hist[begin].copy();ever=ever_hist[begin].copy();present=n>0
    updates=stat_hist[begin,0];skipped=stat_hist[begin,1];jumps=stat_hist[begin,2]
    if events-begin>len(n)-ever.sum():return n,np.array([begin,updates,skipped,jumps,1],dtype=np.int64)
    for ticks in range(begin,events):
        np.random.seed(seeds[ticks]);stable=0;warm=np.ones(len(n));finished=False
        while not finished:
            if n.sum()==0 or ever.all():return n,np.array([ticks,updates,skipped,jumps,1],dtype=np.int64)
            if updates>=max_updates:raise RuntimeError('Event branch update limit, not zero likelihood')
            new=base_collapsed(n/K,K,wp,wi,gamma);sf=site_factor(n,K,wp,wi,ww)
            pi=needed_survival(sf,new,present,K,cp,ci,qp,qi,qw,warm);warm=pi
            lam=np.zeros(len(n));predicted=np.zeros(len(n));stationary=True
            for d in range(len(n)):
                if present[d]:
                    predicted[d]=K[d]*new[d]
                    if abs(predicted[d]-n[d])>=1e-9 or 0<predicted[d]<1:stationary=False
                else:lam[d]=K[d]*new[d]*pi[d]
            hit=np.zeros(len(n),dtype=np.bool_)
            if jump and stable>=10 and stationary:
                total=lam.sum()
                if total<=0:return n,np.array([ticks,updates,skipped,jumps,1],dtype=np.int64)
                waiting=np.random.geometric(-np.expm1(-total));skipped+=waiting;jumps+=1
                count=positive_poisson_jit(total)
                for _ in range(count):
                    u=np.random.random()*total;cumulative=0.
                    for d in range(len(n)):
                        cumulative+=lam[d]
                        if u<cumulative:hit[d]=True;break
            else:
                for d in range(len(n)):
                    if lam[d]>0:hit[d]=np.random.random() < -np.expm1(-lam[d])
                updates+=1
            for d in range(len(n)):
                if hit[d]:predicted[d]=min(1/max(pi[d],1e-3),K[d]);present[d]=True
                if predicted[d]<1:predicted[d]=0.;present[d]=False
                if predicted[d]>0 and not ever[d]:ever[d]=True;finished=True
            if not finished:stable=stable+1 if stationary else 0
            n=predicted
        n_hist[ticks+1]=n;ever_hist[ticks+1]=ever
        stat_hist[ticks+1]=np.array([updates,skipped,jumps],dtype=np.int64)
    return n,np.array([events,updates,skipped,jumps,0],dtype=np.int64)


class EventTree(JitTree):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.offset=np.r_[0,np.cumsum(self.data['times'])].astype(int)
        self.seed_count=int(self.offset[-1]+4)

    def simulate_branch(self,n,edge,seeds,cache=None,begin=0):
        events=int(self.data['times'][edge]);M=len(n)
        if cache is None:
            nh=np.zeros((events+1,M));eh=np.zeros((events+1,M),dtype=bool)
            sh=np.zeros((events+1,3),dtype=np.int64);nh[0]=n;eh[0]=n>0
        else:
            nh,eh,sh=(x.copy() for x in cache)
        m=self.landscape;w=m.W;q=m.clonal
        end,stats=event_branch(nh,eh,sh,begin,events,seeds,m.K,w.indptr,w.indices,w.data,
            m.gamma,self.cp,self.ci,q.indptr,q.indices,q.data,self.jump)
        return end,stats,(nh,eh,sh)

    def run_noise(self,mask,seeds,previous=None,changed=None):
        splitseeds=seeds[self.offset[-1]:]
        if previous is None:
            starts=np.zeros((8,len(self.K)));ends=np.zeros_like(starts)
            stats=np.zeros((8,5),dtype=np.int64);caches=[None]*8
        else:
            starts=previous[0].copy();ends=previous[1].copy();stats=previous[2].copy();caches=list(previous[4])
        event_begin=0
        if changed is None:
            a,b=base._old.seeded_split(self.root(mask),int(splitseeds[0]),self.data['cells'],self.neighbours,self.order)
            starts[6],starts[7]=a,b;todo=set(range(8))
        else:
            edge,event_begin=changed
            if edge<8:
                todo=set({0:[0],1:[1],2:[2,0,1],3:[3],4:[4],5:[5],6:[6,2,0,1,3],7:[7,4,5]}[edge])
            else:
                child={8:(6,7),9:(2,3),10:(0,1),11:(4,5)}[edge]
                parent=self.root(mask) if edge==8 else ends[{9:6,10:2,11:7}[edge]]
                a,b=base._old.seeded_split(parent,int(splitseeds[edge-8]),self.data['cells'],self.neighbours,self.order)
                starts[child[0]],starts[child[1]]=a,b
                todo={8:set(range(8)),9:{2,0,1,3},10:{0,1},11:{4,5}}[edge]
        for e in (6,2,0,1,3,7,4,5):
            if e not in todo:continue
            begin=event_begin if changed is not None and changed[0]==e else 0
            reuse=caches[e] if begin else None
            ends[e],stats[e],caches[e]=self.simulate_branch(starts[e],e,seeds[self.offset[e]:self.offset[e+1]],reuse,begin)
            if stats[e,4]:return starts,ends,stats,1,caches
            children={6:(2,3,9),2:(0,1,10),7:(4,5,11)}.get(e)
            if children:
                left,right,s=children
                a,b=base._old.seeded_split(ends[e],int(splitseeds[s-8]),self.data['cells'],self.neighbours,self.order)
                starts[left],starts[right]=a,b
        return starts,ends,stats,0,caches

    def manifest(self):
        import hashlib
        from pathlib import Path
        m=super().manifest();m.update(innovation_parameterization='one independent uniform seed per event interval plus four splits',
            innovation_seeds=self.seed_count,prefix_preserved_when_later_event_seed_changes=True,
            sha256_event_adapter=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        return m


