"""Read-only timing-head checks, independently reproducible for each gun sample."""
import sys,pathlib,json,argparse
import numpy as np
import torch
sys.path.insert(0,'/home/comet/davidz/ReconDev/ecal_clustering_gnn_trainer')
from evaluate_ecal_gnn import build_model_from_checkpoint
from train_ecal_gnn import hierarchical_model_kwargs,stats_tensors
from torch_geometric.loader import DataLoader
ap=argparse.ArgumentParser();ap.add_argument('species',choices=['mum','pim']);args=ap.parse_args()
torch.set_num_threads(1)
base=pathlib.Path('/home/comet/davidz/ReconDev/ReconData');setup=args.species+'_20to120MeV_10cmPlateGun_0To10deg'
ck=torch.load(base/'runs/ecal_clustering'/setup/'step1/checkpoints/best.pt',map_location='cpu',weights_only=False)
model=build_model_from_checkpoint(ck).eval();tm,ts=stats_tensors(ck['target_normalization'],torch.device('cpu'))
path=next((base/'datasets/ecal_clustering/processed/pyg_step1'/setup).glob('test_000_*.pt'))
print('Loading',path,flush=True);allg=torch.load(path,weights_only=False,map_location='cpu',mmap=True)
ids=np.linspace(0,len(allg)-1,4096,dtype=int);graphs=[allg[i] for i in ids];print('Selected',len(graphs),'of',len(allg),flush=True)
cache={};hook=model.step1_timing_head.net[1].register_forward_hook(lambda m,i,o:cache.update(hidden=o.detach(),pre=i[0].detach()))
parts={k:[] for k in ['prediction','target','truth_t0','pid','hidden','pre']}
with torch.no_grad():
 for b in DataLoader(graphs,batch_size=128,shuffle=False):
  o=model(b.x,b.edge_index,b.edge_attr,batch=b.batch,edge_weight=getattr(b,'edge_weight',None),**hierarchical_model_kwargs(b,'1',None,None,{},torch.device('cpu')))
  ix=o['target_global_window_index'];parts['prediction'].append((o['prediction']*ts+tm)[:,0]);parts['target'].append(b.y[:,0]);parts['truth_t0'].append(b.target_incident_time_ns.view(-1));parts['pid'].append(b.target_pid.view(-1));parts['hidden'].append(cache['hidden'][ix]);parts['pre'].append(cache['pre'][ix])
hook.remove();v={k:torch.cat(a).numpy().astype(float) for k,a in parts.items()};p=v['prediction'];t=v['target'];r=t-p;h=v['hidden'];pre=v['pre']
# Exact local head derivative wrt prompt_head_input; this does not include the encoder Jacobian.
w0=model.step1_timing_head.net[0].weight.detach().numpy();w1=model.step1_timing_head.net[2].weight.detach().numpy()[0]
grad=((pre>0)*w1)@w0*float(ts[0]);gn=np.linalg.norm(grad,axis=1)
bias_only=float((model.step1_timing_head.net[2].bias.detach()*ts[0]+tm[0])[0])
res=dict(species=args.species,epoch=ck['epoch'],sample_events=len(graphs),sample_objects=len(t),target_mean_ns=float(tm[0]),target_std_ns=float(ts[0]),bias_only_correction_ns=bias_only,subsets={})
subsets=[('all',np.ones(len(t),bool)),('small_dt',abs(t)<=1),('small_dt_early',(abs(t)<=1)&(v['truth_t0']<=30)),('small_dt_late',(abs(t)<=1)&(v['truth_t0']>30)),('large_dt',t>30)]
for pid in np.unique(v['pid']):subsets.append((f'pid_{int(pid)}_small_dt',(abs(t)<=1)&(v['pid']==pid)))
for name,sel in subsets:
 n=int(sel.sum())
 if not n:continue
 q=np.quantile(r[sel],[.16,.5,.84]);active=(h[sel]>0).mean(0)
 res['subsets'][name]=dict(n=n,all_hidden_zero_fraction=float((h[sel]==0).all(1).mean()),near_bias_only_fraction=float((abs(p[sel]-bias_only)<.001).mean()),active_units=int((active>0).sum()),mean_active_units=float((h[sel]>0).sum(1).mean()),local_head_gradient_norm_median=float(np.median(gn[sel])),pred_std=float(p[sel].std()),pred_central68_halfwidth=float((np.quantile(p[sel],.84)-np.quantile(p[sel],.16))/2),residual_mean=float(r[sel].mean()),residual_std=float(r[sel].std()),residual_mae=float(abs(r[sel]).mean()),residual_median=float(q[1]),residual_central68_halfwidth=float((q[2]-q[0])/2),over_50ns_fraction=float((abs(r[sel])>50).mean()))
small_active=(h[(abs(t)<=1)&(v['truth_t0']<=30)]>0).mean(0)>.01
large_active=(h[t>30]>0).mean(0)>.01
res['unit_overlap']=dict(small_active_over1pct=int(small_active.sum()),large_active_over1pct=int(large_active.sum()),shared_active_over1pct=int((small_active&large_active).sum()))
print(json.dumps(res,indent=2),flush=True)
json.dump(res,open(pathlib.Path(__file__).with_name(args.species+'_timing_head.json'),'w'),indent=2)
