"""Compare timing predictions on one deterministic sample; no training or cuts to production data."""
import sys,pathlib,json
import numpy as np
import torch
sys.path.insert(0,'/home/comet/davidz/ReconDev/ecal_clustering_gnn_trainer')
from evaluate_ecal_gnn import build_model_from_checkpoint
from train_ecal_gnn import hierarchical_model_kwargs,stats_tensors
from torch_geometric.loader import DataLoader
torch.set_num_threads(1)
base=pathlib.Path('/home/comet/davidz/ReconDev/ReconData')
setup='mum_20to120MeV_10cmPlateGun_0To10deg'
path=next((base/'datasets/ecal_clustering/processed/pyg_step1'/setup).glob('test_000_*.pt'))
print('Loading',path,flush=True)
allg=torch.load(path,weights_only=False,map_location='cpu',mmap=True)
ids=np.linspace(0,len(allg)-1,4096,dtype=int);graphs=[allg[i] for i in ids]
print('Loaded',len(allg),'selected',len(graphs),flush=True)
results=[]
for ep in [5,15,25,35,48,50]:
 ck=torch.load(base/'runs/ecal_clustering'/setup/f'step1/checkpoints/epoch_{ep:03d}.pt',map_location='cpu',weights_only=False)
 model=build_model_from_checkpoint(ck).eval();tm,ts=stats_tensors(ck['target_normalization'],torch.device('cpu'))
 preds=[];truths=[];hidden=[]
 hook=model.step1_timing_head.net[1].register_forward_hook(lambda m,i,o:hidden.append(o.detach()))
 with torch.no_grad():
  for b in DataLoader(graphs,batch_size=128,shuffle=False):
   out=model(b.x,b.edge_index,b.edge_attr,batch=b.batch,edge_weight=getattr(b,'edge_weight',None),**hierarchical_model_kwargs(b,'1',None,None,{},torch.device('cpu')))
   preds.append((out['prediction']*ts+tm)[:,0]);truths.append(b.y[:,0])
 hook.remove();p=torch.cat(preds).numpy().astype(float);t=torch.cat(truths).numpy().astype(float);h=torch.cat(hidden);r=t-p
 result={'epoch':ep,'n':len(t),'target_std_ns':float(ts[0]),'hidden_all_zero_fraction':float((h==0).all(1).float().mean()),'active_units':int((h.max(0).values>0).sum()),'subsets':{}}
 for name,sel in [('all',np.ones(len(t),dtype=bool)),('target_abs_le_1ns',abs(t)<=1),('target_abs_gt_30ns',abs(t)>30)]:
  x=r[sel];q=np.quantile(x,[.16,.5,.84]);result['subsets'][name]={'n':len(x),'bias':float(x.mean()),'std':float(x.std()),'mae':float(abs(x).mean()),'rms':float(np.sqrt(np.mean(x*x))),'central68_halfwidth':float((q[2]-q[0])/2),'median':float(q[1]),'fraction_abs_gt_50ns':float(np.mean(abs(x)>50))}
 results.append(result);print(json.dumps(result),flush=True)
json.dump(results,open(pathlib.Path(__file__).with_suffix('.json'),'w'),indent=2)
