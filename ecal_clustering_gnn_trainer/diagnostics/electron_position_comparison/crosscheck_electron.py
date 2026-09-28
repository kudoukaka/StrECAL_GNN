import sys,pathlib,json,gc,time
import torch,numpy as np
sys.path.insert(0,'/home/comet/davidz/ReconDev/ecal_clustering_gnn_trainer')
from evaluate_ecal_gnn import build_model_from_checkpoint
from train_ecal_gnn import hierarchical_model_kwargs,stats_tensors
from torch_geometric.loader import DataLoader
torch.set_num_threads(1)
base=pathlib.Path('/home/comet/davidz/ReconDev/ReconData')
new='em_20to120MeV_10cmPlateGun_0To10deg';old='em_regression_20to120MeV_10cmPlateGun_0To10deg'
cache=pathlib.Path('/tmp/electron_new_test_sample.pt')
# Deterministic, evenly spaced sample from the first new test shard.
metabase=base/'datasets/ecal_clustering/processed/pyg_step1'
newstats=json.load(open(metabase/new/'metadata.json'))['normalization']
oldstats=json.load(open(metabase/old/'metadata.json'))['normalization']
if cache.exists():graphs=torch.load(cache,weights_only=False)
else:
 p=next((base/'datasets/ecal_clustering/processed/pyg_step1'/new).glob('test_000_*.pt'))
 print('Loading',p,flush=True);allg=torch.load(p,weights_only=False,map_location='cpu');ids=np.linspace(0,len(allg)-1,4096,dtype=int);graphs=[allg[i] for i in ids];print('Loaded',len(allg),'sample',len(graphs),flush=True);torch.save(graphs,cache);del allg;gc.collect()
results=[]
for setup,ep,adapt_time in [(old,'036',False),(old,'036',True),(new,'025',False),(new,'046',False),(new,'049',False),(new,'050',False)]:
 c=torch.load(base/'runs/ecal_clustering'/setup/f'step1/checkpoints/epoch_{ep}.pt',map_location='cpu',weights_only=False);model=build_model_from_checkpoint(c).eval();tm,ts=stats_tensors(c['target_normalization'],torch.device('cpu'))
 pred=[];truth=[];hidden=[]
 handle=model.step1_position_head.net[1].register_forward_hook(lambda m,i,o:hidden.append(o.detach()))
 with torch.no_grad():
  for b in DataLoader(graphs,batch_size=128,shuffle=False):
   if adapt_time:
    b.x[:,2]=(b.x[:,2]*newstats['std'][0]+newstats['mean'][0]-oldstats['mean'][0])/oldstats['std'][0]
    b.window_x[:,2:4]=(b.window_x[:,2:4]*newstats['std'][0]+newstats['mean'][0]-oldstats['mean'][0])/oldstats['std'][0]
   o=model(b.x,b.edge_index,b.edge_attr,batch=b.batch,edge_weight=getattr(b,'edge_weight',None),**hierarchical_model_kwargs(b,'1',None,None,{},torch.device('cpu')))
   pred.append(o['prediction']*ts+tm);truth.append(b.y)
 handle.remove();p=torch.cat(pred).numpy()[:,1:3];t=torch.cat(truth).numpy()[:,1:3];h=torch.cat(hidden);r=t-p
 result=dict(setup=setup,epoch=ep,adapt_time_to_checkpoint=adapt_time,n=len(t),mae=np.abs(r).mean(0).tolist(),sd=r.std(0).tolist(),baseline_sd=t.std(0).tolist(),correction_sd=p.std(0).tolist(),correlation=[float(np.corrcoef(p[:,j],t[:,j])[0,1]) for j in range(2)],hidden_windows=int(h.shape[0]),all_hidden_zero_fraction=float((h==0).all(1).float().mean()),active_neurons=int((h.max(0).values>0).sum()),frequently_active_neurons=int(((h>0).float().mean(0)>.01).sum()))
 results.append(result);print(json.dumps(result),flush=True)
json.dump(results,open(pathlib.Path(__file__).with_name('crosscheck_electron.json'),'w'),indent=2)
