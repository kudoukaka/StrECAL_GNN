import numpy as np,pathlib,json
base=pathlib.Path(__file__).resolve().parent
for s in ['mum','em','pim']:
 a=np.genfromtxt(base/(s+'.csv'),delimiter=',',names=True);b=a['tmin']-a['truth_t0'];p=a['oracle_t0']-a['truth_t0'];mr=a['reco_t0']-a['truth_t0'];m=a['matched']==1
 print('\nSPECIES',s,'N',len(a),'matched',sum(m),'sameanchor',sum(m&(a['same_anchor']==1)))
 for name,x in [('truth_t0',a['truth_t0']),('baseline',b),('correction',b-p),('oracle_res',p),('matched_res',mr[m])]:
  print(name,'mean/std/MAE',np.mean(x),np.std(x),np.mean(abs(x)),'q',np.quantile(x,[0,.001,.01,.16,.5,.84,.99,.999,1]).round(6))
 for name,sel in [('all',np.ones(len(a),dtype=bool)),('matched',m),('unmatched',~m),('matched_delayed',m&(a['has_delayed']==1)),('matched_no_delayed',m&(a['has_delayed']==0)),('base_abs_le_1',abs(b)<=1),('base_abs_gt_1',abs(b)>1),('base_abs_gt_30',abs(b)>30),('matched_base_abs_le_1',m&(abs(b)<=1)),('matched_base_abs_gt_30',m&(abs(b)>30))]:
  r=p[sel];t=b[sel]
  if not len(r):
   print(name,'N 0');continue
  print(name,'N',len(r),'baseMean/std/MAE',round(t.mean(),6),round(t.std(),6),round(abs(t).mean(),6),'oracleMean/std/MAE',round(r.mean(),6),round(r.std(),6),round(abs(r).mean(),6))
 print('tails matched: threshold count fraction sumsq fraction')
 for k in [.1,1,5,10,30,50,80]:
  r=mr[m];tail=abs(r)>k;print(k,int(sum(tail)),round(tail.mean(),6),round((r[tail]**2).sum()/(r*r).sum(),6))
 print('TOP oracle',[(int(a['file_'][i]),int(a['event'][i]),round(b[i],3),round(p[i],3),int(m[i]),int(a['has_delayed'][i])) for i in np.argsort(abs(p))[-8:]])
