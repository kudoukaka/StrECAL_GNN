"""Full-test oracle diagnostics: no matched-only or residual-based selection."""
import pathlib,json,numpy as np
base=pathlib.Path(__file__).resolve().parent
out={}
for s in ['mum','pim']:
 a=np.genfromtxt(base/(s+'.csv'),delimiter=',',names=True)
 target=a['tmin']-a['truth_t0'];pred=a['tmin']-a['oracle_t0'];residual=target-pred
 out[s]={}
 for label,mask in [('small_dt',abs(target)<=1),('small_dt_early',(abs(target)<=1)&(a['truth_t0']<=30)),('small_dt_late',(abs(target)<=1)&(a['truth_t0']>30))]:
  out[s][label]={'n':int(mask.sum()),'matched':int(a['matched'][mask].sum()),'fraction_abs_residual_gt50':float(np.mean(abs(residual[mask])>50))}
  for name,array in [('target',target),('prediction',pred),('residual',residual)]:
   x=array[mask];q=np.quantile(x,[.16,.5,.84]);out[s][label][name]={'mean':float(x.mean()),'std':float(x.std()),'mae':float(abs(x).mean()),'median':float(q[1]),'central68_halfwidth':float((q[2]-q[0])/2)}
json.dump(out,open(base/'timing_populations.json','w'),indent=2)
print(json.dumps(out,indent=2))
