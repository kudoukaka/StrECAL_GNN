"""Truth-defined diagnostic subsets; never used as inference-time selections."""
import pathlib,json,numpy as np
p=pathlib.Path(__file__).resolve().parent
a=np.genfromtxt(p/'pim.csv',delimiter=',',names=True)
t=a['tmin']-a['truth_t0'];pred=a['tmin']-a['oracle_t0'];err=t-pred
constant=abs(pred+1.065601)<.001;matched=a['matched']==1
result={}
for name,sel in [('all',np.ones(len(t),bool)),('abs_dt_le_1ns',abs(t)<=1),('dt_gt_30ns',t>30),('small_dt_early_incident',(abs(t)<=1)&(a['truth_t0']<=30)),('small_dt_late_incident',(abs(t)<=1)&(a['truth_t0']>30))]:
 result[name]=dict(n=int(sel.sum()),constant_count=int((constant&sel).sum()),constant_fraction=float(constant[sel].mean()),matched_count=int((matched&sel).sum()),matched_fraction=float(matched[sel].mean()),predicted_correction_median=float(np.median(pred[sel])),error_mae=float(abs(err[sel]).mean()),prediction_target_correlation=float(np.corrcoef(t[sel],pred[sel])[0,1]))
print(json.dumps(result,indent=2))
