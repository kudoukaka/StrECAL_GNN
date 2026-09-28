#!/usr/bin/env python3
"""Evaluate physics-aware multi-object hierarchical ECAL GNN."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from ecal_gnn_model import ECALClusteringGNN, NODE_INPUT_DIM, WINDOW_INPUT_DIM
from train_ecal_gnn import (
    MultiObjectMetrics, association_pos_weight_tensor, find_split_files, get_device,
    hierarchical_model_kwargs, load_graphs, regression_elementwise_loss,
    role_class_weight_tensor, stats_tensors, step1_loss_weights,
    reconstruct_step1_absolute, step1_truth_absolute,
)


def parse_args():
    p=argparse.ArgumentParser(description="Evaluate physics-aware hierarchical ECAL GNN")
    p.add_argument("--dataset-dir",required=True);p.add_argument("--run-dir",required=True);p.add_argument("--checkpoint",required=True)
    p.add_argument("--split",default="test",choices=["train","val","test"]);p.add_argument("--batch-size",type=int,default=256);p.add_argument("--num-workers",type=int,default=0);p.add_argument("--device",default="auto",choices=["auto","cpu","cuda"]);p.add_argument("--max-files",type=int,default=-1)
    p.add_argument("--root-output",default="auto");p.add_argument("--root-max-events",type=int,default=-1)
    p.add_argument("--prompt-threshold",type=float,default=0.5);p.add_argument("--association-threshold",type=float,default=0.5)
    p.add_argument("--min-pred-cluster-energy-mev",type=float,default=20.0)
    p.add_argument("--match-time-scale-ns",type=float,default=5.0);p.add_argument("--match-position-scale-mm",type=float,default=30.0);p.add_argument("--match-angle-scale-deg",type=float,default=10.0)
    p.add_argument("--match-max-time-ns",type=float,default=100.0);p.add_argument("--match-max-position-mm",type=float,default=150.0);p.add_argument("--match-max-angle-deg",type=float,default=45.0)
    return p.parse_args()


def _get(a,n,d):
    v=a.get(n,d);return d if v is None else v


def build_model_from_checkpoint(c):
    a=c.get("args",{});step=str(c.get("reconstruction_step",a.get("reconstruction_step","legacy")));vocab={int(k):int(v) for k,v in c.get("pdg_vocab",{}).items()}
    kw=dict(
        reconstruction_step=step,node_input_dim=int(_get(a,"node_input_dim",NODE_INPUT_DIM)),edge_input_dim=int(_get(a,"edge_input_dim",6)),
        node_hidden_dim=int(_get(a,"node_hidden_dim",64)),edge_hidden_dim=int(_get(a,"edge_hidden_dim",64)),node_encoder_mid_dim=int(_get(a,"node_encoder_mid_dim",32)),edge_encoder_mid_dim=int(_get(a,"edge_encoder_mid_dim",32)),message_hidden_dim=int(_get(a,"message_hidden_dim",128)),regression_hidden_dim=int(_get(a,"regression_hidden_dim",128)),track_hidden_dim=int(_get(a,"track_hidden_dim",64)),track_embedding_dim=int(_get(a,"track_embedding_dim",64)),pdg_embedding_dim=int(_get(a,"pdg_embedding_dim",8)),pdg_vocab_size=max(max(vocab.values(),default=0)+1,1),num_gnn_layers=int(_get(a,"num_gnn_layers",3)),
        node_encoder_depth=int(_get(a,"node_encoder_depth",2)),edge_encoder_depth=int(_get(a,"edge_encoder_depth",2)),message_mlp_depth=int(_get(a,"message_mlp_depth",2)),update_mlp_depth=int(_get(a,"update_mlp_depth",2)),node_head_depth=int(_get(a,"node_head_depth",2)),edge_head_depth=int(_get(a,"edge_head_depth",2)),regression_head_depth=int(_get(a,"regression_head_depth",2)),
        step1_timing_head_depth=int(_get(a,"step1_timing_head_depth",_get(a,"regression_head_depth",2))),step1_position_head_depth=int(_get(a,"step1_position_head_depth",_get(a,"regression_head_depth",2))),step1_direction_head_depth=int(_get(a,"step1_direction_head_depth",_get(a,"regression_head_depth",2))),step1_energy_head_depth=int(_get(a,"step1_energy_head_depth",_get(a,"regression_head_depth",2))),track_encoder_depth=int(_get(a,"track_encoder_depth",2)),fusion_head_depth=int(_get(a,"fusion_head_depth",2)),
        window_input_dim=int(_get(a,"window_input_dim",WINDOW_INPUT_DIM)),window_hidden_dim=int(_get(a,"window_hidden_dim",64)),event_edge_hidden_dim=int(_get(a,"event_edge_hidden_dim",32)),event_message_hidden_dim=int(_get(a,"event_message_hidden_dim",128)),association_edge_hidden_dim=int(_get(a,"association_edge_hidden_dim",32)),association_hidden_dim=int(_get(a,"association_hidden_dim",128)),num_event_gnn_layers=int(_get(a,"num_event_gnn_layers",2)),window_encoder_depth=int(_get(a,"window_encoder_depth",2)),window_fusion_depth=int(_get(a,"window_fusion_depth",2)),event_edge_encoder_depth=int(_get(a,"event_edge_encoder_depth",2)),event_message_mlp_depth=int(_get(a,"event_message_mlp_depth",2)),event_update_mlp_depth=int(_get(a,"event_update_mlp_depth",2)),window_role_head_depth=int(_get(a,"window_role_head_depth",2)),association_edge_encoder_depth=int(_get(a,"association_edge_encoder_depth",2)),association_head_depth=int(_get(a,"association_head_depth",2)),
        graph_pool=str(_get(a,"graph_pool","add_mean_max")),event_pool=str(_get(a,"event_pool","add_mean_max")),event_edge_mode=str(_get(a,"event_edge_mode","hybrid")),event_hybrid_bidirectional_ns=float(_get(a,"event_hybrid_bidirectional_ns",30.0)),step1_prompt_time_gate_ns=float(_get(a,"step1_prompt_time_gate_ns",30.0)),dropout=float(_get(a,"dropout",0.0)),aggr=str(_get(a,"aggr","add")),event_aggr=str(_get(a,"event_aggr","add")),
    )
    m=ECALClusteringGNN(**kw);m.load_state_dict(c["model_state_dict"]);return m


def checkpoint_train_args(c):
    a=c.get("args",{});d=dict(regression_loss="huber",huber_beta=1.0,step1_time_weight=2.0,step1_position_weight=2.0,step1_direction_weight=1.0,step1_cluster_energy_weight=1.0,step1_max_crystal_energy_weight=1.0,window_role_loss_weight=.1,window_role_class_weighting="auto",max_window_role_class_weight=10.,association_loss_weight=.2,association_pos_weight="auto",max_association_pos_weight=20.)
    d.update({k:a[k] for k in d if k in a and a[k] is not None});return SimpleNamespace(**d)


def direction_angle_deg(a,b):
    na=np.linalg.norm(a);nb=np.linalg.norm(b)
    if na<=0 or nb<=0:return 180.0
    return float(np.degrees(np.arccos(np.clip(np.dot(a,b)/(na*nb),-1,1))))


def hungarian(cost):
    if cost.size==0:return np.empty(0,dtype=int),np.empty(0,dtype=int)
    trans=False;c=np.asarray(cost,dtype=float)
    if c.shape[0]>c.shape[1]:c=c.T;trans=True
    n,m=c.shape;u=np.zeros(n+1);v=np.zeros(m+1);p=np.zeros(m+1,dtype=int);way=np.zeros(m+1,dtype=int)
    for i in range(1,n+1):
        p[0]=i;j0=0;minv=np.full(m+1,np.inf);used=np.zeros(m+1,dtype=bool)
        while True:
            used[j0]=True;i0=p[j0];delta=np.inf;j1=0
            for j in range(1,m+1):
                if used[j]:continue
                cur=c[i0-1,j-1]-u[i0]-v[j]
                if cur<minv[j]:minv[j]=cur;way[j]=j0
                if minv[j]<delta:delta=minv[j];j1=j
            for j in range(m+1):
                if used[j]:u[p[j]]+=delta;v[j]-=delta
                else:minv[j]-=delta
            j0=j1
            if p[j0]==0:break
        while True:
            j1=way[j0];p[j0]=p[j1];j0=j1
            if j0==0:break
    rows=[];cols=[]
    for j in range(1,m+1):
        if p[j]>0:rows.append(p[j]-1);cols.append(j-1)
    rows=np.asarray(rows,dtype=int);cols=np.asarray(cols,dtype=int)
    return (cols,rows) if trans else (rows,cols)


def match_step1(pred,truth,args):
    npred,ntruth=len(pred),len(truth);cost=np.full((npred,ntruth),1e9,dtype=float)
    for i in range(npred):
        for j in range(ntruth):
            dt=abs(pred[i,0]-truth[j,0]);dr=float(np.linalg.norm(pred[i,1:3]-truth[j,1:3]));ang=direction_angle_deg(pred[i,3:6],truth[j,3:6])
            if dt>args.match_max_time_ns or dr>args.match_max_position_mm or ang>args.match_max_angle_deg:continue
            cost[i,j]=(dt/args.match_time_scale_ns)**2+(dr/args.match_position_scale_mm)**2+(ang/args.match_angle_scale_deg)**2
    if npred==0 or ntruth==0:return []
    N=max(npred,ntruth);C=np.full((N,N),1e6);C[:npred,:ntruth]=cost;rr,cc=hungarian(C);out=[]
    for r,c in zip(rr,cc):
        if r<npred and c<ntruth and cost[r,c]<1e6:out.append((int(r),int(c),float(cost[r,c])))
    return out


def association_matrix(n,pairs,prob):
    M=np.eye(n,dtype=float)
    for (s,d),p in zip(pairs.T,prob):M[int(s),int(d)]=M[int(d),int(s)]=float(p)
    return M


def select_candidates(window_abs,prompt_prob,assoc,args):
    selected=[];rejected_energy=0
    for w in np.argsort(-prompt_prob):
        if prompt_prob[w]<args.prompt_threshold:continue
        if args.min_pred_cluster_energy_mev>=0 and window_abs[w,6]<args.min_pred_cluster_energy_mev:
            rejected_energy+=1;continue
        if any(assoc[w,q]>=args.association_threshold for q in selected):continue
        selected.append(int(w))
    return selected,rejected_energy


def empty_root():
    scalar=["event_index","event","file_index","graph_index_in_file","reconstruction_step","n_node","n_window","n_truth_object","n_reco_object","n_matched","n_association_pair","n_energy_cut_rejected"]
    ints=["truth_cluster_id","truth_primary_id","truth_pid","truth_prompt_window","reco_anchor_window","reco_match_truth_index","truth_match_reco_index","association_src","association_dst","association_target","event_edge_src","event_edge_dst","window_role_target","window_role_pred"]
    floats=["truth_t0","truth_y","truth_z","truth_dx","truth_dy","truth_dz","truth_ecluster","truth_emax","truth_eincident","pred_oracle_t0","pred_oracle_y","pred_oracle_z","pred_oracle_dx","pred_oracle_dy","pred_oracle_dz","pred_oracle_ecluster","pred_oracle_emax","pred_oracle_eincident","reco_t0","reco_y","reco_z","reco_dx","reco_dy","reco_dz","reco_ecluster","reco_emax","reco_prompt_prob","reco_match_cost","association_prob","association_dt","event_edge_dt","window_prompt_prob","window_delayed_prob","window_prompt_attention","window_seed_amp","window_total_amp","window_truth_t0_ns","window_centroid_y_mm","window_centroid_z_mm","window_tmin_ns","window_esum_mev"]
    ints += ["window_primary_id"]
    return {k:[] for k in scalar+ints+floats}


def _event_scalar(batch,name,i,default=-1):
    if not hasattr(batch,name):return default
    v=getattr(batch,name).detach().cpu().view(-1)
    return int(v[i]) if v.numel()==batch.num_graphs else default


def append_root(root,batch,out,pred_phys,truth_phys,step,event_start,file_idx,graph_start,args,window_abs=None):
    nwin=batch.num_windows.detach().cpu().long().view(-1);woff=torch.cumsum(nwin,0)-nwin
    nt=batch.num_targets.detach().cpu().long().view(-1);toff=torch.cumsum(nt,0)-nt
    roles=batch.window_role_target.detach().cpu().long();rlog=out["window_role_logits"].detach().cpu();rprob=torch.softmax(rlog,1);rpred=rlog.argmax(1);att=out["prompt_attention"].detach().cpu()
    api=out["association_pair_index"].detach().cpu().long();ap=out["association_prob"].detach().cpu()
    eei=out["event_edge_index"].detach().cpu().long()
    wt=batch.window_tmin_ns.detach().cpu().float();wcy=batch.window_centroid_y_mm.detach().cpu().float();wcz=batch.window_centroid_z_mm.detach().cpu().float();wes=batch.window_esum_mev.detach().cpu().float()
    pred=pred_phys.detach().cpu().numpy();truth=truth_phys.detach().cpu().numpy();wabs=window_abs.detach().cpu().numpy() if window_abs is not None else None
    appended=0
    for i in range(int(batch.num_graphs)):
        if args.root_max_events>=0 and len(root["event_index"])>=args.root_max_events:break
        ws=int(woff[i]);we=ws+int(nwin[i]);ts=int(toff[i]);te=ts+int(nt[i]);nw=we-ws
        pmask=(api[0]>=ws)&(api[0]<we) if api.numel() else torch.zeros((0,),dtype=torch.bool);qi=torch.nonzero(pmask,as_tuple=False).view(-1)
        lp=(api[:,qi]-ws).numpy() if qi.numel() else np.empty((2,0),dtype=int);prob=ap[qi].numpy() if qi.numel() else np.empty(0);assoc=association_matrix(nw,lp,prob)
        emask=(eei[0]>=ws)&(eei[0]<we) if eei.numel() else torch.zeros((0,),dtype=torch.bool);eq=torch.nonzero(emask,as_tuple=False).view(-1);lep=(eei[:,eq]-ws).numpy() if eq.numel() else np.empty((2,0),dtype=int)
        truth_event=truth[ts:te];pred_event=pred[ts:te]
        if step=="1":
            selected,rejected=select_candidates(wabs[ws:we],rprob[ws:we,1].numpy(),assoc,args);reco=wabs[ws:we][selected] if selected else np.empty((0,8),dtype=float);matches=match_step1(reco,truth_event,args)
            rm=np.full(len(selected),-1,dtype=int);tm=np.full(len(truth_event),-1,dtype=int);mc=np.full(len(selected),np.nan,dtype=float)
            for a,b,c in matches:rm[a]=b;tm[b]=a;mc[a]=c
        else:
            selected=list(range(len(truth_event)));rejected=0;reco=np.empty((0,8));matches=[(j,j,0.) for j in range(len(truth_event))];rm=np.arange(len(truth_event));tm=np.arange(len(truth_event));mc=np.zeros(len(truth_event))

        for k,v in [("event_index",event_start+appended),("event",_event_scalar(batch,"event",i,-1)),("file_index",file_idx),("graph_index_in_file",graph_start+i),("reconstruction_step",int(step)),("n_node",int(batch.ptr[i+1]-batch.ptr[i])),("n_window",nw),("n_truth_object",len(truth_event)),("n_reco_object",len(selected)),("n_matched",len(matches)),("n_association_pair",len(qi)),("n_energy_cut_rejected",rejected)]:root[k].append(np.int64(v) if k in {"event_index","event","graph_index_in_file"} else np.int32(v))
        def iarr(n,v):root[n].append(np.asarray(v,dtype=np.int64))
        def farr(n,v):root[n].append(np.asarray(v,dtype=np.float32))
        iarr("window_primary_id", batch.window_primary_id.detach().cpu().view(-1)[ws:we].numpy()
             if hasattr(batch, "window_primary_id") else [])
        # Truth metadata and physical truth.
        for n,field in [("truth_cluster_id","target_prompt_cluster_id"),("truth_primary_id","target_primary_id"),("truth_pid","target_pid"),("truth_prompt_window","target_prompt_window_local")]:
            vals=getattr(batch,field).detach().cpu().view(-1)[ts:te].numpy();iarr(n,vals)
        if step=="1":
            for n,col in [("truth_t0",0),("truth_y",1),("truth_z",2),("truth_dx",3),("truth_dy",4),("truth_dz",5),("truth_ecluster",6),("truth_emax",7)]:farr(n,truth_event[:,col])
            farr("truth_eincident",[])
            for n,col in [("pred_oracle_t0",0),("pred_oracle_y",1),("pred_oracle_z",2),("pred_oracle_dx",3),("pred_oracle_dy",4),("pred_oracle_dz",5),("pred_oracle_ecluster",6),("pred_oracle_emax",7)]:farr(n,pred_event[:,col])
            farr("pred_oracle_eincident",[])
            if len(selected):
                for n,col in [("reco_t0",0),("reco_y",1),("reco_z",2),("reco_dx",3),("reco_dy",4),("reco_dz",5),("reco_ecluster",6),("reco_emax",7)]:farr(n,reco[:,col])
            else:
                for n in ("reco_t0","reco_y","reco_z","reco_dx","reco_dy","reco_dz","reco_ecluster","reco_emax"):farr(n,[])
            iarr("reco_anchor_window",selected);iarr("reco_match_truth_index",rm);iarr("truth_match_reco_index",tm);farr("reco_match_cost",mc);farr("reco_prompt_prob",rprob[ws:we,1].numpy()[selected] if selected else [])
        else:
            farr("truth_eincident",truth_event[:,0]);farr("pred_oracle_eincident",pred_event[:,0])
            for n in ("truth_t0","truth_y","truth_z","truth_dx","truth_dy","truth_dz","truth_ecluster","truth_emax","pred_oracle_t0","pred_oracle_y","pred_oracle_z","pred_oracle_dx","pred_oracle_dy","pred_oracle_dz","pred_oracle_ecluster","pred_oracle_emax","reco_t0","reco_y","reco_z","reco_dx","reco_dy","reco_dz","reco_ecluster","reco_emax","reco_prompt_prob","reco_match_cost"):farr(n,[])
            iarr("reco_anchor_window",[]);iarr("reco_match_truth_index",[]);iarr("truth_match_reco_index",[])
        iarr("association_src",lp[0] if lp.size else []);iarr("association_dst",lp[1] if lp.size else []);iarr("association_target",batch.association_target.detach().cpu().view(-1)[sum(int(x*(x-1)//2) for x in nwin[:i]):sum(int(x*(x-1)//2) for x in nwin[:i+1])].numpy())
        farr("association_prob",prob);farr("association_dt",[(float(wt[ws+d]-wt[ws+s])) for s,d in lp.T] if lp.size else [])
        iarr("event_edge_src",lep[0] if lep.size else []);iarr("event_edge_dst",lep[1] if lep.size else []);farr("event_edge_dt",[(float(wt[ws+d]-wt[ws+s])) for s,d in lep.T] if lep.size else [])
        iarr("window_role_target",roles[ws:we].numpy());iarr("window_role_pred",rpred[ws:we].numpy());farr("window_prompt_prob",rprob[ws:we,1].numpy());farr("window_delayed_prob",rprob[ws:we,2].numpy());farr("window_prompt_attention",att[ws:we].numpy())
        for n,field in [("window_seed_amp","window_seed_amp"),("window_total_amp","window_total_amp"),("window_truth_t0_ns","window_truth_t0_ns")]:farr(n,getattr(batch,field).detach().cpu().view(-1)[ws:we].numpy() if hasattr(batch,field) else [])
        farr("window_centroid_y_mm",wcy[ws:we].numpy());farr("window_centroid_z_mm",wcz[ws:we].numpy());farr("window_tmin_ns",wt[ws:we].numpy());farr("window_esum_mev",wes[ws:we].numpy())
        appended+=1
    return appended


def write_root(path,root,record):
    import awkward as ak, uproot
    scalar_types={"event_index":np.int64,"event":np.int64,"file_index":np.int32,"graph_index_in_file":np.int64,"reconstruction_step":np.int32,"n_node":np.int32,"n_window":np.int32,"n_truth_object":np.int32,"n_reco_object":np.int32,"n_matched":np.int32,"n_association_pair":np.int32,"n_energy_cut_rejected":np.int32}
    int_keys=[k for k in root if k not in scalar_types and (k == "window_primary_id" or k.startswith("truth_cluster") or k.startswith("truth_primary") or k.startswith("truth_pid") or k.startswith("truth_prompt") or k.startswith("reco_anchor") or k.endswith("_index") or k in {"association_src","association_dst","association_target","event_edge_src","event_edge_dst","window_role_target","window_role_pred","truth_match_reco_index"})]
    arr={k:np.asarray(root[k],dtype=t) for k,t in scalar_types.items()}
    types=dict(scalar_types)
    for k in root:
        if k in scalar_types:continue
        typ="var * int64" if k in int_keys else "var * float32";types[k]=typ;arr[k]=ak.Array(root[k])
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with uproot.recreate(path) as f:
        tr=f.mktree("events",types);tr.extend(arr)
        meta=np.frombuffer(json.dumps(record,indent=2).encode(),dtype=np.uint8);mt=f.mktree("metadata",{"json_bytes":"var * uint8"});mt.extend({"json_bytes":ak.Array([meta])})


def evaluate(model,files,step,dev,args,ck,ta,root):
    tstats=ck["target_normalization"];tm,ts=stats_tensors(tstats,dev);xstats=ck.get("step2_track_normalization");xm=xs=None
    if step=="2":xm,xs=stats_tensors(xstats,dev)
    vocab={int(k):int(v) for k,v in ck.get("pdg_vocab",{}).items()};rstats=ck["window_role_stats"];astats=ck["association_stats"]
    sw=step1_loss_weights(ta,dev) if step=="1" else None;rw=role_class_weight_tensor(rstats,ta,dev);apw=association_pos_weight_tensor(astats,ta,dev);metrics=MultiObjectMetrics(step);model.eval();events=truth_tot=reco_tot=matched_tot=ecut=0
    with torch.no_grad():
        for fi,path in enumerate(files):
            graphs=load_graphs(path);loader=DataLoader(graphs,batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers,pin_memory=dev.type=="cuda");gstart=0
            for b in loader:
                b=b.to(dev);target=b.y.float().reshape(-1,8 if step=="1" else 1);tn=(target-tm)/ts;rt=b.window_role_target.long().view(-1);at=b.association_target.float().view(-1)
                out=model(b.x,b.edge_index,b.edge_attr,batch=b.batch,edge_weight=getattr(b,"edge_weight",None),**hierarchical_model_kwargs(b,step,xm,xs,vocab,dev));pn=out["prediction"]
                elem=regression_elementwise_loss(pn,tn,ta.regression_loss,ta.huber_beta);reg=(elem*sw).sum()/(sw.sum()*max(int(elem.shape[0]),1)) if step=="1" else elem.mean();role=F.cross_entropy(out["window_role_logits"],rt,weight=rw) if rt.numel() else reg.new_tensor(0.);assoc=F.binary_cross_entropy_with_logits(out["association_logits"],at,pos_weight=apw) if at.numel() else reg.new_tensor(0.);total=reg+ta.window_role_loss_weight*role+ta.association_loss_weight*assoc
                pr=pn*ts+tm
                window_abs=None
                if step=="1":
                    pred_phys=reconstruct_step1_absolute(pr,b);truth_phys=step1_truth_absolute(b)
                    wr=out["window_prediction"]*ts+tm
                    window_abs=wr.clone();window_abs[:,0]=b.window_tmin_ns.view(-1)-wr[:,0];window_abs[:,1]=b.window_centroid_y_mm.view(-1)-wr[:,1];window_abs[:,2]=b.window_centroid_z_mm.view(-1)-wr[:,2]
                else:pred_phys=pr;truth_phys=target
                metrics.update(total,reg,role,assoc,pred_phys,truth_phys,out["window_role_logits"],rt,out["association_logits"],at,int(b.num_graphs))
                if root is not None:
                    before=len(root["event_index"]);events+=append_root(root,b,out,pred_phys,truth_phys,step,events,fi,gstart,args,window_abs)
                    for q in range(before,len(root["event_index"])):truth_tot+=int(root["n_truth_object"][q]);reco_tot+=int(root["n_reco_object"][q]);matched_tot+=int(root["n_matched"][q]);ecut+=int(root["n_energy_cut_rejected"][q])
                gstart+=int(b.num_graphs)
            del graphs,loader
    sm=metrics.summary();sm.update({"reco_truth_objects":truth_tot,"reco_pred_objects":reco_tot,"reco_matched_objects":matched_tot,"reco_efficiency":matched_tot/max(truth_tot,1),"reco_purity":matched_tot/max(reco_tot,1),"reco_energy_cut_rejected":ecut})
    return sm


def main():
    args=parse_args();dev=get_device(args.device);ck=torch.load(args.checkpoint,map_location="cpu",weights_only=False);step=str(ck.get("reconstruction_step",ck.get("args",{}).get("reconstruction_step","1")));model=build_model_from_checkpoint(ck).to(dev);files=find_split_files(Path(args.dataset_dir),args.split,args.max_files);ta=checkpoint_train_args(ck);root=empty_root() if args.root_output.lower()!="none" and step in {"1","2"} else None
    metrics=evaluate(model,files,step,dev,args,ck,ta,root);epoch=int(ck.get("epoch",-1));rec={"checkpoint":args.checkpoint,"epoch":epoch,"reconstruction_step":step,"split":args.split,"step1_target":"residual dt/dy/dz reconstructed back to absolute t0/y/z","metrics":metrics}
    out=Path(args.run_dir)/"outputs";out.mkdir(parents=True,exist_ok=True);jp=out/f"eval_{args.split}_epoch_{epoch:03d}.json";jp.write_text(json.dumps(rec,indent=2));print(json.dumps(rec,indent=2));print(f"[write] {jp}")
    if root is not None:
        rp=out/f"eval_{args.split}_epoch_{epoch:03d}.root" if args.root_output=="auto" else Path(args.root_output);write_root(rp,root,rec);print(f"[write] {rp}")

if __name__=="__main__":main()
