#!/usr/bin/env python3
"""Train the physics-aware multi-object hierarchical ECAL GNN."""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from ecal_gnn_model import (
    ECALClusteringGNN,
    NODE_INPUT_DIM,
    WINDOW_INPUT_DIM,
    WINDOW_ROLE_CLASSES,
)

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

STEP1_TARGET_NAMES = [
    "dt_prompt_min_minus_t0_ns",
    "dy_prompt_centroid_minus_incident_y_mm",
    "dz_prompt_centroid_minus_incident_z_mm",
    "incident_dir_x",
    "incident_dir_y",
    "incident_dir_z",
    "cluster_energy_mev",
    "most_significant_crystal_energy_mev",
]
STEP2_TARGET_NAMES = ["incident_energy_mev"]
STEP2_TRACK_FEATURE_NAMES = [
    "track_entry_x_mm", "track_entry_y_mm", "track_entry_z_mm",
    "track_dir_x", "track_dir_y", "track_dir_z",
]
WINDOW_ROLE_NAMES = ["other_or_noise", "prompt_any_object", "nonprompt_truth_activity"]


def parse_args():
    p = argparse.ArgumentParser(description="Train physics-aware hierarchical ECAL GNN")
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--reconstruction-step", default="auto", choices=["auto","legacy","1","2"])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--device", default="auto", choices=["auto","cpu","cuda"])
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--save-every", type=int, default=1)
    p.add_argument("--resume", default="")
    p.add_argument("--resume-reset-optimizer", action="store_true")
    p.add_argument("--max-train-files", type=int, default=-1)
    p.add_argument("--max-val-files", type=int, default=-1)

    # Random stats sampling (does not alter training RNG state).
    p.add_argument("--stats-max-events", type=int, default=100000,
                   help="Randomly sample at most N training events for normalization/class stats; <=0 scans all")
    p.add_argument("--stats-seed", type=int, default=None,
                   help="Independent stats-sampling seed; default uses --seed")

    # Pulse GNN.
    p.add_argument("--node-input-dim", type=int, default=NODE_INPUT_DIM)
    p.add_argument("--edge-input-dim", type=int, default=6)
    p.add_argument("--node-hidden-dim", type=int, default=64)
    p.add_argument("--edge-hidden-dim", type=int, default=64)
    p.add_argument("--node-encoder-mid-dim", type=int, default=32)
    p.add_argument("--edge-encoder-mid-dim", type=int, default=32)
    p.add_argument("--message-hidden-dim", type=int, default=128)
    p.add_argument("--num-gnn-layers", type=int, default=3)
    p.add_argument("--aggr", default="add", choices=["add","mean","max"])

    # Window/event hierarchy.
    p.add_argument("--window-input-dim", type=int, default=WINDOW_INPUT_DIM)
    p.add_argument("--window-hidden-dim", type=int, default=64)
    p.add_argument("--event-edge-hidden-dim", type=int, default=32)
    p.add_argument("--event-message-hidden-dim", type=int, default=128)
    p.add_argument("--association-edge-hidden-dim", type=int, default=32)
    p.add_argument("--association-hidden-dim", type=int, default=128)
    p.add_argument("--num-event-gnn-layers", type=int, default=2)
    p.add_argument("--event-aggr", default="add", choices=["add","mean","max"])
    pools=["add","mean","max","add_mean","mean_max","add_max","add_mean_max"]
    p.add_argument("--graph-pool", default="add_mean_max", choices=pools)
    p.add_argument("--event-pool", default="add_mean_max", choices=pools)
    p.add_argument("--event-edge-mode", default="hybrid",
                   choices=["hybrid","time_directed","bidirectional_complete"])
    p.add_argument("--event-hybrid-bidirectional-ns", type=float, default=30.0,
                   help="Hybrid Event GNN: windows within this |dt| exchange messages both ways")
    p.add_argument("--step1-prompt-time-gate-ns", type=float, default=30.0,
                   help="Step1 prompt-context hard gate and exponential time scale")

    # Heads.
    p.add_argument("--regression-hidden-dim", type=int, default=128)
    p.add_argument("--track-hidden-dim", type=int, default=64)
    p.add_argument("--track-embedding-dim", type=int, default=64)
    p.add_argument("--pdg-embedding-dim", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.0)

    # Configurable MLP depth.
    p.add_argument("--mlp-depth", type=int, default=2)
    depth_names = [
        "node-encoder","edge-encoder","message-mlp","update-mlp","node-head","edge-head",
        "regression-head","step1-timing-head","step1-position-head","step1-direction-head",
        "step1-energy-head","track-encoder","fusion-head","window-encoder","window-fusion",
        "event-edge-encoder","event-message-mlp","event-update-mlp","window-role-head",
        "association-edge-encoder","association-head",
    ]
    for name in depth_names:
        p.add_argument(f"--{name}-depth", type=int, default=None)

    # Legacy classifier options.
    p.add_argument("--node-loss-weight", type=float, default=1.0)
    p.add_argument("--edge-loss-weight", type=float, default=1.0)
    p.add_argument("--max-pos-weight", type=float, default=50.0)

    # Regression loss. Defaults intentionally emphasize timing and position.
    p.add_argument("--regression-loss", default="huber", choices=["huber","mse","l1"])
    p.add_argument("--huber-beta", type=float, default=1.0)
    p.add_argument("--step1-time-weight", type=float, default=2.0)
    p.add_argument("--step1-position-weight", type=float, default=2.0)
    p.add_argument("--step1-direction-weight", type=float, default=1.0)
    p.add_argument("--step1-cluster-energy-weight", type=float, default=1.0)
    p.add_argument("--step1-max-crystal-energy-weight", type=float, default=1.0)

    # Auxiliary heads.
    p.add_argument("--window-role-loss-weight", type=float, default=0.10)
    p.add_argument("--window-role-class-weighting", default="auto", choices=["auto","none"])
    p.add_argument("--max-window-role-class-weight", type=float, default=10.0)
    p.add_argument("--association-loss-weight", type=float, default=0.20)
    p.add_argument("--association-pos-weight", default="auto")
    p.add_argument("--max-association-pos-weight", type=float, default=20.0)
    return p.parse_args()


def resolve_depth_args(a):
    names=[
        "node_encoder_depth","edge_encoder_depth","message_mlp_depth","update_mlp_depth",
        "node_head_depth","edge_head_depth","regression_head_depth","step1_timing_head_depth",
        "step1_position_head_depth","step1_direction_head_depth","step1_energy_head_depth",
        "track_encoder_depth","fusion_head_depth","window_encoder_depth","window_fusion_depth",
        "event_edge_encoder_depth","event_message_mlp_depth","event_update_mlp_depth",
        "window_role_head_depth","association_edge_encoder_depth","association_head_depth",
    ]
    if a.mlp_depth < 1: raise ValueError("--mlp-depth must be >=1")
    for n in names:
        v=getattr(a,n)
        if v is None: setattr(a,n,int(a.mlp_depth))
        elif v < 1: raise ValueError(f"--{n.replace('_','-')} must be >=1")


def set_seed(seed):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def get_device(name):
    if name=="cpu": return torch.device("cpu")
    if name=="cuda":
        if not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_split_files(dataset_dir: Path, split: str, max_files=-1):
    files=sorted(Path(dataset_dir).glob(f"{split}_*.pt"))
    if max_files>0: files=files[:max_files]
    if not files: raise FileNotFoundError(f"No {split}_*.pt in {dataset_dir}")
    return files


def load_graphs(path):
    obj=torch.load(path,map_location="cpu",weights_only=False)
    return obj if isinstance(obj,list) else [obj]


def graph_step(g):
    if hasattr(g,"reconstruction_step") and torch.is_tensor(g.reconstruction_step) and g.reconstruction_step.numel():
        v=int(g.reconstruction_step.view(-1)[0]); return str(v) if v in (1,2) else "legacy"
    return "legacy"


def infer_dataset_step(files):
    g=load_graphs(files[0])
    if not g: raise RuntimeError("empty dataset")
    return graph_step(g[0])


def verify_requested_step(req,det):
    if req=="auto": return det
    if req!=det: raise RuntimeError(f"requested step {req}, dataset step {det}")
    return req


class RunningVectorStats:
    def __init__(self,dim):
        self.dim=dim;self.count=0
        self.sum=torch.zeros(dim,dtype=torch.float64)
        self.sumsq=torch.zeros(dim,dtype=torch.float64)
    def update(self,x):
        x=x.detach().cpu().double().reshape(-1,self.dim)
        x=x[torch.isfinite(x).all(1)]
        if not x.numel(): return
        self.count+=int(x.shape[0]);self.sum+=x.sum(0);self.sumsq+=(x*x).sum(0)
    def finish(self,names):
        if self.count==0: raise RuntimeError("no finite stats samples")
        mean=self.sum/self.count
        var=torch.clamp(self.sumsq/self.count-mean*mean,min=0.0)
        std=torch.sqrt(var);std[std<1e-12]=1.0
        return {"names":list(names),"count":self.count,"mean":mean.tolist(),"std":std.tolist()}


def compute_regression_normalization(files,step,max_events=100000,seed=12345):
    target_stats=RunningVectorStats(8 if step=="1" else 1)
    track_stats=RunningVectorStats(6) if step=="2" else None
    pdgs=set();roles=torch.zeros(3,dtype=torch.long);apos=aneg=0;nevent=nobj=0;nfiles=0
    rng=random.Random(seed);sample_files=list(files);rng.shuffle(sample_files)
    print(f"[stats] randomly sampling up to {max_events} training events (seed={seed})" if max_events>0 else f"[stats] scanning full training set (seed={seed})")
    for fi,path in enumerate(sample_files):
        if max_events>0 and nevent>=max_events: break
        graphs=load_graphs(path);nfiles+=1
        remaining=(max_events-nevent) if max_events>0 else len(graphs)
        take=min(len(graphs),remaining) if max_events>0 else len(graphs)
        idx=rng.sample(range(len(graphs)),take) if take<len(graphs) else list(range(len(graphs)))
        if take==len(graphs): rng.shuffle(idx)
        for gi in idx:
            g=graphs[gi]
            if graph_step(g)!=step: raise RuntimeError(f"mixed reconstruction step in {path}")
            k=int(g.num_targets.view(-1)[0]);target_stats.update(g.y);nobj+=k;nevent+=1
            r=g.window_role_target.long().view(-1);roles+=torch.bincount(r,minlength=3)
            at=g.association_target.float().view(-1);apos+=int((at>.5).sum());aneg+=int((at<=.5).sum())
            if step=="2":
                track_stats.update(g.step2_track_features)
                pdgs.update(int(x) for x in g.step2_pdg.view(-1).tolist())
            if max_events>0 and nevent>=max_events: break
        print(f"  [stats] files_loaded={nfiles}/{len(files)} sampled_events={nevent} objects={nobj}",flush=True)
        del graphs
    return (
        target_stats.finish(STEP1_TARGET_NAMES if step=="1" else STEP2_TARGET_NAMES),
        track_stats.finish(STEP2_TRACK_FEATURE_NAMES) if track_stats is not None else None,
        {c:i+1 for i,c in enumerate(sorted(pdgs))},
        {"names":WINDOW_ROLE_NAMES,"counts":roles.tolist(),"sampled_events":nevent,"sampled_files":nfiles,"stats_seed":seed},
        {"positive":apos,"negative":aneg,"events":nevent,"objects":nobj,"sampled_files":nfiles,"stats_seed":seed},
    )


def stats_tensors(stats,device):
    return torch.tensor(stats["mean"],dtype=torch.float32,device=device),torch.tensor(stats["std"],dtype=torch.float32,device=device)


def pdg_to_index(pdg,pdg_vocab,device):
    return torch.tensor([pdg_vocab.get(int(x),0) for x in pdg.detach().cpu().view(-1).tolist()],dtype=torch.long,device=device)


def step1_loss_weights(a,device):
    return torch.tensor([
        a.step1_time_weight,
        a.step1_position_weight,a.step1_position_weight,
        a.step1_direction_weight,a.step1_direction_weight,a.step1_direction_weight,
        a.step1_cluster_energy_weight,a.step1_max_crystal_energy_weight,
    ],dtype=torch.float32,device=device)


def role_class_weight_tensor(stats,a,device):
    if a.window_role_class_weighting=="none": return None
    counts=torch.tensor(stats["counts"],dtype=torch.float32,device=device)
    present=counts>0
    if not present.any(): return None
    total=counts[present].sum();w=torch.zeros_like(counts)
    w[present]=total/(present.sum()*counts[present])
    w=torch.clamp(w,max=a.max_window_role_class_weight)
    return w


def association_pos_weight_tensor(stats,a,device):
    if str(a.association_pos_weight).lower()!="auto":
        return torch.tensor([float(a.association_pos_weight)],dtype=torch.float32,device=device)
    pos=max(int(stats["positive"]),1);neg=int(stats["negative"])
    return torch.tensor([min(max(neg/pos,1.0),a.max_association_pos_weight)],dtype=torch.float32,device=device)


def regression_elementwise_loss(pred,target,name,beta):
    if name=="mse": return (pred-target)**2
    if name=="l1": return torch.abs(pred-target)
    return F.smooth_l1_loss(pred,target,beta=beta,reduction="none")


def hierarchical_model_kwargs(batch,step,track_mean,track_std,pdg_vocab,device):
    kw={
        "node_window_local":batch.node_window_local.long(),
        "num_windows":batch.num_windows.long().view(-1),
        "window_x":batch.window_x.float().reshape(-1,WINDOW_INPUT_DIM),
        "window_tmin_ns":batch.window_tmin_ns.float().view(-1),
        "num_targets":batch.num_targets.long().view(-1),
        "target_prompt_window_local":batch.target_prompt_window_local.long().view(-1),
    }
    if step=="2":
        tr=batch.step2_track_features.float().reshape(-1,6)
        kw["step2_track_features"]=(tr-track_mean)/track_std
        kw["step2_pdg_index"]=pdg_to_index(batch.step2_pdg,pdg_vocab,device)
    return kw


def reconstruct_step1_absolute(pred_residual,batch):
    """Convert predicted residual targets back to physical t0/y/z."""
    tmin=batch.target_prompt_tmin_ns.float().view(-1)
    yc=batch.target_prompt_centroid_y_mm.float().view(-1)
    zc=batch.target_prompt_centroid_z_mm.float().view(-1)
    out=pred_residual.clone()
    out[:,0]=tmin-pred_residual[:,0]
    out[:,1]=yc-pred_residual[:,1]
    out[:,2]=zc-pred_residual[:,2]
    return out


def step1_truth_absolute(batch):
    return torch.cat([
        batch.target_incident_time_ns.float().view(-1,1),
        batch.target_incident_y_mm.float().view(-1,1),
        batch.target_incident_z_mm.float().view(-1,1),
        batch.target_incident_direction.float().reshape(-1,3),
        batch.target_cluster_energy_mev.float().view(-1,1),
        batch.target_most_significant_crystal_energy_mev.float().view(-1,1),
    ],dim=1)


class MultiObjectMetrics:
    def __init__(self,step):
        self.step=step;self.total_sum=self.reg_sum=self.role_sum=self.assoc_sum=0.0;self.nb=0;self.nevent=0;self.nobj=0
        self.abs_sum=None;self.sq_sum=None;self.rel_sq_energy=0.0;self.rel_sq_emax=0.0;self.angle_sum=0.0
        self.role_conf=torch.zeros((3,3),dtype=torch.long);self.assoc_conf=torch.zeros((2,2),dtype=torch.long)
    def update(self,total,reg,role,assoc,pred_phys,target_phys,role_logits,role_target,assoc_logits,assoc_target,n_events):
        self.total_sum+=float(total.detach().cpu());self.reg_sum+=float(reg.detach().cpu());self.role_sum+=float(role.detach().cpu());self.assoc_sum+=float(assoc.detach().cpu());self.nb+=1;self.nevent+=int(n_events);self.nobj+=int(target_phys.shape[0])
        p=pred_phys.detach().cpu();t=target_phys.detach().cpu();d=p-t
        if self.abs_sum is None:self.abs_sum=torch.zeros(d.shape[1],dtype=torch.float64);self.sq_sum=torch.zeros(d.shape[1],dtype=torch.float64)
        self.abs_sum+=torch.abs(d).double().sum(0);self.sq_sum+=(d.double()**2).sum(0)
        if self.step=="1" and len(t):
            pn=p[:,3:6]/torch.clamp(torch.linalg.norm(p[:,3:6],dim=1,keepdim=True),min=1e-9)
            tn=t[:,3:6]/torch.clamp(torch.linalg.norm(t[:,3:6],dim=1,keepdim=True),min=1e-9)
            self.angle_sum+=float(torch.rad2deg(torch.acos(torch.clamp((pn*tn).sum(1),-1,1))).sum())
            self.rel_sq_energy+=float((((p[:,6]-t[:,6])/torch.clamp(torch.abs(t[:,6]),min=1e-6))**2).sum())
            self.rel_sq_emax+=float((((p[:,7]-t[:,7])/torch.clamp(torch.abs(t[:,7]),min=1e-6))**2).sum())
        elif self.step=="2" and len(t):
            self.rel_sq_energy+=float((((p[:,0]-t[:,0])/torch.clamp(torch.abs(t[:,0]),min=1e-6))**2).sum())
        if role_target.numel():
            rp=role_logits.detach().cpu().argmax(1);rt=role_target.detach().cpu().long()
            for a,b in zip(rt.tolist(),rp.tolist()):self.role_conf[a,b]+=1
        if assoc_target.numel():
            ap=(torch.sigmoid(assoc_logits.detach().cpu())>.5).long();at=(assoc_target.detach().cpu()>.5).long()
            for a,b in zip(at.tolist(),ap.tolist()):self.assoc_conf[a,b]+=1
    def summary(self):
        n=max(self.nobj,1);out={
            "loss":self.total_sum/max(self.nb,1),"regression_loss":self.reg_sum/max(self.nb,1),
            "window_role_loss":self.role_sum/max(self.nb,1),"association_loss":self.assoc_sum/max(self.nb,1),
            "n_events":self.nevent,"n_objects":self.nobj,"objects_per_event":self.nobj/max(self.nevent,1),
        }
        if self.abs_sum is not None:
            mae=(self.abs_sum/n).tolist();rms=torch.sqrt(self.sq_sum/n).tolist()
            if self.step=="1":
                out.update({
                    "incident_time_ns_mae":mae[0],"incident_time_ns_rms":rms[0],
                    "incident_y_mm_mae":mae[1],"incident_z_mm_mae":mae[2],
                    "incident_position_yz_mae_mm":float(math.sqrt(mae[1]**2+mae[2]**2)),
                    "incident_position_yz_rms_mm":float(math.sqrt(rms[1]**2+rms[2]**2)),
                    "incident_direction_angle_mean_deg":self.angle_sum/n,
                    "cluster_energy_mev_mae":mae[6],"cluster_energy_mev_rms":rms[6],
                    "cluster_energy_relative_rms":math.sqrt(self.rel_sq_energy/n),
                    "max_crystal_energy_relative_rms":math.sqrt(self.rel_sq_emax/n),
                })
            else:
                out.update({"incident_energy_mev_mae":mae[0],"incident_energy_mev_rms":rms[0],"incident_energy_relative_rms":math.sqrt(self.rel_sq_energy/n)})
        out["window_role_accuracy"]=float(torch.diag(self.role_conf).sum())/max(int(self.role_conf.sum()),1)
        out["association_accuracy"]=float(torch.diag(self.assoc_conf).sum())/max(int(self.assoc_conf.sum()),1)
        out["association_same_recall"]=int(self.assoc_conf[1,1])/max(int(self.assoc_conf[1,:].sum()),1)
        out["association_same_precision"]=int(self.assoc_conf[1,1])/max(int(self.assoc_conf[:,1].sum()),1)
        return out


# Minimal legacy helpers kept for compatibility.
def masked_bce_with_logits(logits,targets,mask,max_pos_weight):
    if mask is not None:logits=logits[mask];targets=targets[mask]
    if targets.numel()==0:return logits.new_tensor(0.0),0
    targets=targets.float();pos=targets.sum();neg=targets.numel()-pos
    pw=torch.clamp(neg/pos,min=1.0,max=max_pos_weight).reshape(1).to(logits.device) if pos>0 else None
    return F.binary_cross_entropy_with_logits(logits,targets,pos_weight=pw),int(targets.numel())

def batch_masks(batch):
    node_mask=~batch.is_ambiguous.bool() if hasattr(batch,"is_ambiguous") else torch.ones_like(batch.y_node,dtype=torch.bool)
    if batch.edge_index.numel()==0:edge_mask=torch.empty((0,),dtype=torch.bool,device=batch.x.device)
    else:edge_mask=node_mask[batch.edge_index[0]]&node_mask[batch.edge_index[1]]
    return node_mask,edge_mask

class MetricAccumulator:
    def __init__(self,nw,ew):self.nw=nw;self.ew=ew;self.loss=0.;self.nb=0
    def update(self,total_loss,*args,**kwargs):self.loss+=float(total_loss.detach().cpu());self.nb+=1
    def summary(self):return {"loss":self.loss/max(self.nb,1),"best_selection_score":-self.loss/max(self.nb,1)}


def build_model(a,step,pdg_vocab_size):
    return ECALClusteringGNN(
        reconstruction_step=step,
        node_input_dim=a.node_input_dim,edge_input_dim=a.edge_input_dim,node_hidden_dim=a.node_hidden_dim,edge_hidden_dim=a.edge_hidden_dim,
        node_encoder_mid_dim=a.node_encoder_mid_dim,edge_encoder_mid_dim=a.edge_encoder_mid_dim,message_hidden_dim=a.message_hidden_dim,
        regression_hidden_dim=a.regression_hidden_dim,track_hidden_dim=a.track_hidden_dim,track_embedding_dim=a.track_embedding_dim,
        pdg_embedding_dim=a.pdg_embedding_dim,pdg_vocab_size=max(int(pdg_vocab_size),1),num_gnn_layers=a.num_gnn_layers,
        node_encoder_depth=a.node_encoder_depth,edge_encoder_depth=a.edge_encoder_depth,message_mlp_depth=a.message_mlp_depth,
        update_mlp_depth=a.update_mlp_depth,node_head_depth=a.node_head_depth,edge_head_depth=a.edge_head_depth,
        regression_head_depth=a.regression_head_depth,step1_timing_head_depth=a.step1_timing_head_depth,
        step1_position_head_depth=a.step1_position_head_depth,step1_direction_head_depth=a.step1_direction_head_depth,
        step1_energy_head_depth=a.step1_energy_head_depth,track_encoder_depth=a.track_encoder_depth,fusion_head_depth=a.fusion_head_depth,
        window_input_dim=a.window_input_dim,window_hidden_dim=a.window_hidden_dim,event_edge_hidden_dim=a.event_edge_hidden_dim,
        event_message_hidden_dim=a.event_message_hidden_dim,association_edge_hidden_dim=a.association_edge_hidden_dim,
        association_hidden_dim=a.association_hidden_dim,num_event_gnn_layers=a.num_event_gnn_layers,window_encoder_depth=a.window_encoder_depth,
        window_fusion_depth=a.window_fusion_depth,event_edge_encoder_depth=a.event_edge_encoder_depth,event_message_mlp_depth=a.event_message_mlp_depth,
        event_update_mlp_depth=a.event_update_mlp_depth,window_role_head_depth=a.window_role_head_depth,
        association_edge_encoder_depth=a.association_edge_encoder_depth,association_head_depth=a.association_head_depth,
        graph_pool=a.graph_pool,event_pool=a.event_pool,event_edge_mode=a.event_edge_mode,
        event_hybrid_bidirectional_ns=a.event_hybrid_bidirectional_ns,step1_prompt_time_gate_ns=a.step1_prompt_time_gate_ns,
        dropout=a.dropout,aggr=a.aggr,event_aggr=a.event_aggr,
    )


def run_one_split_regression(model,files,step,device,batch_size,num_workers,tstats,xstats,pdg_vocab,rstats,astats,a,optimizer=None,train=False,epoch=0,split_name="train"):
    model.train(train);metrics=MultiObjectMetrics(step);tm,ts=stats_tensors(tstats,device);xm=xs=None
    if step=="2":xm,xs=stats_tensors(xstats,device)
    loss_w=step1_loss_weights(a,device) if step=="1" else None
    role_w=role_class_weight_tensor(rstats,a,device);assoc_pw=association_pos_weight_tensor(astats,a,device)
    usefiles=list(files);random.shuffle(usefiles) if train else None;nb=0;start=time.time()
    for fi,path in enumerate(usefiles):
        graphs=load_graphs(path)
        loader=DataLoader(graphs,batch_size=batch_size,shuffle=train,num_workers=num_workers,pin_memory=device.type=="cuda")
        for batch in loader:
            batch=batch.to(device);target=batch.y.float().reshape(-1,8 if step=="1" else 1);target_norm=(target-tm)/ts
            role_target=batch.window_role_target.long().view(-1);assoc_target=batch.association_target.float().view(-1)
            if train:optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(train):
                out=model(batch.x,batch.edge_index,batch.edge_attr,batch=batch.batch,edge_weight=getattr(batch,"edge_weight",None),**hierarchical_model_kwargs(batch,step,xm,xs,pdg_vocab,device))
                pred_norm=out["prediction"]
                elem=regression_elementwise_loss(pred_norm,target_norm,a.regression_loss,a.huber_beta)
                reg=(elem*loss_w).sum()/(loss_w.sum()*max(int(elem.shape[0]),1)) if step=="1" else elem.mean()
                role=F.cross_entropy(out["window_role_logits"],role_target,weight=role_w) if role_target.numel() else reg.new_tensor(0.)
                assoc=F.binary_cross_entropy_with_logits(out["association_logits"],assoc_target,pos_weight=assoc_pw) if assoc_target.numel() else reg.new_tensor(0.)
                total=reg+a.window_role_loss_weight*role+a.association_loss_weight*assoc
                if train:total.backward();optimizer.step()
            pred_raw=pred_norm.detach()*ts+tm
            if step=="1":pred_phys=reconstruct_step1_absolute(pred_raw,batch);truth_phys=step1_truth_absolute(batch)
            else:pred_phys=pred_raw;truth_phys=target
            metrics.update(total,reg,role,assoc,pred_phys,truth_phys,out["window_role_logits"],role_target,out["association_logits"],assoc_target,int(batch.num_graphs))
            nb+=1
            if train and a.log_every>0 and nb%a.log_every==0:
                sm=metrics.summary();print(f"[{split_name}] epoch={epoch:03d} file={fi+1}/{len(usefiles)} batch={nb} total={sm['loss']:.5g} reg={sm['regression_loss']:.5g} objects={sm['n_objects']} elapsed={time.time()-start:.1f}s",flush=True)
        del graphs,loader
        if device.type=="cuda":torch.cuda.empty_cache()
    return metrics.summary()


def run_one_split_legacy(model,files,device,batch_size,num_workers,a,optimizer=None,train=False):
    model.train(train);m=MetricAccumulator(a.node_loss_weight,a.edge_loss_weight)
    for path in files:
        graphs=load_graphs(path);loader=DataLoader(graphs,batch_size=batch_size,shuffle=train,num_workers=num_workers)
        for b in loader:
            b=b.to(device)
            if train:optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(train):
                o=model(b.x,b.edge_index,b.edge_attr,batch=b.batch,edge_weight=getattr(b,"edge_weight",None));nm,em=batch_masks(b)
                nl,_=masked_bce_with_logits(o["node_logits"],b.y_node,nm,a.max_pos_weight);el,_=masked_bce_with_logits(o["edge_logits"],b.y_edge,em,a.max_pos_weight)
                loss=a.node_loss_weight*nl+a.edge_loss_weight*el
                if train:loss.backward();optimizer.step()
            m.update(loss)
    return m.summary()


def save_checkpoint(path,model,opt,epoch,a,step,tr,va,tstats=None,xstats=None,pdg=None,rstats=None,astats=None):
    torch.save({
        "epoch":epoch,"reconstruction_step":step,"model_state_dict":model.state_dict(),"optimizer_state_dict":opt.state_dict(),
        "args":vars(a),"train_metrics":tr,"val_metrics":va,"target_normalization":tstats,"step2_track_normalization":xstats,
        "pdg_vocab":{str(k):int(v) for k,v in (pdg or {}).items()},"window_role_stats":rstats,"association_stats":astats,
        "step1_target_names":STEP1_TARGET_NAMES,
    },path);print(f"[checkpoint] saved {path}")


def main():
    a=parse_args();resolve_depth_args(a);set_seed(a.seed)
    dataset=Path(a.dataset_dir);run=Path(a.run_dir);ck=run/"checkpoints";lg=run/"logs";out=run/"outputs";tb=run/"tensorboard"
    for d in (ck,lg,out,tb):d.mkdir(parents=True,exist_ok=True)
    dev=get_device(a.device);trainf=find_split_files(dataset,"train",a.max_train_files);valf=find_split_files(dataset,"val",a.max_val_files)
    step=verify_requested_step(a.reconstruction_step,infer_dataset_step(trainf));a.reconstruction_step=step
    print(f"[device] {dev}\n[task] step={step}\n[data] train={len(trainf)} val={len(valf)}")
    if step in {"1","2"}:
        print(f"[hierarchy] event_edge_mode={a.event_edge_mode} hybrid_dt={a.event_hybrid_bidirectional_ns} ns")
        print(f"[Step1 prompt context] P_same * P_prompt * exp(-|dt|/{a.step1_prompt_time_gate_ns} ns), hard gate |dt|<={a.step1_prompt_time_gate_ns} ns")

    resume=torch.load(a.resume,map_location="cpu",weights_only=False) if a.resume else None
    tstats=xstats=rstats=astats=None;pdg={}
    if step in {"1","2"}:
        if resume and resume.get("target_normalization") is not None:
            tstats=resume["target_normalization"];xstats=resume.get("step2_track_normalization");pdg={int(k):int(v) for k,v in resume.get("pdg_vocab",{}).items()};rstats=resume.get("window_role_stats");astats=resume.get("association_stats")
        if tstats is None or rstats is None or astats is None:
            stats_seed=a.seed if a.stats_seed is None else a.stats_seed
            tstats,xstats,pdg,rstats,astats=compute_regression_normalization(trainf,step,a.stats_max_events,stats_seed)
        print("[stats] target normalization")
        for n,m,sd in zip(tstats["names"],tstats["mean"],tstats["std"]):print(f"  {n:48s} mean={m: .6g} std={sd: .6g}")
        print(f"[stats] sampled events={astats['events']} objects={astats['objects']} files={astats.get('sampled_files',-1)} assoc_pos={astats['positive']} assoc_neg={astats['negative']}")

    model=build_model(a,step,max(pdg.values(),default=0)+1).to(dev);print(model)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=a.weight_decay)
    start_epoch=1;best=-float("inf") if step=="legacy" else float("inf");tie=float("inf")
    if resume:
        model.load_state_dict(resume["model_state_dict"])
        if not a.resume_reset_optimizer and "optimizer_state_dict" in resume:opt.load_state_dict(resume["optimizer_state_dict"])
        start_epoch=int(resume.get("epoch",0))+1
    writer=SummaryWriter(log_dir=str(tb)) if SummaryWriter else None
    with open(lg/"metrics.jsonl","a",encoding="utf-8") as f:
        for epoch in range(start_epoch,a.epochs+1):
            print(f"\n[epoch {epoch}/{a.epochs}] train")
            if step=="legacy":
                tr=run_one_split_legacy(model,trainf,dev,a.batch_size,a.num_workers,a,opt,True);print("[validation]");va=run_one_split_legacy(model,valf,dev,a.batch_size,a.num_workers,a,None,False)
            else:
                tr=run_one_split_regression(model,trainf,step,dev,a.batch_size,a.num_workers,tstats,xstats,pdg,rstats,astats,a,opt,True,epoch,"train")
                print("[validation]")
                va=run_one_split_regression(model,valf,step,dev,a.batch_size,a.num_workers,tstats,xstats,pdg,rstats,astats,a,None,False,epoch,"val")
            rec={"epoch":epoch,"reconstruction_step":step,"train":tr,"val":va};f.write(json.dumps(rec)+"\n");f.flush()
            if step=="1":
                print(f"[epoch {epoch:03d}] train_loss={tr['loss']:.6g} val_loss={va['loss']:.6g} reg={va['regression_loss']:.6g}\n  val: t0_MAE={va['incident_time_ns_mae']:.4g} ns posYZ_MAE={va['incident_position_yz_mae_mm']:.4g} mm angle={va['incident_direction_angle_mean_deg']:.4g} deg Ecl_relRMS={va['cluster_energy_relative_rms']:.4g}")
            elif step=="2":
                print(f"[epoch {epoch:03d}] train_loss={tr['loss']:.6g} val_loss={va['loss']:.6g} reg={va['regression_loss']:.6g}\n  val: Einc_MAE={va['incident_energy_mev_mae']:.4g} MeV relRMS={va['incident_energy_relative_rms']:.4g}")
            else:print(f"[epoch {epoch:03d}] train={tr['loss']:.6g} val={va['loss']:.6g}")
            if writer:
                for pre,mm in (("train",tr),("val",va)):
                    for k,v in mm.items():
                        if isinstance(v,(int,float)) and math.isfinite(float(v)):writer.add_scalar(f"{pre}/{k}",v,epoch)
            if epoch%a.save_every==0:save_checkpoint(ck/f"epoch_{epoch:03d}.pt",model,opt,epoch,a,step,tr,va,tstats,xstats,pdg,rstats,astats)
            criterion=float(va.get("best_selection_score",-1e99)) if step=="legacy" else float(va["regression_loss"]);total=float(va["loss"])
            better=(criterion>best if step=="legacy" else criterion<best) or (math.isclose(criterion,best,rel_tol=0,abs_tol=1e-12) and total<tie)
            if better:
                best=criterion;tie=total;save_checkpoint(ck/"best.pt",model,opt,epoch,a,step,tr,va,tstats,xstats,pdg,rstats,astats);print(f"[best] epoch={epoch} criterion={best:.6g}")
    if writer:writer.close()


if __name__=="__main__":main()
