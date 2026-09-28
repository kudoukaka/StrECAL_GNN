#!/usr/bin/env python3
"""Structural checker for the physics-aware hierarchical ECAL PyG dataset."""
from __future__ import annotations
import argparse, glob, json
from pathlib import Path
import torch
from torch_geometric.loader import DataLoader

NODE_DIM = 8
WINDOW_DIM = 13
STEP1_DIM = 8


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("dataset_dir")
    p.add_argument("--split",default="train",choices=["train","val","test"])
    p.add_argument("--batch-size",type=int,default=16)
    p.add_argument("--max-files",type=int,default=-1)
    p.add_argument("--max-graphs",type=int,default=-1)
    p.add_argument("--strict",action="store_true")
    return p.parse_args()


def graph_step(g):
    if hasattr(g,"reconstruction_step") and g.reconstruction_step.numel():
        v=int(g.reconstruction_step.view(-1)[0])
        return str(v) if v in (1,2) else "legacy"
    return "legacy"


def load(files,n):
    out=[]
    for f in files:
        x=torch.load(f,map_location="cpu",weights_only=False)
        out.extend(x if isinstance(x,list) else [x])
        if n>0 and len(out)>=n:return out[:n]
    return out


def check(g,i):
    e=[]; st=graph_step(g)
    for name in ("x", "edge_attr", "edge_weight", "window_x", "y"):
        value = getattr(g, name, None)
        if value is not None and not torch.isfinite(value).all():
            e.append(f"event {i}: nonfinite {name}")
    for name in ("node_y", "node_z", "window_centroid_y_mm", "window_centroid_z_mm"):
        value = getattr(g, name, None)
        if value is not None and (not torch.isfinite(value).all() or (value.abs() >= 1e9).any()):
            e.append(f"event {i}: invalid geometry in {name}")
    for name in ("sequence_id", "block_id", "crystal_id"):
        value = getattr(g, name, None)
        if value is not None and (value < 0).any():
            e.append(f"event {i}: invalid {name}")
    channel = getattr(g, "channel", None)
    if channel is not None and ((channel <= 0) | (((channel.long() >> 25) & 63) != 4)).any():
        e.append(f"event {i}: invalid ECAL channel")
    if st=="legacy":
        return e
    required=(
        "x","edge_index","edge_attr","node_window_local","num_windows","window_x",
        "window_tmin_ns","window_centroid_y_mm","window_centroid_z_mm","window_esum_mev",
        "window_role_target","association_target","num_association_pairs","num_targets",
        "target_prompt_window_local","target_prompt_cluster_id","target_primary_id","y",
    )
    for n in required:
        if not hasattr(g,n):e.append(f"event {i}: missing {n}")
    if e:return e
    nw=int(g.num_windows.view(-1)[0]); nt=int(g.num_targets.view(-1)[0]); npair=nw*(nw-1)//2
    if tuple(g.x.shape)[1:] != (NODE_DIM,): e.append(f"event {i}: x {tuple(g.x.shape)} expected [N,{NODE_DIM}]")
    if tuple(g.window_x.shape)!=(nw,WINDOW_DIM):e.append(f"event {i}: window_x {tuple(g.window_x.shape)} != ({nw},{WINDOW_DIM})")
    for n in ("window_tmin_ns","window_centroid_y_mm","window_centroid_z_mm","window_esum_mev"):
        if getattr(g,n).numel()!=nw:e.append(f"event {i}: {n} count != windows")
    if g.window_role_target.numel()!=nw:e.append(f"event {i}: role count != windows")
    if g.association_target.numel()!=npair:e.append(f"event {i}: association_target={g.association_target.numel()} expected {npair}")
    if int(g.num_association_pairs.view(-1)[0])!=npair:e.append(f"event {i}: num_association_pairs mismatch")
    if g.target_prompt_window_local.numel()!=nt:e.append(f"event {i}: prompt-anchor count != targets")
    if g.target_prompt_cluster_id.numel()!=nt or g.target_primary_id.numel()!=nt:e.append(f"event {i}: truth object id count != targets")
    dim=STEP1_DIM if st=="1" else 1
    if tuple(g.y.shape)!=(nt,dim):e.append(f"event {i}: y shape {tuple(g.y.shape)} != ({nt},{dim})")
    local=g.node_window_local.long().view(-1)
    if local.numel()!=g.num_nodes:e.append(f"event {i}: node_window_local != num_nodes")
    if local.numel() and (int(local.min())<0 or int(local.max())>=nw):e.append(f"event {i}: invalid node_window_local range")
    if g.edge_index.numel() and (local[g.edge_index[0]]!=local[g.edge_index[1]]).any():e.append(f"event {i}: pulse edge crosses seed windows")
    anchors=g.target_prompt_window_local.long().view(-1)
    if anchors.numel() and (int(anchors.min())<0 or int(anchors.max())>=nw):e.append(f"event {i}: invalid prompt anchor")
    if int((g.window_role_target==1).sum())<1:e.append(f"event {i}: no prompt-role window")
    if st=="1":
        for n in ("target_dt_ns","target_dy_mm","target_dz_mm","target_prompt_centroid_y_mm","target_prompt_centroid_z_mm","target_prompt_tmin_ns","target_incident_time_ns","target_incident_y_mm","target_incident_z_mm"):
            if not hasattr(g,n): e.append(f"event {i}: missing Step1 field {n}")
            elif getattr(g,n).numel()!=nt: e.append(f"event {i}: {n} count != targets")
        if hasattr(g,"target_dt_ns") and not torch.allclose(g.y[:,0],g.target_dt_ns): e.append(f"event {i}: y[:,0] != target_dt_ns")
        if hasattr(g,"target_dy_mm") and not torch.allclose(g.y[:,1],g.target_dy_mm): e.append(f"event {i}: y[:,1] != target_dy_mm")
        if hasattr(g,"target_dz_mm") and not torch.allclose(g.y[:,2],g.target_dz_mm): e.append(f"event {i}: y[:,2] != target_dz_mm")
    if st=="2":
        if tuple(g.step2_track_features.shape)!=(nt,6):e.append(f"event {i}: step2_track_features {tuple(g.step2_track_features.shape)} != ({nt},6)")
        if g.step2_pdg.numel()!=nt:e.append(f"event {i}: step2_pdg count != targets")
    return e


def main():
    a=parse_args();d=Path(a.dataset_dir)
    meta=json.loads((d/"metadata.json").read_text()) if (d/"metadata.json").exists() else {}
    files=sorted(glob.glob(str(d/f"{a.split}_*.pt")));files=files[:a.max_files] if a.max_files>0 else files
    if not files:raise SystemExit("no pt files")
    gs=load(files,a.max_graphs);st=graph_step(gs[0]);errs=[]
    for i,g in enumerate(gs):errs+=check(g,i)
    print(f"dataset={d}\nsplit={a.split}\nstep={st}\nevents={len(gs)}\nmetadata task={meta.get('task')}")
    if st!="legacy":
        nw=sum(int(g.num_windows.view(-1)[0]) for g in gs);nt=sum(int(g.num_targets.view(-1)[0]) for g in gs);npair=sum(int(g.association_target.numel()) for g in gs)
        print(f"node_dim={gs[0].x.shape[1]} window_dim={gs[0].window_x.shape[1]}")
        print(f"windows={nw} mean_windows/event={nw/max(len(gs),1):.4g}")
        print(f"truth objects={nt} mean_objects/event={nt/max(len(gs),1):.4g}\nassociation pairs={npair}")
        roles=torch.cat([g.window_role_target.view(-1) for g in gs]);print("window roles [other,prompt,nonprompt] =",torch.bincount(roles,minlength=3).tolist())
        if st=="1":
            y=torch.cat([g.y for g in gs],0)
            print("Step1 residual target mean [dt,dy,dz] =",y[:,:3].mean(0).tolist())
            print("Step1 residual target std  [dt,dy,dz] =",y[:,:3].std(0,unbiased=False).tolist())
        print("first event:",gs[0]);print("  y",tuple(gs[0].y.shape),"num_targets",int(gs[0].num_targets),"prompt windows",gs[0].target_prompt_window_local.tolist())
    print("invariants:","PASS" if not errs else "FAIL")
    for x in errs[:50]:print(" [error]",x)
    if st!="legacy":
        b=next(iter(DataLoader(gs,batch_size=min(a.batch_size,len(gs)),shuffle=False)))
        print("batch shapes:","x",tuple(b.x.shape),"window_x",tuple(b.window_x.shape),"window_tmin_ns",tuple(b.window_tmin_ns.shape),"y",tuple(b.y.shape))
        assert b.y.shape[0]==int(b.num_targets.sum());assert b.window_x.shape[0]==int(b.num_windows.sum())
    if errs and a.strict:raise SystemExit(2)

if __name__=="__main__":main()
