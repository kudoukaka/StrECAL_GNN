#!/usr/bin/env python3
"""
ecal_gnn_model.py

Physics-aware multi-object hierarchical ECAL GNN.

Regression hierarchy
--------------------
Pulse nodes -> Pulse encoder -> Pulse GNN x N -> pool pulses by seed window
-> Window encoder/fusion -> Event GNN x M -> window-role + association heads
-> per-anchor object contexts -> Step-1 / Step-2 heads.

Step 1 predicts residual targets (8 values per object):
  [dt = t_prompt_min - t0,
   dy = y_prompt_centroid - y_inc,
   dz = z_prompt_centroid - z_inc,
   dir_x, dir_y, dir_z,
   E_cluster, E_max]

Step 1 context is prompt-only.  For anchor i, member-window weight is
  w_ij = P_same(i,j) * P_prompt(j) * exp(-|dt_ij|/tau)
with a hard timing gate |dt_ij| <= prompt_time_gate_ns and w_ii = 1.
This suppresses delayed same-primary activity from Step-1 regression.

Step 2 uses the full same-primary object context (no prompt/timing suppression)
plus the full-event context, track ECAL-entry position/direction and PDG.
Track momentum is never an input.

Event GNN edge mode defaults to "hybrid":
  |dt| <= hybrid_bidirectional_ns : i <-> j
  |dt| >  hybrid_bidirectional_ns : earlier -> later
so simultaneous/overlapping prompt windows exchange information while delayed
activity remains causal in time.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.utils import softmax as pyg_softmax

STEP1_OUTPUT_DIM = 8
STEP1_TIMING_DIM = 1
STEP1_POSITION_DIM = 2
STEP1_DIRECTION_DIM = 3
STEP1_ENERGY_DIM = 2
STEP2_OUTPUT_DIM = 1

NODE_INPUT_DIM = 8
WINDOW_INPUT_DIM = 13
EVENT_EDGE_INPUT_DIM = 6
ASSOCIATION_EDGE_INPUT_DIM = 6
WINDOW_ROLE_CLASSES = 3


class MLP(nn.Module):
    """Configurable-depth MLP; depth is the number of Linear layers."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        depth: int = 2,
        final_activation: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError(f"MLP depth must be >= 1, got {depth}")
        dims = [int(input_dim)]
        if depth > 1:
            dims += [int(hidden_dim)] * (depth - 1)
        dims += [int(output_dim)]
        layers = []
        for i in range(depth):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            is_last = i == depth - 1
            if (not is_last) or final_activation:
                layers.append(nn.ReLU())
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)
        self.depth = int(depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ECALMessagePassingLayer(MessagePassing):
    def __init__(
        self,
        node_hidden_dim: int,
        edge_hidden_dim: int,
        message_hidden_dim: int,
        message_mlp_depth: int = 2,
        update_mlp_depth: int = 2,
        dropout: float = 0.0,
        aggr: str = "add",
    ):
        super().__init__(aggr=aggr)
        self.message_mlp = MLP(
            2 * node_hidden_dim + edge_hidden_dim,
            message_hidden_dim,
            node_hidden_dim,
            depth=message_mlp_depth,
            final_activation=True,
            dropout=dropout,
        )
        self.update_mlp = MLP(
            2 * node_hidden_dim,
            node_hidden_dim,
            node_hidden_dim,
            depth=update_mlp_depth,
            final_activation=True,
            dropout=dropout,
        )

    def forward(self, h, edge_index, edge_h, edge_weight=None):
        if edge_weight is None:
            edge_weight = h.new_ones((edge_index.shape[1],))
        else:
            edge_weight = edge_weight.to(dtype=h.dtype, device=h.device).view(-1)
        if edge_index.numel() == 0:
            m = h.new_zeros(h.shape)
        else:
            m = self.propagate(
                edge_index=edge_index,
                h=h,
                edge_h=edge_h,
                edge_weight=edge_weight,
            )
        return self.update_mlp(torch.cat([h, m], dim=-1))

    def message(self, h_i, h_j, edge_h, edge_weight):
        msg = self.message_mlp(torch.cat([h_i, h_j, edge_h], dim=-1))
        return msg * edge_weight.view(-1, 1)


def graph_pool_dimension(hidden_dim: int, pool: str) -> int:
    if pool in {"add", "mean", "max"}:
        return hidden_dim
    if pool in {"add_mean", "mean_max", "add_max"}:
        return 2 * hidden_dim
    if pool == "add_mean_max":
        return 3 * hidden_dim
    raise ValueError(f"Unknown pool: {pool}")


def pool_graph(h: torch.Tensor, batch: torch.Tensor, pool: str) -> torch.Tensor:
    add = global_add_pool(h, batch) if "add" in pool else None
    mean = global_mean_pool(h, batch) if "mean" in pool else None
    mx = global_max_pool(h, batch) if "max" in pool else None
    if pool == "add": return add
    if pool == "mean": return mean
    if pool == "max": return mx
    if pool == "add_mean": return torch.cat([add, mean], dim=-1)
    if pool == "mean_max": return torch.cat([mean, mx], dim=-1)
    if pool == "add_max": return torch.cat([add, mx], dim=-1)
    if pool == "add_mean_max": return torch.cat([add, mean, mx], dim=-1)
    raise ValueError(f"Unknown pool: {pool}")


def _hierarchy_indices(pulse_batch, node_window_local, num_windows):
    num_windows = num_windows.long().view(-1)
    batch_size = int(num_windows.numel())
    device = pulse_batch.device
    window_batch = torch.repeat_interleave(
        torch.arange(batch_size, device=device, dtype=torch.long), num_windows
    )
    offsets = torch.cumsum(num_windows, dim=0) - num_windows
    global_window = node_window_local.long().view(-1) + offsets[pulse_batch.long()]
    return global_window, window_batch, offsets


def _pair_features(wx_i: torch.Tensor, wx_j: torch.Tensor) -> torch.Tensor:
    """Event/association edge features from model-scale window_x.

    window_x columns used here:
      0 yc, 1 zc, 2 tmin, 3 tmean, 4 log1p(Esum), ...
    """
    dy = wx_j[0] - wx_i[0]
    dz = wx_j[1] - wx_i[1]
    dr = torch.sqrt(torch.clamp(dy * dy + dz * dz, min=0.0))
    dt = wx_j[2] - wx_i[2]
    dmean_t = wx_j[3] - wx_i[3]
    dactivity = wx_j[4] - wx_i[4]
    return torch.stack([dy, dz, dr, dt, dmean_t, dactivity])


def _build_event_edges(
    window_x: torch.Tensor,
    window_tmin_ns: torch.Tensor,
    num_windows: torch.Tensor,
    mode: str = "hybrid",
    hybrid_bidirectional_ns: float = 30.0,
):
    """Construct event-level window edges.

    hybrid:
      if |dt_physical| <= hybrid_bidirectional_ns, keep both i->j and j->i;
      otherwise keep only earlier->later.
    """
    if mode not in {"hybrid", "time_directed", "bidirectional_complete"}:
        raise ValueError(f"Unknown event_edge_mode: {mode}")
    src_all, dst_all, attr_all, weight_all = [], [], [], []
    offset = 0
    for nt in num_windows.long().view(-1):
        n = int(nt.item())
        wx = window_x[offset:offset+n]
        wt = window_tmin_ns[offset:offset+n]
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                ti = float(wt[i].detach().item())
                tj = float(wt[j].detach().item())
                keep = False
                if mode == "bidirectional_complete":
                    keep = True
                elif mode == "time_directed":
                    keep = (tj > ti) or (tj == ti and j > i)
                else:  # hybrid
                    if abs(tj - ti) <= float(hybrid_bidirectional_ns):
                        keep = True
                    else:
                        keep = (tj > ti) or (tj == ti and j > i)
                if not keep:
                    continue
                feat = _pair_features(wx[i], wx[j])
                src_all.append(i + offset)
                dst_all.append(j + offset)
                attr_all.append(feat)
                weight_all.append(1.0 / (1.0 + torch.abs(feat[2])))
        offset += n
    if not src_all:
        return (
            torch.empty((2, 0), dtype=torch.long, device=window_x.device),
            window_x.new_empty((0, EVENT_EDGE_INPUT_DIM)),
            window_x.new_empty((0,)),
        )
    return (
        torch.tensor([src_all, dst_all], dtype=torch.long, device=window_x.device),
        torch.stack(attr_all, dim=0).to(window_x),
        torch.stack(weight_all, dim=0).to(window_x),
    )


def _build_association_pairs(window_x, window_tmin_ns, num_windows):
    """One unordered pair per window pair, encoded earlier -> later."""
    src_all, dst_all, attr_all = [], [], []
    offset = 0
    for nt in num_windows.long().view(-1):
        n = int(nt.item())
        wx = window_x[offset:offset+n]
        wt = window_tmin_ns[offset:offset+n]
        for a in range(n):
            for b in range(a + 1, n):
                ta = float(wt[a].detach().item())
                tb = float(wt[b].detach().item())
                if tb < ta or (tb == ta and b < a):
                    i, j = b, a
                else:
                    i, j = a, b
                src_all.append(i + offset)
                dst_all.append(j + offset)
                attr_all.append(_pair_features(wx[i], wx[j]))
        offset += n
    if not src_all:
        return (
            torch.empty((2, 0), dtype=torch.long, device=window_x.device),
            window_x.new_empty((0, ASSOCIATION_EDGE_INPUT_DIM)),
        )
    return (
        torch.tensor([src_all, dst_all], dtype=torch.long, device=window_x.device),
        torch.stack(attr_all, dim=0).to(window_x),
    )


def _association_matrix_for_event(
    ids: torch.Tensor,
    window_batch: torch.Tensor,
    pair_index: torch.Tensor,
    association_prob: torch.Tensor,
) -> torch.Tensor:
    n = int(ids.numel())
    W = torch.eye(n, dtype=association_prob.dtype if association_prob.numel() else torch.float32, device=ids.device)
    if n == 0 or pair_index.numel() == 0:
        return W
    # ids are contiguous within a batched event, but keep mapping explicit.
    global_to_local = {int(g.item()): k for k, g in enumerate(ids)}
    pair_src = pair_index[0]
    event_id = int(window_batch[ids[0]].item())
    mask = window_batch[pair_src] == event_id
    for q in torch.nonzero(mask, as_tuple=False).view(-1).tolist():
        gs = int(pair_index[0, q].item())
        gd = int(pair_index[1, q].item())
        if gs not in global_to_local or gd not in global_to_local:
            continue
        s = global_to_local[gs]
        d = global_to_local[gd]
        p = association_prob[q]
        W[s, d] = p
        W[d, s] = p
    return W


def _weighted_add_mean_max(h: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """Return per-anchor [weighted add | weighted mean | weighted max]."""
    weighted = W.unsqueeze(-1) * h.unsqueeze(0)  # [anchor, member, hidden]
    add = weighted.sum(dim=1)
    mean = add / torch.clamp(W.sum(dim=1, keepdim=True), min=1.0e-6)
    mx = weighted.max(dim=1).values
    return torch.cat([add, mean, mx], dim=-1)


def _build_full_and_prompt_object_contexts(
    window_h: torch.Tensor,
    window_batch: torch.Tensor,
    window_tmin_ns: torch.Tensor,
    pair_index: torch.Tensor,
    association_prob: torch.Tensor,
    prompt_prob: torch.Tensor,
    prompt_time_gate_ns: float,
):
    """Build full same-primary and prompt-only object contexts for each anchor."""
    full_pieces = []
    prompt_pieces = []
    nevent = int(window_batch.max().item()) + 1 if window_batch.numel() else 0
    gate = float(prompt_time_gate_ns)
    tau = max(gate, 1.0e-6)

    for event_id in range(nevent):
        ids = torch.nonzero(window_batch == event_id, as_tuple=False).view(-1)
        if ids.numel() == 0:
            continue
        h = window_h[ids]
        W_full = _association_matrix_for_event(ids, window_batch, pair_index, association_prob).to(h)
        full_pieces.append(_weighted_add_mean_max(h, W_full))

        # Prompt-only context: association * promptness * temporal gate.
        t = window_tmin_ns[ids].to(h)
        pp = prompt_prob[ids].to(h)
        dt = torch.abs(t.unsqueeze(1) - t.unsqueeze(0))
        temporal = torch.exp(-dt / tau)
        if gate > 0.0:
            temporal = temporal * (dt <= gate).to(h.dtype)
        # Member prompt probability multiplies each column j.
        W_prompt = W_full * pp.unsqueeze(0) * temporal
        # Always retain the anchor itself so the context never vanishes.
        diag = torch.arange(ids.numel(), device=h.device)
        W_prompt[diag, diag] = 1.0
        prompt_pieces.append(_weighted_add_mean_max(h, W_prompt))

    if not full_pieces:
        empty = window_h.new_empty((0, 3 * window_h.shape[1]))
        return empty, empty
    return torch.cat(full_pieces, dim=0), torch.cat(prompt_pieces, dim=0)


def _target_global_indices(num_windows, num_targets, target_prompt_window_local):
    num_windows = num_windows.long().view(-1)
    num_targets = num_targets.long().view(-1)
    target_local = target_prompt_window_local.long().view(-1)
    if int(target_local.numel()) != int(num_targets.sum().item()):
        raise ValueError("target_prompt_window_local count != sum(num_targets)")
    win_offsets = torch.cumsum(num_windows, dim=0) - num_windows
    target_batch = torch.repeat_interleave(
        torch.arange(num_targets.numel(), device=num_targets.device, dtype=torch.long),
        num_targets,
    )
    return target_local + win_offsets[target_batch], target_batch


class ECALGNN(nn.Module):
    def __init__(
        self,
        reconstruction_step="1",
        node_input_dim=NODE_INPUT_DIM,
        edge_input_dim=6,
        node_hidden_dim=64,
        edge_hidden_dim=64,
        node_encoder_mid_dim=32,
        edge_encoder_mid_dim=32,
        message_hidden_dim=128,
        regression_hidden_dim=128,
        track_hidden_dim=64,
        track_embedding_dim=64,
        pdg_embedding_dim=8,
        pdg_vocab_size=1,
        num_gnn_layers=3,
        node_encoder_depth=2,
        edge_encoder_depth=2,
        message_mlp_depth=2,
        update_mlp_depth=2,
        node_head_depth=2,
        edge_head_depth=2,
        regression_head_depth=2,
        step1_timing_head_depth=None,
        step1_position_head_depth=None,
        step1_direction_head_depth=None,
        step1_energy_head_depth=None,
        track_encoder_depth=2,
        fusion_head_depth=2,
        window_input_dim=WINDOW_INPUT_DIM,
        window_hidden_dim=64,
        event_edge_hidden_dim=32,
        event_message_hidden_dim=128,
        association_edge_hidden_dim=32,
        association_hidden_dim=128,
        num_event_gnn_layers=2,
        window_encoder_depth=2,
        window_fusion_depth=2,
        event_edge_encoder_depth=2,
        event_message_mlp_depth=2,
        event_update_mlp_depth=2,
        window_role_head_depth=2,
        association_edge_encoder_depth=2,
        association_head_depth=2,
        graph_pool="add_mean_max",
        event_pool="add_mean_max",
        event_edge_mode="hybrid",
        event_hybrid_bidirectional_ns=30.0,
        step1_prompt_time_gate_ns=30.0,
        dropout=0.0,
        aggr="add",
        event_aggr="add",
    ):
        super().__init__()
        step = str(reconstruction_step).lower()
        if step in {"step1", "1"}: step = "1"
        elif step in {"step2", "2"}: step = "2"
        elif step in {"legacy", "0"}: step = "legacy"
        else: raise ValueError(f"invalid reconstruction_step {reconstruction_step}")
        self.reconstruction_step = step
        self.graph_pool = str(graph_pool)
        self.event_pool = str(event_pool)
        self.event_edge_mode = str(event_edge_mode)
        self.event_hybrid_bidirectional_ns = float(event_hybrid_bidirectional_ns)
        self.step1_prompt_time_gate_ns = float(step1_prompt_time_gate_ns)

        self.node_encoder = MLP(node_input_dim, node_encoder_mid_dim, node_hidden_dim, node_encoder_depth, True, dropout)
        self.edge_encoder = MLP(edge_input_dim, edge_encoder_mid_dim, edge_hidden_dim, edge_encoder_depth, True, dropout)
        self.gnn_layers = nn.ModuleList([
            ECALMessagePassingLayer(
                node_hidden_dim, edge_hidden_dim, message_hidden_dim,
                message_mlp_depth, update_mlp_depth, dropout, aggr
            ) for _ in range(num_gnn_layers)
        ])

        if step == "legacy":
            self.node_head = MLP(node_hidden_dim, max(node_hidden_dim // 2, 1), 1, node_head_depth, False, dropout)
            self.edge_head = MLP(2 * node_hidden_dim + edge_hidden_dim, node_hidden_dim, 1, edge_head_depth, False, dropout)
            return

        pulse_pool_dim = graph_pool_dimension(node_hidden_dim, graph_pool)
        self.window_encoder = MLP(window_input_dim, window_hidden_dim, window_hidden_dim, window_encoder_depth, True, dropout)
        self.window_fusion = MLP(pulse_pool_dim + window_hidden_dim, window_hidden_dim, window_hidden_dim, window_fusion_depth, True, dropout)
        self.event_edge_encoder = MLP(EVENT_EDGE_INPUT_DIM, event_edge_hidden_dim, event_edge_hidden_dim, event_edge_encoder_depth, True, dropout)
        self.event_gnn_layers = nn.ModuleList([
            ECALMessagePassingLayer(
                window_hidden_dim, event_edge_hidden_dim, event_message_hidden_dim,
                event_message_mlp_depth, event_update_mlp_depth, dropout, event_aggr
            ) for _ in range(num_event_gnn_layers)
        ])
        self.window_role_head = MLP(window_hidden_dim, window_hidden_dim, WINDOW_ROLE_CLASSES, window_role_head_depth, False, dropout)
        self.association_edge_encoder = MLP(
            ASSOCIATION_EDGE_INPUT_DIM, association_edge_hidden_dim, association_edge_hidden_dim,
            association_edge_encoder_depth, True, dropout,
        )
        self.association_head = MLP(
            2 * window_hidden_dim + association_edge_hidden_dim,
            association_hidden_dim, 1, association_head_depth, False, dropout,
        )

        object_pool_dim = 3 * window_hidden_dim
        event_dim = graph_pool_dimension(window_hidden_dim, event_pool)

        if step == "1":
            td = regression_head_depth if step1_timing_head_depth is None else step1_timing_head_depth
            pd = regression_head_depth if step1_position_head_depth is None else step1_position_head_depth
            dd = regression_head_depth if step1_direction_head_depth is None else step1_direction_head_depth
            ed = regression_head_depth if step1_energy_head_depth is None else step1_energy_head_depth
            prompt_head_input = window_hidden_dim + object_pool_dim
            self.step1_timing_head = MLP(prompt_head_input, regression_hidden_dim, STEP1_TIMING_DIM, td, False, dropout)
            self.step1_position_head = MLP(prompt_head_input, regression_hidden_dim, STEP1_POSITION_DIM, pd, False, dropout)
            self.step1_direction_head = MLP(prompt_head_input, regression_hidden_dim, STEP1_DIRECTION_DIM, dd, False, dropout)
            self.step1_energy_head = MLP(object_pool_dim, regression_hidden_dim, STEP1_ENERGY_DIM, ed, False, dropout)
        else:
            if pdg_vocab_size < 1:
                raise ValueError("pdg_vocab_size must be >=1")
            self.track_encoder = MLP(6, track_hidden_dim, track_embedding_dim, track_encoder_depth, True, dropout)
            self.pdg_embedding = nn.Embedding(pdg_vocab_size, pdg_embedding_dim)
            fusion_dim = object_pool_dim + event_dim + track_embedding_dim + pdg_embedding_dim
            self.fusion_head = MLP(fusion_dim, regression_hidden_dim, STEP2_OUTPUT_DIM, fusion_head_depth, False, dropout)

    def encode_pulses(self, x, edge_index, edge_attr, edge_weight=None):
        h = self.node_encoder(x)
        eh = self.edge_encoder(edge_attr)
        for layer in self.gnn_layers:
            h = layer(h, edge_index, eh, edge_weight)
        return {"node_hidden": h, "edge_hidden": eh}

    def encode_hierarchy(
        self,
        pulse_h,
        pulse_batch,
        node_window_local,
        num_windows,
        window_x,
        window_tmin_ns,
    ):
        global_window, window_batch, _ = _hierarchy_indices(pulse_batch, node_window_local, num_windows)
        if int(window_x.shape[0]) != int(num_windows.long().sum().item()):
            raise ValueError("window_x rows != sum(num_windows)")
        if int(window_tmin_ns.numel()) != int(window_x.shape[0]):
            raise ValueError("window_tmin_ns count != window_x rows")

        pulse_window_h = pool_graph(pulse_h, global_window, self.graph_pool)
        window_h = self.window_fusion(torch.cat([pulse_window_h, self.window_encoder(window_x)], dim=-1))

        event_edge_index, event_edge_attr, event_edge_weight = _build_event_edges(
            window_x, window_tmin_ns, num_windows,
            mode=self.event_edge_mode,
            hybrid_bidirectional_ns=self.event_hybrid_bidirectional_ns,
        )
        event_edge_h = self.event_edge_encoder(event_edge_attr)
        for layer in self.event_gnn_layers:
            window_h = layer(window_h, event_edge_index, event_edge_h, event_edge_weight)

        role_logits = self.window_role_head(window_h)
        prompt_prob = torch.softmax(role_logits, dim=1)[:, 1]
        prompt_attention = pyg_softmax(role_logits[:, 1], window_batch)

        pair_index, pair_attr = _build_association_pairs(window_x, window_tmin_ns, num_windows)
        pair_edge_h = self.association_edge_encoder(pair_attr)
        if pair_index.numel():
            association_logits = self.association_head(
                torch.cat([window_h[pair_index[0]], window_h[pair_index[1]], pair_edge_h], dim=-1)
            ).squeeze(-1)
            association_prob = torch.sigmoid(association_logits)
        else:
            association_logits = window_h.new_empty((0,))
            association_prob = window_h.new_empty((0,))

        full_object, prompt_object = _build_full_and_prompt_object_contexts(
            window_h=window_h,
            window_batch=window_batch,
            window_tmin_ns=window_tmin_ns,
            pair_index=pair_index,
            association_prob=association_prob,
            prompt_prob=prompt_prob,
            prompt_time_gate_ns=self.step1_prompt_time_gate_ns,
        )
        event_h = pool_graph(window_h, window_batch, self.event_pool)

        return {
            "window_hidden": window_h,
            "window_batch": window_batch,
            "event_edge_index": event_edge_index,
            "event_edge_attr": event_edge_attr,
            "event_edge_weight": event_edge_weight,
            "window_role_logits": role_logits,
            "window_prompt_prob": prompt_prob,
            "prompt_attention": prompt_attention,
            "association_pair_index": pair_index,
            "association_pair_attr": pair_attr,
            "association_logits": association_logits,
            "association_prob": association_prob,
            "full_object_add_mean_max": full_object,
            "prompt_object_add_mean_max": prompt_object,
            "event_hidden": event_h,
        }

    def forward(
        self,
        x,
        edge_index,
        edge_attr,
        batch=None,
        edge_weight=None,
        node_window_local=None,
        num_windows=None,
        window_x=None,
        window_tmin_ns=None,
        num_targets=None,
        target_prompt_window_local=None,
        step2_track_features=None,
        step2_pdg_index=None,
    ):
        if batch is None:
            batch = x.new_zeros((x.shape[0],), dtype=torch.long)
        enc = self.encode_pulses(x, edge_index, edge_attr, edge_weight)
        h = enc["node_hidden"]
        eh = enc["edge_hidden"]

        if self.reconstruction_step == "legacy":
            node_logits = self.node_head(h).squeeze(-1)
            if edge_index.numel():
                edge_logits = self.edge_head(
                    torch.cat([h[edge_index[0]], h[edge_index[1]], eh], dim=-1)
                ).squeeze(-1)
            else:
                edge_logits = x.new_empty((0,))
            enc.update({"node_logits": node_logits, "edge_logits": edge_logits})
            return enc

        if node_window_local is None or num_windows is None or window_x is None or window_tmin_ns is None:
            raise ValueError("hierarchical regression requires node_window_local,num_windows,window_x,window_tmin_ns")

        hier = self.encode_hierarchy(
            h, batch, node_window_local, num_windows, window_x, window_tmin_ns
        )
        enc.update(hier)

        if self.reconstruction_step == "1":
            prompt_head_input = torch.cat(
                [hier["window_hidden"], hier["prompt_object_add_mean_max"]], dim=-1
            )
            timing = self.step1_timing_head(prompt_head_input)
            position = self.step1_position_head(prompt_head_input)
            direction = self.step1_direction_head(prompt_head_input)
            energy = self.step1_energy_head(hier["prompt_object_add_mean_max"])
            candidate = torch.cat([timing, position, direction, energy], dim=-1)
            enc["window_prediction"] = candidate
            if num_targets is not None and target_prompt_window_local is not None:
                gidx, target_batch = _target_global_indices(
                    num_windows, num_targets, target_prompt_window_local
                )
                enc["target_global_window_index"] = gidx
                enc["target_batch"] = target_batch
                enc["prediction"] = candidate[gidx]
            else:
                enc["prediction"] = candidate
            return enc

        if num_targets is None or target_prompt_window_local is None:
            raise ValueError("Step2 requires num_targets and target_prompt_window_local")
        if step2_track_features is None or step2_pdg_index is None:
            raise ValueError("Step2 requires track entry/direction and PDG for each target")

        gidx, target_batch = _target_global_indices(
            num_windows, num_targets, target_prompt_window_local
        )
        track = step2_track_features.reshape(-1, 6)
        pdg_idx = step2_pdg_index.long().view(-1)
        if int(track.shape[0]) != int(gidx.numel()) or int(pdg_idx.numel()) != int(gidx.numel()):
            raise ValueError("Step2 track/PDG count != sum(num_targets)")

        track_h = self.track_encoder(track)
        pdg_h = self.pdg_embedding(pdg_idx)
        object_h = hier["full_object_add_mean_max"][gidx]
        event_h = hier["event_hidden"][target_batch]
        fused = torch.cat([object_h, event_h, track_h, pdg_h], dim=-1)
        enc.update({
            "target_global_window_index": gidx,
            "target_batch": target_batch,
            "track_hidden": track_h,
            "pdg_hidden": pdg_h,
            "fusion_hidden": fused,
            "prediction": self.fusion_head(fused),
        })
        return enc


ECALClusteringGNN = ECALGNN
