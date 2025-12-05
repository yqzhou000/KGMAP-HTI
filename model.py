# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphOperation(nn.Module):
    def __init__(self):
        super(GraphOperation, self).__init__()
    
    def forward(self, x, adj_matrices, idx):
        return torch.spmm(adj_matrices[idx], x)


class AdaptiveMetaGraphCell(nn.Module):
    def __init__(self, n_step, n_hid_prev, n_hid, use_norm=True, use_nl=True):
        super(AdaptiveMetaGraphCell, self).__init__()
        
        self.n_step = n_step
        self.n_hid = n_hid
        self.use_norm = use_norm
        self.use_nl = use_nl
        
        self.affine = nn.Linear(n_hid_prev, n_hid)
        
        self.norm = nn.LayerNorm(n_hid) if use_norm else nn.Identity()
        
        self.ops_seq = nn.ModuleList()
        for i in range(self.n_step):
            self.ops_seq.append(GraphOperation())

        self.ops_res = nn.ModuleList()
        for i in range(1, self.n_step):
            for j in range(i):
                self.ops_res.append(GraphOperation())
    
    def forward(self, x, adjs, idxes_seq, idxes_res):
        x = self.affine(x)
        
        states = [x]
        offset = 0
        
        for i in range(self.n_step):
            seqi = self.ops_seq[i](states[i], adjs[:-1], idxes_seq[i])
            
            resi = sum(
                self.ops_res[offset + j](h, adjs, idxes_res[offset + j])
                for j, h in enumerate(states[:i])
            ) if i > 0 else 0
            
            offset += i
            
            states.append(seqi + resi)
        
        output = self.norm(states[-1])
        
        if self.use_nl:
            output = F.gelu(output)
        
        return output


class PathEncoder(nn.Module):
    def __init__(self, n_hid, dropout=0.2):
        super(PathEncoder, self).__init__()
        self.n_hid = n_hid
        
        self.path_lstm = nn.LSTM(n_hid, n_hid, batch_first=True,
                                 bidirectional=True)
        self.path_proj = nn.Linear(2 * n_hid, n_hid)
        
        self.path_attention = nn.MultiheadAttention(
            n_hid, num_heads=4, dropout=dropout
        )
        
        self.disease_context_encoder = nn.Linear(n_hid, n_hid)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, herb_emb, ingredient_embs, target_emb, disease_embs=None):
        path_embeddings = []
        
        for ingredient_emb in ingredient_embs:
            path_seq = torch.stack([
                herb_emb.squeeze(0),
                ingredient_emb,
                target_emb.squeeze(0)
            ], dim=0).unsqueeze(0)
            
            path_out, _ = self.path_lstm(path_seq)
            path_emb = self.path_proj(path_out[:, -1, :])
            
            if disease_embs is not None and len(disease_embs) > 0:
                disease_context = torch.mean(disease_embs, dim=0).unsqueeze(0)
                disease_context = self.disease_context_encoder(disease_context)
                path_emb = path_emb + 0.1 * disease_context
            
            path_embeddings.append(path_emb)
        
        if len(path_embeddings) == 0:
            return herb_emb + target_emb
        
        path_embeddings = torch.cat(path_embeddings, dim=0)
        
        if len(path_embeddings) == 1:
            return path_embeddings
        
        path_embeddings = path_embeddings.unsqueeze(1)
        attended_paths, _ = self.path_attention(
            path_embeddings, path_embeddings, path_embeddings
        )
        
        final_path_emb = torch.mean(attended_paths, dim=0)
        
        return final_path_emb


class CompletePathEncoder(nn.Module):
    def __init__(self, n_hid, dropout=0.2):
        super(CompletePathEncoder, self).__init__()
        self.n_hid = n_hid
        
        self.complete_path_lstm = nn.LSTM(n_hid, n_hid, batch_first=True,
                                          bidirectional=True)
        self.complete_path_proj = nn.Linear(2 * n_hid, n_hid)
        
        self.path_attention = nn.MultiheadAttention(
            n_hid, num_heads=4, dropout=dropout
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, herb_emb, ingredient_emb, target_emb, disease_emb):
        path_seq = torch.stack([
            herb_emb.squeeze(0) if herb_emb.dim() > 1 else herb_emb,
            ingredient_emb,
            target_emb.squeeze(0) if target_emb.dim() > 1 else target_emb,
            disease_emb
        ], dim=0).unsqueeze(0)
        
        path_out, _ = self.complete_path_lstm(path_seq)
        path_emb = self.complete_path_proj(path_out[:, -1, :])
        
        return path_emb


class KGMAP_HTI(nn.Module):

    def __init__(self, in_dims, n_hid, n_steps, dropout=0.2, attn_dim=64,
                 use_norm=True, out_nl=True, use_path_enhancement=True):
        super(KGMAP_HTI, self).__init__()
        
        self.n_hid = n_hid
        self.use_path_enhancement = use_path_enhancement
        
        self.node_embeddings = nn.ModuleList()
        assert isinstance(in_dims, list), "in_dims must be a list"
        
        for i in range(len(in_dims)):
            self.node_embeddings.append(nn.Linear(in_dims[i], n_hid))
        
        assert isinstance(n_steps, list), "n_steps must be a list"
        self.gcn_layers = nn.ModuleList()
        
        for i in range(len(n_steps)):
            self.gcn_layers.append(
                AdaptiveMetaGraphCell(
                    n_steps[i], n_hid, n_hid,
                    use_norm=use_norm, use_nl=out_nl
                )
            )
        

        self.attn_fc1 = nn.Linear(n_hid, attn_dim)
        self.attn_fc2 = nn.Linear(attn_dim, 1)
        
        if self.use_path_enhancement:
            self.path_encoder = PathEncoder(n_hid, dropout)
            self.complete_path_encoder = CompletePathEncoder(n_hid, dropout)
        
        self.predictor = nn.Sequential(
            nn.Linear(n_hid, n_hid // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_hid // 2, 1),
            nn.Sigmoid()
        )
        
        self.feats_drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, node_feats, node_types, adjs, idxes_seq, idxes_res, 
                herb_target_pairs=None, paths_info=None):

        hid = torch.zeros((node_types.size(0), self.n_hid), device=node_types.device)
        
        for i in range(len(node_feats)):
            node_mask = (node_types == i)
            hid[node_mask] = self.node_embeddings[i](node_feats[i])
        
        hid = self.feats_drop(hid)
        
        layer_outputs = []
        layer_attns = []
        
        for i, gcn_layer in enumerate(self.gcn_layers):
            hid_i = gcn_layer(hid, adjs, idxes_seq[i], idxes_res[i])
            layer_outputs.append(hid_i)
            
            attn_i = self.attn_fc2(torch.tanh(self.attn_fc1(hid_i)))
            layer_attns.append(attn_i)
        
        hids = torch.stack(layer_outputs, dim=0).transpose(0, 1)
        attns = F.softmax(torch.cat(layer_attns, dim=-1), dim=-1)
        
        out = (attns.unsqueeze(dim=-1) * hids).sum(dim=1)

        if herb_target_pairs is None:
            return out
        
        predictions = []
        batch_size = herb_target_pairs.size(0)
        
        for i in range(batch_size):
            herb_id = herb_target_pairs[i, 0]
            target_id = herb_target_pairs[i, 1]

            herb_emb = out[herb_id].unsqueeze(0)
            target_emb = out[target_id].unsqueeze(0)
            
            if self.use_path_enhancement and paths_info is not None and \
               i < len(paths_info) and len(paths_info[i]) > 0:
                
                path_info_list = paths_info[i]
                ingredient_embs = []
                disease_embs = []
                
                herb_offset = 0
                target_offset = len(node_feats[0])
                ingredient_offset = target_offset + len(node_feats[1])
                disease_offset = ingredient_offset + len(node_feats[2])
                
                for path_info in path_info_list:
                    ingredient_id = path_info['ingredient_id']
                    
                    ingredient_node_id = ingredient_offset + \
                                        (ingredient_id % len(node_feats[2]))
                    
                    if ingredient_node_id < out.size(0):
                        ingredient_embs.append(out[ingredient_node_id])
                    
                    for disease_id in path_info.get('related_diseases', []):
                        disease_node_id = disease_offset + \
                                         (disease_id % len(node_feats[3]))
                        if disease_node_id < out.size(0):
                            disease_embs.append(out[disease_node_id])
                
                if len(ingredient_embs) > 0:
                    ingredient_embs = torch.stack(ingredient_embs)
                    disease_embs = torch.stack(disease_embs) if len(disease_embs) > 0 else None
                    
                    path_emb = self.path_encoder(
                        herb_emb, ingredient_embs, target_emb, disease_embs
                    )
                    prediction = self.predictor(path_emb).squeeze()
                else:
                    combined_emb = herb_emb + target_emb
                    prediction = self.predictor(combined_emb).squeeze()
            else:
                combined_emb = herb_emb + target_emb
                prediction = self.predictor(combined_emb).squeeze()
            
            predictions.append(prediction)
        
        return torch.stack(predictions)


class KGMAP_HTI_Baseline(nn.Module):
    def __init__(self, in_dims, n_hid, n_steps, dropout=0.2, attn_dim=64,
                 use_norm=True, out_nl=True):
        super(KGMAP_HTI_Baseline, self).__init__()
        
        self.n_hid = n_hid
        
        self.node_embeddings = nn.ModuleList()
        assert isinstance(in_dims, list)
        
        for i in range(len(in_dims)):
            self.node_embeddings.append(nn.Linear(in_dims[i], n_hid))
        
        assert isinstance(n_steps, list)
        self.gcn_layers = nn.ModuleList()
        
        for i in range(len(n_steps)):
            self.gcn_layers.append(
                AdaptiveMetaGraphCell(
                    n_steps[i], n_hid, n_hid,
                    use_norm=use_norm, use_nl=out_nl
                )
            )
        
        self.attn_fc1 = nn.Linear(n_hid, attn_dim)
        self.attn_fc2 = nn.Linear(attn_dim, 1)
        
        self.feats_drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, node_feats, node_types, adjs, idxes_seq, idxes_res):
        hid = torch.zeros((node_types.size(0), self.n_hid), device=node_types.device)
        
        for i in range(len(node_feats)):
            node_mask = (node_types == i)
            hid[node_mask] = self.node_embeddings[i](node_feats[i])
        
        hid = self.feats_drop(hid)
        
        layer_outputs = []
        layer_attns = []
        
        for i, gcn_layer in enumerate(self.gcn_layers):
            hid_i = gcn_layer(hid, adjs, idxes_seq[i], idxes_res[i])
            layer_outputs.append(hid_i)
            
            attn_i = self.attn_fc2(torch.tanh(self.attn_fc1(hid_i)))
            layer_attns.append(attn_i)
        
        hids = torch.stack(layer_outputs, dim=0).transpose(0, 1)
        attns = F.softmax(torch.cat(layer_attns, dim=-1), dim=-1)
        
        out = (attns.unsqueeze(dim=-1) * hids).sum(dim=1)
        
        return out
