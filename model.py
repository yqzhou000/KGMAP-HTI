import torch
import torch.nn as nn
import torch.nn.functional as F

class Op(nn.Module):
    """图操作"""
    def __init__(self):
        super(Op, self).__init__()
    
    def forward(self, x, adjs, idx):
        return torch.spmm(adjs[idx], x)

class Cell(nn.Module):
    """图神经网络单元"""
    def __init__(self, n_step, n_hid_prev, n_hid, use_norm=True, use_nl=True):
        super(Cell, self).__init__()
        self.affine = nn.Linear(n_hid_prev, n_hid)
        self.n_step = n_step
        self.norm = nn.LayerNorm(n_hid) if use_norm else lambda x: x
        self.use_nl = use_nl
        
        self.ops_seq = nn.ModuleList()
        self.ops_res = nn.ModuleList()
        
        for i in range(self.n_step):
            self.ops_seq.append(Op())
        
        for i in range(1, self.n_step):
            for j in range(i):
                self.ops_res.append(Op())
    
    def forward(self, x, adjs, idxes_seq, idxes_res):
        x = self.affine(x)
        states = [x]
        offset = 0
        
        for i in range(self.n_step):
            seqi = self.ops_seq[i](states[i], adjs[:-1], idxes_seq[i])
            resi = sum(self.ops_res[offset + j](h, adjs, idxes_res[offset + j])
                       for j, h in enumerate(states[:i]))
            offset += i
            states.append(seqi + resi)
        
        output = self.norm(states[-1])
        if self.use_nl:
            output = F.gelu(output)
        return output

class PathEncoder(nn.Module):
    """路径编码器 - 编码草药→成分→靶标路径"""
    def __init__(self, n_hid, dropout=0.2):
        super(PathEncoder, self).__init__()
        self.n_hid = n_hid
        
        # 路径序列编码器
        self.path_lstm = nn.LSTM(n_hid, n_hid, batch_first=True, bidirectional=True)
        self.path_proj = nn.Linear(2 * n_hid, n_hid)
        
        # 多路径注意力聚合
        self.path_attention = nn.MultiheadAttention(n_hid, num_heads=4, dropout=dropout)
        
        # 疾病上下文编码器
        self.disease_context_encoder = nn.Linear(n_hid, n_hid)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, herb_emb, ingredient_embs, target_emb, disease_embs=None):
        """
        编码草药→成分→靶标路径
        Args:
            herb_emb: 草药嵌入 [1, n_hid]
            ingredient_embs: 成分嵌入列表 [num_ingredients, n_hid]
            target_emb: 靶标嵌入 [1, n_hid]
            disease_embs: 疾病嵌入列表（可选） [num_diseases, n_hid]
        Returns:
            path_embedding: 路径嵌入 [1, n_hid]
        """
        path_embeddings = []
        
        # 为每个成分构建一条路径
        for ingredient_emb in ingredient_embs:
            # 构建路径序列：herb → ingredient → target
            path_seq = torch.stack([
                herb_emb.squeeze(0),
                ingredient_emb,
                target_emb.squeeze(0)
            ], dim=0).unsqueeze(0)  # [1, 3, n_hid]
            
            # LSTM编码路径
            path_out, _ = self.path_lstm(path_seq)  # [1, 3, 2*n_hid]
            path_emb = self.path_proj(path_out[:, -1, :])  # [1, n_hid]
            
            # 如果有疾病上下文信息，融入疾病信息
            if disease_embs is not None and len(disease_embs) > 0:
                disease_context = torch.mean(disease_embs, dim=0).unsqueeze(0)  # [1, n_hid]
                disease_context = self.disease_context_encoder(disease_context)
                path_emb = path_emb + 0.1 * disease_context  # 轻微融入疾病信息
            
            path_embeddings.append(path_emb)
        
        if len(path_embeddings) == 0:
            # 如果没有路径，直接返回草药和靶标的组合
            return herb_emb + target_emb
        
        # 多路径注意力聚合
        path_embeddings = torch.cat(path_embeddings, dim=0)  # [num_paths, n_hid]
        
        if len(path_embeddings) == 1:
            return path_embeddings
        
        # 使用自注意力聚合多条路径
        path_embeddings = path_embeddings.unsqueeze(1)  # [num_paths, 1, n_hid]
        attended_paths, _ = self.path_attention(
            path_embeddings, path_embeddings, path_embeddings
        )  # [num_paths, 1, n_hid]
        
        # 平均池化得到最终路径表示
        final_path_emb = torch.mean(attended_paths, dim=0)  # [1, n_hid]
        
        return final_path_emb

class HerbTargetModel(nn.Module):
    """草药-靶标相互作用预测模型（通过成分路径增强）"""
    def __init__(self, in_dims, n_hid, n_steps, dropout=None, attn_dim=64,
                 use_norm=True, out_nl=True):
        super(HerbTargetModel, self).__init__()
        self.n_hid = n_hid
        
        # 节点类型嵌入层
        self.ws = nn.ModuleList()
        assert isinstance(in_dims, list)
        for i in range(len(in_dims)):
            self.ws.append(nn.Linear(64, n_hid))
        
        # 图神经网络层
        assert isinstance(n_steps, list)
        self.metas = nn.ModuleList()
        for i in range(len(n_steps)):
            self.metas.append(Cell(n_steps[i], n_hid, n_hid,
                                   use_norm=use_norm, use_nl=out_nl))
        
        # 注意力机制
        self.attn_fc1 = nn.Linear(n_hid, attn_dim)
        self.attn_fc2 = nn.Linear(attn_dim, 1)
        
        # 路径编码器
        self.path_encoder = PathEncoder(n_hid, dropout)
        
        # 最终预测层
        self.predictor = nn.Sequential(
            nn.Linear(n_hid, n_hid // 2),
            nn.ReLU(),
            nn.Dropout(dropout if dropout else 0.0),
            nn.Linear(n_hid // 2, 1),
            nn.Sigmoid()
        )
        
        self.feats_drop = nn.Dropout(dropout) if dropout is not None else lambda x: x
    
    def forward(self, node_feats, node_types, adjs, idxes_seq, idxes_res, 
                herb_target_pairs=None, paths_info=None):
        # 初始化节点隐藏状态
        hid = torch.zeros((node_types.size(0), self.n_hid)).cuda()
        for i in range(len(node_feats)):
            hid[node_types == i] = self.ws[i](node_feats[i])
        
        hid = self.feats_drop(hid)
        
        # 多层图卷积
        temps = []
        attns = []
        for i, meta in enumerate(self.metas):
            hidi = meta(hid, adjs, idxes_seq[i], idxes_res[i])
            temps.append(hidi)
            attni = self.attn_fc2(torch.tanh(self.attn_fc1(temps[-1])))
            attns.append(attni)
        
        # 注意力聚合
        hids = torch.stack(temps, dim=0).transpose(0, 1)
        attns = F.softmax(torch.cat(attns, dim=-1), dim=-1)
        out = (attns.unsqueeze(dim=-1) * hids).sum(dim=1)
        
        # 如果没有提供具体的herb-target对，直接返回节点嵌入
        if herb_target_pairs is None:
            return out
        
        # 基于路径的预测
        predictions = []
        batch_size = herb_target_pairs.size(0)
        
        for i in range(batch_size):
            herb_id = herb_target_pairs[i, 0]
            target_id = herb_target_pairs[i, 1]
            
            # 获取草药和靶标的嵌入
            herb_emb = out[herb_id].unsqueeze(0)  # [1, n_hid]
            target_emb = out[target_id].unsqueeze(0)  # [1, n_hid]
            
            # 获取路径信息
            if paths_info is not None and i < len(paths_info) and len(paths_info[i]) > 0:
                # 有路径信息时，使用路径编码器
                path_info_list = paths_info[i]
                
                # 收集所有相关的成分嵌入
                ingredient_embs = []
                disease_embs = []
                
                # 获取节点偏移量
                herb_offset = 0
                target_offset = len(node_feats[0])  # 草药节点数量
                ingredient_offset = target_offset + len(node_feats[1])  # + 靶标节点数量
                disease_offset = ingredient_offset + len(node_feats[2])  # + 成分节点数量
                
                for path_info in path_info_list:
                    ingredient_id = path_info['ingredient_id']
                    
                    # 成分在图中的实际位置
                    # 这里需要根据实际的成分ID映射来确定正确的节点位置
                    # 暂时使用简化的映射，假设ingredient_id是连续的
                    ingredient_node_id = ingredient_offset + (ingredient_id % len(node_feats[2]))
                    
                    if ingredient_node_id < out.size(0):
                        ingredient_embs.append(out[ingredient_node_id])
                    
                    # 收集疾病嵌入
                    for disease_id in path_info.get('related_diseases', []):
                        disease_node_id = disease_offset + (disease_id % len(node_feats[3]))
                        if disease_node_id < out.size(0):
                            disease_embs.append(out[disease_node_id])
                
                # 使用路径编码器
                if len(ingredient_embs) > 0:
                    ingredient_embs = torch.stack(ingredient_embs)
                    disease_embs = torch.stack(disease_embs) if len(disease_embs) > 0 else None
                    
                    path_emb = self.path_encoder(herb_emb, ingredient_embs, target_emb, disease_embs)
                    prediction = self.predictor(path_emb).squeeze()
                else:
                    # 没有有效成分信息时，使用直接连接
                    combined_emb = herb_emb + target_emb
                    prediction = self.predictor(combined_emb).squeeze()
            else:
                # 没有路径信息时，使用直接连接
                combined_emb = herb_emb + target_emb
                prediction = self.predictor(combined_emb).squeeze()
            
            predictions.append(prediction)
        
        return torch.stack(predictions)

class HerbTargetModelWithoutPath(nn.Module):
    """不使用路径信息的基线模型"""
    def __init__(self, in_dims, n_hid, n_steps, dropout=None, attn_dim=64,
                 use_norm=True, out_nl=True):
        super(HerbTargetModelWithoutPath, self).__init__()
        self.n_hid = n_hid
        
        # 节点类型嵌入层
        self.ws = nn.ModuleList()
        assert isinstance(in_dims, list)
        for i in range(len(in_dims)):
            self.ws.append(nn.Linear(64, n_hid))
        
        # 图神经网络层
        assert isinstance(n_steps, list)
        self.metas = nn.ModuleList()
        for i in range(len(n_steps)):
            self.metas.append(Cell(n_steps[i], n_hid, n_hid,
                                   use_norm=use_norm, use_nl=out_nl))
        
        # 注意力机制
        self.attn_fc1 = nn.Linear(n_hid, attn_dim)
        self.attn_fc2 = nn.Linear(attn_dim, 1)
        
        self.feats_drop = nn.Dropout(dropout) if dropout is not None else lambda x: x
    
    def forward(self, node_feats, node_types, adjs, idxes_seq, idxes_res):
        # 初始化节点隐藏状态
        hid = torch.zeros((node_types.size(0), self.n_hid)).cuda()
        for i in range(len(node_feats)):
            hid[node_types == i] = self.ws[i](node_feats[i])
        
        hid = self.feats_drop(hid)
        
        # 多层图卷积
        temps = []
        attns = []
        for i, meta in enumerate(self.metas):
            hidi = meta(hid, adjs, idxes_seq[i], idxes_res[i])
            temps.append(hidi)
            attni = self.attn_fc2(torch.tanh(self.attn_fc1(temps[-1])))
            attns.append(attni)
        
        # 注意力聚合
        hids = torch.stack(temps, dim=0).transpose(0, 1)
        attns = F.softmax(torch.cat(attns, dim=-1), dim=-1)
        out = (attns.unsqueeze(dim=-1) * hids).sum(dim=1)
        
        return out


class TransE(nn.Module):
    """TransE知识图谱嵌入模块"""
    def __init__(self, num_entities, num_relations, dim, p=2):
        super(TransE, self).__init__()
        self.ent_emb = nn.Embedding(num_entities, dim)
        self.rel_emb = nn.Embedding(num_relations, dim)
        self.p = p
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.ent_emb.weight)
        nn.init.xavier_uniform_(self.rel_emb.weight)

    def score(self, heads, rels, tails):
        h = self.ent_emb(heads)
        r = self.rel_emb(rels)
        t = self.ent_emb(tails)
        return torch.norm(h + r - t, p=self.p, dim=-1)

    def loss(self, pos_triples, neg_triples, margin=1.0):
        pos_score = self.score(pos_triples[:, 0], pos_triples[:, 1], pos_triples[:, 2])
        neg_score = self.score(neg_triples[:, 0], neg_triples[:, 1], neg_triples[:, 2])
        return torch.relu(margin + pos_score - neg_score).mean()


class AdaptiveMetaGraphGCN(nn.Module):
    """自适应元图结构搜索的GCN聚合模块"""
    def __init__(self, num_layers, num_relations, n_hid, dropout=0.0, use_norm=True,
                 use_adaptive=True, use_residual=True):
        super(AdaptiveMetaGraphGCN, self).__init__()
        self.num_layers = num_layers
        self.num_relations = num_relations
        self.use_adaptive = use_adaptive
        self.use_residual = use_residual
        self.dropout = nn.Dropout(dropout)
        self.norms = nn.ModuleList([
            nn.LayerNorm(n_hid) if use_norm else nn.Identity()
            for _ in range(num_layers)
        ])

        self.alphas = nn.ModuleList()
        self.res_alphas = nn.ModuleList()
        for i in range(1, num_layers + 1):
            layer_alpha = nn.ParameterList([
                nn.Parameter(torch.zeros(num_relations)) for _ in range(i)
            ])
            self.alphas.append(layer_alpha)
            if i > 1:
                self.res_alphas.append(nn.ParameterList([
                    nn.Parameter(torch.zeros(1)) for _ in range(i - 1)
                ]))
            else:
                self.res_alphas.append(nn.ParameterList())

    def forward(self, h0, adjs):
        h_list = [h0]
        for i in range(1, self.num_layers + 1):
            agg = 0.0
            for t in range(i):
                if self.use_adaptive:
                    theta = F.softmax(self.alphas[i - 1][t], dim=-1)
                else:
                    theta = torch.ones(self.num_relations, device=h_list[0].device) / self.num_relations
                for r in range(self.num_relations):
                    agg = agg + theta[r] * torch.spmm(adjs[r], h_list[t])

            if self.use_residual and i > 1 and len(self.res_alphas[i - 1]) > 0:
                res_weights = torch.stack([p for p in self.res_alphas[i - 1]]).squeeze(-1)
                res_weights = F.softmax(res_weights, dim=0)
                res = 0.0
                for j in range(i - 1):
                    res = res + res_weights[j] * h_list[j]
                agg = agg + res

            out = self.norms[i - 1](agg)
            out = F.gelu(out)
            out = self.dropout(out)
            h_list.append(out)

        return h_list[1:]


class MultiHopSemanticPropagation(nn.Module):
    """多跳语义传播模块（关系注意+变换矩阵）"""
    def __init__(self, num_layers, num_relations, n_hid, dropout=0.0, use_relation_attention=True):
        super(MultiHopSemanticPropagation, self).__init__()
        self.num_layers = num_layers
        self.num_relations = num_relations
        self.use_relation_attention = use_relation_attention
        self.dropout = nn.Dropout(dropout)

        self.alphas = nn.ModuleList()
        self.transforms = nn.ModuleList()
        for t in range(1, num_layers + 1):
            layer_alpha = nn.ParameterList()
            layer_w = nn.ModuleList()
            for i in range(t):
                layer_alpha.append(nn.Parameter(torch.zeros(num_relations)))
                layer_w.append(nn.ModuleList([nn.Linear(n_hid, n_hid) for _ in range(num_relations)]))
            self.alphas.append(layer_alpha)
            self.transforms.append(layer_w)

    def forward(self, h_list, adjs):
        # h_list: list of H_i, i from 0..t-1
        outputs = []
        for t in range(1, self.num_layers + 1):
            agg = 0.0
            for i in range(t):
                if self.use_relation_attention:
                    alpha = F.softmax(self.alphas[t - 1][i], dim=-1)
                else:
                    alpha = torch.ones(self.num_relations, device=h_list[0].device) / self.num_relations
                for r in range(self.num_relations):
                    h_proj = self.transforms[t - 1][i][r](h_list[i])
                    agg = agg + alpha[r] * torch.spmm(adjs[r], h_proj)
            out = F.gelu(agg)
            out = self.dropout(out)
            outputs.append(out)
            h_list.append(out)
        return outputs


class LayerAttention(nn.Module):
    """层级注意力融合不同层输出"""
    def __init__(self, n_hid, attn_dim=64):
        super(LayerAttention, self).__init__()
        self.fc1 = nn.Linear(n_hid, attn_dim)
        self.fc2 = nn.Linear(attn_dim, 1)

    def forward(self, h_layers):
        h_stack = torch.stack(h_layers, dim=0)  # [L, N, D]
        attn = self.fc2(torch.tanh(self.fc1(h_stack)))  # [L, N, 1]
        attn = F.softmax(attn, dim=0)
        out = (attn * h_stack).sum(dim=0)
        return out, attn


class PathAttention(nn.Module):
    """路径级注意力聚合"""
    def __init__(self, n_hid, use_attention=True):
        super(PathAttention, self).__init__()
        self.w = nn.Linear(n_hid, n_hid)
        self.q = nn.Parameter(torch.zeros(n_hid))
        self.use_attention = use_attention
        nn.init.xavier_uniform_(self.w.weight)

    def forward(self, path_embs):
        if not self.use_attention:
            beta = torch.ones(path_embs.size(0), device=path_embs.device) / path_embs.size(0)
            return path_embs.mean(dim=0), beta
        scores = torch.matmul(torch.tanh(self.w(path_embs)), self.q)
        beta = F.softmax(scores, dim=0)
        return (beta.unsqueeze(-1) * path_embs).sum(dim=0), beta


class HTIPredictor(nn.Module):
    """草药-靶标相互作用预测模块"""
    def __init__(self, n_hid, dropout=0.2):
        super(HTIPredictor, self).__init__()
        self.herb_encoder = nn.Sequential(
            nn.Linear(n_hid, n_hid),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.target_encoder = nn.Sequential(
            nn.Linear(n_hid, n_hid),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.predictor = nn.Sequential(
            nn.Linear(n_hid * 2 + 1, n_hid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_hid, 1),
            nn.Sigmoid()
        )

    def forward(self, herb_emb, target_emb, path_weight):
        h = self.herb_encoder(herb_emb)
        t = self.target_encoder(target_emb)
        x = torch.cat([h, t, path_weight], dim=-1)
        return self.predictor(x).squeeze(-1)


class KGMAPHTIModel(nn.Module):
    def __init__(self, num_entities, num_relations, n_hid=256, num_layers=3,
                 kge_dim=128, dropout=0.2,
                 use_adaptive_search=True,
                 use_semantic_edges=True,
                 use_residual=True,
                 use_relation_attention=True,
                 use_metapath_attention=True,
                 use_transe_pretrain=True):
        super(KGMAPHTIModel, self).__init__()
        self.transe = TransE(num_entities, num_relations, kge_dim)
        self.proj = nn.Linear(kge_dim, n_hid) if kge_dim != n_hid else nn.Identity()
        self.gcn = AdaptiveMetaGraphGCN(
            num_layers, num_relations, n_hid, dropout=dropout,
            use_adaptive=use_adaptive_search,
            use_residual=use_residual
        )
        self.semantic_prop = MultiHopSemanticPropagation(
            num_layers, num_relations, n_hid, dropout=dropout,
            use_relation_attention=use_relation_attention
        )
        self.layer_attn = LayerAttention(n_hid)
        self.path_attn = PathAttention(n_hid, use_attention=use_metapath_attention)
        self.predictor = HTIPredictor(n_hid, dropout=dropout)
        self.use_transe_pretrain = use_transe_pretrain

    def kge_loss(self, pos_triples, neg_triples, margin=1.0):
        return self.transe.loss(pos_triples, neg_triples, margin=margin)

    def reset_transe_embeddings(self):
        nn.init.normal_(self.transe.ent_emb.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.transe.rel_emb.weight, mean=0.0, std=0.1)

    def encode(self, adjs):
        h0 = self.proj(self.transe.ent_emb.weight)
        gcn_layers = self.gcn(h0, adjs)
        h_layers = self.semantic_prop([h0] + gcn_layers, adjs)
        h_final, _ = self.layer_attn(h_layers)
        return h_final

    def _build_path_embs(self, h_final, path_info_list, offsets):
        if not path_info_list:
            return None
        path_embs = []
        for path_info in path_info_list:
            herb_idx = int(path_info['herb_id']) + offsets['herb']
            ing_idx = int(path_info['ingredient_id']) + offsets['ingredient']
            target_idx = int(path_info['target_id']) + offsets['target']
            path_emb = (h_final[herb_idx] + h_final[ing_idx] + h_final[target_idx]) / 3.0
            diseases = path_info.get('related_diseases', [])
            if diseases:
                disease_embs = []
                for d_id in diseases:
                    d_idx = int(d_id) + offsets['disease']
                    if d_idx < h_final.size(0):
                        disease_embs.append(h_final[d_idx])
                if disease_embs:
                    disease_ctx = torch.stack(disease_embs, dim=0).mean(dim=0)
                    path_emb = path_emb + 0.1 * disease_ctx
            path_embs.append(path_emb)
        if not path_embs:
            return None
        return torch.stack(path_embs, dim=0)

    def path_attention_confidence(self, h_final, path_info_list, offsets):
        path_embs = self._build_path_embs(h_final, path_info_list, offsets)
        if path_embs is None:
            return 0.0
        if path_embs.size(0) == 1:
            return 1.0
        _, beta = self.path_attn(path_embs)
        return float(beta.mean().detach().cpu().item())

    def _path_context(self, h_final, path_info_list, offsets):
        path_embs = self._build_path_embs(h_final, path_info_list, offsets)
        if path_embs is None:
            return None
        path_ctx, _ = self.path_attn(path_embs)
        return path_ctx

    def predict_pairs(self, h_final, pairs, path_infos, path_weights, offsets):
        preds = []
        for idx in range(pairs.size(0)):
            herb_id = int(pairs[idx, 0].item())
            target_id = int(pairs[idx, 1].item())
            herb_emb = h_final[herb_id].unsqueeze(0)
            target_emb = h_final[target_id].unsqueeze(0)

            path_ctx = self._path_context(h_final, path_infos[idx], offsets)
            if path_ctx is not None:
                path_ctx = path_ctx.unsqueeze(0)
                herb_emb = herb_emb + path_ctx
                target_emb = target_emb + path_ctx

            w = path_weights[idx].view(1, 1)
            pred = self.predictor(herb_emb, target_emb, w)
            preds.append(pred.squeeze(0))

        return torch.stack(preds)

    def final_score(self, y_hat, confidence, path_weight):
        """融合预测概率与路径置信度的最终评分"""
        return y_hat * confidence * (1.0 + path_weight)
