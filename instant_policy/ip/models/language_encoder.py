import torch
from torch import nn
from torch_geometric.data import HeteroData
from ip.models.graph_transformer import GraphTransformer
from ip.utils.common_utils import PositionalEncoder


class LanguageConditionedEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config['hidden_dim']
        self.lang_emb_dim = config['lang_emb_dim']
        self.num_layers = config['lang_num_layers']
        self.device = config['device']

        self.pos_embd = PositionalEncoder(3,
                                          config['local_num_freq'],
                                          log_space=True,
                                          add_original_x=True,
                                          scale=1.0)
        self.edge_dim = self.pos_embd.d_output * 2

        self.lang_proj = nn.Linear(self.lang_emb_dim, self.hidden_dim)
        self.lang_edge_emb = nn.Parameter(torch.randn(1, self.edge_dim))

        self.transformer = GraphTransformer(
            in_channels=self.hidden_dim,
            hidden_channels=self.hidden_dim,
            heads=self.hidden_dim // 64,
            num_layers=self.num_layers,
            metadata=(
                ['scene', 'gripper', 'language'],
                [
                    ('scene', 'rel', 'scene'),
                    ('scene', 'rel', 'gripper'),
                    ('gripper', 'rel', 'gripper'),
                    ('language', 'lang_to_scene', 'scene'),
                    ('language', 'lang_to_gripper', 'gripper'),
                ]
            ),
            edge_dim=self.edge_dim,
            dropout=0.0,
            norm='layer',
        )

    def forward(self, scene_x, scene_pos, gripper_x, gripper_pos, lang_emb):
        """
        Args:
            scene_x: [B, Ns, H] local-encoder features for current scene nodes
            scene_pos: [B, Ns, 3] positions for current scene nodes
            gripper_x: [B, Ng, H] local-encoder features for current gripper nodes
            gripper_pos: [B, Ng, 3] positions for current gripper nodes
            lang_emb: [B, D] precomputed language embeddings
        Returns:
            gripper_out: [B, Ng, H] language-conditioned gripper features
        """
        batch_size, num_scene, _ = scene_x.shape
        num_gripper = gripper_x.shape[1]

        lang_emb = self.lang_proj(lang_emb.to(scene_x.device))

        graph = self._build_language_graph(
            scene_x=scene_x,
            scene_pos=scene_pos,
            gripper_x=gripper_x,
            gripper_pos=gripper_pos,
            lang_emb=lang_emb,
            batch_size=batch_size,
            num_scene=num_scene,
            num_gripper=num_gripper,
            device=scene_x.device,
        )

        x_dict = self.transformer(graph.x_dict, graph.edge_index_dict, graph.edge_attr_dict)
        gripper_out = x_dict['gripper'].view(batch_size, num_gripper, -1)
        return gripper_out

    def _build_language_graph(self, scene_x, scene_pos, gripper_x, gripper_pos,
                              lang_emb, batch_size, num_scene, num_gripper, device):
        graph = HeteroData()

        graph['scene'].x = scene_x.reshape(batch_size * num_scene, -1)
        graph['scene'].pos = scene_pos.reshape(batch_size * num_scene, 3)

        graph['gripper'].x = gripper_x.reshape(batch_size * num_gripper, -1)
        graph['gripper'].pos = gripper_pos.reshape(batch_size * num_gripper, 3)

        graph['language'].x = lang_emb.reshape(batch_size, -1)

        scene_edges = self._dense_batched_edges(batch_size, num_scene, num_scene, device)
        gripper_edges = self._dense_batched_edges(batch_size, num_gripper, num_gripper, device)
        scene_gripper_edges = self._dense_batched_edges(batch_size, num_scene, num_gripper, device)

        graph[('scene', 'rel', 'scene')].edge_index = scene_edges
        graph[('gripper', 'rel', 'gripper')].edge_index = gripper_edges
        graph[('scene', 'rel', 'gripper')].edge_index = scene_gripper_edges

        graph[('scene', 'rel', 'scene')].edge_attr = self._rel_edge_attr(
            graph['scene'].pos, graph['scene'].pos, scene_edges)
        graph[('gripper', 'rel', 'gripper')].edge_attr = self._rel_edge_attr(
            graph['gripper'].pos, graph['gripper'].pos, gripper_edges)
        graph[('scene', 'rel', 'gripper')].edge_attr = self._rel_edge_attr(
            graph['scene'].pos, graph['gripper'].pos, scene_gripper_edges)

        lang_scene_edges = self._language_edges(batch_size, num_scene, device)
        lang_gripper_edges = self._language_edges(batch_size, num_gripper, device)

        graph[('language', 'lang_to_scene', 'scene')].edge_index = lang_scene_edges
        graph[('language', 'lang_to_gripper', 'gripper')].edge_index = lang_gripper_edges

        graph[('language', 'lang_to_scene', 'scene')].edge_attr = self.lang_edge_emb.expand(
            lang_scene_edges.shape[1], -1)
        graph[('language', 'lang_to_gripper', 'gripper')].edge_attr = self.lang_edge_emb.expand(
            lang_gripper_edges.shape[1], -1)

        return graph

    def _rel_edge_attr(self, pos_src, pos_dst, edge_index):
        pos_src = pos_src[edge_index[0]]
        pos_dst = pos_dst[edge_index[1]]
        delta = pos_dst - pos_src
        pos_embd = self.pos_embd(delta)
        return torch.cat([pos_embd, pos_embd], dim=-1)

    def _dense_batched_edges(self, batch_size, num_src, num_dst, device, src_offset=0, dst_offset=0):
        edges = []
        for b in range(batch_size):
            src = torch.arange(num_src, device=device) + b * num_src + src_offset
            dst = torch.arange(num_dst, device=device) + b * num_dst + dst_offset
            edge = torch.cartesian_prod(src, dst).t()
            edges.append(edge)
        return torch.cat(edges, dim=1)

    def _language_edges(self, batch_size, num_dst, device, dst_offset=0):
        src = torch.arange(batch_size, device=device).repeat_interleave(num_dst)
        dst = torch.arange(batch_size * num_dst, device=device) + dst_offset
        return torch.stack([src, dst], dim=0)
