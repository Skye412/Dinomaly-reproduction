"""Frozen retrieval context for V5. Runtime banks never enter checkpoints or Git."""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from crmc import build_memory_bank, load_bank, sha256_file


class RetrievalProvider:
    def __init__(self, bank, rows, device, k=16):
        metadata, descriptors, groups = bank
        if metadata['sample_ids'] != [r['sample'] for r in rows]:
            raise ValueError('Bank and training manifest order differ')
        self.bank = bank
        self.k, self.device = k, device
        self.parents = np.asarray([r['parent_id'] for r in rows])
        self.descriptors = F.normalize(descriptors.to(device).float(), dim=-1)
        # Mean of the two raw target fusion groups gives one 768-dim context.
        # Pool only patches; no CLS/register token enters memory context.
        pooled = []
        for start in range(0, len(rows), 32):
            g = (groups[0][start:start+32].float() + groups[1][start:start+32].float()) * .5
            side = int(g.shape[1] ** .5)
            if side != 28 or g.shape[1] != side * side:
                raise ValueError('V5 expects 28x28 training patches')
            p = F.avg_pool2d(g.transpose(1, 2).reshape(-1, 768, side, side), 2, 2)
            pooled.append(p.flatten(2).transpose(1, 2).half())
        self.context = torch.cat(pooled).contiguous()  # CPU fp16 [N,196,768]
        self.training_neighbors = self.rank(descriptors.to(device).float(), np.arange(len(rows)))
        self.training = False

    def rank(self, cls, identities=None):
        if not torch.isfinite(cls).all():
            raise RuntimeError('Non-finite query descriptors')
        similarity = (F.normalize(cls.float(), dim=-1) @ self.descriptors.T).detach().cpu().numpy()
        if identities is not None:
            identities = np.asarray(identities, dtype=np.int64)
            if len(identities) != len(cls) or np.any(identities < 0) or np.any(identities >= len(self.parents)):
                raise ValueError('Invalid training identity')
            forbidden = self.parents[None, :] == self.parents[identities, None]
            forbidden[np.arange(len(identities)), identities] = True
            similarity[forbidden] = -np.inf
            if np.any(np.isfinite(similarity).sum(1) < self.k):
                raise ValueError('Fewer than K eligible normals after self/parent exclusion')
        ids = np.argsort(-similarity, axis=1, kind='mergesort')[:, :self.k].copy()
        if identities is not None and np.any(self.parents[ids] == self.parents[identities, None]):
            raise AssertionError('Same-parent neighbor leaked into training')
        return torch.from_numpy(ids)

    def __call__(self, cls, identities=None):
        if self.training:
            if identities is None:
                raise RuntimeError('V5 training requires original manifest identities')
            ids = self.training_neighbors[torch.as_tensor(identities).cpu().long()]
        else:
            if identities is not None:
                raise ValueError('Evaluation cannot apply training identity exclusions')
            ids = self.rank(cls)
        result = self.context[ids].reshape(len(cls), self.k * 196, 768).to(self.device).float()
        if result.shape[1] != 3136 or not torch.isfinite(result).all():
            raise RuntimeError('Invalid V5 retrieved context')
        return result.detach()


def attach_retrieval(model, bank, rows, device):
    provider = RetrievalProvider(bank, rows, device)
    # Provider is deliberately not nn.Module: bank is referenced via provenance,
    # never copied into model.state_dict/checkpoints.
    model.retrieval_provider = provider
    return provider


def prepare_retrieval(model, dataset, loader, output, device, provenance, source=None):
    bank = load_bank(source) if source else build_memory_bank(model, loader, device, output, provenance)
    if source:
        meta = bank[0]
        if meta['provenance']['manifest_sha256'] != provenance['manifest_sha256']:
            raise ValueError('V5 source bank fingerprint mismatch')
    provider = attach_retrieval(model, bank, dataset.rows, device)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output/'training_neighbors.npy', provider.training_neighbors.numpy())
    (output/'retrieval.json').write_text(json.dumps(dict(k=16, patches_per_reference=196,
        context_tokens=3136, group_fusion='mean of two raw V0 target groups before 2x2 pooling',
        training_exclusion='self and equal manifest parent_id', parent_field='parent_id (DACL group_id)',
        parent_limit='DACL group_id is content identity, not a verified bridge/session identifier',
        samples=[r['sample'] for r in dataset.rows], parents=provider.parents.tolist(),
        source_bank=str(source) if source else str(output),
        bank_file_sha256=bank[0]['file_sha256'], provenance=provenance), indent=2), encoding='utf-8')
    return provider
