"""CPU unit checks only: no dataset sampling, training or GPU experiments."""
import unittest
import numpy as np
import torch
from models.retrieval_attention import RetrievalCrossAttention
from retrieval_v5 import RetrievalProvider
from crmc_supplement import coreset_indices, stitch

class RetrievalTests(unittest.TestCase):
    def test_linear_attention_matches_dense_kernel_and_gradients(self):
        torch.manual_seed(7)
        block=RetrievalCrossAttention(24,24,3)
        x=torch.randn(2,7,24,requires_grad=True); context=torch.randn(2,13,24)
        q=block.q(block.norm_q(x)).reshape(2,7,3,8).transpose(1,2)
        c=block.norm_context(context)
        k=block.k(c).reshape(2,13,3,8).transpose(1,2)
        v=block.v(c).reshape(2,13,3,8).transpose(1,2)
        weights=(torch.nn.functional.elu(q)+1)@(torch.nn.functional.elu(k)+1).transpose(-1,-2)
        dense=(weights@v)/(weights.sum(-1,keepdim=True)+1e-6)
        expected=block.norm_out(x+block.proj(dense.transpose(1,2).reshape(2,7,24)))
        actual=block(x,context)
        torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-5)
        actual.square().sum().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in block.parameters()))

    def test_training_exclusion_pooling_and_eval_access(self):
        rows=[dict(sample=str(i),parent_id=str(i//2)) for i in range(20)]
        descriptors=torch.eye(20).half()
        patches=torch.arange(784).float().reshape(1,784,1).expand(20,-1,768).half()
        provider=RetrievalProvider(({'sample_ids':[r['sample'] for r in rows]},descriptors,[patches,patches]),rows,'cpu')
        ids=provider.training_neighbors.numpy()
        self.assertTrue(np.all(provider.parents[ids]!=provider.parents[:,None]))
        self.assertTrue(torch.all(provider.rank(descriptors)[:,0]==torch.arange(20)))
        provider.training=True
        with self.assertRaises(RuntimeError):provider(descriptors[:1])
        context=provider(descriptors[:2],torch.tensor([0,1]))
        self.assertEqual(tuple(context.shape),(2,3136,768))
        self.assertAlmostEqual(context[0,0,0].item(),14.5)
        # A single-parent library cannot silently reduce K.
        with self.assertRaises(ValueError):
            RetrievalProvider(({'sample_ids':[r['sample'] for r in rows]},descriptors,[patches,patches]),
                              [dict(r,parent_id='same') for r in rows],'cpu')

    def test_coreset_stitch_and_memory_exclusion_from_checkpoints(self):
        torch.manual_seed(2);groups=[torch.randn(7,9,8).half(),torch.randn(7,9,8).half()]
        a=coreset_indices(groups,'cpu');b=coreset_indices(groups,'cpu')
        self.assertTrue(np.array_equal(a,b));self.assertEqual(len(a),6);self.assertEqual(len(set(a)),6)
        torch.testing.assert_close(stitch(torch.ones(2,4,1,28,28)),torch.ones(2,1,512,512))
        self.assertEqual(sum(p.numel() for p in RetrievalCrossAttention().parameters()),2366976)

if __name__=='__main__':unittest.main()
