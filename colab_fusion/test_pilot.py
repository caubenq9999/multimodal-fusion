import unittest
import numpy as np
from pilot import reciprocal, ranks, metrics, grid_weights, make_model, parse_weights, symmetric_positive_loss

class TestPilot(unittest.TestCase):
    def test_missing_and_ties(self):
        s=np.array([[[1.,100.],[1.,2.],[0.,1.]]])
        available=np.array([[1,0],[1,1],[1,1]],bool)
        r=reciprocal(s,available,60)
        self.assertEqual(r[0,0,1],0)
        np.testing.assert_allclose(r[0,:,0],[1/61,1/62,1/63])
        np.testing.assert_allclose(r[0,1:,1],[1/61,1/62])

    def test_metrics(self):
        r=ranks(np.array([[3,2,1],[1,3,2]]),np.array([0,2]))
        np.testing.assert_array_equal(r,[1,2])
        self.assertEqual(metrics(r)['MRR'],.75)

    def test_simplex(self):
        w=grid_weights(4)
        self.assertEqual(len(w),286)
        np.testing.assert_allclose(w.sum(1),1,atol=1e-6)
        self.assertTrue((w>=0).all())

    def test_absent_branch_is_miss(self):
        r=ranks(np.full((2,3),-np.inf),np.array([0,2]))
        self.assertTrue(np.isinf(r).all())
        self.assertEqual(metrics(r)['MRR'],0)

    def test_models_and_mask(self):
        try: import torch
        except ImportError: self.skipTest('PyTorch unavailable')
        torch.manual_seed(42)
        q=torch.randn(5,16); v=torch.randn(7,4,16); mask=torch.ones(7,4); mask[:,2]=0
        r=torch.rand(5,7,4)/60; r[:,:,2]=0
        for kind in ['linear','mlp','moe','latent']:
            model=make_model(kind,16,4,8); model.eval()
            pred=model(q,v,mask,r)
            self.assertEqual(tuple(pred.shape),(5,7)); self.assertTrue(torch.isfinite(pred).all())
            other=v.clone(); other[:,2]=torch.randn_like(other[:,2])*100
            torch.testing.assert_close(pred,model(q,other,mask,r))
            pred.square().mean().backward()
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters()))

    def test_llm_validation(self):
        np.testing.assert_allclose(parse_weights('{"image":0.6,"caption":0.4}',['image','caption']),[.6,.4])
        for text in ['{}','{"image":-1,"caption":2}','{"image":true,"caption":0}','not JSON']:
            with self.assertRaises(ValueError): parse_weights(text,['image','caption'])

    def test_adafuse_and_symmetric_loss(self):
        import torch
        torch.manual_seed(42)
        model=make_model('adafuse_adapted',16,2,8,projection=12,expert_hidden=32)
        q=torch.randn(4,16); v=torch.nn.functional.normalize(torch.randn(3,2,16),dim=-1)
        logits=model(q,v,torch.ones(3,2),torch.zeros(4,3,2))
        loss=symmetric_positive_loss(logits,torch.tensor([0,0,1,2]))
        self.assertTrue(torch.isfinite(loss)); loss.backward()
        self.assertTrue(model.out.weight.grad.abs().sum()>0)
        self.assertTrue(model.reliability[0].weight.grad.abs().sum()>0)
        model.eval()
        # Candidate vectors are independently cacheable; chunking does not change scores.
        torch.testing.assert_close(model(q,v,torch.ones(3,2),None),torch.cat([
            model(q,v[i:i+1],torch.ones(1,2),None) for i in range(3)],dim=1))

if __name__=='__main__': unittest.main()
