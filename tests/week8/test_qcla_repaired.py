"""Correctness checks for the real QCLA path; GPU smoke validates SARA integration."""
from types import SimpleNamespace
import pandas as pd
import pytest
import torch
import torch.nn.functional as F
from src.week8.qcla import build_qcla_adapter
from src.week8.qcla_training import QCLATrainingDataset,gold_target_inputs,answer_loss,single_query_collate


def test_conditioning_is_required_not_zero_filled():
    r={'question_id':'q','paper_id':'p','question':'question'}
    route={'q':pd.DataFrame([{'paper_id':'p','question':'question'}])}
    with pytest.raises(ValueError,match='query embedding'):
        QCLATrainingDataset([r],{'q':torch.ones(10,4096)},routing_by_qid=route)
    with pytest.raises(ValueError,match='draft embedding'):
        QCLATrainingDataset([r],{'q':torch.ones(10,4096)},{'q':torch.ones(4096)},routing_by_qid=route)


@pytest.mark.parametrize('variant',['qcla_1','qcla_2'])
def test_identity_initialization_and_actual_residual_gate(variant):
    a=build_qcla_adapter(variant,dim=8,hidden_dim=4)
    x=torch.randn(6,8)*.01;q=torch.randn(8);y=torch.randn(8)
    assert torch.equal(a(x,q,y),x)
    with torch.no_grad():
        a.mlp_f[-1].weight.normal_();a.mlp_f[-1].bias.normal_()
        a.gate_mlp[-2].bias.fill_(-100.)
    assert torch.equal(a(x,q,y),x)
    with pytest.raises(ValueError,match='query'):a(x)
    if variant=='qcla_2':
        with pytest.raises(ValueError,match='draft'):a(x,q)


def test_answer_loss_causal_positions_and_gradient_flow():
    class TinyFrozenGenerator(torch.nn.Module):
        dtype=torch.float32
        def __init__(self):
            super().__init__();self.head=torch.nn.Linear(3,7,bias=False)
            self.head.requires_grad_(False)
        def forward(self,input_ids,retrieval_embeds,logits_to_keep,**kwargs):
            logits=self.head(retrieval_embeds).reshape(1,1,7).expand(1,input_ids.shape[1],7)
            self.full_logits=logits
            return SimpleNamespace(logits=logits[:,-logits_to_keep:])
    model=TinyFrozenGenerator();adapter=torch.nn.Linear(3,3)
    embeds=adapter(torch.randn(1,3));ids=torch.tensor([[1,2,3,4,5,6]])
    targets=torch.tensor([[4,5,6]])
    loss=answer_loss(model,ids,targets,embeds)
    expected=F.cross_entropy(model.full_logits[:,2:5].reshape(-1,7),targets.reshape(-1))
    assert torch.allclose(loss,expected)
    loss.backward()
    assert adapter.weight.grad.abs().sum()>0
    assert model.head.weight.grad is None


def test_gold_target_preserves_prefix_and_no_second_bos():
    class Tokenizer:
        eos_token_id=9
        def __call__(self,text,return_tensors=None,add_special_tokens=True):
            if add_special_tokens:
                assert text=='canonical prompt'
                return {'input_ids':torch.tensor([[1,2,3]])}
            assert text==' real answer'
            return {'input_ids':[4,5]}
    ids,targets=gold_target_inputs(Tokenizer(),'canonical prompt',{'gold_answers':['real answer']},'cpu')
    assert ids.tolist()==[[1,2,3,4,5,9]] and targets.tolist()==[[4,5,9]]
    with pytest.raises(ValueError,match='gold-answer'):
        gold_target_inputs(Tokenizer(),'canonical prompt',{'gold_answers':[]},'cpu')


def test_single_query_batch_keeps_record_structure():
    record={'record':{'passages':[{'text':'text'}]},'question_id':'q'}
    assert single_query_collate([record]) is record
    with pytest.raises(ValueError):single_query_collate([record,record])


def test_gradient_accumulation_normalizes_partial_group(monkeypatch):
    import src.week8.qcla_training as training
    class Adapter(torch.nn.Module):
        def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.tensor(0.))
    class Frozen(torch.nn.Linear):
        def gradient_checkpointing_enable(self,**kwargs):pass
    a=Adapter();model=Frozen(1,1)
    monkeypatch.setattr(training,'prepare_example',lambda batch,tok,adapter,device:(None,adapter.weight,None,None))
    monkeypatch.setattr(training,'gold_target_inputs',lambda tok,prompt,record,device:(None,torch.tensor(record['target'])))
    monkeypatch.setattr(training,'answer_loss',lambda model,ids,targets,embeds:(embeds-targets).square())
    batches=[{'record':{'target':.1}} for _ in range(4)]+[{'record':{'target':.3}}]
    opt=torch.optim.SGD(a.parameters(),lr=.1)
    loss=training.train_qcla_epoch(a,model,None,batches,opt,gradient_accumulation_steps=4,device='cpu')
    # First four examples update 0 -> .02; final query updates .02 -> .076.
    assert float(a.weight.detach())==pytest.approx(.076,abs=1e-6)
    assert loss==pytest.approx((4*.01+.28**2)/5,abs=1e-6)
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())
