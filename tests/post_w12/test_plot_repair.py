"""CPU-only regression checks; no answers generated or scientific identities edited."""
import pytest
import matplotlib
matplotlib.use('Agg')
from src.post_w12.bea_qbgf.analysis import plots,METHODS

def cells():
    return [dict(dataset=d,budget=b,method=m,quality=.1+i,evidence_tokens=50+i,
                 aspect_coverage=.2+i,mean_pairwise_redundancy=None if i==0 else .3)
            for d in ['HotpotQA','QASPER','SciFact'] for b in [256,1024] for i,m in enumerate(METHODS)]

def test_method_specific_values_and_four_figures(tmp_path,monkeypatch):
    import matplotlib.axes
    original=matplotlib.axes.Axes.bar;seen=[]
    def capture(self,x,height,*a,**kw):
        seen.append(list(height));return original(self,x,height,*a,**kw)
    monkeypatch.setattr(matplotlib.axes.Axes,'bar',capture)
    plots(tmp_path/'nested',cells(),cells())
    assert seen[0]==[.1,1.1,2.1]
    assert len(list((tmp_path/'nested/figures').glob('*.png')))==4
    assert len(seen)==24

def test_missing_cell_is_explicit(tmp_path):
    with pytest.raises(ValueError,match='Missing plotting cells'):plots(tmp_path,cells()[:-1],cells())

def test_duplicate_cell_is_explicit(tmp_path):
    with pytest.raises(ValueError,match='Duplicate plotting cell'):plots(tmp_path,cells()+[cells()[0]],cells())
