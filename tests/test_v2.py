"""Deterministic tests. No live credentials, prices, or network are required."""
import importlib.util
import sys
from pathlib import Path
from datetime import datetime, timezone
import json
import numpy as np
import pandas as pd
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('terminal_v2',ROOT/'test.py')
a=importlib.util.module_from_spec(spec);sys.modules[spec.name]=a;spec.loader.exec_module(a)

@pytest.fixture
def frame():
    index=pd.bdate_range('2025-01-01',periods=100)
    price=np.linspace(100,130,len(index))+np.sin(np.arange(len(index))/5)
    frame=pd.DataFrame({'Open':price,'High':price+2,'Low':price-2,'Close':price+.5,'Volume':10000.},index=index)
    return a.add_indicators(frame)

@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    a.st.cache_data.clear()
    def fail(*args,**kwargs):raise ConnectionError('offline test')
    monkeypatch.setattr(a,'request_bytes',fail)
    monkeypatch.setattr(a,'fdr_table',fail)
    monkeypatch.setattr(a,'v2_sec_json',fail)

def test_precursor_bounded_and_explained(frame):
    p=a.v2_precursor(frame)
    assert 0<=p['score']<=100
    assert sum(r['배점'] for r in p['rules'])==100
    assert p['score']==max(0,sum(r['득점'] for r in p['rules'])-p['penalty'])
    assert '기관 매집' not in p['comment']

def test_volume_baseline_excludes_signal_bar(frame):
    frame.loc[frame.index[-1],'Volume']=30000
    p=a.v2_precursor(frame)
    assert p['volume_ratio']==3
    assert p['rules'][0]['득점']==25

def test_flat_or_missing_history_not_scored(frame):
    frame.Volume=0
    with pytest.raises(ValueError,match='거래량'):a.v2_precursor(frame)
    with pytest.raises(ValueError,match='66'):a.v2_precursor(frame.head(20))

def test_overheat_penalty_is_explicit(frame):
    frame.loc[frame.index[-1],'RSI']=85
    p=a.v2_precursor(frame)
    assert p['penalty']==10 and p['status']!='관심 조건 충족'

def test_trailing_streak_treats_zero_and_missing_as_break():
    assert a.v2_streak([-5,2,3])==2
    assert a.v2_streak([2,3,0])==0
    assert a.v2_streak([1,np.nan,4])==1

def test_performance_never_enters_before_observation():
    index=pd.bdate_range('2026-01-01',periods=15)
    df=pd.DataFrame({'Open':100.,'High':220.,'Low':90.,'Close':110.,'Volume':1000.},index=index)
    # A stale signal price cannot earn a pre-observation jump.
    rec={'region':'KR','signal_date':'2026-01-01','recorded_at':'2026-01-08T20:00:00+09:00'}
    df.loc[:'2026-01-08','Close']=200
    result=a.v2_signal_outcome(rec,df,5,0)
    assert result['entry_date']=='2026-01-09'
    assert result['exit_date']=='2026-01-15'
    assert result['return_pct']==pytest.approx(10)
    fee=a.v2_signal_outcome(rec,df,5,10)
    assert fee['return_pct']<10

def test_outcome_waits_and_skips_suspension():
    idx=pd.bdate_range('2026-01-01',periods=8)
    df=pd.DataFrame({'Open':100.,'High':110.,'Low':90.,'Close':105.,'Volume':1000.},index=idx)
    rec={'region':'US','signal_date':'2026-01-01','recorded_at':'2026-01-01T20:00:00-05:00'}
    df.attrs['region']='US'
    df.loc['2026-01-02','Volume']=0
    got=a.v2_signal_outcome(rec,df,5,0)
    assert got['entry_date']=='2026-01-05'
    assert a.v2_signal_outcome(rec,df.head(3),5)['return_pct'] is None

def test_backup_has_no_nonfinite_or_cross_symbol_positions():
    doc={'schema':2,'portfolio':[{'region':'KR','code':'005930','name':'삼성전자','quantity':2,'average':100,'stop':90,'target':120}]}
    clean=a.v2_validate_backup(json.dumps(doc).encode())
    assert clean['portfolio'][0]['average']==100
    doc['portfolio'][0]['quantity']=float('nan')
    with pytest.raises(ValueError):a.v2_validate_backup(json.dumps(doc).encode())

def test_duplicate_portfolio_and_unversioned_signal_rejected():
    p={'region':'US','code':'AAPL','quantity':2,'average':100}
    with pytest.raises(ValueError):a.v2_validate_backup(json.dumps({'schema':2,'portfolio':[p,p]}).encode())
    with pytest.raises(ValueError):a.v2_validate_backup(json.dumps({'schema':2,'signals':[{'region':'KR','code':'005930','model':'old'}]}).encode())

def test_portfolio_does_not_convert_or_sum_currencies():
    positions=[{'region':'KR','code':'005930','quantity':2,'average':100,'stop':90},
               {'region':'US','code':'AAPL','quantity':1,'average':50,'target':70}]
    got=a.v2_portfolio_value(positions,{('KR','005930'):{'close':80,'asof':'2026-01-01'}})
    assert got[0]['currency']=='KRW' and got[0]['pnl']==-40 and got[0]['alert']=='손절 기준 이하'
    assert got[1]['currency']=='USD' and got[1]['value'] is None

def test_sector_comparison_requires_same_date_and_two_members():
    ps=[{'code':'NVDA','asof':'2026-01-02','signal':{'score':80,'volume_ratio':2},'delta5':10},
        {'code':'AMD','asof':'2026-01-02','signal':{'score':60,'volume_ratio':1},'delta5':-2},
        {'code':'AVGO','asof':'2026-01-01','signal':{'score':100,'volume_ratio':3},'delta5':50}]
    t=a.v2_sector_table(ps,'US').set_index('섹터')
    assert t.loc['반도체','강도']==70 and t.loc['반도체','5일 변화']==4
    assert t.loc['반도체','확인 종목']==2
    assert pd.isna(t.loc['에너지','강도'])

def test_schedule_only_due_weekdays_no_catchup_or_duplicate():
    now=datetime(2026,9,14,9,31,tzinfo=a.KST)
    assert a.v2_due_slot(now,['09:30','11:00'],[])=='2026-09-14 09:30'
    assert a.v2_due_slot(now,['09:30'],['2026-09-14 09:30']) is None
    assert a.v2_due_slot(now.replace(hour=10),['09:30'],[]) is None
    assert a.v2_due_slot(now.replace(day=12),['09:30'],[]) is None

def test_disclosure_failures_do_not_leak_credentials(monkeypatch):
    monkeypatch.setattr(a,'v2_dart_codes',lambda key:(_ for _ in ()).throw(ValueError('secret=DO_NOT_PRINT')))
    r=a.v2_disclosures('KR','005930','DO_NOT_PRINT')
    assert r.status=='error' and 'DO_NOT_PRINT' not in ' '.join(r.notes)

def test_sec_disclosure_primary_links(monkeypatch):
    monkeypatch.setattr(a,'v2_sec_tickers',lambda agent:{'AAPL':320193})
    monkeypatch.setattr(a,'v2_sec_json',lambda *args:{'filings':{'recent':{'form':['10-Q'],'filingDate':['2026-08-01'],'accessionNumber':['0000320193-26-000001'],'primaryDocument':['aapl.htm']}}})
    got=a.v2_disclosures('US','AAPL','Fixture contact@example.com')
    assert got.data[0]['link']=='https://www.sec.gov/Archives/edgar/data/320193/000032019326000001/aapl.htm'

def test_news_is_clues_not_predicted_impact():
    assert a.v2_news_clues('수주 증가에도 소송 부담')[0]=='혼재·원문 확인'
    assert a.v2_news_clues('회사 정기 주주총회')[0]=='방향 판단 보류'
