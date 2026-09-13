"""Causal levels, observed turnover, missing-flow exclusions and automatic execution."""
import numpy as np
import pandas as pd
from test_v2 import a, frame, no_network


def investors(df, foreign=None, institution=None):
    n=5
    return pd.DataFrame({'Date':df.index[-n:],'ForeignNet':foreign or [200]*n,
                         'InstitutionNet':institution or [100]*n})


def test_turnover_uses_actual_dated_value_and_rejects_gaps(frame):
    raw=pd.DataFrame({'Amount':1e7,'MarCap':1e9},index=frame.index[-21:])
    raw.iloc[-1,0]=3e7
    result=a.v21_turnover(raw,frame)
    assert result['turnover']==3 and result['turnover_multiple']==3
    assert result['turnover_ok']
    changed=frame.copy();changed.Close*=100
    assert a.v21_turnover(raw,changed)==result  # Never synthesize traded value from close.
    assert not a.v21_turnover(raw.iloc[:-1],frame)['turnover_ok']
    raw.iloc[-1,1]=0
    assert a.v21_turnover(raw,frame)['turnover'] is None


def test_joint_buying_is_joint_not_combined_or_missing(frame):
    result=a.v21_joint_flow(investors(frame),frame)
    assert result['flow_ok'] and result['joint_streak']==5
    # Positive combined flows every day, but only two actual joint buying days.
    split=investors(frame,[500,-100,500,-100,500],[-100,500,-100,500,500])
    assert not a.v21_joint_flow(split,frame)['flow_ok']
    assert not a.v21_joint_flow(pd.DataFrame(),frame)['flow_known']
    assert not a.v21_joint_flow(investors(frame).iloc[:-1],frame)['flow_ok']
    impossible=investors(frame,[20000]*5)
    assert not a.v21_joint_flow(impossible,frame)['flow_known']


def test_levels_exclude_signal_bar_and_stop_stays_below_support(frame):
    before=a.v21_trade_levels(frame)
    changed=frame.copy();changed.loc[changed.index[-1],'High']*=2
    after=a.v21_trade_levels(changed)
    assert before['breakout']==after['breakout']
    assert before['support']==after['support']
    assert 0<before['stop']<before['support']<=frame.Close.iloc[-1]


def profile(frame,code,asof='2026-09-11',flow=True):
    e=a.v21_evidence(frame,investors(frame) if flow else None)
    return {'code':code,'name':code,'region':'KR','asof':asof,'close':float(frame.Close.iloc[-1]),
            'signal':{'score':80},'source':'synthetic test','evidence':e}


def test_sector_expansion_requires_same_constituents_and_coverage(frame):
    codes=a.V2_SECTORS['KR']['반도체']
    rows=[profile(frame,c) for c in codes]
    for p in rows:
        p['evidence'].update(above20=True,above20_past=False,positive5=True,quality=True)
    table=a.v21_sector_breadth(rows,'KR').set_index('섹터')
    assert table.loc['반도체','확산 충족']
    assert len(a.v21_candidates(rows,'KR','sector'))==3
    rows[-1]['asof']='2026-09-10'
    table=a.v21_sector_breadth(rows,'KR').set_index('섹터')
    assert not table.loc['반도체','확산 충족']
    assert a.v21_candidates(rows,'KR','sector')==[]


def test_candidates_do_not_promote_missing_flow_or_old_dates(frame):
    rows=[profile(frame,'005930'),profile(frame,'000660',flow=False),profile(frame,'042700','2026-09-10')]
    assert [r['코드'] for r in a.v21_candidates(rows,'KR','flow')]==['005930']


def test_automatic_chunks_complete_without_manual_start(frame,monkeypatch):
    monkeypatch.setattr(a,'v21_profile',lambda region,code,name:profile(frame,code))
    job={'records':[{'Code':str(n),'Name':str(n)} for n in range(9)],'cursor':0,'profiles':[],'errors':[],'halted':False}
    for _ in range(3): a.v21_advance(job,'KR')
    assert job['cursor']==9 and len(job['profiles'])==9
    a.v21_advance(job,'KR')
    assert len(job['profiles'])==9


def test_total_source_failure_stops_bounded_requests(monkeypatch):
    def fail(*args):raise ValueError('no historical data')
    monkeypatch.setattr(a,'v21_profile',fail)
    job={'records':[{'Code':str(n),'Name':str(n)} for n in range(50)],'cursor':0,'profiles':[],'errors':[],'halted':False}
    for _ in range(5): a.v21_advance(job,'KR')
    assert job['halted'] and job['cursor']==12 and not job['profiles']
