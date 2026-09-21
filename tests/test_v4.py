"""Entry guide: no financial requests, no invented reward signal, causal reference levels."""
import numpy as np
import pandas as pd
import pytest
from test_v2 import a,no_network

NOW=pd.Timestamp('2026-09-21 16:00',tz='America/New_York')

def history(direction=1):
    index=pd.bdate_range(end='2026-09-18',periods=300)
    price=100+direction*np.arange(300)*.05+np.sin(np.arange(300)*.5)
    frame=pd.DataFrame({'Open':price-.15,'High':price+.7,'Low':price-.7,'Close':price,'Volume':10000.},index=index)
    frame.attrs['region']='US'
    return frame


def test_reward_is_observed_and_conservative_for_entry_range():
    p=a.v4_trade_plan('test',100,102,97,[101,120],'condition')
    assert p['target']==101 and p['rr']<0 and not p['rr_pass']
    at_level=a.v4_trade_plan('test',100,100,97,[100,120],'condition')
    assert at_level['rr']==0 and not at_level['rr_pass']
    p=a.v4_trade_plan('test',100,102,97,[112,120],'condition')
    assert p['rr']==2 and p['risk_pct']==pytest.approx(5/102*100)
    assert p['rr_pass']


def test_two_r_management_line_never_proves_entry_edge():
    p=a.v4_trade_plan('test',100,102,97,[],'condition')
    assert p['target']==112 and p['rr']==2
    assert p['target_kind']=='2R 관리선(가정)' and not p['rr_pass']
    assert not a.v4_trade_plan('test',100,102,103,[],'')['valid']


def test_downtrend_and_stale_quotes_do_not_recommend_entry():
    result=a.v4_entry_guide(history(-1),now=NOW)
    assert result['status']=='관망'
    stale=a.v4_entry_guide(history(),now='2027-01-01')
    assert not stale['valid'] and stale['plans']==[]


def test_manual_price_changes_risk_reward_without_changing_historical_levels():
    frame=history();first=a.v4_entry_guide(frame,now=NOW)
    assert first['valid']
    higher=a.v4_entry_guide(frame,reference_price=first['reference']+2*first['atr'],now=NOW)
    assert higher['support']==first['support']
    assert higher['resistance20']==first['resistance20']
    assert higher['plans'][0]['risk_pct']>first['plans'][0]['risk_pct']
    below=a.v4_entry_guide(frame,reference_price=first['plans'][0]['stop']-.1,now=NOW)
    assert below['status']=='관망' and not below['plans'][0]['valid']


def test_signal_high_not_used_as_prior_breakout_level_and_today_excluded():
    frame=history();first=a.v4_entry_guide(frame,now=NOW)
    changed=frame.copy();changed.loc[changed.index[-1],'High']*=2
    assert a.v4_entry_guide(changed,now=NOW)['resistance20']==first['resistance20']
    future=frame.copy();future.loc[pd.Timestamp('2026-09-21')]=[150,151,149,150,10000]
    assert a.v4_entry_guide(future,now=NOW)['asof']==first['asof']


def test_unavailable_benchmark_is_not_marked_as_outperformance():
    result=a.v4_entry_guide(history(),pd.DataFrame(),now=NOW)
    assert result['relative20'] is None
    assert all(r['점검']!='시장 대비 상대강도' for r in result['checks'])


def test_ready_verdict_requires_observed_reward_and_bounded_risk():
    # Many deterministic histories cover mixed prices; check safety invariant in every case.
    for shift in range(16):
        f=history();f.Close+=np.sin(np.arange(len(f))*.5+shift)*.3
        f.High=np.maximum(f.High,f.Close+.1);f.Low=np.minimum(f.Low,f.Close-.1)
        result=a.v4_entry_guide(f,now=NOW)
        if result['status']=='분할 진입 검토':
            ready=[p for p in result['plans'] if p.get('state')=='조건 충족']
            assert ready and all(p['rr_pass'] and p['target_kind']=='관측 저항' and p['risk_pct']<=8 for p in ready)
        for plan in result['plans']:
            if plan['valid']: assert 0<plan['stop']<plan['low']<=plan['high']


def test_us_screen_has_entry_guide_and_never_requests_financials():
    from streamlit.testing.v1 import AppTest
    from test_v2_ui import SOURCE,FIXTURE
    a.st.cache_data.clear()
    fixture=FIXTURE.rsplit('main()',1)[0]+'''
def forbidden_financials(*args): raise AssertionError('Financial request must not execute')
v3_fundamentals=forbidden_financials
v3_yahoo_snapshot=forbidden_financials
main()
'''
    at=AppTest.from_string(SOURCE+fixture).run(timeout=30)
    at.radio(key='market_region').set_value('🇺🇸 미국주식').run(timeout=30)
    at.text_input(key='us_query').input('AVGO').run(timeout=30)
    next(b for b in at.button if b.label=='미국 종목 정밀 분석').click().run(timeout=30)
    assert not at.exception
    labels=[m.label for m in at.metric]
    assert '기업 실적·가치' not in labels and 'TTM ROE' not in labels
    assert '현재 손익비' in labels
    tables=[f.value for f in at.dataframe if '진입 하단(USD)' in f.value.columns]
    assert len(tables)==1 and len(tables[0])>=2
    at.checkbox(key='v4_custom_AVGO').check().run(timeout=30)
    at.number_input(key='v4_price_AVGO').set_value(160.).run(timeout=30)
    assert not at.exception
    assert any('160.00' in c.value for c in at.caption)
