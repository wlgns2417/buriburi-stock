import numpy as np
import pandas as pd
import pytest
from test_v2 import a,no_network
from streamlit.testing.v1 import AppTest
from test_v2_ui import SOURCE,FIXTURE

NOW=pd.Timestamp('2026-09-29',tz='Asia/Seoul')
def history():
    index=pd.bdate_range(end='2026-09-28',periods=400)
    price=100+np.arange(400)*.07+np.sin(np.arange(400)/6)*3
    d=pd.DataFrame({'Open':price-.2,'High':price+1,'Low':price-1,'Close':price,'Volume':10000.},index=index)
    d.attrs['region']='KR';return d

def test_period_mdd_and_current_new_record():
    d=history();d.iloc[-1,d.columns.get_loc('Close')]=40;d.iloc[-1,d.columns.get_loc('Low')]=39
    p=a.v5_analyze(d,NOW)
    assert p['reach']>1 and not p['checks']['표본 MDD 70~100%']
    assert not p['eligible']

def test_no_current_day_or_short_stale_data():
    d=history();row=d.iloc[-1:].copy();row.index=pd.DatetimeIndex(['2026-09-29'])
    ext=pd.concat([d,row]);ext.attrs=d.attrs
    assert a.v5_analyze(ext,NOW)['asof']=='2026-09-28'
    with pytest.raises(ValueError):a.v5_analyze(d.tail(200),NOW)
    with pytest.raises(ValueError):a.v5_analyze(d,pd.Timestamp('2026-10-15',tz='Asia/Seoul'))

def test_price_plan_and_score_invariants():
    for shift in range(10):
        d=history();d.Close+=shift;d.Open+=shift;d.High+=shift;d.Low+=shift
        p=a.v5_analyze(d,NOW)
        assert 0<=p['score']<=100 and p['count']==sum(p['checks'].values())
        for plan in p['plans']:
            if plan.get('valid'):assert 0<plan['stop']<plan['low']<=plan['high']
        if p['eligible']:
            assert p['bull'] and not p['bear'] and p['plans'][0]['rr']>=2
            assert p['plans'][0]['target_kind']=='관측 저항'

def test_confirmed_divergence_and_invalidation():
    n=60;d=pd.DataFrame({'Low':np.ones(n)*100,'High':np.ones(n)*110,'Close':np.ones(n)*105,'MA200':np.ones(n)*90,'RSI':np.ones(n)*40,'MACD_HIST':np.zeros(n),'ATR14':np.ones(n)})
    d.index=pd.bdate_range('2026-07-01',periods=n)
    d.iloc[40,d.columns.get_loc('Low')]=90;d.iloc[55,d.columns.get_loc('Low')]=85
    d.iloc[40,d.columns.get_loc('RSI')]=20;d.iloc[55,d.columns.get_loc('RSI')]=30
    assert not a.v5_divergences(d.iloc[:58])
    events=a.v5_divergences(d)
    assert any(e['유형']=='일반 상승' and e['확인일']==str(d.index[58].date()) for e in events)
    d.iloc[-1,d.columns.get_loc('Low')]=84
    assert not a.v5_divergences(d)

def test_filters_and_missing_universe():
    rows,source=a.v5_universe(a.V5_MARKETS[2])
    assert rows and '대체 표본' in source
    p=a.v5_analyze(history(),NOW)
    assert a.v5_matches(p,'전체')
    assert a.v5_matches(p,'🛡️ 5대 조건 충족')==(p['count']==5)

def test_screener_ui_filters_markets_and_details():
    at=AppTest.from_string(SOURCE+FIXTURE).run(timeout=30)
    assert not at.exception
    assert at.radio(key='v2_menu').value=='🔎 반등 스크리너'
    for market in a.V5_MARKETS:
        at.radio(key='v5_market').set_value(market).run(timeout=30)
        assert not at.exception
        assert at.session_state['v5_job_'+market]['profiles']
    for preset in a.V5_FILTERS:
        at.radio(key='v5_preset').set_value(preset).run(timeout=30)
        assert not at.exception
    at.radio(key='v5_preset').set_value('전체').run(timeout=30)
    buttons=[b for b in at.button if str(b.key).startswith('v5_detail_')]
    assert buttons
    buttons[0].click().run(timeout=30)
    assert not at.exception
