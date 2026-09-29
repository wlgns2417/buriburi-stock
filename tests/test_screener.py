from pathlib import Path
import importlib.util,sys
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('screener',ROOT/'test.py');a=importlib.util.module_from_spec(spec);sys.modules['screener']=a;spec.loader.exec_module(a)
SOURCE=(ROOT/'test.py').read_text().rsplit("if __name__=='__main__':",1)[0]
FIXTURE='''
def fixture_history(code, days=800, region='KR'):
    idx=pd.bdate_range(end=pd.Timestamp.now().normalize()-pd.Timedelta(days=2),periods=400)
    price=100+np.arange(400)*.1+np.sin(np.arange(400)/7)*2
    df=pd.DataFrame({'Open':price,'High':price+2,'Low':price-2,'Close':price+.2,'Volume':10000.},index=idx)
    df.attrs['region']=region
    return Result(df,'TEST FIXTURE · 합성 데이터')
def offline(*args,**kwargs):raise ConnectionError('offline test')
request_bytes=offline
fdr_table=offline
fetch_history=fixture_history
main()
'''
@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    a.st.cache_data.clear()
    def fail(*args,**kwargs):raise ConnectionError('offline test')
    monkeypatch.setattr(a,'request_bytes',fail);monkeypatch.setattr(a,'fdr_table',fail)

NOW=pd.Timestamp('2026-09-29',tz='Asia/Seoul')
def history():
    index=pd.bdate_range(end='2026-09-28',periods=400)
    price=100+np.arange(400)*.07+np.sin(np.arange(400)/6)*3
    d=pd.DataFrame({'Open':price-.2,'High':price+1,'Low':price-1,'Close':price,'Volume':10000.},index=index)
    d.attrs['region']='KR';return d

def test_period_mdd_and_new_record():
    d=history();d.iloc[-1,d.columns.get_loc('Close')]=40;d.iloc[-1,d.columns.get_loc('Low')]=39
    p=a.analyze(d,NOW)
    assert p['reach']>1 and not p['checks']['표본 MDD 70~100%'] and not p['eligible']

def test_causal_current_bar_and_stale_history():
    d=history();row=d.iloc[-1:].copy();row.index=pd.DatetimeIndex(['2026-09-29'])
    ext=pd.concat([d,row]);ext.attrs=d.attrs
    assert a.analyze(ext,NOW)['asof']=='2026-09-28'
    with pytest.raises(ValueError):a.analyze(d.tail(200),NOW)
    with pytest.raises(ValueError):a.analyze(d,pd.Timestamp('2026-10-15',tz='Asia/Seoul'))

def test_confirmed_divergence_and_invalidation():
    n=60;d=pd.DataFrame({'Low':np.ones(n)*100,'High':np.ones(n)*110,'Close':np.ones(n)*105,'MA200':np.ones(n)*90,'RSI':np.ones(n)*40,'MACD_HIST':np.zeros(n),'ATR14':np.ones(n)})
    d.index=pd.bdate_range('2026-07-01',periods=n)
    d.iloc[40,d.columns.get_loc('Low')]=90;d.iloc[55,d.columns.get_loc('Low')]=85
    d.iloc[40,d.columns.get_loc('RSI')]=20;d.iloc[55,d.columns.get_loc('RSI')]=30
    assert not a.find_divergences(d.iloc[:58])
    assert any(e['유형']=='일반 상승' and e['확인일']==str(d.index[58].date()) for e in a.find_divergences(d))
    d.iloc[-1,d.columns.get_loc('Low')]=84
    assert not a.find_divergences(d)

def test_reward_does_not_skip_barrier_or_count_assumed_target():
    p=a.trade_plan('test',100,102,97,[101,120],'condition')
    assert p['target']==101 and p['rr']<0 and not p['rr_pass']
    p=a.trade_plan('test',100,100,97,[],'condition')
    assert p['target_kind']=='2R 관리선(가정)' and not p['rr_pass']
    assert not a.trade_plan('test',100,100,100,[],'condition')['valid']

def test_filters_sort_html_escape_and_fallback():
    p=a.analyze(history(),NOW);p.update(code='TEST',name='<script>x</script>',region='US',source='fixture')
    assert '&lt;script&gt;' in a.card_html(p,a.MARKETS[2])
    assert '<script>' not in a.card_html(p,a.MARKETS[2])
    assert not a.filter_profiles([p],'전체 종목','no-match',[],a.SORTS[0])
    assert len(a.filter_profiles([p],'전체 종목','test',[],a.SORTS[0]))==1
    rows,source=a.market_universe(a.MARKETS[2]);assert rows and '대체 표본' in source

def test_new_interface_has_no_legacy_navigation_and_all_markets():
    at=AppTest.from_string(SOURCE+FIXTURE).run(timeout=30)
    assert not at.exception
    assert not any(x.key in ['v2_menu','market_region'] for x in at.radio)
    for market in a.MARKETS:
        at.radio(key='market').set_value(market).run(timeout=30)
        assert not at.exception
        assert at.session_state['scan_'+market]['profiles']
    for preset in a.PRESETS:
        at.radio(key='preset').set_value(preset).run(timeout=30);assert not at.exception
    at.radio(key='preset').set_value('전체 종목').run(timeout=30)
    next(b for b in at.button if str(b.key).startswith('detail_')).click().run(timeout=30)
    assert not at.exception

def test_pause_search_and_sort():
    at=AppTest.from_string(SOURCE+FIXTURE).run(timeout=30)
    at.toggle(key='auto_scan').set_value(False).run(timeout=30)
    cursor=at.session_state['scan_'+a.MARKETS[0]]['cursor']
    at.text_input(key='search').set_value('nonsense').run(timeout=30)
    assert not at.exception and at.session_state['scan_'+a.MARKETS[0]]['cursor']==cursor
    assert not any(str(b.key).startswith('detail_') for b in at.button)
    for sort in a.SORTS:
        at.selectbox(key='sort').set_value(sort).run(timeout=30);assert not at.exception

def test_failing_data_stops_without_fabricated_cards():
    fixture=FIXTURE.replace('fetch_history=fixture_history','fetch_history=lambda *a,**k:Result(pd.DataFrame(),"offline","error")')
    at=AppTest.from_string(SOURCE+fixture).run(timeout=30)
    for i in range(4):at.run(timeout=30)
    job=at.session_state['scan_'+a.MARKETS[0]]
    assert not job['profiles'] and job['cursor']==6 and len(job['errors'])==6
    assert at.warning


def test_add_symbol_and_guide_dialog():
    at=AppTest.from_string(SOURCE+FIXTURE).run(timeout=30)
    next(b for b in at.button if b.label=='＋ 종목 추가').click().run(timeout=30)
    at.text_input(key='add_code').set_value('005930')
    at.text_input(key='add_name').set_value('삼성전자')
    next(b for b in at.button if b.label=='추가하고 분석').click().run(timeout=30)
    assert not at.exception
    assert at.session_state['custom_'+a.MARKETS[0]][0]['Code']=='005930'
    next(b for b in at.button if b.label=='▣ 전략 / 설명서').click().run(timeout=30)
    assert not at.exception
