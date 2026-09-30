from pathlib import Path
import importlib.util,sys,time,threading,tempfile
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
@st.cache_resource
def fixture_service():return ScanService(tempfile.mkdtemp(),loader=load_profile,universe_loader=market_universe)
scan_service=fixture_service
main()
'''
@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    a.st.cache_data.clear();a.st.cache_resource.clear()
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


def finish(service,region,records):
    for _ in range(300):
        snap=service.poll(region,records)
        if snap['cursor']==snap['total'] or snap['blocked']:return snap
        time.sleep(.01)
    raise AssertionError('worker did not complete')

def sample_profile(code='005930'):
    d=history();d.index=pd.bdate_range(end=pd.Timestamp.now().normalize()-pd.Timedelta(days=2),periods=len(d))
    p=a.analyze(d);p.update(code=code,name=code,region='KR',source='fixture');return p

def test_bounded_parallel_nonblocking_and_singleflight(tmp_path):
    gate=threading.Event();lock=threading.Lock();calls=[];active=[0,0]
    def loader(region,code,name):
        with lock:active[0]+=1;active[1]=max(active);calls.append(code)
        gate.wait(3)
        try:return sample_profile(code)
        finally:
            with lock:active[0]-=1
    service=a.ScanService(tmp_path,workers=4,loader=loader)
    records=[{'Code':f'{i:06d}','Name':str(i)} for i in range(8)]
    try:
        start=time.monotonic();first=service.poll('KR',records)
        assert time.monotonic()-start<.5 and first['pending']==4
        service.poll('KR',records);assert len(service.pending)==4
        gate.set();out=finish(service,'KR',records+records)
        assert out['cursor']==8 and out['total']==8 and out['progress']==1
        assert len(calls)==8 and active[1]<=4
        service.poll('KR',records,paused=True);assert len(calls)==8
    finally:gate.set();service.close()

def test_persistent_reuse_refresh_and_no_names_on_disk(tmp_path):
    calls=[]
    def loader(region,code,name):calls.append(code);return sample_profile(code)
    records=[{'Code':'005930','Name':'PRIVATE_CUSTOM_NAME'}]
    service=a.ScanService(tmp_path,loader=loader)
    out=finish(service,'KR',records);service.close()
    assert out['profiles'][0]['name']=='PRIVATE_CUSTOM_NAME'
    assert 'PRIVATE_CUSTOM_NAME' not in next(tmp_path.glob('*.json')).read_text()
    service=a.ScanService(tmp_path,loader=loader)
    try:
        out=service.poll('KR',records);assert out['progress']==1 and len(calls)==1
        service.invalidate('KR',records);finish(service,'KR',records);assert len(calls)==2
    finally:service.close()

def test_failure_pause_empty_and_changing_universe_progress(tmp_path):
    def fail(*args):raise ValueError('offline')
    service=a.ScanService(tmp_path,loader=fail)
    records=[{'Code':str(i),'Name':str(i)} for i in range(20)]
    try:
        assert service.poll('KR',records,paused=True)['pending']==0
        out=finish(service,'KR',records);assert out['blocked'] and out['cursor']==4
        for subset in [records[:1],[],records+records]:
            snap=service.poll('KR',subset,paused=True);assert 0<=snap['progress']<=1 and snap['cursor']<=snap['total']
    finally:service.close()

def test_no_chart_detail_and_group_scores():
    p=sample_profile();text=a.detail_html(p)
    assert '5대 기술 조건' in text and '손익비' in text and '기술적 반등 점수' in text
    assert '<svg' not in text and 'plotly' not in text
    assert 'KODEX' not in ''.join(a.MARKETS) and len(a.MARKETS)==3

def settle(at):
    for _ in range(20):
        at.run(timeout=10)
        if any(str(b.key).startswith('detail_') for b in at.button):return at
        time.sleep(.03)
    return at

def test_async_ui_market_filter_detail_and_add():
    at=settle(AppTest.from_string(SOURCE+FIXTURE))
    assert not at.exception
    for market in a.MARKETS:
        at.radio(key='market').set_value(market);settle(at);assert not at.exception
    for preset in a.PRESETS:
        at.radio(key='preset').set_value(preset).run(timeout=10);assert not at.exception
    at.radio(key='preset').set_value('전체 종목');settle(at)
    next(b for b in at.button if str(b.key).startswith('detail_')).click().run(timeout=10)
    assert not at.exception and len(at.get('plotly_chart'))==0
    at.radio(key='market').set_value(a.MARKETS[0]).run(timeout=10)
    next(b for b in at.button if b.label=='＋ 종목 추가').click().run(timeout=10)
    at.text_input(key='add_code').set_value('005930');at.text_input(key='add_name').set_value('삼성전자')
    next(b for b in at.button if b.label=='추가하고 분석').click().run(timeout=10)
    assert not at.exception
