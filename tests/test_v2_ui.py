from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

ROOT=Path(__file__).resolve().parents[1]
SOURCE=(ROOT/'test.py').read_text().rsplit('if __name__ == "__main__":',1)[0]
FIXTURE=r'''
def fixture_history(code, days=800, region='KR'):
    idx=pd.bdate_range(end=pd.Timestamp.now().normalize()-pd.Timedelta(days=2),periods=400)
    price=100+np.arange(400)*.1+np.sin(np.arange(400)/7)*2
    df=pd.DataFrame({'Open':price,'High':price+2,'Low':price-2,'Close':price+.2,'Volume':10000.},index=idx)
    df.attrs['region']=region
    return Result(df,'offline fixture')
def offline(*a,**k): raise ConnectionError('offline fixture')
request_bytes=offline
fdr_table=offline
fetch_history=fixture_history
v2_secret=lambda name:''
_naver_market=lambda *a:Result(pd.DataFrame(columns=MARKET_COLS),'fixture','error',['0종목 시세 표본'])
fetch_main=lambda *a:Result({'fund':empty_fund(),'quote':{}},'fixture','error')
fetch_investors=lambda *a:Result(pd.DataFrame(columns=INV_COLS),'fixture','error')
fetch_reports=lambda *a:Result([],'fixture','error')
main()
'''

def app():
    return AppTest.from_string(SOURCE+FIXTURE).run(timeout=30)

def test_all_menus_and_markets_survive_offline():
    at=app()
    assert not at.exception
    for region in ['🇰🇷 한국주식','🇺🇸 미국주식']:
        at.radio(key='market_region').set_value(region).run(timeout=30)
        for menu in ['🧭 시장 상황판','🚨 급등 전조','🔥 섹터 순환','🐋 수급 추적','📰 뉴스·공시','💼 내 종목','🧪 백테스트·성과','📊 종목 분석']:
            at.radio(key='v2_menu').set_value(menu).run(timeout=30)
            assert not at.exception,(region,menu)

def test_radar_batch_record_navigation_and_original_analysis():
    at=app()
    at.radio(key='v2_menu').set_value('🚨 급등 전조').run(timeout=30)
    at.button(key='v2_start_scan').click().run(timeout=30)
    assert not at.exception
    assert at.session_state['v2_job_KR']['cursor']==9
    assert len(at.session_state['v2_profiles_KR'])==9
    at.button(key='v2_manual_record_radar').click().run(timeout=30)
    assert not at.exception
    assert len(at.session_state.v2_signals)>=1
    count=len(at.session_state.v2_signals)
    at.button(key='v2_manual_record_radar').click().run(timeout=30)
    assert len(at.session_state.v2_signals)==count
    at.button(key='v2_open_radar').click().run(timeout=30)
    assert not at.exception
    assert at.session_state.selected_code=='009830'
    assert at.radio(key='v2_menu').value=='📊 종목 분석'

def test_portfolio_form_value_delete_and_performance_waiting():
    at=app()
    at.radio(key='v2_menu').set_value('💼 내 종목').run(timeout=30)
    at.text_input(key='v2_position_code').input('005930')
    at.number_input(key='v2_position_quantity').set_value(2.)
    at.number_input(key='v2_position_average').set_value(100.)
    next(b for b in at.button if b.label=='보유 종목 추가·수정').click().run(timeout=30)
    assert not at.exception
    assert len(at.session_state.v2_portfolio)==1
    at.button(key='v2_value_portfolio').click().run(timeout=30)
    assert not at.exception
    assert at.session_state.v2_portfolio_values[0]['value']>0
    at.button(key='v2_delete_position').click().run(timeout=30)
    assert not at.exception and len(at.session_state.v2_portfolio)==0
    at.radio(key='v2_menu').set_value('🧪 백테스트·성과').run(timeout=30)
    at.button(key='v2_evaluate_signals').click().run(timeout=30)
    assert not at.exception

def test_sector_flow_news_and_us_scan():
    at=app()
    at.radio(key='v2_menu').set_value('🔥 섹터 순환').run(timeout=30)
    at.button(key='v2_sector_scan').click().run(timeout=30)
    assert not at.exception
    at.radio(key='v2_menu').set_value('🐋 수급 추적').run(timeout=30)
    at.text_input(key='v2_flow_query').input('005930').run(timeout=30)
    at.button(key='v2_load_flow').click().run(timeout=30)
    assert not at.exception
    at.radio(key='v2_menu').set_value('📰 뉴스·공시').run(timeout=30)
    at.text_input(key='v2_news_query').input('005930').run(timeout=30)
    at.button(key='v2_load_news').click().run(timeout=30)
    assert not at.exception
    at.radio(key='market_region').set_value('🇺🇸 미국주식').run(timeout=30)
    at.radio(key='v2_menu').set_value('🚨 급등 전조').run(timeout=30)
    at.button(key='v2_start_scan').click().run(timeout=30)
    assert not at.exception
    at.button(key='v2_next_batch').click().run(timeout=30)
    assert not at.exception
    assert at.session_state['v2_job_US']['cursor']==15
    assert all(p['region']=='US' for p in at.session_state['v2_profiles_US'])
