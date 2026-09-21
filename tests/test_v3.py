"""Financial identities, missingness, causal momentum and scenario sensitivity."""
import json
import numpy as np
import pandas as pd
import pytest
from test_v2 import a,no_network

NOW=pd.Timestamp('2026-09-21 12:00:00',tz='America/New_York')

def packed(rows,dates):
    frame=pd.DataFrame(rows,index=dates).T
    return json.loads(frame.to_json(orient='split',date_format='iso'))

@pytest.fixture
def payload():
    dates=pd.to_datetime(['2026-06-30','2026-03-31','2025-12-31','2025-09-30','2025-06-30'])
    return {'info':{'symbol':'TEST','quoteType':'EQUITY','currency':'USD','financialCurrency':'USD',
                    'sector':'Technology','marketCap':2000.,'regularMarketTime':int(pd.Timestamp('2026-09-18',tz='UTC').timestamp()),'sharesOutstanding':20},
            'tables':{'income':packed({'TotalRevenue':[150,140,130,120,100],'NetIncome':[30,28,26,24,15]},dates),
                      'cash':packed({'OperatingCashFlow':[40,35,30,25,20],'CapitalExpenditure':[-5]*5},dates),
                      'balance':packed({'StockholdersEquity':[300,280,260,240,220],'TotalDebt':[80]*5,'CashAndCashEquivalents':[50]*5},dates)}}


def test_statement_ratios_use_matched_ttm_and_average_equity(payload):
    c=a.v3_company_metrics(payload,NOW);m=c['metrics']
    assert m['revenue']==540 and m['net_income']==108 and m['fcf']==110
    assert m['roe']==pytest.approx(108/260)
    assert m['revenue_growth']==.5 and m['earnings_growth']==1
    assert m['fcf_yield']==pytest.approx(.055)
    score=a.v3_company_score(c)
    assert sum(r['배점'] for r in score['rows'])==100
    assert score['score']==pytest.approx(sum(r['득점'] for r in score['rows']))


def test_missing_zero_negative_are_not_interchangeable(payload):
    payload['tables'].pop('cash')
    score=a.v3_company_score(a.v3_company_metrics(payload,NOW))
    assert score['score'] is None and score['coverage']==60
    payload['tables']['income']['data'][1][-1]=-1
    c=a.v3_company_metrics(payload,NOW)
    assert c['metrics']['earnings_growth'] is None
    assert '기저효과' in ' '.join(c['notes'])
    payload['tables']['income']['data'][1][-1]=15
    payload['tables']['income']['data'][1][0]=0
    c=a.v3_company_metrics(payload,NOW)
    assert c['metrics']['earnings_growth']==-1


def test_mismatched_period_capex_sign_currency_and_staleness(payload):
    payload['tables']['cash']['columns'][0]='2026-07-31T00:00:00.000'
    c=a.v3_company_metrics(payload,NOW)
    assert c['metrics']['fcf'] is None
    payload['info']['financialCurrency']='EUR'
    assert not a.v3_company_metrics(payload,NOW)['supported']
    payload['info']['financialCurrency']='USD'
    assert not a.v3_company_metrics(payload,'2027-06-01')['supported']
    payload['info']['sector']='Financial Services'
    assert not a.v3_company_metrics(payload,NOW)['supported']


def test_no_consensus_or_ticker_specific_score_boost(payload):
    original=a.v3_company_score(a.v3_company_metrics(payload,NOW))
    payload['info'].update(symbol='AVGO',targetMeanPrice=999999,recommendationKey='strong_buy',numberOfAnalystOpinions=100)
    assert a.v3_company_score(a.v3_company_metrics(payload,NOW))==original


def test_high_company_low_timing_is_explained_without_false_bearish_claim(payload):
    c=a.v3_company_metrics(payload,NOW);score=a.v3_company_score(c)
    text=a.v3_commentary(score,{'score':60},{'score':39},c)
    assert '기업 지표는 우호적' in text and '단기 가격 추세' in text
    assert '상승 확률' not in text


def test_skip_month_momentum_and_benchmark_date_integrity():
    idx=pd.bdate_range('2025-01-01',periods=300)
    df=pd.DataFrame({'Close':np.arange(300)+100.,'Volume':100.},index=idx)
    baseline=a.v3_price_factors(df,df)
    assert baseline['metrics']['relative12']==0
    change=df.copy();change.iloc[-21:,0]*=1.5
    updated=a.v3_price_factors(change,df)
    assert baseline['metrics']['momentum12']==updated['metrics']['momentum12']
    assert baseline['metrics']['momentum6']==updated['metrics']['momentum6']
    assert updated['metrics']['volatility']>baseline['metrics']['volatility']
    incomplete=a.v3_price_factors(df,df.iloc[:-1])
    assert incomplete['coverage']==80 and incomplete['score'] is None
    assert a.v3_price_factors(df.tail(60),None)['coverage']==0


def test_scenario_dilution_monotonicity_and_invalid_inputs():
    base=a.v3_five_year_scenario(1000,.2,100,50,.1,25,0)
    diluted=a.v3_five_year_scenario(1000,.2,100,50,.1,25,.05)
    assert diluted['price']<base['price']
    assert base['price']==pytest.approx(1000*1.1**5*.2*25/100)
    with pytest.raises(ValueError):a.v3_five_year_scenario(1000,-.2,100,50,.1,25,0)
    with pytest.raises(ValueError):a.v3_five_year_scenario(1000,.2,0,50,.1,25,0)


def test_provider_failure_is_visible_and_not_scored(monkeypatch):
    def fail(code):raise RuntimeError('secret detail must not appear')
    monkeypatch.setattr(a,'v3_yahoo_snapshot',fail)
    result=a.v3_fundamentals('TEST')
    assert result.status=='error' and 'secret detail' not in str(result.notes)
    assert a.v3_company_score(a.v3_company_metrics(result.data,NOW))['coverage']==0


def test_suspension_does_not_receive_low_volatility_bonus():
    index=pd.bdate_range('2025-01-01',periods=300)
    frame=pd.DataFrame({'Close':100.,'Volume':0.},index=index)
    result=a.v3_price_factors(frame,frame)
    assert result['score'] is None and result['coverage']==0


def test_wrong_symbol_is_rejected(payload,monkeypatch):
    monkeypatch.setattr(a,'v3_yahoo_snapshot',lambda code:payload)
    result=a.v3_fundamentals('AVGO')
    assert result.status=='error'
