"""개미 투자전략실 — 독립형 주식 퀀트 스크리너. 실행: streamlit run test.py"""
from __future__ import annotations
import math, re, json, sys, subprocess, threading, html, time, tempfile, hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from dataclasses import dataclass, field
from io import StringIO
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import streamlit as st
KST=ZoneInfo('Asia/Seoul')
EXCHANGE_TZ={'KR':KST,'US':ZoneInfo('America/New_York')}
OHLCV=['Open','High','Low','Close','Volume']
_local=threading.local()


SEED_KR = [('009830', '한화솔루션'), ('005930', '삼성전자'), ('000660', 'SK하이닉스'),
           ('035420', 'NAVER'), ('035720', '카카오'), ('005380', '현대차'),
           ('000270', '기아'), ('042660', '한화오션'), ('012450', '한화에어로스페이스')]

SEED_US = [('NVDA', 'NVIDIA 엔비디아'), ('AAPL', 'Apple 애플'), ('TSLA', 'Tesla 테슬라'),
           ('PLTR', 'Palantir 팔란티어'), ('AMD', 'Advanced Micro Devices AMD'),
           ('MSFT', 'Microsoft 마이크로소프트'), ('AMZN', 'Amazon 아마존'),
           ('GOOGL', 'Alphabet 알파벳 구글'), ('META', 'Meta 메타'),
           ('AVGO', 'Broadcom 브로드컴'), ('JPM', 'JPMorgan 제이피모건'),
           ('V', 'Visa 비자'), ('WMT', 'Walmart 월마트'), ('XOM', 'Exxon Mobil 엑슨모빌'),
           ('BRK.B', 'Berkshire Hathaway 버크셔 해서웨이')]

_DATA_WORKER = r'''
import sys, json, contextlib
import FinanceDataReader as fdr
with contextlib.redirect_stdout(sys.stderr):
    df = fdr.StockListing(sys.argv[2]) if sys.argv[1] == 'listing' else fdr.DataReader(sys.argv[2], sys.argv[3], sys.argv[4])
print(df.to_json(orient='split', date_format='iso'))
'''

MARKETS=['🇰🇷 코스피 대형 100','🇺🇸 S&P 500','🇺🇸 나스닥 100']


def number(value):
    """None is missing; zero is a valid observation."""
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None

def clean_history(frame):
    if frame is None or frame.empty:
        raise ValueError("일봉을 가져오지 못했습니다.")
    if not set(OHLCV).issubset(frame.columns):
        raise ValueError("일봉에 Open, High, Low, Close, Volume 열이 필요합니다.")
    df = frame[OHLCV].copy()
    df.index = pd.to_datetime(df.index, errors="coerce")
    if df.index.isna().any():
        raise ValueError("해석할 수 없는 일봉 날짜가 있습니다.")
    if df.index.tz is not None:
        df.index = df.index.tz_convert(KST).tz_localize(None)
    df.index = df.index.normalize()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.apply(pd.to_numeric, errors="coerce")
    # Some providers encode suspended sessions with zero O/H/L. Keep the
    # session and its last mark, but the simulator never trades zero-volume bars.
    suspended = (df.Volume == 0) & (df.Close > 0)
    for col in ["Open", "High", "Low"]:
        df.loc[suspended & (df[col] == 0), col] = df.loc[suspended & (df[col] == 0), "Close"]
    valid = (
        np.isfinite(df).all(axis=1)
        & (df[["Open", "High", "Low", "Close"]] > 0).all(axis=1)
        & (df.Volume >= 0)
        & (df.High >= df[["Open", "Close", "Low"]].max(axis=1))
        & (df.Low <= df[["Open", "Close", "High"]].min(axis=1))
    )
    if not valid.all():
        raise ValueError(f"OHLCV 검증 실패: {int((~valid).sum())}개 일봉. 잘못된 봉을 건너뛰어 수익률을 연결하지 않습니다.")
    df.index.name = "Date"
    return df.astype(float)

def _oscillator(positive, negative):
    total = positive + negative
    return (100 * positive / total.where(total != 0)).mask(total == 0, 50.0)

def add_indicators(frame):
    df = frame.copy()
    for period in [5, 20, 60]:
        df[f"MA{period}"] = df.Close.rolling(period, min_periods=period).mean()
    std = df.Close.rolling(20, min_periods=20).std(ddof=0)
    df["BB_Upper"], df["BB_Lower"] = df.MA20 + 2 * std, df.MA20 - 2 * std
    width = df.BB_Upper - df.BB_Lower
    df["BB_pct"] = ((df.Close - df.BB_Lower) / width.where(width != 0)).mask(width == 0, 0.5)
    previous = df.Close.shift(1)
    tr = pd.concat([df.High - df.Low, (df.High - previous).abs(), (df.Low - previous).abs()], axis=1).max(axis=1)
    df["ATR14"] = tr.rolling(14, min_periods=14).mean()
    df["MACD"] = df.Close.ewm(span=12, adjust=False, min_periods=12).mean() - df.Close.ewm(span=26, adjust=False, min_periods=26).mean()
    df["MACD_SIGNAL"] = df.MACD.ewm(span=9, adjust=False, min_periods=9).mean()
    df["MACD_HIST"] = df.MACD - df.MACD_SIGNAL
    delta = df.Close.diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=14).mean()
    df["RSI"] = _oscillator(gain, loss)
    df["OBV"] = (np.sign(delta).fillna(0) * df.Volume).cumsum()
    tp = (df.High + df.Low + df.Close) / 3
    flow = tp * df.Volume
    pos = flow.where(tp.diff() > 0, 0).rolling(14, min_periods=14).sum()
    neg = flow.where(tp.diff() < 0, 0).rolling(14, min_periods=14).sum()
    df["MFI"] = _oscillator(pos, neg)
    return df

@dataclass
class Result:
    data: object
    source: str
    status: str = "ok"
    notes: list[str] = field(default_factory=list)
    fetched_at: str = field(default_factory=lambda: datetime.now(KST).isoformat(timespec="seconds"))

def valid_code(code):
    return bool(re.fullmatch(r"\d{6}", str(code)))

def _require_code(code):
    if not valid_code(code):
        raise ValueError("국내 종목의 6자리 숫자 코드를 입력해 주십시오.")

def _session():
    if not hasattr(_local, "session"):
        session = requests.Session()
        session.headers.update({"User-Agent": "Mozilla/5.0", "Accept-Language": "ko-KR,ko;q=0.9"})
        retry = Retry(total=1, connect=1, read=0, status=0, backoff_factor=0.2)
        session.mount("https://", HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=4))
        _local.session = session
    return _local.session

def request_bytes(url, params=None):
    response = _session().get(url, params=params, timeout=(3.05, 7))
    response.raise_for_status()
    if len(response.content) > 5_000_000:
        raise ValueError("응답 크기가 예상 범위를 초과했습니다.")
    return response.content

def failure(source, empty, error):
    return Result(empty, source, "error", [f"데이터를 확인하지 못했습니다 ({type(error).__name__})."])

def fdr_table(kind, symbol, days=800):
    today = datetime.now(KST).date()
    result = subprocess.run([sys.executable, '-c', _DATA_WORKER, kind, symbol,
                             str(today - timedelta(days=days)), str(today)],
                            capture_output=True, text=True, encoding='utf-8', timeout=22, check=True)
    payload = json.loads(result.stdout)
    return pd.DataFrame(payload['data'], columns=payload['columns'], index=payload['index'])

def normalize_directory(frame, region):
    df = frame.rename(columns={'Symbol': 'Code', 'Security Name': 'Name', '회사명': 'Name', '종목코드': 'Code'}).copy()
    if not {'Code', 'Name'}.issubset(df):
        raise ValueError('종목 목록의 필수 열을 확인하지 못했습니다.')
    df = df.dropna(subset=['Code', 'Name'])
    df['Code'] = df.Code.astype(str).str.strip().str.upper()
    if region == 'KR':
        df['Code'] = df.Code.str.zfill(6)
        df = df[df.Code.map(valid_code)]
    else:
        df = df[df.Code.str.fullmatch(r'[A-Z]{1,6}(?:[.-][A-Z]{1,2})?')]
    return df.drop_duplicates('Code').reset_index(drop=True)

def yahoo_history(symbol, region, days=800):
    from urllib.parse import quote
    tz = EXCHANGE_TZ[region]
    end = datetime.now(tz)
    url = 'https://query1.finance.yahoo.com/v8/finance/chart/' + quote(symbol, safe='')
    payload = json.loads(request_bytes(url, {'period1': int((end - timedelta(days=days)).timestamp()),
                                           'period2': int(end.timestamp()), 'interval': '1d'}))
    chart = payload['chart']
    if chart.get('error') or not chart.get('result'):
        raise ValueError('Yahoo 일봉 없음')
    item = chart['result'][0]
    if region == 'US' and item.get('meta', {}).get('currency') != 'USD':
        raise ValueError('USD 종목이 아닙니다.')
    values = item['indicators']['quote'][0]
    index = pd.to_datetime(item['timestamp'], unit='s', utc=True).tz_convert(tz).tz_localize(None).normalize()
    frame = pd.DataFrame({col: values[col.lower()] for col in OHLCV}, index=index)
    # 완전히 비어 있는 공급원 행만 제거; 일부 결측/비정상 봉은 clean_history에서 거부합니다.
    frame = frame.dropna(how='all')
    return clean_history(frame)

def fetch_history(code, days=800, region='KR'):
    if region == 'KR':
        _require_code(code)
        sources = [('fdr', 'NAVER:' + code), ('fdr', 'KRX:' + code)]
    else:
        if not re.fullmatch(r'[A-Z]{1,6}(?:[.-][A-Z]{1,2})?', str(code)):
            return failure('US', pd.DataFrame(), ValueError('티커 형식 오류'))
        sources = [('yahoo', code.replace('.', '-')), ('fdr', 'YAHOO:' + code.replace('.', '-'))]
    errors = []
    for kind, symbol in sources:
        try:
            df = yahoo_history(symbol, region, days) if kind == 'yahoo' else fdr_table('history', symbol, days)
            df = clean_history(df)
            df.attrs['region'] = region
            return Result(df, ('Yahoo chart / ' if kind == 'yahoo' else 'FinanceDataReader / ') + symbol,
                          notes=errors + ['공급원별 기업행사/수정주가 방식이 다를 수 있습니다.'])
        except Exception as exc:
            errors.append(symbol + ': ' + type(exc).__name__)
    return Result(pd.DataFrame(), ' / '.join(s for _, s in sources), 'error', errors)

def completed_history(frame, now=None):
    region = frame.attrs.get('region', 'KR')
    tz = EXCHANGE_TZ[region]
    now = now or datetime.now(tz)
    if now.tzinfo is None:
        now = now.replace(tzinfo=tz)
    copy = frame.copy()
    if isinstance(copy.index, pd.DatetimeIndex) and copy.index.tz is not None:
        copy.index = copy.index.tz_convert(tz).tz_localize(None)
    df = clean_history(copy)
    df = df.loc[df.index < pd.Timestamp(now.astimezone(tz).date())].copy()
    df.attrs['region'] = region
    return df

def trade_plan(name,low,high,stop,resistances,condition):
    values=[number(x) for x in [low,high,stop]]
    if any(x is None for x in values) or not 0<stop<low<=high:
        return {'name':name,'valid':False,'condition':'가격 구조가 성립하지 않아 이 시나리오는 제외합니다.'}
    obstacles=[float(x) for x in resistances if number(x) is not None and x>=low]
    risk=high-stop
    target=min(obstacles) if obstacles else high+2*risk
    kind='관측 저항' if obstacles else '2R 관리선(가정)'
    rr=(target-high)/risk
    risk_pct=risk/high*100
    return {'name':name,'valid':True,'low':float(low),'high':float(high),'stop':float(stop),
            'target':float(target),'target_kind':kind,'rr':float(rr),'risk_pct':float(risk_pct),
            'rr_pass':bool(kind=='관측 저항' and rr>=1.5 and risk_pct<=8),
            'condition':condition,'state':'조건 대기'}

def market_universe(market):
    region='KR' if market==MARKETS[0] else 'US'
    try:
        if market==MARKETS[0]:
            raw=fdr_table('listing','KOSPI')
            if 'Marcap' not in raw: raise ValueError('시총 없음')
            raw['Marcap']=pd.to_numeric(raw.Marcap,errors='coerce')
            d=normalize_directory(raw.dropna(subset=['Marcap']).sort_values('Marcap',ascending=False).head(100),'KR')
            source='코스피 시가총액 상위 100개 (KOSPI100 공식 지수 구성과 다를 수 있음)'
        elif market==MARKETS[1]:
            d=normalize_directory(fdr_table('listing','S&P500'),'US');source='FDR S&P 500 구성 목록'
        else:
            body=request_bytes('https://en.wikipedia.org/wiki/Nasdaq-100')
            tables=pd.read_html(StringIO(body.decode('utf-8')))
            d=next(t for t in tables if 'Ticker' in t.columns and 'Company' in t.columns)
            d=normalize_directory(d.rename(columns={'Ticker':'Code','Company':'Name'}),'US')
            if len(d)<90: raise ValueError('불완전한 지수 구성')
            source='공개 Nasdaq-100 구성 표 (조회 시점 목록)'
        if d.empty: raise ValueError('빈 목록')
        return d[['Code','Name']].to_dict('records'),source
    except Exception:
        seeds=SEED_KR if region=='KR' else SEED_US
        return [{'Code':c,'Name':n} for c,n in seeds], '전체 목록 조회 실패 · 주요 종목 대체 표본 (지수 전체 또는 구성 보장 아님)'

def find_divergences(df):
    """Two-sided pivots: signal is available only after three confirming bars."""
    events=[]
    for side,column in [('low','Low'),('high','High')]:
        values=df[column].to_numpy();pivots=[]
        for i in range(max(3,len(df)-100),len(df)-3):
            around=np.r_[values[i-3:i],values[i+1:i+4]]
            if (values[i]<around.min() if side=='low' else values[i]>around.max()):pivots.append(i)
        if len(pivots)<2:continue
        b=pivots[-1]
        earlier=[i for i in pivots[:-1] if 5<=b-i<=60]
        if not earlier or len(df)-1-(b+3)>10:continue
        a=earlier[-1]
        for indicator in ['RSI','MACD_HIST']:
            x,y=number(df[indicator].iloc[a]),number(df[indicator].iloc[b])
            if x is None or y is None:continue
            delta=y-x;minimum=2 if indicator=='RSI' else float(df.ATR14.iloc[b])*.02
            if abs(delta)<minimum or abs(values[b]/values[a]-1)<.002:continue
            kind=None
            if side=='low' and values[b]<values[a] and delta>0:kind='일반 상승'
            if side=='low' and values[b]>values[a] and delta<0 and df.Close.iloc[b]>df.MA200.iloc[b]:kind='히든 상승'
            if side=='high' and values[b]>values[a] and delta<0:kind='일반 하락'
            if side=='high' and values[b]<values[a] and delta>0 and df.Close.iloc[b]<df.MA200.iloc[b]:kind='히든 하락'
            if kind:
                invalid=(df.Low.iloc[b+1:]<values[b]).any() if side=='low' else (df.High.iloc[b+1:]>values[b]).any()
                if not invalid:events.append({'유형':kind,'지표':indicator,'첫 피벗':str(df.index[a].date()),'둘째 피벗':str(df.index[b].date()),'확인일':str(df.index[b+3].date()),'가격1':values[a],'가격2':values[b],'지표1':x,'지표2':y})
    return events

def analyze(frame,now=None):
    df=add_indicators(completed_history(frame,now)).tail(756).copy()
    if len(df)<252:raise ValueError('252개 이상 확정 일봉이 필요합니다.')
    region=frame.attrs.get('region','KR');today=(now or datetime.now(EXCHANGE_TZ[region])).date()
    if (today-df.index[-1].date()).days>7:raise ValueError('일봉이 7일 넘게 경과했습니다.')
    if df.Volume.iloc[-1]<=0 or (df.Volume.tail(60)>0).mean()<.9:raise ValueError('거래 정지 또는 거래량 부족')
    df['MA200']=df.Close.rolling(200).mean()
    # Wilder RSI, seeded using the first 14 changes; neutral flat prices are 50.
    delta=df.Close.diff();g=delta.clip(lower=0);l=-delta.clip(upper=0)
    for series in [g,l]:
        series.iloc[14]=series.iloc[1:15].mean();series.iloc[:14]=np.nan
    ag=g.ewm(alpha=1/14,adjust=False).mean();al=l.ewm(alpha=1/14,adjust=False).mean()
    df['RSI']=_oscillator(ag,al)
    lo=df.Low.rolling(14).min();hi=df.High.rolling(14).max()
    df['STO_K']=((df.Close-lo)/(hi-lo).replace(0,np.nan)*100).fillna(50)
    df['STO_D']=df.STO_K.rolling(3).mean()
    r=df.iloc[-1];atr=float(r.ATR14)
    if atr<=0:raise ValueError('변동성이 없어 가격 시나리오를 계산하지 못했습니다.')
    dd=1-df.Close/df.Close.cummax();current=float(dd.iloc[-1])
    # Exclude last bar from historical MDD to let a fresh record exceed 100%.
    historical=float(dd.iloc[:-1].max());reach=current/historical if historical>0 else None
    volume=float(r.Volume/df.Volume.iloc[-21:-1].mean())
    checks={'볼린저 하단~중단':bool(r.BB_Lower<=r.Close<=r.MA20),'200일선 위':bool(r.Close>r.MA200),
            'RSI 40 이하':bool(r.RSI<=40),'표본 MDD 70~100%':bool(reach is not None and .7<=reach<=1),
            '거래량 회복':bool(volume>=1.2 and r.Close>=df.Close.iloc[-2])}
    events=find_divergences(df);bull=any('상승' in e['유형'] for e in events);bear=any('하락' in e['유형'] for e in events)
    rebound=bool(r.Close>r.Open and r.Close>df.Close.iloc[-2])
    points=[('가격 위치',15 if checks['볼린저 하단~중단'] else 0),('장기 추세',20 if checks['200일선 위'] else 0),
            ('RSI 조정',10 if checks['RSI 40 이하'] else 0),('표본 낙폭',10 if checks['표본 MDD 70~100%'] else 0),
            ('거래량 회복',15 if checks['거래량 회복'] else 0),('상승 다이버전스',20 if bull else 0),('반등 확인',10 if rebound else 0),('하락 다이버전스',-20 if bear else 0)]
    score=max(0,sum(p for _,p in points))
    support=float(df.Low.iloc[-21:-1].min());stop=support-.5*atr
    barriers=[float(x) for x in [r.MA20,r.BB_Upper,df.High.iloc[-253:-1].max()]]
    plan=trade_plan('1차 검토 40%',float(r.Close),float(r.Close),stop,barriers,'확정 양봉 반등 + 상승 다이버전스 + 200일선 위 + 손익비 2 이상')
    second=min(float(r.BB_Lower),support+.25*atr)
    plan2=trade_plan('2차 검토 60%',second,second,stop,barriers,'지지 재확인 때만 검토 · 1차보다 낮지 않거나 손절 이하이면 미제공') if stop<second<r.Close else {'valid':False}
    rr=plan.get('rr',0)
    eligible=bool(plan.get('valid') and plan.get('target_kind')=='관측 저항' and rr>=2 and plan.get('risk_pct',100)<=8 and bull and not bear and rebound and checks['200일선 위'] and checks['거래량 회복'])
    status='진입 검토' if eligible else ('하락 경계' if bear else ('반등 관찰' if bull else '조건 대기'))
    return {'df':df,'close':float(r.Close),'asof':str(df.index[-1].date()),'checks':checks,'count':sum(checks.values()),'events':events,'bull':bull,'bear':bear,'score':score,'s_grade':score>=82 and not bear,
            'status':status,'eligible':eligible,'plans':[plan,plan2],'points':points,'rsi':float(r.RSI),'volume_ratio':volume,'dd':current,'mdd':historical,'reach':reach,'support':support,'stop':stop}

def load_profile(region,code,name):
    result=fetch_history(code,days=1200,region=region)
    if result.data.empty and region=='KR':
        try:
            frame=yahoo_history(code+'.KS','KR',days=1200);frame.attrs['region']='KR'
            result=Result(frame,'Yahoo chart / '+code+'.KS')
        except Exception:pass
    if result.data.empty:raise ValueError('일봉 공급원 조회 실패')
    p=analyze(result.data);p.update(code=code,name=name,region=region,source=result.source)
    return p

MODEL_VERSION='fast-1'
class ScanService:
    """Bounded process-wide workers; no Streamlit calls or session data in workers."""
    def __init__(self,root=None,workers=4,loader=None,universe_loader=None):
        self.pool=ThreadPoolExecutor(max_workers=workers,thread_name_prefix='stock-data')
        self.workers=workers;self.lock=threading.RLock();self.values={};self.pending={};self.errors={};self.generations={};self.universes={};self.universe_pending={}
        self.loader=loader or load_profile;self.universe_loader=universe_loader or market_universe
        self.root=Path(root or Path(tempfile.gettempdir())/'ant-stock-cache'/MODEL_VERSION)
        self.root.mkdir(parents=True,exist_ok=True)
        for old in self.root.glob('*.json'):
            try:
                if time.time()-old.stat().st_mtime>8*86400:old.unlink()
            except OSError:pass
    def key(self,region,code):
        return (region,code,str(datetime.now(EXCHANGE_TZ[region]).date()))
    def path(self,key):return self.root/(hashlib.sha256('|'.join(key).encode()).hexdigest()+'.json')
    def read(self,key):
        if key in self.values:return self.values[key]
        try:
            data=json.loads(self.path(key).read_text());p=data['profile']
            p['df']=pd.read_json(StringIO(data['frame']),orient='split');p['df'].attrs['region']=key[0]
            if p['region']!=key[0] or p['code']!=key[1] or (datetime.now(EXCHANGE_TZ[key[0]]).date()-pd.Timestamp(p['asof']).date()).days>7: return None
            self.values[key]=p;return p
        except (OSError,ValueError,KeyError,TypeError):return None
    def save(self,key,p):
        try:
            target=self.path(key);temp=target.with_suffix('.tmp')
            data={'profile':{k:v for k,v in p.items() if k!='df'},'frame':p['df'].to_json(orient='split',date_format='iso')}
            temp.write_text(json.dumps(data,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)));temp.replace(target)
        except (OSError,ValueError,TypeError):pass
    def harvest(self):
        for key,(generation,future) in list(self.pending.items()):
            if not future.done():continue
            del self.pending[key]
            if self.generations.get(key,0)!=generation:continue
            try:
                p=future.result();self.values[key]=p;self.errors.pop(key,None);self.save(key,p)
            except Exception as exc:self.errors[key]=(time.monotonic(),str(exc)[:180])
        # Bound memory and prune obsolete disk files without deleting today's results.
        if len(self.values)>1500:
            for key in list(self.values)[:len(self.values)-1500]:self.values.pop(key,None)
    def invalidate(self,region,records):
        with self.lock:
            for r in records:
                key=self.key(region,r['Code']);self.generations[key]=self.generations.get(key,0)+1
                self.values.pop(key,None);self.errors.pop(key,None)
                try:self.path(key).unlink(missing_ok=True)
                except OSError:pass
    def universe(self,market):
        key=(market,str(datetime.now(KST).date()))
        with self.lock:
            if key in self.universes:return self.universes[key]
            future=self.universe_pending.get(key)
            if future is None:self.universe_pending[key]=self.pool.submit(self.universe_loader,market);return None
            if not future.done():return None
            try:result=future.result()
            except Exception:
                seeds=SEED_KR if market==MARKETS[0] else SEED_US
                result=([{'Code':c,'Name':n} for c,n in seeds],'목록 조회 실패 · 주요 종목 대체 표본')
            self.universes[key]=result;del self.universe_pending[key];return result
    def poll(self,region,records,paused=False):
        unique=list({r['Code']:r for r in records}.values())
        with self.lock:
            self.harvest();profiles=[];errors=[];missing=[]
            for r in unique:
                key=self.key(region,r['Code']);p=self.read(key)
                if p is not None:profiles.append(dict(p,name=r['Name']))
                elif key in self.errors and time.monotonic()-self.errors[key][0]<120:errors.append({'종목':r['Code'],'사유':self.errors[key][1]})
                else:missing.append((key,r))
            blocked=not profiles and len(errors)>=self.workers
            if not paused and not blocked:
                for key,r in missing:
                    if key in self.pending:continue
                    if len(self.pending)>=self.workers:break
                    self.pending[key]=(self.generations.get(key,0),self.pool.submit(self.loader,region,r['Code'],r['Code']))
            done=len(profiles)+len(errors);total=len(unique)
            return {'profiles':profiles,'errors':errors,'cursor':done,'total':total,'progress':min(1.,max(0.,done/max(1,total))),'blocked':blocked,'pending':sum(self.key(region,r['Code']) in self.pending for r in unique),'checked_at':datetime.now(KST).strftime('%H:%M:%S')}
    def close(self):self.pool.shutdown(wait=True,cancel_futures=True)

@st.cache_resource
def scan_service():return ScanService()


def detail_html(p):
    e=html.escape;currency='$' if p['region']=='US' else '₩'
    def price(value):return '—' if value is None else currency+f'{value:,.2f}'
    plan=p['plans'][0];valid=plan.get('valid',False)
    signal=' · '.join(sorted(set(x['유형'] for x in p['events']))) or '최근 확정 다이버전스 없음'
    explanation=('상승 신호와 반등·거래량·손익비 조건이 함께 충족된 구간입니다. 손절 기준을 전제로 분할 접근을 검토합니다.' if p['eligible'] else
                 '하락 다이버전스가 확인되어 신규 진입보다 추세 회복 확인이 우선입니다.' if p['bear'] else
                 '일부 반등 조건은 보이지만 진입 요건이 모두 충족되지는 않았습니다. 가격만 보고 진입하기보다 거래량과 반등 확인을 기다리는 구간입니다.')
    facts=[('검토 진입가',price(plan.get('low'))),('1차 저항 / 목표',price(plan.get('target'))),('손절 참고가',price(plan.get('stop'))),('손익비',f"1 : {plan['rr']:.2f}" if valid else '산정 불가')]
    cards=''.join(f'<div class="detail-metric"><small>{label}</small><b>{value}</b></div>' for label,value in facts)
    r=p['df'].iloc[-1]
    obs=[f"BB {price(float(r.BB_Lower))} ~ {price(float(r.MA20))}",f"SMA200 {price(float(r.MA200))}",f"RSI {p['rsi']:.1f}",f"현재 낙폭 {p['dd']*100:.1f}% / 과거 최대 {p['mdd']*100:.1f}%",f"직전20일 대비 {p['volume_ratio']:.2f}배"]
    checks=''.join(f'<div class="condition {"passed" if ok else "waiting"}"><small>{i+1}. {e(label)}</small><b>{"✓ 충족" if ok else "확인 대기"}</b><span>{e(value)}</span></div>' for i,((label,ok),value) in enumerate(zip(p['checks'].items(),obs)))
    # Explain the actual existing eight factors; do not invent 24-indicator measurements.
    groups=[('추세 · 위치',35,['가격 위치','장기 추세']),('반등 모멘텀',40,['RSI 조정','상승 다이버전스','반등 확인']),('낙폭 · 위험',10,['표본 낙폭']),('거래량',15,['거래량 회복'])]
    pts=dict(p['points']);group_html=''
    for title,maximum,labels in groups:
        points=sum(pts.get(k,0) for k in labels)
        rows=''.join(f'<li>{e(k)} <strong>{pts.get(k,0)}점</strong></li>' for k in labels)
        group_html+=f'<div class="factor"><h4>{title}<span>{points}/{maximum}</span></h4><div class="factor-bar"><i style="width:{points/maximum*100:.0f}%"></i></div><ul>{rows}</ul></div>'
    return f'''<section class="detail-panel"><div class="detail-title"><h3>종목 매매 타이밍 · {e(p['name'])}</h3><span class="badge">{e(p['status'])}</span></div><div class="detail-signal">{e(signal)}</div><p>{explanation}</p><div class="detail-grid">{cards}</div><small>기준 {e(p['asof'])} 확정 종가 · {e(plan.get('target_kind','유효 가격 구조 없음'))}</small></section><h4 class="section-title">◎ 5대 기술 조건 충족 결과　{p['count']} / 5</h4><div class="condition-grid">{checks}</div><section class="detail-panel"><div class="detail-title"><h3>기술적 반등 점수 종합 평가</h3><span class="score-big">{p['score']}<small>/100</small></span></div><p>{explanation}</p><div class="factor-grid">{group_html}</div><p>하락 다이버전스 조정: {pts.get('하락 다이버전스',0)}점 · 총점은 0점 미만으로 내려가지 않습니다.</p></section>'''


def render_detail(p):
    st.markdown(detail_html(p),unsafe_allow_html=True)
    st.caption(p['source']+' · 상승 확률이 아닌 자체 기술 조건식입니다. 기업 흑자·24개 지표·AI 분석으로 표시하지 않습니다.')
    rows=[{'시나리오':x['name'],'검토 가격':x['low'],'손절선':x['stop'],'저항/목표':x['target'],'손익비':round(x['rr'],2),'조건':x['condition']} for x in p['plans'] if x.get('valid')]
    with st.expander('분할 진입 가격과 신호 근거'):
        if rows:st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
        if p['events']:st.dataframe(pd.DataFrame(p['events']),hide_index=True,use_container_width=True)
        st.caption('2R 관리선은 가정이며 관측 저항이 아닙니다. 40%/60%는 참고 비중이고 가격 도달만으로 추가 매수하지 않습니다.')

PRESETS=['전체 종목','↗ 상승 다이버전스','↘ 하락 다이버전스','✓ 5개 조건 충족','♛ S등급','◉ 진입 검토','▥ 거래량 급증']
SORTS=['퀀트 반등 점수 높은순','조건 충족 많은순','거래량 증가순','낙폭 큰순','종목명순']
CSS='''
<style>
:root{--bg:#0b1020;--panel:#11182b;--border:#25304a;--muted:#8493b0;--green:#00d5a3;--cyan:#00d5e8;--red:#ff6183;--yellow:#ffd34d}
html,body,[class*="css"]{font-family:Inter,"Noto Sans KR",Arial,sans-serif}
.stApp{background:var(--bg);color:#eef3ff}
header[data-testid="stHeader"]{background:transparent;height:0}
[data-testid="stToolbar"]{display:none}
.block-container{max-width:1700px;padding:12px 24px 40px}
[data-testid="stVerticalBlock"]{gap:.65rem}
[data-testid="stHorizontalBlock"]{gap:12px}
h1,h2,h3,p{letter-spacing:-.025em}
.topline{background:linear-gradient(100deg,#103d63,#047a82);border:1px solid #0c7a86;color:#baf8ff;font-size:11px;font-weight:650;padding:7px 12px;border-radius:6px;margin-bottom:8px;display:flex;justify-content:space-between}
.brand{display:flex;align-items:center;gap:12px;padding:9px 0 13px}
.logo{display:grid;place-items:center;width:42px;height:42px;border-radius:12px;background:#063b37;border:1px solid #006b58;color:var(--green);font-size:25px}
.brand h1{font-size:23px!important;margin:0;padding:0;font-weight:850;line-height:1.3}
.brand p{font-size:11px;color:var(--muted);margin:3px 0 0}
.badge{display:inline-block;border:1px solid #125d54;background:#07332f;color:#52edc8;border-radius:6px;padding:3px 7px;font-size:10px;font-weight:650;white-space:nowrap}
.toolbar-label{font-size:11px;color:#8b9bb8;padding:2px 0}
.intro{border-top:1px solid #202a42;border-bottom:1px solid #202a42;padding:19px 0 16px;margin:6px 0 4px}
.eyebrow{color:var(--green);font-size:10px;font-weight:800;letter-spacing:1px}
.intro h2{font-size:19px!important;padding:6px 0;margin:0;font-weight:800}
.intro p{font-size:12px;color:#91a4c5;margin:0;line-height:1.7}
.keyrules{display:flex;gap:6px;margin:12px 0 0;flex-wrap:wrap}
.keyrules span{border:1px solid #24314a;border-radius:7px;padding:7px 11px;font-size:11px;background:#101b2d;color:#70dfc7}
[data-testid="stBaseButton-secondary"],[data-testid="stBaseButton-primary"]{border-radius:7px!important;min-height:35px!important;font-size:12px!important;border:1px solid #2b3852!important;background:#192239!important;color:#c5d3ec!important;padding:4px 10px!important}
[data-testid="stBaseButton-primary"]{background:#008c6d!important;border-color:#00c89b!important;color:white!important}
[data-testid="stBaseButton-secondary"]:hover{border-color:#00b994!important;color:#58f7ce!important}
[data-testid="stTextInput"] input,[data-baseweb="select"]>div{background:#090f20!important;color:#c3d3ed!important;border-color:#2c3851!important;font-size:12px!important;min-height:35px!important}
[data-testid="stRadio"] label p{font-size:11px!important}
[data-testid="stRadio"] [role="radiogroup"]{background:#080e1d;border:1px solid #253149;border-radius:9px;padding:5px;gap:5px!important}
[data-testid="stRadio"] [role="radiogroup"]>label{border:1px solid transparent;border-radius:6px;padding:6px 9px;margin:0!important;background:#111a2b}
[data-testid="stRadio"] [role="radiogroup"]>label:has(input:checked){background:#443bfa;border-color:#645cff;color:#fff}
[data-testid="stRadio"] [role="radiogroup"]>label>div:first-child{display:none}
[data-testid="stRadio"] [role="radiogroup"]>label:focus-within{outline:2px solid #8ecbff}
.st-key-preset [role="radiogroup"]>label:has(input:checked){background:#073a35;border-color:#009c7e;color:#55edc2}
.summary{display:flex;gap:8px;flex-wrap:wrap;padding:9px 0;border-bottom:1px solid #202a42}
.summary span{color:#8396b5;font-size:11px;border-right:1px solid #2c3750;padding-right:14px}
.summary b{color:#53e9c4;font-size:13px;margin-left:5px}
.resultline{display:flex;justify-content:space-between;align-items:center;color:#8b9ebd;font-size:11px;margin:10px 0}
.resultline strong{color:#fff}.resultline em{color:#29dbc5;font-style:normal}
.stock{border:1px solid #2c354e;border-radius:11px;background:#10182a;overflow:hidden;min-height:283px;padding:17px 17px 13px;border-top:4px solid var(--cyan)}
.stock.bull{border-top-color:var(--green)}.stock.bear{border-top-color:var(--red)}.stock.gold{border-top-color:var(--yellow)}
.stockhead{display:flex;justify-content:space-between;align-items:flex-start;gap:5px}
.symbol{font-size:20px;font-weight:850;color:#fff;line-height:1.15}.symbol small{font-size:9px;margin-left:5px;vertical-align:middle;background:#1f3658;padding:3px 5px;border-radius:3px;color:#9bcaff}
.name{font-size:11px;color:#7e92b3;margin-top:6px;min-height:16px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:205px}
.price{text-align:right;font-size:21px;font-weight:850;white-space:nowrap;line-height:1.15}
.change{font-size:11px;font-weight:650;margin-top:6px}.up{color:#00dcb0}.down{color:#ff6183}
.signal{display:flex;justify-content:space-between;align-items:center;margin:16px 0 12px;font-size:11px;color:#a4b8d5}
.signal b{color:#eef7ff;font-size:19px}.signal .pill{font-size:10px;color:#47e9c0;background:#093a32;border:1px solid #0c6450;border-radius:5px;padding:4px 7px}
.spark{height:37px;width:100%;margin:0 0 10px}
.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;border-top:1px solid #25304a;padding-top:10px}.stats small{display:block;color:#8192ad;font-size:9px;margin-bottom:4px}.stats b{font-size:12px;color:#dbe8fc}
.checks{display:flex;gap:4px;margin:13px 0 0;flex-wrap:wrap}.checks span{font-size:9px;padding:3px 6px;border:1px solid #303b50;color:#6c7e9c;border-radius:4px}.checks .yes{border-color:#08785f;background:#083a31;color:#48e1b5}
.cardfoot{font-size:9px;color:#7385a4;display:flex;justify-content:space-between;margin-top:10px}
.empty{border:1px dashed #32405a;border-radius:12px;background:#10182a;padding:35px 20px;text-align:center;color:#8498b9}.empty b{display:block;color:#e7efff;font-size:16px;margin-bottom:9px}
[data-testid="stExpander"]{border-color:#253149;border-radius:7px}
[data-testid="stDialog"] [role="dialog"]{max-width:1100px;width:94vw;background:#0d1425;color:#edf5ff}
[data-testid="stCaptionContainer"] p{font-size:10px;color:#7f91ad}
[data-testid="stProgress"]{margin:0!important}

[data-testid="stRadioOption"]{border:1px solid transparent!important;border-radius:6px;padding:7px 10px!important;background:#111a2b!important;color:#a7bddb!important}
[data-testid="stRadioOption"]:has(input:checked){background:#443bfa!important;border-color:#645cff!important;color:white!important}
[data-testid="stRadioOption"]>div>div:first-child{display:none}
[data-testid="stRadioOption"]:focus-within{outline:2px solid #8ecbff}
.st-key-preset [data-testid="stRadioOption"]:has(input:checked){background:#073a35!important;border-color:#009c7e!important;color:#55edc2!important}
[data-testid="stSelectbox"] [role="group"],[data-testid="stSelectbox"] input,[data-testid="stSelectbox"] button,[data-testid="stNumberInput"] input,[data-testid="stNumberInput"] button{background:#10192c!important;color:#bed1ed!important;border-color:#2b3852!important}
[data-testid="stWidgetLabel"],[data-testid="stCheckbox"] label{color:#a7bddb!important}
[data-testid="stTextInput"] input::placeholder{color:#7589a9!important}
[role="listbox"]{background:#152039!important;color:#d7e7ff!important}

@media(max-width:700px){.block-container{padding:9px 12px}.brand h1{font-size:20px!important}.intro h2{font-size:16px!important}.topline span:last-child{display:none}.resultline{display:block}.stock{min-height:270px}.name{max-width:160px}}

.detail-panel{border:1px solid #08785f;background:#10192b;border-radius:12px;padding:18px;margin:8px 0 18px}
.detail-title{display:flex;align-items:center;justify-content:space-between;gap:12px}.detail-title h3{font-size:18px!important;padding:0;margin:0}
.detail-panel p{color:#aabbd6;font-size:12px;line-height:1.8}.detail-panel small{font-size:10px;color:#8296b8}
.detail-signal{color:#00e5b2;background:#102b2a;border:1px solid #285548;border-radius:6px;padding:10px;margin-top:13px;font-weight:700;font-size:12px}
.detail-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin:12px 0}.detail-metric{border:1px solid #2c3852;background:#111b2e;border-radius:7px;padding:12px}.detail-metric small{display:block;margin-bottom:7px}.detail-metric b{font-size:19px;color:#25e2b3}.detail-metric:nth-child(2) b{color:#2ad1f1}.detail-metric:nth-child(3) b{color:#ff6383}.detail-metric:nth-child(4) b{color:#ffd24c}
.section-title{font-size:14px!important}.condition-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin:12px 0 20px}.condition{border:1px solid #344158;background:#111b2e;border-radius:9px;padding:12px;min-width:0}.condition.passed{border-color:#009f7e}.condition small{font-size:10px;color:#97a9c3;display:block}.condition b{display:block;font-size:14px;margin:8px 0;color:#d4e5ff}.condition.passed b{color:#1fe5b1}.condition span{font-size:10px;color:#889dbc;overflow-wrap:anywhere}
.score-big{color:#22dfd5;font-size:29px;font-weight:800;white-space:nowrap}.score-big small{font-size:12px;margin-left:5px}
.factor-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:10px}.factor{border:1px solid #22425b;border-radius:9px;padding:14px;background:#0e1a2b}.factor h4{font-size:13px!important;color:#20debb;padding:0;margin:0}.factor h4 span{float:right;color:#a9dff6}.factor li{font-size:11px;color:#9cb0ce;line-height:2}.factor li strong{float:right;color:#50dfc4}.factor ul{padding-left:14px}.factor-bar{height:4px;background:#243550;margin:12px 0}.factor-bar i{display:block;height:4px;background:#06ccac}
@media(max-width:700px){.detail-grid,.condition-grid{grid-template-columns:repeat(2,1fr)}.factor-grid{grid-template-columns:1fr}.detail-title h3{font-size:15px!important}}
</style>
'''
GUIDE='''### 스크리너 사용 순서
마켓 선택 → 자동 분석 → 조건 필터 → 종목 카드의 상세 분석.

**5대 기술 조건:** 볼린저 하단~중단 / SMA200 위 / Wilder RSI14≤40 / 표본 MDD 도달률70~100% / 거래량 회복(직전20일 평균1.2배 이상, 전일 종가 이상). 재무 조회는 사용하지 않으며 기업 흑자 대신 거래량 회복을 사용합니다.

**다이버전스:** 좌우3봉으로 확인된 가격 피벗에서 RSI 또는 MACD 히스토그램의 방향 차이를 비교합니다. 피벗 간격5~60봉, 확인 후10봉 이내만 표시합니다. 저점/고점이 무너지면 취소하며 히든 상승/하락은200일선 위/아래를 추가로 요구합니다.

**기술 S등급:** 점수82 이상, 하락 다이버전스 없음. **진입 검토:** 상승 신호, 장기 추세, 양봉 반등, 거래량 회복, 관측 저항까지 손익비2 이상, 손절 거리8% 이하를 모두 요구합니다. S등급만으로 진입을 권하지 않습니다.

**MDD:** 최대 최근756거래일 표본이며 역대 전체가 아닙니다. 현재 낙폭/전일까지 표본 최대 낙폭입니다.100%를 넘을 수 있으며 미래 손실의 한도가 아닙니다.

**가격:** 거래소 현지 당일 봉을 제외한 확정 일봉입니다. 갱신은 자료 재조회이며 실시간 체결가가 아닙니다. 공급원별 수정주가/기업행사 처리 차이가 있습니다. 기준일은 카드에 표시합니다.

**분할 시나리오:** 1차40%·2차60%는 참고 비중입니다. 2차 가격에 도달했다는 이유만으로 추가 매수하지 않습니다. 손절=직전20일 저점−0.5ATR, 목표 후보는20일선·BB상단·직전252봉 고점 중 가까운 저항입니다.2R 가정선은 진입 조건 충족 근거에서 제외합니다.

**운영:** 최대4개 백그라운드 작업으로 병렬 조회하고 화면은1초마다 완료 결과만 확인합니다. 초기 조회가 모두 실패하면 추가 요청을 중단합니다. 같은 날 저장된 결과를 재사용하며 서버 재배포·절전으로 저장소가 초기화되면 다시 조회합니다. 일시정지는 신규 요청만 중단하며 진행 중 작업은 완료됩니다. 날짜가 바뀌면 새로 분석합니다. 직접 추가한 목록은 현재 세션에 보관합니다. 목록 조회 실패 시 주요 종목 표본임을 표시합니다. 수익성은 검증되지 않은 자체 기술 조건식입니다.
'''

@st.dialog('전략 지표 · 사용설명서')
def guide_dialog():
    st.markdown(GUIDE)

@st.dialog('종목 정밀 분석')
def detail_dialog(profile):
    render_detail(profile)

def register_symbol(market):
    region='KR' if market==MARKETS[0] else 'US'
    code=st.session_state.get('add_code','').strip().upper()
    name=st.session_state.get('add_name','').strip()
    valid=valid_code(code) if region=='KR' else bool(re.fullmatch(r'[A-Z]{1,6}(?:[.-][A-Z]{1,2})?',code))
    st.session_state['add_error']='' if valid else '종목 코드 형식을 확인해 주십시오.'
    if valid:
        records=st.session_state.setdefault('custom_'+market,[])
        if code not in [r['Code'] for r in records]:records.insert(0,{'Code':code,'Name':name or code})


@st.dialog('새 종목 추가')
def add_dialog(market):
    region='KR' if market==MARKETS[0] else 'US'
    st.caption('현재 마켓의 사용자 목록에 추가합니다. 지수 구성 종목 여부는 보장하지 않습니다.')
    with st.form('add_form'):
        st.text_input('6자리 종목코드' if region=='KR' else '미국 주식 티커',key='add_code')
        st.text_input('표시 이름 (선택)',key='add_name')
        submitted=st.form_submit_button('추가하고 분석',type='primary',use_container_width=True,on_click=register_symbol,args=(market,))
    if submitted:
        if st.session_state.get('add_error'):st.error(st.session_state['add_error'])
        else:st.rerun()


def sparkline(values,color):
    vals=np.asarray(values,dtype=float);lo,hi=vals.min(),vals.max()
    coords=' '.join(f'{i*300/max(1,len(vals)-1):.1f},{33-(v-lo)/max(hi-lo,1e-9)*28:.1f}' for i,v in enumerate(vals))
    return f'<svg class="spark" viewBox="0 0 300 38" preserveAspectRatio="none" aria-label="최근 40거래일 종가 추이"><polyline fill="none" stroke="{color}" stroke-width="1.6" points="{coords}"/></svg>'


def card_html(p,market):
    e=html.escape
    currency='$' if p['region']=='US' else '₩'
    change=(p['close']/p['df'].Close.iloc[-2]-1)*100
    tone='gold' if p['s_grade'] else ('bear' if p['bear'] else ('bull' if p['bull'] else ''))
    color='#ff6183' if p['bear'] else '#00d5b5'
    signals=' · '.join(sorted(set(x['유형'] for x in p['events']))) or '신호 대기'
    checks=''.join(f'<span class="{"yes" if value else ""}">{"✓" if value else "·"} {label}</span>' for label,value in zip(['BB','200MA','RSI','MDD','거래량'],p['checks'].values()))
    reach=f"{p['reach']*100:.0f}%" if p['reach'] is not None else '—'
    label=['KR','S&P','NDX'][MARKETS.index(market)]
    price=f"{p['close']:,.2f}" if p['region']=='US' else f"{p['close']:,.0f}"
    return f'''<article class="stock {tone}"><div class="stockhead"><div><div class="symbol">{e(p['code'])}<small>{label}</small></div><div class="name" title="{e(p['name'])}">{e(p['name'])}</div></div><div class="price">{currency}{price}<div class="change {'up' if change>=0 else 'down'}">{change:+.2f}%</div></div></div><div class="signal"><span class="pill">{e(signals)}</span><span>{'♛ S · ' if p['s_grade'] else ''}<b>{p['score']}</b> / 100</span></div>{sparkline(p['df'].Close.tail(40),color)}<div class="stats"><div><small>표본 MDD 도달률</small><b>{reach}</b></div><div><small>RSI · 14</small><b>{p['rsi']:.1f}</b></div><div><small>거래량 / 20일</small><b>{p['volume_ratio']:.2f}x</b></div></div><div class="checks">{checks}</div><div class="cardfoot"><span>{e(p['status'])} · 조건 {p['count']}/5</span><span>{e(p['asof'])} 종가</span></div></article>'''


def filter_profiles(profiles,preset,query,requirements,sort):
    def match(p):
        flags={'전체 종목':True,'↗ 상승 다이버전스':p['bull'],'↘ 하락 다이버전스':p['bear'],'✓ 5개 조건 충족':p['count']==5,'♛ S등급':p['s_grade'],'◉ 진입 검토':p['eligible'],'▥ 거래량 급증':p['volume_ratio']>=2}
        return flags[preset] and (not query or query.casefold() in (p['code']+' '+p['name']).casefold()) and all(p['checks'].get(k,False) for k in requirements)
    out=[p for p in profiles if match(p)]
    keys={'퀀트 반등 점수 높은순':lambda p:(-p['score'],p['code']),'조건 충족 많은순':lambda p:(-p['count'],-p['score']),'거래량 증가순':lambda p:-p['volume_ratio'],'낙폭 큰순':lambda p:-p['dd'],'종목명순':lambda p:p['name']}
    return sorted(out,key=keys[sort])


@st.fragment(run_every='1s')
def screen_results(market,custom,preset,query,requirements,sort,paused):
    region='KR' if market==MARKETS[0] else 'US';service=scan_service()
    universe=service.universe(market)
    if universe is None:
        st.info('종목 목록을 백그라운드에서 불러오고 있습니다. 화면 조작은 계속 가능합니다.')
        return
    records,source=universe;codes={r['Code'] for r in custom}
    records=custom+[r for r in records if r['Code'] not in codes]
    if custom:source+=f' + 사용자 추가 {len(custom)}개'
    job=service.poll(region,records,paused)
    profiles=job['profiles'];selected=filter_profiles(profiles,preset,query,requirements,sort)
    st.markdown(f'<div class="summary"><span>분석 완료<b>{len(profiles)}</b></span><span>상승 신호<b>{sum(p["bull"] for p in profiles)}</b></span><span>하락 경계<b>{sum(p["bear"] for p in profiles)}</b></span><span>5개 충족<b>{sum(p["count"]==5 for p in profiles)}</b></span><span>S등급<b>{sum(p["s_grade"] for p in profiles)}</b></span><span>진입 검토<b>{sum(p["eligible"] for p in profiles)}</b></span></div>',unsafe_allow_html=True)
    st.progress(job['progress'])
    st.caption(f"{source} · {job['cursor']}/{len(records)} 처리 · 실패 {len(job['errors'])} · 최근 처리 {job['checked_at'] or '대기'} KST")
    if job['blocked']:st.warning('공급원 조회가 4개 이상 실패하여 자동 요청을 중단했습니다. 실패 사유 확인 후 즉시 갱신으로 재시도해 주십시오.')
    st.markdown(f'<div class="resultline"><span>스크리닝 결과: <strong>{len(selected)}개 종목</strong>　|　<em>{"분석 일시정지" if paused else "확정 일봉 자동 분석"}</em></span><span>종목 상세에서 매매 타이밍 · 5대 조건 · 점수 근거를 확인하세요</span></div>',unsafe_allow_html=True)
    if not selected:
        title='현재 조건에 맞는 종목이 없습니다' if profiles else ('분석을 일시정지했습니다' if paused else '시장 데이터를 확인하고 있습니다')
        st.markdown(f'<div class="empty"><b>{title}</b>조건을 변경하시거나 분석 진행 상태와 수집 실패 사유를 확인해 주십시오.</div>',unsafe_allow_html=True)
    pages=max(1,(len(selected)+11)//12);page_key='page_'+market
    if st.session_state.get(page_key,1)>pages:st.session_state[page_key]=1
    # Place pagination below cards visually while retaining a stable, bounded session state.
    page=st.session_state.get(page_key,1);visible=selected[(page-1)*12:page*12]
    for offset in range(0,len(visible),3):
        for col,p in zip(st.columns(3),visible[offset:offset+3]):
            with col:
                st.markdown(card_html(p,market),unsafe_allow_html=True)
                if st.button('매매 타이밍 · 종합 평가 ↗',key='detail_'+market+p['code'],use_container_width=True):detail_dialog(p)
    bottom=st.columns([1,2,1])
    bottom[0].number_input('결과 페이지',min_value=1,max_value=pages,step=1,key=page_key)
    bottom[1].caption(f'페이지 {page}/{pages} · 한 페이지 12종목 · 조건별 점수는 상승 확률이 아닙니다.')
    if selected:
        export=pd.DataFrame([{k:p[k] for k in ['code','name','asof','close','score','count','status']} for p in selected])
        bottom[2].download_button('↓ 결과 CSV',export.to_csv(index=False).encode('utf-8-sig'),'screener.csv','text/csv',use_container_width=True)
    if job['errors']:
        with st.expander(f'데이터 수집 상태 · 실패 {len(job["errors"])}개'):st.dataframe(pd.DataFrame(job['errors']),hide_index=True,use_container_width=True)


def main():
    st.set_page_config(page_title='개미 투자전략실 | 퀀트 스크리너',page_icon='📈',layout='wide',initial_sidebar_state='collapsed')
    st.markdown(CSS,unsafe_allow_html=True)
    st.markdown('<div class="topline"><span>◉ ANT QUANT LAB　 ·　주식 데이터 기반 리서치</span><span>확정 일봉 · 공개 조건식 · 가상화폐 제외</span></div>',unsafe_allow_html=True)
    st.markdown('<div class="brand"><div class="logo">↗</div><div><h1>개미 투자전략실 <span class="badge">QUANT SCREENER</span></h1><p>5대 기술 조건 · RSI/MACD 다이버전스 · 거래량 · 손익비</p></div></div>',unsafe_allow_html=True)
    if st.session_state.get('market') not in MARKETS:st.session_state.market=MARKETS[0]
    market=st.session_state.market
    bar=st.columns([1.25,2.1,1.1,1.1,1.1])
    paused=bar[0].toggle('자동 분석',value=True,key='auto_scan') is False
    bar[1].markdown('<div class="toolbar-label">● 확정 일봉 스캔　<span style="color:#00d5a3">최대 4종목 병렬 · 저장 결과 우선</span><br>종가 기준 · 장중 실시간 가격 아님</div>',unsafe_allow_html=True)
    if bar[2].button('＋ 종목 추가',use_container_width=True):add_dialog(market)
    if bar[3].button('▣ 전략 / 설명서',use_container_width=True):guide_dialog()
    if bar[4].button('⟳ 즉시 갱신',type='primary',use_container_width=True):
        service=scan_service();universe=service.universe(market)
        if universe is not None:service.invalidate('KR' if market==MARKETS[0] else 'US',universe[0]+st.session_state.get('custom_'+market,[]))
        st.rerun()
    st.markdown('<section class="intro"><div class="eyebrow">● GLOBAL STOCKS QUANT SCREENER</div><h2>낙폭과 다이버전스로 찾는 주식 반등 후보</h2><p>국내 대형주 · 미국 주요 지수 구성 종목을 자동으로 살펴봅니다.<br><span style="color:#00d5a3">상승 다이버전스</span>와 <span style="color:#ff6183">하락 경계 신호</span>를 구분하고, 진입 조건과 손익비를 함께 확인합니다.</p><div class="keyrules"><span>1. 볼린저 위치</span><span>2. 200일선</span><span>3. RSI ≤ 40</span><span>4. 표본 MDD</span><span>5. 거래량 회복</span></div></section>',unsafe_allow_html=True)
    controls=st.columns([2.3,1,1])
    market=controls[0].radio('마켓 선택',MARKETS,horizontal=True,key='market',label_visibility='collapsed')
    query=controls[1].text_input('종목 검색',placeholder='티커, 종목명 (NVDA, 삼성전자)',label_visibility='collapsed',key='search')
    sort=controls[2].selectbox('정렬',SORTS,label_visibility='collapsed',key='sort')
    preset=st.radio('조건별 프리셋',PRESETS,horizontal=True,label_visibility='collapsed',key='preset')
    with st.expander('세부 5대 조건 · 원하는 조건을 추가로 선택하세요'):
        labels=['볼린저 하단~중단','200일선 위','RSI 40 이하','표본 MDD 70~100%','거래량 회복']
        requirements=[label for col,label in zip(st.columns(5),labels) if col.checkbox(label,key='require_'+label)]
        st.caption('흑자 조건은 재무 조회 없이 판단할 수 없어 거래량 회복으로 대체했습니다. ')
    screen_results(market,st.session_state.get('custom_'+market,[]),preset,query,requirements,sort,paused)


if __name__=='__main__':
    main()
