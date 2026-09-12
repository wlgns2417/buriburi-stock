# -*- coding: utf-8 -*-
"""퀀트 투자전략실 — 데이터 기반 주식 분석 플랫폼

GitHub에 있는 기존 실행 .py 파일의 내용을 이 파일 전체로 교체하십시오.
기존 파일명과 실행 설정은 유지하실 수 있습니다.
core.py, providers.py, fdr_worker.py, 테마 설정 파일을 별도로 올릴 필요가 없습니다.

실행: python -m streamlit run test.py
필요 패키지(기존 앱과 동일):
    streamlit, finance-datareader, numpy, pandas, plotly, requests, beautifulsoup4
권장 Python: 3.12. 기존 requirements.txt는 설치 목록이므로 유지하십시오.
추가 API 키는 필요하지 않습니다. 미국 뉴스는 영문 헤드라인입니다.
출처 문서:
    https://github.com/FinanceData/FinanceDataReader
    https://www.nasdaqtrader.com/trader.aspx?id=symboldirdefs
미국 탐지기 순위는 사용자가 입력한 최대 30개 종목의 표본 순위입니다.
공통 기술 점수와 한국 수급 점수는 별도이며 합산하지 않습니다. 탐지기와 시장 랭킹은 목적과 배점이 다른 별도 모델입니다.

데이터 수집 실패는 미확인으로 표시하며 임의 가격으로 대체하지 않습니다.
공매도 자동 수집은 미제공이며 선택적 CSV 입력을 사용합니다.
V2: 기존 v5.5 + 전조 레이더·시장 상황판·섹터 표본·수급 추적·뉴스/공시·보유 종목·관찰 신호 성과.
자동 스캔은 활성 화면에서만 동작합니다. 개인 기록은 세션 보관 + JSON 백업입니다.
OpenDART 자동 공시는 DART_API_KEY, SEC 공시는 SEC_USER_AGENT를 Secrets에 설정하십시오.
미국 재무/수급 미제공 항목은 미확인 처리합니다. 외부 공급원 접근은 배포 환경에 따라 실패할 수 있습니다.
"""
from __future__ import annotations



# ===== 1. 계산 엔진 =====
from datetime import datetime
from zoneinfo import ZoneInfo
import math

import numpy as np
import pandas as pd

KST = ZoneInfo("Asia/Seoul")
OHLCV = ["Open", "High", "Low", "Close", "Volume"]
STRATEGIES = {
    "5일·20일 이동평균 추세 추종": "trend_following",
    "볼린저 중심선 회복 스윙": "bollinger_reversal",
    "RSI 42~68 구간 보유": "rsi_momentum",
}


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


def period_return(df, sessions):
    if len(df) <= sessions:
        return None
    return (df.Close.iloc[-1] / df.Close.iloc[-sessions - 1] - 1) * 100


def investor_window(investors, history, sessions, columns):
    """Require the exact latest sessions; never label 3 or 18 rows as 5/20 days."""
    if investors.empty or len(history) < sessions or not set(columns).issubset(investors.columns):
        return None
    inv = investors.copy()
    inv["Date"] = pd.to_datetime(inv["Date"], errors="coerce")
    inv = inv.dropna(subset=["Date"]).drop_duplicates("Date", keep="last").set_index("Date")
    window = inv.reindex(history.index[-sessions:])
    if not window[columns].apply(pd.to_numeric, errors="coerce").notna().all().all():
        return None
    return window


def evaluate_score(df, investors, fund, short=None):
    """100 possible points, no normalization or implicit reward for missing data.

    Every row is positive evidence in [0, weight]. 'None' means unavailable.
    Full grading is withheld unless all 100 possible points are observable.
    """
    latest, prev = df.iloc[-1], df.iloc[-2]
    rows = []

    def add(group, label, weight, points, detail):
        rows.append({"영역": group, "항목": label, "배점": weight,
                     "득점": points, "상태": "미확인" if points is None else "확인",
                     "근거": detail})

    ready = all(number(latest.get(k)) is not None for k in ["MA5", "MA20", "MA60"])
    pts = (12 if latest.MA5 > latest.MA20 > latest.MA60 else 6 if latest.Close > latest.MA20 else 0) if ready else None
    add("추세", "이동평균", 12, pts, "5>20>60: 12점 / 종가>20일선: 6점 / 나머지: 0점")
    bb = number(latest.get("BB_pct"))
    add("추세", "볼린저 위치", 7, (7 if 0.75 <= bb <= 1.05 else 0) if bb is not None else None, "%b 0.75~1.05: 7점")
    macd_ok = all(number(x) is not None for x in [latest.MACD_HIST, prev.MACD_HIST])
    pts = (6 if latest.MACD_HIST > 0 >= prev.MACD_HIST else 3 if latest.MACD_HIST > 0 else 0) if macd_ok else None
    add("추세", "MACD", 6, pts, "히스토그램 상향 돌파: 6점 / 양수 유지: 3점")

    w5 = investor_window(investors, df, 5, ["ForeignNet", "InstitutionNet"])
    if w5 is None:
        pts, detail = None, "일봉과 일치하는 최근 5거래일 수급 필요"
    else:
        foreign, institution = w5.ForeignNet.sum(), w5.InstitutionNet.sum()
        pts = 12 if foreign > 0 and institution > 0 else 6 if foreign > 0 or institution > 0 else 0
        detail = f"5일 순매수 수량: 외국인 {foreign:+,.0f}주 / 기관 {institution:+,.0f}주"
    add("수급", "외국인·기관", 12, pts, detail)
    w10 = investor_window(investors, df, 10, ["ForeignRate"])
    add("수급", "외국인 보유율", 5, (5 if w10.ForeignRate.iloc[-1] > w10.ForeignRate.iloc[0] else 0) if w10 is not None else None,
        "일봉과 일치하는 10거래일 중 첫날 대비 보유율 증가: 5점")
    mfi = number(latest.MFI)
    add("수급", "MFI", 5, (5 if 50 <= mfi <= 75 and df.Volume.tail(14).sum() > 0 else 0) if mfi is not None else None,
        "MFI 50~75 및 거래량 존재: 5점. 특정 투자자의 매집을 증명하지 않음")
    add("수급", "OBV", 3, 3 if latest.OBV > df.OBV.tail(20).mean() else 0, "OBV가 최근 20일 평균 상회: 3점")

    target, roe, per, industry, pbr = [number(fund.get(k)) for k in ["Target", "ROE", "PER", "IndustryPER", "PBR"]]
    upside = (target / latest.Close - 1) * 100 if target is not None and target > 0 else None
    add("가치", "컨센서스 목표가", 10, (10 if upside >= 25 else 6 if upside >= 10 else 0) if upside is not None else None,
        f"종가 대비 괴리율 {upside:+.1f}%" if upside is not None else "목표가 미제공")
    add("가치", "최근 확정 연간 ROE", 8, (8 if roe >= 15 else 4 if roe >= 8 else 0) if roe is not None else None,
        f"{fund.get('ROEPeriod', '')} ROE {roe:.2f}%" if roe is not None else "확정 연간 실적을 확인하지 못함")
    comparable = per is not None and industry is not None and per > 0 and industry > 0
    add("가치", "업종 대비 PER", 4, (4 if per <= industry * 0.7 else 0) if comparable else None,
        f"PER {per:g} / 업종 {industry:g}" if comparable else "양수 PER끼리만 비교; 적자·미제공·음수 업종PER은 비교 제외")
    add("가치", "PBR", 3, (3 if 0 < pbr < 0.9 else 0) if pbr is not None else None,
        f"PBR {pbr:g}배" if pbr is not None else "미제공")

    rsi = number(latest.RSI)
    add("모멘텀", "RSI", 10, (10 if 45 <= rsi <= 65 else 5 if 30 <= rsi < 45 else 0) if rsi is not None else None,
        f"RSI(단순 14일) {rsi:.1f}" if rsi is not None else "준비 기간 부족")
    cutoff = df.index[-1] - pd.Timedelta(weeks=52)
    window = df.loc[df.index >= cutoff]
    year_ready = df.index[0] <= cutoff
    dist = (latest.Close / window.High.max() - 1) * 100
    add("모멘텀", "52주 고가 근접", 10, (10 if dist >= -7 else 0) if year_ready else None,
        f"52주 고가 대비 {dist:+.1f}%" if year_ready else "52주 이력이 부족하여 평가 제외")
    ratio = number((short or {}).get("ShortRatio"))
    short_date = pd.to_datetime((short or {}).get("Date"), errors="coerce")
    short_ok = ratio is not None and 0 <= ratio <= 100 and short_date == df.index[-1]
    add("모멘텀", "공매도 거래량 비중", 5, (5 if ratio < 7 else 0) if short_ok else None,
        f"{ratio:.2f}% (7% 미만: 5점)" if short_ok else "분석 기준일과 일치하는 공매도 데이터 미확인")

    points = sum(row["득점"] for row in rows if row["득점"] is not None)
    possible = sum(row["배점"] for row in rows if row["득점"] is not None)
    missing = 100 - possible
    grade = "일부 항목 미확인 · 종합등급 보류"
    if not missing:
        grade = "조건 충족도 높음" if points >= 80 else "조건 충족도 보통" if points >= 50 else "조건 충족도 낮음"
    return {"points": points, "possible": possible, "coverage": possible,
            "score": points if not missing else None, "upper_bound": points + missing,
            "grade": grade, "logs": pd.DataFrame(rows)}


def price_scenario(df):
    """ATR-based reference scenario. No claimed trend support or executable ticks."""
    price, atr = float(df.Close.iloc[-1]), number(df.ATR14.iloc[-1])
    if df.attrs.get("region") == "US":
        if atr is None or atr <= 0 or price <= 0:
            return None
        d = min(max(atr, price * 0.01), price * 0.15)
        values = [round(v, 2) for v in [price - d * 2.5, price - d, price - d * .5, price + d * 1.5, price + d * 3]]
        stop, e2, e1, t1, t2 = values
        if not (0 < stop < e2 < e1 < price < t1 < t2):
            return None
        return {"1차 참고 진입가": e1, "2차 참고 진입가": e2, "1차 참고 목표가": t1, "2차 참고 목표가": t2, "참고 손절가": stop, "reward_risk": (t1-e1)/(e1-stop)}
    if atr is None or atr <= 0 or price < 10:
        return None
    distance = min(max(atr, price * 0.01), price * 0.15)
    entry1 = math.floor(price - 0.5 * distance)
    entry2 = min(math.floor(price - distance), entry1 - 1)
    stop = math.floor(entry2 - 1.5 * distance)
    target1 = math.ceil(price + 1.5 * distance)
    target2 = max(math.ceil(price + 3 * distance), target1 + 1)
    if not (0 < stop < entry2 < entry1 < price < target1 < target2):
        return None
    return {"1차 참고 진입가": entry1, "2차 참고 진입가": entry2,
            "1차 참고 목표가": target1, "2차 참고 목표가": target2,
            "참고 손절가": stop,
            "reward_risk": (target1 - entry1) / (entry1 - stop)}


def strategy_signals(df, strategy):
    valid = df[["MA5", "MA20", "MA60", "RSI", "BB_pct"]].notna().all(axis=1)
    if strategy == "trend_following":
        return ((df.MA5 > df.MA20) & valid).astype(int)
    if strategy == "rsi_momentum":
        return (df.RSI.between(42, 68) & valid).astype(int)
    if strategy != "bollinger_reversal":
        raise ValueError("지원하지 않는 전략입니다.")
    position, output = 0, []
    for i in range(len(df)):
        if not valid.iloc[i]:
            position = 0
        elif position:
            # Exit is evaluated before entry. The entry region excludes >=1.05.
            if df.BB_pct.iloc[i] >= 1.05 or df.Close.iloc[i] < df.MA20.iloc[i]:
                position = 0
        elif df.Close.iloc[i] > df.MA20.iloc[i] and 0.4 <= df.BB_pct.iloc[i] < 1.05:
            position = 1
        output.append(position)
    return pd.Series(output, index=df.index, dtype=int)


def _simulate(df, desired, capital, fee, slippage, sell_tax):
    cash, units, entry = float(capital), 0.0, None
    equity, positions, trades = [], [], []
    for i, (day, bar) in enumerate(df.iterrows()):
        target = int(desired.iloc[i])
        tradable = bar.Volume > 0 and bar.Open > 0
        if units == 0 and target == 1 and tradable:
            fill = bar.Open * (1 + slippage)
            allocated = cash
            units = cash / (fill * (1 + fee))
            cash = 0.0
            entry = {"EntryDate": day, "EntryPrice": fill, "Capital": allocated, "Units": units}
        elif units > 0 and target == 0 and tradable:
            fill = bar.Open * (1 - slippage)
            cash = units * fill * (1 - fee - sell_tax)
            trades.append({**entry, "ExitDate": day, "ExitPrice": fill,
                           "PnL": cash - entry["Capital"], "ReturnPct": (cash / entry["Capital"] - 1) * 100})
            units, entry = 0.0, None
        equity.append(cash + units * bar.Close)
        positions.append(int(units > 0))
    return pd.Series(equity, index=df.index), pd.Series(positions, index=df.index), pd.DataFrame(trades), entry


def run_backtest(df, strategy="trend_following", start=None, capital=10_000_000,
                 fee_bps=1.5, slippage_bps=5.0, sell_tax_bps=0.0):
    """Signal at t close -> fill at t+1 open. Closed trades and equity share fills.

    Fractional units, all-in/all-out, no leverage, cash interest or dividends.
    Open final position is marked at the last close; no artificial final exit.
    """
    if capital <= 0 or any(not math.isfinite(v) or v < 0 or v >= 1000 for v in [fee_bps, slippage_bps, sell_tax_bps]):
        raise ValueError("투자금과 비용 설정을 확인해 주십시오.")
    signal = strategy_signals(df, strategy)
    # Shared warm-up window for all strategies and the benchmark.
    eligible = df[["MA60", "RSI", "BB_pct", "MACD_SIGNAL"]].notna().all(axis=1).shift(1, fill_value=False)
    if start is not None:
        eligible &= df.index >= pd.Timestamp(start)
    eligible_indices = df.index[eligible]
    if not len(eligible_indices):
        raise ValueError("준비 기간 이후의 백테스트 데이터가 부족합니다.")
    first = eligible_indices[0]
    bt = df.loc[df.index >= first].copy()
    if len(bt) < 2:
        raise ValueError("백테스트에는 준비 기간 이후 최소 2거래일이 필요합니다.")
    desired = signal.shift(1, fill_value=0).reindex(bt.index)
    fee, slip, tax = [v / 10_000 for v in [fee_bps, slippage_bps, sell_tax_bps]]
    equity, position, trades, open_trade = _simulate(bt, desired, capital, fee, slip, tax)
    hold, _, _, _ = _simulate(bt, pd.Series(1, index=bt.index), capital, fee, slip, tax)
    bt["Signal"] = signal.reindex(bt.index)
    bt["DesiredPosition"], bt["Position"] = desired, position
    bt["Equity"], bt["BenchmarkEquity"] = equity, hold
    bt["Cum_Strategy"], bt["Cum_Market"] = equity / capital, hold / capital
    # Initial cash belongs to the peak series (first entry fee can cause MDD).
    peaks = np.maximum.accumulate(np.r_[capital, equity.to_numpy()])[1:]
    mdd = float((equity.to_numpy() / peaks - 1).min() * 100)
    total, benchmark = (equity.iloc[-1] / capital - 1) * 100, (hold.iloc[-1] / capital - 1) * 100
    win_rate = float((trades.PnL > 0).mean() * 100) if not trades.empty else None
    return {"df": bt, "trades": trades, "open_trade": open_trade,
            "total_return": total, "buy_hold_return": benchmark,
            "excess_pp": total - benchmark, "mdd": mdd,
            "win_rate": win_rate, "trade_count": len(trades)}


def market_rankings(stocks):
    if stocks.empty:
        return stocks.copy()
    df = stocks.copy()
    # Closing price * volume is an estimate, not turnover or investor inflow.
    required = ["Close", "Chg", "Volume", "Marcap"]
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=required)
    df = df[(df.Close > 0) & (df.Volume > 0) & (df.Marcap > 0)].copy()
    df["AmountEstimate"] = df.Close * df.Volume
    if df.empty:
        return df
    # Bounded continuous momentum. Crossing +20% no longer drops ~36 points.
    df["ScreenScore"] = (
        50 * ((df.Chg + 10) / 30).clip(0, 1)
        + 25 * df.Marcap.rank(pct=True)
        + 25 * df.AmountEstimate.rank(pct=True)
    ).round(1)
    df["Momentum"] = df.Chg.rank(pct=True) * 60 + df.AmountEstimate.rank(pct=True) * 40
    return df.sort_values(["ScreenScore", "AmountEstimate"], ascending=False).reset_index(drop=True)

# ===== 2. 데이터 수집 (일봉 수집 코드도 이 파일에 포함) =====

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path
import json
import re
import subprocess
import sys
import threading
from urllib.parse import parse_qs, urljoin, urlparse
import xml.etree.ElementTree as ET

from bs4 import BeautifulSoup
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


BASE = "https://finance.naver.com"
MARKET_COLS = ["Code", "Name", "Market", "Close", "Chg", "Volume", "Marcap"]
INV_COLS = ["Date", "Close", "InstitutionNet", "ForeignNet", "ForeignRate", "InstitutionAmountEstimate", "ForeignAmountEstimate"]
_local = threading.local()


@dataclass
class Result:
    data: object
    source: str
    status: str = "ok"
    notes: list[str] = field(default_factory=list)
    fetched_at: str = field(default_factory=lambda: datetime.now(KST).isoformat(timespec="seconds"))


def compact(value):
    return re.sub(r"\s+", "", str(value))


def numeric(value):
    cleaned = compact(value).replace(",", "").replace("%", "").replace("−", "-")
    return number(cleaned)


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


def safe_url(value, base=BASE):
    resolved = urljoin(base, value)
    parsed = urlparse(resolved)
    return resolved if parsed.scheme in {"https", "http"} and parsed.netloc and not parsed.username else None


def failure(source, empty, error):
    return Result(empty, source, "error", [f"데이터를 확인하지 못했습니다 ({type(error).__name__})."])


def parse_market(body, market):
    soup = BeautifulSoup(body, "html.parser")
    table = soup.select_one("table.type_2")
    if table is None:
        raise ValueError("시가총액 표를 찾지 못했습니다.")
    heads = [compact(x.get_text()) for x in table.select("th")]
    required = {"현재가": "Close", "등락률": "Chg", "거래량": "Volume", "시가총액": "Marcap"}
    if not set(required).issubset(heads):
        raise ValueError("시가총액 표의 필수 열 구성이 달라졌습니다.")
    positions = {label: heads.index(label) for label in required}
    rows = []
    for tr in table.select("tr"):
        cells = tr.find_all("td", recursive=False)
        link = tr.select_one("a[href*='code=']")
        if not link or len(cells) != len(heads):
            continue
        code = parse_qs(urlparse(link.get("href", "")).query).get("code", [""])[0]
        if not valid_code(code):
            continue
        row = {"Code": code, "Name": link.get_text(strip=True), "Market": market}
        for label, field_name in required.items():
            row[field_name] = numeric(cells[positions[label]].get_text())
        if any(row[key] is None for key in required.values()):
            continue
        row["Marcap"] *= 100_000_000
        if row["Close"] > 0 and row["Volume"] >= 0 and row["Marcap"] > 0:
            rows.append(row)
    if not rows:
        raise ValueError("검증된 종목 행이 없습니다.")
    return pd.DataFrame(rows, columns=MARKET_COLS)


def _naver_market(pages_per_market=2):
    pages_per_market = max(1, min(int(pages_per_market), 5))
    jobs = [(sosok, page) for sosok in (0, 1) for page in range(1, pages_per_market + 1)]

    def get(job):
        sosok, page = job
        market = "KOSPI" if sosok == 0 else "KOSDAQ"
        try:
            body = request_bytes(BASE + "/sise/sise_market_sum.naver", {"sosok": sosok, "page": page})
            return parse_market(body, market), None
        except Exception as exc:
            return None, f"{market} {page}페이지: {type(exc).__name__}"

    with ThreadPoolExecutor(max_workers=4) as pool:
        responses = list(pool.map(get, jobs))
    frames, errors = [x for x, _ in responses if x is not None], [err for _, err in responses if err]
    df = pd.concat(frames, ignore_index=True).drop_duplicates("Code") if frames else pd.DataFrame(columns=MARKET_COLS)
    notes = [f"시장별 시가총액 상위 {pages_per_market}페이지 중 {len(frames)}/{len(jobs)}페이지 수집; 실제 {len(df)}종목 표본.",
             "페이지별 조회 시점이 다를 수 있으며 기준 체결 시각은 미제공입니다."] + errors
    return Result(df, BASE + "/sise/sise_market_sum.naver", "ok" if not errors else "partial" if frames else "error", notes)


def _original_local_search(query, stocks):
    query = query.strip()
    if not query or stocks.empty:
        return pd.DataFrame(columns=["Code", "Name"])
    matches = (stocks.Code.astype(str).str.startswith(query) if query.isdigit()
               else stocks.Name.astype(str).str.contains(query, case=False, regex=False, na=False))
    return stocks.loc[matches, ["Code", "Name"]].drop_duplicates("Code").head(10)


def parse_autocomplete(payload):
    rows = []

    def scalar(value):
        while isinstance(value, list) and value:
            value = value[0]
        return value if isinstance(value, str) else ""

    def walk(value):
        if isinstance(value, dict):
            code = value.get("code") or value.get("itemCode") or value.get("symbolCode")
            name = value.get("name") or value.get("itemName") or value.get("stockName")
            if valid_code(code) and isinstance(name, str):
                rows.append({"Code": str(code), "Name": name})
            else:
                for child in value.values():
                    if isinstance(child, (dict, list)):
                        walk(child)
        elif isinstance(value, list):
            if len(value) >= 2 and valid_code(scalar(value[0])) and scalar(value[1]) and not valid_code(scalar(value[1])):
                rows.append({"Code": scalar(value[0]), "Name": scalar(value[1])})
            else:
                for child in value:
                    if isinstance(child, (dict, list)):
                        walk(child)
    walk(payload)
    return pd.DataFrame(rows, columns=["Code", "Name"]).drop_duplicates("Code").head(10)


def search_remote(query):
    source = "https://ac.finance.naver.com/ac"
    try:
        payload = json.loads(request_bytes(source, {"q": query[:80], "target": "stock"}))
        return Result(parse_autocomplete(payload), source)
    except Exception as exc:
        return failure(source, pd.DataFrame(columns=["Code", "Name"]), exc)




def empty_fund():
    return {
        "PER": None, "PBR": None, "DividendYield": None, "IndustryPER": None,
        "Target": None, "ROE": None, "ROEPeriod": None, "Summary": None,
        "ForwardPeriod": None, "ForwardOperatingProfitGrowth": None,
        "ForwardOperatingProfitTurnaround": False,
        "ForwardEPSGrowth": None, "ForwardEPSTurnaround": False,
        "ForwardROE": None,
    }


def parse_main(body, code):
    soup = BeautifulSoup(body, "html.parser")
    # Validate identity, not just the existence of some price elsewhere on page.
    company = soup.select_one(".wrap_company")
    accessible = next((tag for tag in soup.select("div.blind")
                       if "종목코드" in tag.get_text() and code in tag.get_text()), None)
    identity = company.get_text(" ", strip=True) if company else (accessible.get_text(" ", strip=True) if accessible else "")
    if code not in identity:
        raise ValueError("응답의 종목코드를 검증하지 못했습니다.")
    name_tag = company.select_one("h2 a") if company else None
    name = name_tag.get_text(strip=True) if name_tag else code
    quote = {"Code": code, "Name": name, "Price": None, "Previous": None, "AsOf": None, "Market": "KRX"}
    # Prefer the explicitly labelled accessibility block; do not mix KRX/NXT panels.
    if accessible:
        text = accessible.get_text(" ", strip=True)
        price = re.search(r"현재가\s*([\d,]+)", text)
        previous = re.search(r"전일가\s*([\d,]+)", text)
        quote["Price"] = numeric(price.group(1)) if price else None
        quote["Previous"] = numeric(previous.group(1)) if previous else None
        timestamp = re.search(r"(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일\s*(\d{1,2})시\s*(\d{1,2})분", text)
        if timestamp:
            quote["AsOf"] = datetime(*map(int, timestamp.groups()), tzinfo=KST).isoformat(timespec="minutes")
    fund = empty_fund()
    for field_name, selector in [("PER", "#_per"), ("PBR", "#_pbr"), ("DividendYield", "#_dvr"), ("Target", "#_target_money")]:
        tag = soup.select_one(selector)
        fund[field_name] = numeric(tag.get_text()) if tag else None
    # _cper can be forecast PER; industry PER must be found by its label.
    for th in soup.select("th"):
        label = compact(th.get_text())
        td = th.find_next_sibling("td")
        if not td:
            continue
        if label.startswith("동일업종PER"):
            em = td.select_one("em")
            fund["IndustryPER"] = numeric(em.get_text() if em else td.get_text().replace("배", ""))
        if "목표주가" in label and fund["Target"] is None:
            ems = td.select("em")
            if ems:
                fund["Target"] = numeric(ems[-1].get_text())
    summary = soup.select(".summary_info p")
    if summary:
        fund["Summary"] = "\n\n".join(x.get_text(" ", strip=True) for x in summary)
    table = soup.select_one(".cop_analysis table")
    if table:
        # Parse annual actual/estimate columns. Forward fields remain None when consensus is not published.
        annual = next((th for th in table.select("thead th") if "최근 연간 실적" in th.get_text(" ", strip=True)), None)
        count = int(annual.get("colspan", "0")) if annual else 0
        periods = [compact(th.get_text()) for th in table.select("thead th") if re.search(r"\d{4}\.\d{2}", th.get_text())]
        if count > 0 and periods:
            annual_periods = periods[:count]
            annual_rows = {}
            for tr in table.select("tbody tr"):
                th = tr.select_one("th")
                cells = tr.find_all("td", recursive=False)
                if not th or len(cells) < count:
                    continue
                annual_rows[compact(th.get_text())] = [numeric(cells[i].get_text()) for i in range(count)]

            def row_values(prefix, exclude=None):
                for label, values in annual_rows.items():
                    if label.startswith(prefix) and (exclude is None or exclude not in label):
                        return values
                return None

            roe_values = row_values("ROE")
            if roe_values:
                for i in reversed(range(min(count, len(annual_periods)))):
                    value = roe_values[i]
                    if "(E)" not in annual_periods[i] and value is not None:
                        fund["ROE"], fund["ROEPeriod"] = value, annual_periods[i]
                        break

            forecast_idx = next((i for i, period in enumerate(annual_periods) if "(E)" in period), None)
            if forecast_idx is not None:
                fund["ForwardPeriod"] = annual_periods[forecast_idx]
                actual_idx = next((i for i in range(forecast_idx - 1, -1, -1)
                                   if "(E)" not in annual_periods[i]), None)
                op_values = row_values("영업이익", exclude="률")
                eps_values = row_values("EPS")
                if op_values and actual_idx is not None:
                    actual, estimate = op_values[actual_idx], op_values[forecast_idx]
                    if estimate is not None:
                        if actual is not None and actual > 0:
                            fund["ForwardOperatingProfitGrowth"] = (estimate / actual - 1) * 100
                        elif actual is not None and actual <= 0 < estimate:
                            fund["ForwardOperatingProfitTurnaround"] = True
                if eps_values and actual_idx is not None:
                    actual, estimate = eps_values[actual_idx], eps_values[forecast_idx]
                    if estimate is not None:
                        if actual is not None and actual > 0:
                            fund["ForwardEPSGrowth"] = (estimate / actual - 1) * 100
                        elif actual is not None and actual <= 0 < estimate:
                            fund["ForwardEPSTurnaround"] = True
                if roe_values and forecast_idx < len(roe_values):
                    fund["ForwardROE"] = roe_values[forecast_idx]
    return {"quote": quote, "fund": fund}


def fetch_main(code):
    _require_code(code)
    source = BASE + "/item/main.naver?code=" + code
    try:
        data = parse_main(request_bytes(source), code)
        missing = [key for key in ["PER", "PBR", "IndustryPER", "Target", "ROE"] if data["fund"][key] is None]
        notes = ["조회 화면의 KRX 가격 스냅샷이며 일봉 및 백테스트에 합치지 않습니다."]
        if missing:
            notes.append("재무 미확인 항목: " + ", ".join(missing))
        if data["quote"]["Price"] is None or data["quote"]["AsOf"] is None:
            notes.append("현재가 또는 그 기준 시각을 확인하지 못했습니다.")
        return Result(data, source, "partial" if missing or data["quote"]["Price"] is None else "ok", notes)
    except Exception as exc:
        return failure(source, {"quote": {}, "fund": empty_fund()}, exc)


def parse_investors(body):
    soup = BeautifulSoup(body, "html.parser")
    rows = []
    for table in soup.select("table.type2"):
        labels = compact(" ".join(th.get_text(" ", strip=True) for th in table.select("th")))
        if not all(key in labels for key in ["날짜", "종가", "기관", "외국인", "보유율"]):
            continue
        # Two-level header: 5 common fields + institution net + foreign net/held/rate.
        if not all(key in labels for key in ["전일비", "등락률", "거래량", "보유주수"]):
            continue
        for tr in table.select("tr"):
            cells = tr.find_all("td", recursive=False)
            if len(cells) != 9:
                continue
            values = [x.get_text(" ", strip=True) for x in cells]
            if not re.fullmatch(r"\d{4}\.\d{2}\.\d{2}", values[0]):
                continue
            close, institution, foreign, rate = [numeric(values[i]) for i in [1, 5, 6, 8]]
            if close is None or close <= 0:
                continue
            if rate is not None and not 0 <= rate <= 100:
                rate = None
            rows.append({"Date": pd.Timestamp(values[0].replace(".", "-")), "Close": close,
                         "InstitutionNet": institution, "ForeignNet": foreign, "ForeignRate": rate,
                         "InstitutionAmountEstimate": institution * close / 1e8 if institution is not None else None,
                         "ForeignAmountEstimate": foreign * close / 1e8 if foreign is not None else None})
    if not rows:
        raise ValueError("투자자별 수급 표의 구조 또는 데이터를 검증하지 못했습니다.")
    return pd.DataFrame(rows, columns=INV_COLS).drop_duplicates("Date").sort_values("Date")


def fetch_investors(code):
    _require_code(code)
    source = BASE + "/item/frgn.naver?code=" + code
    frames, errors = [], []
    for page in (1, 2):
        try:
            frames.append(parse_investors(request_bytes(BASE + "/item/frgn.naver", {"code": code, "page": page})))
        except Exception as exc:
            errors.append(f"수급 {page}페이지: {type(exc).__name__}")
    df = pd.concat(frames, ignore_index=True).drop_duplicates("Date").sort_values("Date") if frames else pd.DataFrame(columns=INV_COLS)
    return Result(df, source, "ok" if not errors else "partial" if frames else "error",
                  ["금액은 순매수 수량×해당일 종가의 추정값입니다. 실제 순매수 거래대금과 다릅니다."] + errors)


def parse_reports(body, code):
    soup = BeautifulSoup(body, "html.parser")
    table = soup.select_one("table.type_1")
    if not table or not all(label in table.get_text() for label in ["종목명", "제목", "증권사", "작성일"]):
        raise ValueError("리포트 표를 확인하지 못했습니다.")
    rows = []
    for tr in table.select("tr"):
        cells = tr.find_all("td", recursive=False)
        if len(cells) != 6:
            continue
        stock, title = cells[0].select_one("a"), cells[1].select_one("a")
        if not stock or not title:
            continue
        row_code = parse_qs(urlparse(stock.get("href", "")).query).get("code", [""])[0]
        if row_code != code:
            continue
        link = safe_url(title.get("href", ""), BASE + "/research/")
        if link:
            rows.append({"title": title.get_text(strip=True), "broker": cells[2].get_text(strip=True),
                         "date": cells[4].get_text(strip=True), "link": link})
    return rows[:5]


def fetch_reports(code):
    _require_code(code)
    source = BASE + "/research/company_list.naver"
    try:
        body = request_bytes(source, {"searchType": "itemCode", "itemCode": code})
        return Result(parse_reports(body, code), source)
    except Exception as exc:
        return failure(source, [], exc)


def parse_news(body):
    root = ET.fromstring(body)
    rows = []
    for item in root.findall("./channel/item")[:5]:
        link = safe_url(item.findtext("link", ""), "https://news.google.com")
        if link:
            rows.append({"title": item.findtext("title", "제목 미제공"), "link": link,
                         "date": item.findtext("pubDate", ""), "publisher": item.findtext("source", "")})
    return rows


def fetch_news(name):
    source = "https://news.google.com/rss/search"
    try:
        body = request_bytes(source, {"q": f'"{name[:80]}" 주식', "hl": "ko", "gl": "KR", "ceid": "KR:ko"})
        return Result(parse_news(body), source)
    except Exception as exc:
        return failure(source, [], exc)


def read_short_csv(content, code, asof):
    """Optional explicit data input; no unsupported short-selling endpoint used."""
    frame = pd.read_csv(StringIO(content.decode("utf-8-sig")), dtype={"Code": str})
    required = {"Code", "Date", "ShortRatio"}
    if not required.issubset(frame.columns):
        raise ValueError("공매도 CSV에는 Code, Date, ShortRatio 열이 필요합니다.")
    frame["Code"] = frame.Code.str.strip().str.zfill(6)
    frame["Date"] = pd.to_datetime(frame.Date, errors="coerce")
    matches = frame[(frame.Code == code) & (frame.Date == pd.Timestamp(asof))]
    if len(matches) != 1:
        raise ValueError("종목코드와 분석 기준일이 일치하는 행이 정확히 1개 필요합니다.")
    row = matches.iloc[0].to_dict()
    ratio = numeric(row["ShortRatio"])
    if ratio is None or not 0 <= ratio <= 100:
        raise ValueError("ShortRatio는 0~100 범위의 거래량 비중(%)이어야 합니다.")
    row["ShortRatio"] = ratio
    return row


def fetch_bundle(code):
    jobs = {"history": lambda: fetch_history(code), "main": lambda: fetch_main(code),
            "investors": lambda: fetch_investors(code), "reports": lambda: fetch_reports(code)}
    # Four independent bounded data jobs; UI and Streamlit state stay on main thread.
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {key: pool.submit(fn) for key, fn in jobs.items()}
        return {key: future.result() for key, future in futures.items()}

# ===== 3. Streamlit 화면 =====
from datetime import datetime
from html import escape

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st



@st.cache_data(ttl=300, max_entries=4, show_spinner=False)
def cached_market(pages=2):
    return fetch_market(pages)


@st.cache_data(ttl=300, max_entries=64, show_spinner=False)
def cached_bundle(code):
    return fetch_bundle(code)


@st.cache_data(ttl=300, max_entries=128, show_spinner=False)
def cached_search(query):
    return search_remote(query)


@st.cache_data(ttl=900, max_entries=64, show_spinner=False)
def cached_news(name):
    return fetch_news(name)


def fmt(value, suffix="", digits=0, signed=False):
    value = number(value)
    if value is None:
        return "미확인"
    return format(value, f"{ '+' if signed else ''},.{digits}f") + suffix


def select_stock(code, name):
    st.session_state.selected_code = code
    st.session_state.selected_name = name
    st.session_state.query = name
    st.session_state.candidates = []
    st.session_state.search_message = ""
    st.session_state.needs_search = False


def request_search():
    st.session_state.needs_search = True


def refresh_selected():
    code, name = st.session_state.selected_code, st.session_state.selected_name
    if code:
        cached_bundle.clear(code)
    if name:
        cached_news.clear(name)


def resolve_pending(stocks):
    if not st.session_state.needs_search:
        return
    st.session_state.needs_search = False
    query = st.session_state.query.strip()
    st.session_state.candidates, st.session_state.search_message = [], ""
    if not query:
        return
    local = local_search(query, stocks)
    if valid_code(query):
        match = local.loc[local.Code == query]
        select_stock(query, match.iloc[0].Name if not match.empty else query)
        return
    exact = local[local.Name.str.casefold() == query.casefold()]
    if len(exact) == 1:
        select_stock(exact.iloc[0].Code, exact.iloc[0].Name)
        return
    if len(query) < 2:
        st.session_state.search_message = "종목명은 두 글자 이상, 종목코드는 6자리로 입력해 주십시오."
        return
    remote = cached_search(query)
    matches = pd.concat([local, remote.data], ignore_index=True).drop_duplicates("Code").head(10)
    exact = matches[matches.Name.str.casefold() == query.casefold()]
    if len(exact) == 1:
        select_stock(exact.iloc[0].Code, exact.iloc[0].Name)
    else:
        st.session_state.candidates = matches.to_dict("records")
        if matches.empty:
            st.session_state.search_message = "검색 결과를 확인하지 못했습니다. 6자리 종목코드로 조회해 주십시오."


def render_rankings(result):
    st.markdown("#### 📡 오늘의 시장 흐름")
    st.caption(result.notes[0])
    st.button("시장 데이터 새로고침", key="refresh_market", on_click=refresh_market_sources, use_container_width=True)
    if result.status != "ok":
        st.warning("일부 또는 전체 페이지를 수집하지 못했습니다. 확보된 종목만 표시합니다.")

    ranked = market_rankings(result.data)
    if ranked.empty:
        render_kr_daily_sample()
        return

    st.info(
        "오른쪽 랭킹은 종목의 정밀 분석 점수가 아닙니다. "
        "'시장 주목도'는 당일 등락률 50 + 표본 내 시가총액 25 + 추정 거래대금 25로 계산한 "
        "단기 시장 관심도 지표입니다."
    )

    attention_tab, turnover_tab, momentum_tab = st.tabs([
        "🔥 시장 주목도", "💰 거래대금 주도", "🚀 당일 모멘텀"
    ])

    with attention_tab:
        st.caption("가격 탄력·시가총액·거래 활발도를 함께 반영한 단기 시장 관심도 순위입니다.")
        frame = ranked.sort_values(["ScreenScore", "AmountEstimate"], ascending=False).head(10)
        for i, row in enumerate(frame.itertuples(), start=1):
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
            st.button(
                f"{medal} {row.Name} · 시장 주목도 {row.ScreenScore:.1f}",
                key=f"attention_{row.Code}",
                use_container_width=True,
                on_click=select_stock,
                args=(row.Code, row.Name),
            )
            st.caption(
                f"{row.Close:,.0f}원 · 당일 {row.Chg:+.2f}% · "
                f"추정 거래대금 {row.AmountEstimate / 1e8:,.0f}억원"
            )

    with turnover_tab:
        st.caption("가격×거래량 기준 추정 거래대금이 큰 종목입니다. 실제 체결 거래대금과는 차이가 있을 수 있습니다.")
        frame = ranked.sort_values("AmountEstimate", ascending=False).head(10)
        for i, row in enumerate(frame.itertuples(), start=1):
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
            amount_eok = row.AmountEstimate / 1e8
            st.button(
                f"{medal} {row.Name} · {amount_eok:,.0f}억원",
                key=f"turnover_{row.Code}",
                use_container_width=True,
                on_click=select_stock,
                args=(row.Code, row.Name),
            )
            st.caption(
                f"{row.Close:,.0f}원 · 당일 {row.Chg:+.2f}% · "
                f"시장 주목도 {row.ScreenScore:.1f}"
            )

    with momentum_tab:
        st.caption("당일 등락률이 강한 종목을 우선 보여줍니다. 급등 자체가 매수 적합성을 의미하지는 않습니다.")
        frame = ranked.sort_values(["Chg", "AmountEstimate"], ascending=False).head(10)
        for i, row in enumerate(frame.itertuples(), start=1):
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
            st.button(
                f"{medal} {row.Name} · 당일 {row.Chg:+.2f}%",
                key=f"day_momentum_{row.Code}",
                use_container_width=True,
                on_click=select_stock,
                args=(row.Code, row.Name),
            )
            st.caption(
                f"{row.Close:,.0f}원 · 추정 거래대금 {row.AmountEstimate / 1e8:,.0f}억원 · "
                f"시장 주목도 {row.ScreenScore:.1f}"
            )

    st.caption(
        "※ 위 순위는 현재 수집된 시장 표본 기준입니다. '시장 주목도'와 왼쪽의 종합 분석 점수는 목적과 계산식이 서로 다릅니다."
    )


def render_chart(df):
    view = df.loc[df.index >= df.index[-1] - pd.Timedelta(days=365)]
    fig = make_subplots(rows=2, cols=2, shared_xaxes=True,
                        row_heights=[0.75, 0.25], column_widths=[0.84, 0.16],
                        horizontal_spacing=0.02, vertical_spacing=0.05,
                        specs=[[{}, {}], [{}, None]])
    fig.add_trace(go.Candlestick(x=view.index, open=view.Open, high=view.High, low=view.Low,
                                close=view.Close, name="일봉", increasing_line_color="#ef4444",
                                decreasing_line_color="#38bdf8"), row=1, col=1)
    for col, color in [("MA5", "#f59e0b"), ("MA20", "#38bdf8"), ("MA60", "#10b981")]:
        fig.add_trace(go.Scatter(x=view.index, y=view[col], name=col,
                                line={"color": color, "width": 1.3}), row=1, col=1)
    low, high = view.Low.min(), view.High.max()
    if high <= low:
        low, high = low * 0.99, high * 1.01
    bins = np.linspace(low, high, 17)
    counts, _ = np.histogram(view.Close, bins=bins, weights=view.Volume)
    fig.add_trace(go.Bar(y=(bins[:-1] + bins[1:]) / 2, x=counts, orientation="h",
                         name="종가 구간별 거래량", marker_color="rgba(56,189,248,0.4)"), row=1, col=2)
    for col, color in [("MACD", "#f43f5e"), ("MACD_SIGNAL", "#fbbf24")]:
        fig.add_trace(go.Scatter(x=view.index, y=view[col], name=col,
                                line={"color": color, "width": 1.3}), row=2, col=1)
    fig.update_yaxes(matches="y", showticklabels=False, row=1, col=2)
    fig.update_xaxes(showticklabels=False, row=1, col=2)
    for row in (1, 2):
        fig.update_xaxes(rangebreaks=[{"bounds": ["sat", "mon"]}], row=row, col=1)
    fig.update_layout(template="plotly_dark", height=530, xaxis_rangeslider_visible=False,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin={"l": 5, "r": 5, "t": 15, "b": 5}, legend={"orientation": "h"})
    st.plotly_chart(fig, use_container_width=True)
    st.caption("우측 분포는 일별 거래량 전체를 그날 종가 구간에 배정한 근사치입니다. 실제 체결가격별 매물대가 아닙니다.")


def render_supply(df, investors):
    flow = assess_investor_flow(investors, df)
    st.write(flow['text'])
    if pd.notna(flow['asof']):
        st.caption(f"수급 최근 자료 {flow['asof']:%Y-%m-%d} · 분석 일봉 {df.index[-1]:%Y-%m-%d}")
    columns = st.columns(4)
    for box, sessions, field, title in [
        (columns[0], 5, "ForeignAmountEstimate", "5거래일 외국인 추정"),
        (columns[1], 5, "InstitutionAmountEstimate", "5거래일 기관 추정"),
        (columns[2], 20, "ForeignAmountEstimate", "20거래일 외국인 추정"),
    ]:
        window = investor_window(investors, df, sessions, [field])
        box.metric(title, fmt(window[field].sum() if window is not None else None, "억원", 1, True))
    latest = investor_window(investors, df, 1, ["ForeignRate"])
    columns[3].metric("외국인 보유율", fmt(latest.ForeignRate.iloc[-1] if latest is not None else None, "%", 2))
    st.caption("추정 금액 = 일별 순매수 수량×해당일 종가의 합. 실제 순매수 금액과 다릅니다. 기간 내 모든 거래일이 있어야 합계를 표시합니다.")
    if investors.empty:
        st.info("투자자별 매매 데이터를 확인하지 못했습니다.")
        return
    inv = investors[investors.Date <= df.index[-1]].sort_values("Date", ascending=False).head(20)
    show = inv.rename(columns={"Date": "날짜", "Close": "종가", "InstitutionNet": "기관 순매수(주)",
                               "ForeignNet": "외국인 순매수(주)", "ForeignRate": "외국인 보유율(%)",
                               "InstitutionAmountEstimate": "기관 추정(억원)", "ForeignAmountEstimate": "외국인 추정(억원)"})
    st.dataframe(show, hide_index=True, use_container_width=True)
    if not inv.empty:
        plot = inv.sort_values("Date")
        fig = go.Figure()
        fig.add_bar(x=plot.Date, y=plot.ForeignNet, name="외국인(주)")
        fig.add_bar(x=plot.Date, y=plot.InstitutionNet, name="기관(주)")
        fig.update_layout(template="plotly_dark", height=280, barmode="group", margin={"t": 15, "b": 5})
        st.plotly_chart(fig, use_container_width=True)


def render_fund(fund, reports, close):
    st.markdown("#### 기업 개요")
    st.text(fund.get("Summary") or "기업 개요를 확인하지 못했습니다.")
    cols = st.columns(4)
    target = fund.get("Target")
    cols[0].metric("컨센서스 목표주가", fmt(target, "원"),
                   fmt((target / close - 1) * 100, "%", 1, True) if target and target > 0 else None)
    cols[1].metric("최근 확정 연간 ROE", fmt(fund.get("ROE"), "%", 2))
    cols[2].metric("PER / 업종 PER", f"{fmt(fund.get('PER'), '배', 2)} / {fmt(fund.get('IndustryPER'), '배', 2)}")
    cols[3].metric("PBR", fmt(fund.get("PBR"), "배", 2))
    st.caption(f"ROE 기간: {fund.get('ROEPeriod') or '미확인'} · 배당수익률: {fmt(fund.get('DividendYield'), '%', 2)}")
    st.caption("조회 시점의 재무·컨센서스 정보입니다. 목표가는 증권사 전망 평균이며, 차트·백테스트의 매도 체결가로 사용하지 않습니다.")
    st.markdown("#### 증권사 리포트")
    if not reports.data:
        st.info("리포트가 없거나 수집하지 못했습니다. 데이터 상태에서 확인하실 수 있습니다.")
    for rep in reports.data:
        st.link_button(f"{rep['broker']} · {rep['title']} ({rep['date']})", rep["link"])


def render_backtest(df):
    choice = st.selectbox("전략", list(STRATEGIES), key="bt_strategy")
    c1, c2, c3 = st.columns(3)
    fee = c1.number_input("편도 수수료(%)", min_value=0.0, max_value=5.0, value=0.015, step=0.005, format="%.3f", key="bt_fee")
    slip = c2.number_input("편도 슬리피지(%)", min_value=0.0, max_value=5.0, value=0.05, step=0.01, format="%.3f", key="bt_slippage")
    tax = c3.number_input("매도 시 세금(%)", min_value=0.0, max_value=5.0, value=0.0, step=0.01, format="%.3f", key="bt_tax")
    st.caption("세금 기본값은 0%입니다. 대상 상품과 적용 기간에 맞게 직접 입력해 주십시오. 입력한 단일 세율을 전체 기간에 적용합니다.")
    start = df.index[-1] - pd.Timedelta(days=365)
    try:
        result = run_backtest(df, STRATEGIES[choice], start=start,
                              fee_bps=fee * 100, slippage_bps=slip * 100, sell_tax_bps=tax * 100)
    except ValueError as exc:
        st.warning(str(exc))
        return
    bt = result["df"]
    st.caption(f"평가 구간: {bt.index[0]:%Y-%m-%d} ~ {bt.index[-1]:%Y-%m-%d}. 이전 일봉은 지표 준비에만 사용합니다.")
    boxes = st.columns(5)
    boxes[0].metric("전략 누적수익률", fmt(result["total_return"], "%", 2, True))
    boxes[1].metric("동일 종목 단순보유", fmt(result["buy_hold_return"], "%", 2, True))
    boxes[2].metric("단순보유 대비 차이", fmt(result["excess_pp"], "%p", 2, True))
    boxes[3].metric("최대낙폭(MDD)", fmt(result["mdd"], "%", 2), help="음수 값이며 0에 가까울수록 과거 낙폭이 작습니다.")
    boxes[4].metric("청산 거래 승률", fmt(result["win_rate"], "%", 1), f"청산 {result['trade_count']}회", delta_color="off")
    fig = go.Figure()
    for col, label, color in [("Cum_Strategy", "전략", "#38bdf8"), ("Cum_Market", "단순보유", "#94a3b8")]:
        fig.add_scatter(x=bt.index, y=(bt[col] - 1) * 100, name=label, line={"color": color})
    fig.update_layout(template="plotly_dark", height=340, yaxis_title="누적 수익률(%)", margin={"t": 20, "b": 10})
    st.plotly_chart(fig, use_container_width=True)
    st.caption("전일 종가로 신호를 확정하고 다음 거래일 시가에 체결합니다. 수수료·슬리피지·매도세를 입력값대로 반영합니다. 거래량 0인 날에는 체결하지 않습니다.")
    st.caption("마지막 보유분은 마지막 종가로 평가하며 청산 승률에서 제외합니다. 단순보유에도 같은 시작일과 매수 비용을 적용합니다. 소수점 수량을 허용한 전액 매수·매도 모형입니다.")
    st.caption("현재 종목의 과거 가격 모형이며 배당·현금이자·실제 주문 유동성은 반영하지 않습니다. 데이터의 수정주가·기업행사 처리에 영향을 받으며 미래 성과를 보장하지 않습니다.")
    if result["open_trade"]:
        st.info(f"종료일 미청산 보유분이 있습니다. 진입일: {result['open_trade']['EntryDate']:%Y-%m-%d}")
    if not result["trades"].empty:
        st.dataframe(result["trades"].drop(columns=["Capital", "Units"]), hide_index=True, use_container_width=True)
    st.download_button("백테스트 일별 결과 CSV", bt.to_csv().encode("utf-8-sig"),
                       file_name=f"backtest_{st.session_state.get('us_selected', '') if st.session_state.get('market_region') == '🇺🇸 미국주식' else st.session_state.selected_code}.csv", mime="text/csv")



def _news_momentum_summary(news_result):
    """최근 헤드라인을 보조적으로 요약한다. 정량 점수에는 반영하지 않는다."""
    if news_result is None or getattr(news_result, "status", "error") == "error" or not getattr(news_result, "data", None):
        return {
            "label": "미확인",
            "text": "최근 관련 뉴스 헤드라인을 확인하지 못했습니다.",
            "headlines": [],
        }

    positive_words = [
        "수주", "계약", "공급", "증설", "투자", "흑자", "상향", "성장", "호조",
        "개선", "회복", "승인", "출시", "협력", "최대", "신사업", "증가", "강세",
    ]
    negative_words = [
        "하향", "적자", "부진", "감소", "손실", "리콜", "소송", "규제", "우려",
        "중단", "지연", "철회", "감산", "약세", "급락", "하락",
    ]
    headlines = [str(x.get("title", "")).strip() for x in news_result.data if str(x.get("title", "")).strip()][:5]
    pos = sum(sum(1 for word in positive_words if word in title) for title in headlines)
    neg = sum(sum(1 for word in negative_words if word in title) for title in headlines)

    if pos >= neg + 2:
        label = "긍정 우위"
        text = f"최근 헤드라인에서는 실적·수주·성장 계열의 긍정 키워드가 상대적으로 우세합니다(긍정 {pos} / 부정 {neg})."
    elif neg >= pos + 2:
        label = "부정 우위"
        text = f"최근 헤드라인에서는 실적 둔화·우려·하락 계열의 부정 키워드가 상대적으로 우세합니다(긍정 {pos} / 부정 {neg})."
    else:
        label = "혼조"
        text = f"최근 뉴스 헤드라인의 방향성은 혼조입니다(긍정 {pos} / 부정 {neg}). 단일 뉴스보다 실적·수급과 함께 확인할 필요가 있습니다."
    return {"label": label, "text": text, "headlines": headlines[:3]}




def render_analysis(bundle, code, display_name):
    history = bundle["history"]
    if history.status == "error":
        st.error("일봉 수집에 실패했습니다. 잠시 후 다시 조회해 주십시오.")
        st.caption(" / ".join(history.notes))
        return
    try:
        completed = completed_history(history.data)
        if len(completed) < 61:
            st.warning(f"전 거래일까지의 일봉이 {len(completed)}개입니다. 60일 지표와 전일 비교에는 최소 61개가 필요합니다.")
            return
        df = add_indicators(completed)
    except ValueError as exc:
        st.error(str(exc))
        return
    fund = bundle["main"].data["fund"]
    quote = bundle["main"].data["quote"]
    investors = bundle["investors"].data
    name = quote.get("Name") or display_name
    stale = (datetime.now(KST).date() - df.index[-1].date()).days > 7
    st.markdown(f"### {escape(name)} ({code})")
    st.caption(f"분석 일봉 기준: {df.index[-1]:%Y-%m-%d} · KST 당일 봉은 항상 제외 · 조회: {history.fetched_at}")
    if stale:
        st.warning("최근 일봉이 7일 이상 경과했습니다. 거래정지·휴장·수집 지연 여부를 확인해 주십시오. 최신 재무정보의 점수 반영은 보류합니다.")
    qcols = st.columns(2)
    previous, snapshot = number(quote.get("Previous")), number(quote.get("Price"))
    qcols[0].metric("확정 일봉 종가", fmt(df.Close.iloc[-1], "원"), fmt(period_return(df, 1), "%", 2, True))
    qcols[1].metric("별도 조회 시세 스냅샷(KRX)", fmt(snapshot, "원"),
                   fmt((snapshot / previous - 1) * 100, "%", 2, True) if snapshot and previous and previous > 0 else None)
    st.caption(f"스냅샷 기준 시각: {quote.get('AsOf') or '미확인'} · 조회 시세는 과거 일봉에 덮어쓰지 않습니다.")
    with st.expander("공매도 데이터 추가 및 출처 상태"):
        st.caption("공매도 자동 수집은 검증된 공급원이 없어 미제공 상태입니다. Code, Date, ShortRatio 열이 있는 UTF-8 CSV를 올리시면 기준일이 일치하는 행을 평가에 사용할 수 있습니다. ShortRatio는 공매도 거래량/전체 거래량×100입니다.")
        uploaded = st.file_uploader("공매도 CSV(선택)", type=["csv"], key=f"short_{code}")
        status_rows = [{"데이터": key, "상태": item.status, "조회 시각(KST)": item.fetched_at,
                        "출처": item.source, "설명": " / ".join(item.notes)} for key, item in bundle.items()]
        st.dataframe(pd.DataFrame(status_rows), hide_index=True, use_container_width=True)
    short = None
    if uploaded is not None:
        try:
            short = read_short_csv(uploaded.getvalue(), code, df.index[-1])
        except (ValueError, UnicodeError, pd.errors.ParserError) as exc:
            st.warning(str(exc))
    with st.expander("수급 수집 상태 · 대체 자료 입력"):
        result = bundle['investors']
        st.caption('수급 출처: ' + result.source + ' · 상태: ' + result.status)
        st.caption(' / '.join(result.notes))
        st.caption('수집 실패 시 증권사 등에서 확인한 CSV를 입력하실 수 있습니다. 필수 열: Code, Date, ForeignNet, InstitutionNet. 수량 단위는 주이며, ForeignRate(%)는 선택입니다. 업로드 자료가 자동 수집 자료를 대신합니다.')
        supply_upload=st.file_uploader('수급 CSV (선택)',type=['csv'],key='investor_csv_'+code)
        if supply_upload is not None:
            try:
                investors=read_investor_csv(supply_upload.getvalue(),code,df)
                st.caption(f'사용자 제공 수급 {len(investors)}거래일을 적용했습니다. 금액은 확정 종가로 추정합니다.')
            except (ValueError, UnicodeError, pd.errors.ParserError) as exc:
                st.error(str(exc))
                investors=pd.DataFrame(columns=INV_COLS)
                st.caption('잘못된 업로드 자료로 수급을 해석하지 않습니다. 파일을 제거하시면 자동 수집 자료를 사용합니다.')
    score = evaluate_technical_score(df)
    # 종합 분석 화면의 리서치 코멘트를 위해 최근 헤드라인을 캐시 조회합니다.
    # 뉴스는 점수에는 반영하지 않고 보조 코멘트로만 사용합니다.
    with st.spinner("추세·수급·모멘텀·최근 뉴스까지 종합 해석하고 있습니다…"):
        news_for_comment = cached_news(name)
    research = build_general_research_commentary(
        df, investors, {} if stale else fund, score, news_for_comment, short, stale
    )
    cols = st.columns(3)
    cols[0].metric("기술 점수 (100점 기준)", f"{score['points']} / {score['possible']}")
    cols[1].metric("기술 지표 확보율", f"{score['coverage']}%")
    cols[2].metric("공매도 거래량 비중", fmt((short or {}).get("ShortRatio"), "%", 2))
    st.write(score["grade"])
    st.caption("기술 점수: 추세 35 · 모멘텀 25 · 거래량 20 · 진입 부담 20. 한국·미국에 같은 규칙을 적용합니다. 수급·재무는 별도 해석하며 기술 점수에 합산하지 않습니다. 승률이나 상승 확률이 아닙니다.")
    if score["score"] is None:
        st.caption(f"미확인 항목까지 확보했을 때의 산술적 점수 범위: {score['points']}~{score['upper_bound']} / 100. 신뢰구간이나 전망 범위가 아닙니다.")

    render_research_cards(research)
    flow = research['flow']
    st.metric('외국인·기관 수급 점수', f"{flow['score']} / 100" if flow['score'] is not None else '확인 대기')
    st.caption('수급 점수는 최근 5거래일의 매수 방향 20 · 거래량 대비 순매수 60 · 순매수 지속성 20입니다. 기술 점수와 별도입니다.')
    if flow['detail']:
        with st.expander('수급 점수 산정 근거'):
            st.dataframe(pd.DataFrame(flow['detail']), hide_index=True, use_container_width=True)

    if research["dist20"] is not None or research["dist60"] is not None:
        st.caption(
            f"이평선 이격 · 20일선 {fmt(research['dist20'], '%', 1, True)} · "
            f"60일선 {fmt(research['dist60'], '%', 1, True)}"
        )

    pos_col, risk_col = st.columns(2)
    with pos_col:
        st.markdown("**📈 긍정 요인**")
        if research["positives"]:
            for item in research["positives"]:
                st.markdown(f"- {item}")
        else:
            st.caption("현재 확인 가능한 뚜렷한 긍정 요인이 제한적입니다.")
    with risk_col:
        st.markdown("**⚠️ 점수 제한·리스크 요인**")
        if research["risks"]:
            for item in research["risks"]:
                st.markdown(f"- {item}")
        else:
            st.caption("현재 관측된 주요 리스크가 두드러지지 않습니다.")

    with st.expander("📌 왜 이 점수인가? · 배점 근거 자세히 보기"):
        left_reason, right_reason = st.columns(2)
        with left_reason:
            st.markdown("**점수에 기여한 핵심 항목**")
            if research["contributors"]:
                for item in research["contributors"]:
                    st.markdown(f"- {item}")
            else:
                st.caption("가점 항목을 확인하지 못했습니다.")
        with right_reason:
            st.markdown("**점수를 제한하거나 등급을 보류한 항목**")
            if research["limiters"]:
                for item in research["limiters"]:
                    st.markdown(f"- {item}")
            else:
                st.caption("0점 또는 미확인 항목이 없습니다.")

        if research["headlines"]:
            st.markdown("**최근 뉴스 헤드라인 참고**")
            for title in research["headlines"]:
                st.markdown(f"- {title}")
            st.caption("뉴스 제목만으로 호재·악재를 단정하지 않습니다. 원문의 발표 시점과 실적·공시를 함께 확인해 주십시오. 뉴스는 기술 점수에 반영하지 않습니다.")

    scenario = price_scenario(df)
    if scenario:
        with st.expander("ATR 기준 가격 시나리오"):
            boxes = st.columns(5)
            for box, (label, value) in zip(boxes, [(k, v) for k, v in scenario.items() if k != "reward_risk"]):
                box.metric(label, fmt(value, "원"))
            st.caption(f"1차 진입 기준 보상/위험 비율 {scenario['reward_risk']:.2f}. 최근 ATR로 간격을 정한 계산 예시이며, 실제 지지·저항 검증이나 호가단위 보정은 포함하지 않습니다.")
    tabs = st.tabs(["차트", "외국인·기관", "기업·리포트", "채점표", "백테스트", "뉴스"])
    with tabs[0]:
        boxes = st.columns(3)
        for box, sessions, label in zip(boxes, [5, 20, 252], ["5거래일 수익률", "20거래일 수익률", "252거래일 수익률"]):
            box.metric(label, fmt(period_return(df, sessions), "%", 2, True))
        render_chart(df)
        st.download_button("분석 일봉·지표 CSV", df.to_csv().encode("utf-8-sig"),
                           file_name=f"daily_{code}.csv", mime="text/csv")
    with tabs[1]:
        render_supply(df, investors)
    with tabs[2]:
        render_fund(fund, bundle["reports"], df.Close.iloc[-1])
    with tabs[3]:
        st.dataframe(score["logs"], hide_index=True, use_container_width=True)
        st.caption("가점 조건을 충족하지 않은 관측값은 0점, 데이터가 없는 항목은 미확인입니다. 득점 합계가 위에 표시된 점수와 일치합니다.")
    with tabs[4]:
        render_backtest(df)
    with tabs[5]:
        if st.button("관련 뉴스 조회", key=f"load_news_{code}"):
            st.session_state[f"news_open_{code}"] = True
        if st.session_state.get(f"news_open_{code}"):
            news = cached_news(name)
            if news.status == "error":
                st.info("뉴스를 조회하지 못했습니다.")
            elif not news.data:
                st.info("검색된 뉴스가 없습니다.")
            for item in news.data:
                st.link_button(item["title"], item["link"])
                st.caption(item["date"])
            st.caption(f"Google News RSS · 조회: {news.fetched_at} · 제목 검색 결과이며 기업 공시 확인을 대체하지 않습니다.")



# ===== 4. 달리는 말 탐지기 =====
def add_running_indicators(frame):
    df = add_indicators(frame)
    df["MA120"] = df.Close.rolling(120, min_periods=120).mean()
    df["VOL_MA20"] = df.Volume.rolling(20, min_periods=20).mean()
    df["HIGH20_PREV"] = df.High.shift(1).rolling(20, min_periods=20).max()
    df["HIGH60_PREV"] = df.High.shift(1).rolling(60, min_periods=60).max()
    df["RET5"] = df.Close.pct_change(5) * 100
    df["RET20"] = df.Close.pct_change(20) * 100
    df["DIST_MA20"] = (df.Close / df.MA20 - 1) * 100
    df["VOL_RATIO"] = df.Volume / df.VOL_MA20.replace(0, np.nan)
    df["MA20_SLOPE"] = df.MA20.pct_change(5) * 100
    df["MA60_SLOPE"] = df.MA60.pct_change(10) * 100

    high, low, close = df.High, df.Low, df.Close
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)
    tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr.replace(0, np.nan)
    dx = 100 * (plus_di-minus_di).abs() / (plus_di+minus_di).replace(0, np.nan)
    df["PLUS_DI"] = plus_di
    df["MINUS_DI"] = minus_di
    df["ADX14"] = dx.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    return df


def running_horse_score(frame):
    try:
        completed = completed_history(frame)
        if len(completed) < 130:
            return None
        df = add_running_indicators(completed)
        needed = ["MA120","VOL_MA20","HIGH60_PREV","ADX14","RSI","MACD","MACD_SIGNAL"]
        ready = df.dropna(subset=needed)
        if len(ready) < 2:
            return None
        r, p = ready.iloc[-1], ready.iloc[-2]
    except Exception:
        return None

    score = 0
    detail = []
    def add(cond, pts, label):
        nonlocal score
        ok = bool(cond)
        if ok:
            score += pts
        detail.append({"판정":"✅" if ok else "➖", "점수":pts if ok else 0, "조건":label})

    # 추세 30
    add(r.Close > r.MA20, 5, "현재가 > 20일선")
    add(r.MA20 > r.MA60, 5, "20일선 > 60일선")
    add(r.MA60 > r.MA120, 5, "60일선 > 120일선")
    add(r.MA20_SLOPE > 0, 5, "20일선 상승")
    add(r.MA60_SLOPE > 0, 5, "60일선 상승")
    add(r.Close >= r.HIGH60_PREV * 0.99, 5, "60일 고점 돌파/근접")

    # 거래량 20
    add(r.VOL_RATIO >= 1.2, 5, "거래량 > 20일 평균 1.2배")
    add(r.VOL_RATIO >= 2.0, 5, "거래량 > 20일 평균 2배")
    add((r.Close > p.Close) and (r.Volume > p.Volume), 5, "상승일 거래량 증가")
    recent = ready.tail(5)
    pullback = (recent.Close.pct_change() < 0).any() and r.Volume < r.VOL_MA20
    add(pullback, 5, "최근 눌림에서 거래량 감소")

    # 모멘텀 20
    add(55 <= r.RSI <= 70, 7, "RSI 55~70")
    add((r.MACD > r.MACD_SIGNAL) and (r.MACD > 0), 7, "MACD > Signal 및 0선 위")
    add((r.ADX14 >= 20) and (r.PLUS_DI > r.MINUS_DI), 6, "ADX 20+ 및 +DI 우위")

    # 돌파/위치 20
    add(r.Close > r.HIGH20_PREV, 5, "20일 신고가 돌파")
    add(r.Close > r.HIGH60_PREV, 5, "60일 신고가 돌파")
    add(-1 <= r.DIST_MA20 <= 6, 5, "20일선 이격 적정")
    add(r.RET20 > 0, 5, "20거래일 수익률 플러스")

    # 과열 방지 10
    add(r.RSI < 78, 5, "RSI 극단 과열 아님")
    add(r.DIST_MA20 < 10, 5, "20일선 과도 이격 아님")

    penalties=[]
    if r.RSI >= 80:
        score -= 8; penalties.append("RSI 80 이상 -8")
    if r.DIST_MA20 >= 15:
        score -= 8; penalties.append("20일선 +15% 이상 이격 -8")
    if r.RET5 >= 20:
        score -= 5; penalties.append("5거래일 +20% 이상 급등 -5")
    score=max(0,min(100,int(round(score))))

    healthy_pullback = r.Close > r.MA20 and -1 <= r.DIST_MA20 <= 4 and 50 <= r.RSI <= 68
    breakout = r.Close > r.HIGH60_PREV
    near = r.Close >= r.HIGH60_PREV * 0.97
    if score >= 80 and healthy_pullback:
        status="🟢 우선 관찰 — 상승 추세와 이격 양호"
    elif score >= 80 and breakout and r.RSI < 75:
        status="🚀 강한 돌파 — 추격보다 눌림 대기"
    elif score >= 70 and near:
        status="🟢 달리는 말 후보 — 돌파·지지 확인"
    elif score >= 60:
        status="🟡 관심 종목 — 조건 일부 미충족"
    elif score >= 45:
        status="🟠 관찰 유지 — 추세 확인 필요"
    else:
        status="🔴 우선순위 낮음"

    return {"score":score,"status":status,"row":r,"df":ready,"detail":pd.DataFrame(detail),
            "penalties":penalties,"support1":float(r.MA20),"support2":float(r.MA60),
            "resistance":float(r.HIGH60_PREV)}



def _horse_supply_metrics(investors, history_df):
    """Return verified 5/20-session foreign/institution flow metrics when available."""
    metrics = {
        "foreign5": None, "institution5": None,
        "foreign20": None, "institution20": None,
        "foreign_rate_change10": None,
    }
    if investors is None or getattr(investors, "empty", True) or history_df is None or history_df.empty:
        return metrics

    for sessions, key_f, key_i in [
        (5, "foreign5", "institution5"),
        (20, "foreign20", "institution20"),
    ]:
        window = investor_window(investors, history_df, sessions, ["ForeignNet", "InstitutionNet"])
        if window is not None:
            metrics[key_f] = float(window.ForeignNet.sum())
            metrics[key_i] = float(window.InstitutionNet.sum())

    w10 = investor_window(investors, history_df, 10, ["ForeignRate"])
    if w10 is not None:
        metrics["foreign_rate_change10"] = float(w10.ForeignRate.iloc[-1] - w10.ForeignRate.iloc[0])
    return metrics


def _fmt_shares(v):
    if v is None:
        return "미확인"
    sign = "+" if v > 0 else ""
    av = abs(v)
    if av >= 1_000_000:
        return f"{sign}{v/1_000_000:,.1f}백만주"
    if av >= 10_000:
        return f"{sign}{v/10_000:,.1f}만주"
    return f"{sign}{v:,.0f}주"


def build_horse_commentary(result, investors=None):
    """Deterministic broker-style commentary based only on observed chart/flow data."""
    r = result["row"]
    supply = _horse_supply_metrics(investors, result["df"])
    positives, risks = [], []

    # Trend structure
    if r.Close > r.MA20 > r.MA60 > r.MA120 and r.MA20_SLOPE > 0 and r.MA60_SLOPE > 0:
        positives.append("단·중기 이동평균선이 정배열을 형성하고 20·60일선의 기울기도 우상향해 추세의 연속성이 양호합니다.")
    elif r.Close > r.MA20 and r.MA20 > r.MA60:
        positives.append("단기 추세는 우위에 있으나 120일선까지 포함한 완전한 중기 정배열 여부는 추가 확인이 필요합니다.")
    else:
        risks.append("이동평균선 배열이 완전히 정돈되지 않아 추세 추종 관점의 가격 구조는 아직 확증 단계가 아닙니다.")

    # Breakout / price location
    dist_high = (r.Close / result["resistance"] - 1) * 100
    if r.Close > result["resistance"]:
        positives.append(f"60일 전고점을 {dist_high:+.1f}% 상회해 매물대 돌파 시도가 확인됩니다. 돌파 가격대의 지지 전환 여부가 후속 강도의 핵심입니다.")
    elif dist_high >= -3:
        positives.append(f"60일 전고점까지 {abs(dist_high):.1f}% 이내로 접근해 가격 발견 구간 진입 가능성이 열려 있습니다.")
    else:
        risks.append(f"60일 전고점 대비 {dist_high:.1f}% 위치로, 본격적인 신고가 모멘텀으로 보기에는 아직 상단 매물 소화가 필요합니다.")

    # Momentum / overheating
    if 55 <= r.RSI <= 70 and r.MACD > r.MACD_SIGNAL and r.MACD > 0 and r.ADX14 >= 20 and r.PLUS_DI > r.MINUS_DI:
        positives.append("RSI·MACD·DMI가 동시에 강세 구간을 가리켜 가격 모멘텀의 질은 비교적 양호한 편입니다.")
    elif r.RSI >= 78 or r.DIST_MA20 >= 10:
        risks.append(f"RSI {r.RSI:.1f}, 20일선 이격 {r.DIST_MA20:+.1f}%로 단기 과열 부담이 있어 신규 추격 매수의 손익비는 다소 불리합니다.")
    else:
        risks.append("모멘텀 지표가 일제히 강세를 확인하지 못해 상승 추세의 가속 구간으로 단정하기에는 신호가 혼재돼 있습니다.")

    # Volume
    if r.VOL_RATIO >= 1.5 and r.Close > result["resistance"]:
        positives.append(f"거래량이 20일 평균의 {r.VOL_RATIO:.2f}배로 확대돼 돌파 과정에 거래 참여가 동반되고 있습니다.")
    elif r.VOL_RATIO < 0.8:
        risks.append(f"거래량이 20일 평균의 {r.VOL_RATIO:.2f}배에 그쳐 가격 상승을 뒷받침하는 거래 참여는 다소 제한적입니다.")
    elif r.VOL_RATIO < 1.2:
        risks.append(f"거래량이 20일 평균의 {r.VOL_RATIO:.2f}배 수준으로, 추세 확장 국면으로 보기에는 수급 확산 신호가 아직 약합니다.")

    # Investor flow
    f5, i5, f20, i20 = supply["foreign5"], supply["institution5"], supply["foreign20"], supply["institution20"]
    if f5 is not None and i5 is not None:
        if f5 > 0 and i5 > 0:
            positives.append(f"최근 5거래일 외국인({_fmt_shares(f5)})과 기관({_fmt_shares(i5)})이 동반 순매수해 두 투자자 집단에서 매수 우위가 관측됩니다. 가격 추세와의 일치 여부도 함께 살펴볼 필요가 있습니다.")
        elif f5 > 0 or i5 > 0:
            buyer = "외국인" if f5 > 0 else "기관"
            risks.append(f"최근 5거래일 {buyer}은 순매수이나 외국인·기관 동반 매수는 확인되지 않아 수급의 폭은 제한적입니다.")
        else:
            risks.append(f"최근 5거래일 외국인({_fmt_shares(f5)}), 기관({_fmt_shares(i5)}) 모두 순매도여서 가격 모멘텀 대비 수급 확증이 부족합니다.")
    else:
        risks.append("최근 외국인·기관 수급 데이터가 충분히 확보되지 않아 수급 측면의 확증 여부는 보수적으로 해석할 필요가 있습니다.")

    if f20 is not None and i20 is not None:
        if f20 > 0 and i20 > 0:
            positives.append("20거래일 누적 기준에서도 외국인·기관이 동반 순매수해 단기성 매수보다 지속성 있는 수급으로 해석할 여지가 있습니다.")
        elif f20 < 0 and i20 < 0:
            risks.append("20거래일 누적으로는 외국인·기관 모두 순매도여서 중기 수급 추세가 아직 가격 상승을 지지하지 못하고 있습니다.")

    # Final house view
    score = result["score"]
    if score >= 82 and len(positives) >= 3 and not (r.RSI >= 78 or r.DIST_MA20 >= 10):
        view = "추세와 모멘텀의 기술적 조건은 비교적 우호적입니다. 수급의 동반 여부는 별도 확인이 필요합니다. 다만 돌파 직후 추격보다는 전고점 또는 20일선 부근의 지지 확인 시 손익비가 개선될 수 있습니다."
    elif score >= 70:
        view = "기술적 추세는 우호적이지만 일부 수급 또는 거래량 조건의 추가 확인이 필요합니다. 신규 진입은 돌파 유지와 눌림 구간의 거래량 감소 여부를 함께 확인하는 전략이 적절합니다."
    elif score >= 55:
        view = "상승 후보군에는 포함될 수 있으나 추세·모멘텀·수급 중 적어도 한 축의 확증이 부족합니다. 현 시점에서는 선매수보다 신호 개선을 확인하는 관찰 전략이 우선입니다."
    else:
        view = "현재는 추세 추종형 신규 진입의 우선순위가 낮습니다. 이동평균선 재정렬, 거래량 회복, 외국인·기관 수급 개선 중 두 가지 이상이 동반되는지를 확인할 필요가 있습니다."

    return {
        "view": view,
        "positives": positives[:4],
        "risks": risks[:4],
        "supply": supply,
    }


@st.cache_data(ttl=900, max_entries=256, show_spinner=False)
def cached_horse_investors(code):
    result = fetch_investors(code)
    return result.data if result.status != "error" else pd.DataFrame(columns=INV_COLS)


@st.cache_data(ttl=900, max_entries=256, show_spinner=False)
def cached_running_score(code):
    result = fetch_history(code)
    if result.status == "error" or result.data is None or result.data.empty:
        return None
    return running_horse_score(result.data)


def render_running_chart(result, name):
    df = result["df"].tail(150)
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.035,
                        row_heights=[0.52,0.16,0.16,0.16])
    fig.add_trace(go.Candlestick(x=df.index, open=df.Open, high=df.High, low=df.Low,
                                 close=df.Close, name="Price"), row=1,col=1)
    for ma in [20,60,120]:
        fig.add_trace(go.Scatter(x=df.index,y=df[f"MA{ma}"],mode="lines",name=f"MA{ma}"),row=1,col=1)
    fig.add_hline(y=result["resistance"], line_dash="dot", annotation_text="60일 전고점", row=1,col=1)
    fig.add_trace(go.Bar(x=df.index,y=df.Volume,name="Volume"),row=2,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=df.VOL_MA20,mode="lines",name="Vol MA20"),row=2,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=df.RSI,mode="lines",name="RSI14"),row=3,col=1)
    for level in [70,50,30]: fig.add_hline(y=level,line_dash="dot",row=3,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=df.MACD,mode="lines",name="MACD"),row=4,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=df.MACD_SIGNAL,mode="lines",name="Signal"),row=4,col=1)
    fig.add_trace(go.Bar(x=df.index,y=df.MACD_HIST,name="Histogram"),row=4,col=1)
    fig.update_layout(template="plotly_dark", title=f"{name} — 모멘텀 프로파일", height=900,
                      xaxis_rangeslider_visible=False, legend={"orientation":"h"},
                      margin={"l":10,"r":10,"t":55,"b":10},
                      paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig,use_container_width=True)


def _resolve_horse_query(query, market_result):
    """Resolve a Korean stock name or 6-digit code to a unique candidate list."""
    query = str(query or "").strip()
    if not query:
        return []
    if valid_code(query):
        name = query
        if market_result is not None and not market_result.data.empty:
            hit = market_result.data[market_result.data.Code.astype(str) == query]
            if not hit.empty:
                name = str(hit.iloc[0].Name)
        return [{"Code": query, "Name": name}]
    if len(query) < 2:
        return []

    frames = []
    if market_result is not None:
        local = local_search(query, market_result.data)
        if not local.empty:
            frames.append(local)
    remote = cached_search(query)
    if remote is not None and getattr(remote, "data", None) is not None and not remote.data.empty:
        frames.append(remote.data[["Code", "Name"]])
    if not frames:
        return []

    matches = pd.concat(frames, ignore_index=True).drop_duplicates("Code")
    exact = matches[matches.Name.astype(str).str.casefold() == query.casefold()]
    if not exact.empty:
        matches = pd.concat([exact, matches], ignore_index=True).drop_duplicates("Code")
    return matches.head(10).to_dict("records")


def _horse_rank_badge(rank):
    return {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, f"#{rank}")


def _render_horse_leaderboard(out):
    if out is None or out.empty:
        return
    st.markdown("#### 🏆 달리는 말 순위")
    st.caption("점수 → 20일 수익률 순으로 정렬한 현재 스캔 표본 순위입니다.")
    top = out.head(5).reset_index(drop=True)
    cols = st.columns(len(top))
    for i, row in top.iterrows():
        rank = i + 1
        with cols[i]:
            st.markdown(
                f"""<div class=\"horse-rank-card horse-rank-{rank}\">
                <div class=\"horse-rank-badge\">{_horse_rank_badge(rank)}</div>
                <div class=\"horse-rank-name\">{escape(str(row['종목명']))}</div>
                <div class=\"horse-rank-score\">{int(row['점수'])}<span>/100</span></div>
                <div class=\"horse-rank-meta\">RSI {row['RSI']:.1f} · 거래량 {row['거래량배수']:.2f}x</div>
                <div class=\"horse-rank-meta\">20일 {row['20일수익률(%)']:+.1f}%</div>
                </div>""",
                unsafe_allow_html=True,
            )
    st.markdown("##### 📋 전체 순위")
    display = out.copy().reset_index(drop=True)
    display.insert(0, "순위", np.arange(1, len(display) + 1))
    st.dataframe(
        display,
        hide_index=True,
        use_container_width=True,
        column_config={
            "순위": st.column_config.NumberColumn("순위", width="small", format="%d위"),
            "점수": st.column_config.ProgressColumn("달리는 말 점수", min_value=0, max_value=100, format="%d점"),
            "종목명": st.column_config.TextColumn("종목명", width="medium"),
            "상태": st.column_config.TextColumn("판정", width="large"),
        },
    )



@st.cache_data(ttl=600, max_entries=4, show_spinner=False)
def _naver_horse_universe(max_pages=20):
    """Fetch KOSPI market-cap pages only. Lightweight first-stage universe collection."""
    max_pages = max(1, min(int(max_pages), 25))
    jobs = [(0, page) for page in range(1, max_pages + 1)]

    def get(job):
        sosok, page = job
        try:
            body = request_bytes(BASE + "/sise/sise_market_sum.naver", {"sosok": sosok, "page": page})
            return parse_market(body, "KOSPI"), None
        except Exception as exc:
            return None, f"KOSPI {page}페이지: {type(exc).__name__}"

    with ThreadPoolExecutor(max_workers=5) as pool:
        responses = list(pool.map(get, jobs))
    frames = [x for x, _ in responses if x is not None]
    errors = [err for _, err in responses if err]
    df = pd.concat(frames, ignore_index=True).drop_duplicates("Code") if frames else pd.DataFrame(columns=MARKET_COLS)
    notes = [f"KOSPI 시가총액 표 {len(frames)}/{len(jobs)}페이지 수집 · {len(df)}종목 확보"] + errors
    return Result(df, BASE + "/sise/sise_market_sum.naver", "ok" if not errors else "partial" if frames else "error", notes)


def _horse_prefilter_kospi(stocks, deep_count=60):
    """Diversified lightweight pre-filter so deep OHLCV calls stay bounded."""
    if stocks is None or stocks.empty:
        return stocks
    if 'AsOf' in stocks.columns and stocks.Marcap.isna().all():
        # 시장 전체 수집 장애 시 명시된 기본 종목 표본만 정밀 분석합니다.
        return stocks[(stocks.Close > 0) & (stocks.Volume > 0)].head(deep_count).copy()
    df = stocks.copy()
    for col in ["Close", "Chg", "Volume", "Marcap"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Close", "Chg", "Volume", "Marcap"])
    df = df[(df.Close > 0) & (df.Volume > 0) & (df.Marcap > 0)].copy()
    if df.empty:
        return df
    df["AmountEstimate"] = df.Close * df.Volume

    # Illiquid tail removal: retain upper 70% of turnover or sufficiently large caps.
    turnover_floor = df.AmountEstimate.quantile(0.30)
    liquid = df[(df.AmountEstimate >= turnover_floor) | (df.Marcap.rank(pct=True) >= 0.70)].copy()
    ranked = market_rankings(liquid)

    # Blend several lenses to avoid a pure large-cap or pure one-day gainer bias.
    bucket = max(20, deep_count // 2)
    parts = [
        ranked.head(bucket),
        liquid.sort_values("AmountEstimate", ascending=False).head(bucket),
        liquid.sort_values("Chg", ascending=False).head(bucket),
        liquid.sort_values("Marcap", ascending=False).head(max(15, deep_count // 3)),
    ]
    merged = pd.concat(parts, ignore_index=True).drop_duplicates("Code")
    if "ScreenScore" not in merged.columns:
        merged = merged.merge(ranked[["Code", "ScreenScore"]], on="Code", how="left")
    return merged.sort_values(["ScreenScore", "AmountEstimate"], ascending=False).head(deep_count)


def _deep_scan_horse_candidates(candidates, workers=6):
    if candidates is None or candidates.empty:
        return []
    records = candidates.to_dict("records")

    def analyze(rec):
        code = str(rec["Code"])
        sr = cached_running_score(code)
        if sr is None:
            return None
        rr = sr["row"]
        return {
            "종목명": rec.get("Name", code),
            "코드": code,
            "점수": sr["score"],
            "상태": sr["status"],
            "종가": round(float(rr.Close)),
            "5일수익률(%)": round(float(rr.RET5), 2),
            "20일수익률(%)": round(float(rr.RET20), 2),
            "RSI": round(float(rr.RSI), 1),
            "ADX": round(float(rr.ADX14), 1),
            "거래량배수": round(float(rr.VOL_RATIO), 2),
            "20일선이격(%)": round(float(rr.DIST_MA20), 2),
            "60일고점대비(%)": round((float(rr.Close) / float(sr["resistance"]) - 1) * 100, 2),
            "_result": sr,
        }

    rows = []
    with ThreadPoolExecutor(max_workers=max(2, min(int(workers), 8))) as pool:
        futures = [pool.submit(analyze, rec) for rec in records]
        for future in futures:
            try:
                item = future.result()
                if item is not None:
                    rows.append(item)
            except Exception:
                pass
    return rows


def _enrich_top20_with_supply(raw):
    if raw is None or raw.empty:
        return raw
    enriched = raw.head(20).copy()
    comments, flow_labels = [], []
    for _, row in enriched.iterrows():
        sr = row["_result"]
        inv = cached_horse_investors(str(row["코드"]))
        commentary = build_horse_commentary(sr, inv)
        supply = commentary["supply"]
        f5, i5 = supply["foreign5"], supply["institution5"]
        if f5 is not None and i5 is not None:
            if f5 > 0 and i5 > 0:
                flow = "외인·기관 동반매수"
            elif f5 > 0:
                flow = "외국인 우위"
            elif i5 > 0:
                flow = "기관 우위"
            else:
                flow = "외인·기관 동반매도"
        else:
            flow = "수급 미확인"
        comments.append(commentary["view"])
        flow_labels.append(flow)
    enriched["수급"] = flow_labels
    enriched["전략 코멘트"] = comments
    return enriched


def render_running_horse(market_result):
    st.markdown("### 🐎 달리는 말 탐지기")
    st.caption("가격 추세, 거래량, RSI, MACD, ADX, 신고가와 이격도를 결합해 시장 주도주의 추세 지속성을 평가합니다. 스코어는 매수 신호가 아닌 후보 선별용입니다.")
    single, scanner, rules = st.tabs(["🔎 개별 분석", "🏆 KOSPI TOP 20", "📐 모델 기준"])

    with single:
        default_query = st.session_state.get("selected_name") or st.session_state.get("selected_code") or "삼성전자"
        c1, c2 = st.columns([4, 1])
        query = c1.text_input(
            "종목명 또는 종목코드",
            value=default_query,
            key="horse_query",
            placeholder="예: 삼성전자 또는 005930",
        )
        run = c2.button("달리는 말 분석", key="horse_run", use_container_width=True, type="primary")

        if run:
            matches = _resolve_horse_query(query, market_result)
            st.session_state.horse_matches = matches
            if len(matches) == 1:
                st.session_state.horse_selected_code = matches[0]["Code"]
                st.session_state.horse_selected_name = matches[0]["Name"]
            elif not matches:
                st.session_state.horse_selected_code = ""
                st.session_state.horse_selected_name = ""

        matches = st.session_state.get("horse_matches", [])
        if run and not matches:
            if str(query).strip().isdigit():
                st.warning("6자리 국내 종목코드 또는 정확한 종목명을 입력해 주십시오.")
            else:
                st.warning("종목을 찾지 못했습니다. 종목명을 조금 더 정확하게 입력해 주십시오.")

        if len(matches) > 1:
            st.caption("검색 결과가 여러 개입니다. 분석할 종목을 선택해 주십시오.")
            labels = [f"{x['Name']} ({x['Code']})" for x in matches]
            chosen = st.selectbox("검색 결과", labels, key="horse_match_select", label_visibility="collapsed")
            if st.button("선택 종목 분석", key="horse_match_run", use_container_width=True):
                idx = labels.index(chosen)
                st.session_state.horse_selected_code = matches[idx]["Code"]
                st.session_state.horse_selected_name = matches[idx]["Name"]

        code = st.session_state.get("horse_selected_code", "")
        name = st.session_state.get("horse_selected_name", code)
        if code:
            with st.spinner(f"{name} 모멘텀 조건을 분석하고 있습니다…"):
                result = cached_running_score(code)
            if result is None:
                st.warning("분석 가능한 일봉 데이터가 부족하거나 조회에 실패했습니다.")
            else:
                r = result["row"]
                st.markdown(f"#### {escape(str(name))} ({code})")
                boxes = st.columns(5)
                boxes[0].metric("달리는 말 점수", f"{result['score']} / 100")
                boxes[1].metric("종가", f"{r.Close:,.0f}원")
                boxes[2].metric("RSI", f"{r.RSI:.1f}")
                boxes[3].metric("ADX", f"{r.ADX14:.1f}")
                boxes[4].metric("거래량", f"{r.VOL_RATIO:.2f}x")
                st.subheader(result["status"])
                st.caption(
                    f"20일선 {r.MA20:,.0f}원 · 60일선 {r.MA60:,.0f}원 · "
                    f"60일 전고점 {result['resistance']:,.0f}원 · 20일선 이격 {r.DIST_MA20:+.1f}%"
                )
                with st.spinner("외국인·기관 수급을 함께 점검하고 있습니다…"):
                    horse_inv = cached_horse_investors(code)
                commentary = build_horse_commentary(result, horse_inv)
                st.markdown("##### 🧠 리서치 데스크 코멘트")
                st.info(commentary["view"])
                cc1, cc2 = st.columns(2)
                with cc1:
                    st.markdown("**📈 상승 논거**")
                    if commentary["positives"]:
                        for text in commentary["positives"]:
                            st.markdown(f"- {text}")
                    else:
                        st.caption("현재 확인 가능한 강한 상승 논거가 제한적입니다.")
                with cc2:
                    st.markdown("**⚠️ 리스크 요인**")
                    if commentary["risks"]:
                        for text in commentary["risks"]:
                            st.markdown(f"- {text}")
                    else:
                        st.caption("주요 기술·수급 리스크가 두드러지지 않습니다.")
                render_running_chart(result, name)
                l, rcol = st.columns([3, 2])
                with l:
                    st.dataframe(result["detail"], hide_index=True, use_container_width=True)
                with rcol:
                    st.write(f"**1차 지지:** {result['support1']:,.0f}원 (20일선)")
                    st.write(f"**2차 지지:** {result['support2']:,.0f}원 (60일선)")
                    st.write(f"**주요 돌파선:** {result['resistance']:,.0f}원")
                    if result["penalties"]:
                        st.warning(" / ".join(result["penalties"]))

    with scanner:
        st.markdown("#### 🏆 KOSPI 달리는 말 · TOP 20")
        st.caption(
            "KOSPI 전체 시가총액 표를 1차로 수집한 뒤 거래대금·등락률·시총으로 후보를 압축하고, "
            "일봉 기술지표를 정밀 분석합니다. 최종 상위 20개는 외국인·기관 수급까지 추가 점검합니다."
        )
        c1, c2 = st.columns(2)
        deep_count = c1.slider("정밀 분석 후보 수", 40, 100, 60, 10, key="horse_deep_count",
                               help="클수록 시장 커버리지는 넓어지지만 조회 시간이 늘어납니다.")
        pages = c2.slider("KOSPI 시장 페이지", 10, 25, 20, 5, key="horse_kospi_pages",
                          help="페이지당 종목 수는 공급 화면에 따라 달라질 수 있습니다.")

        if st.button("KOSPI 모멘텀 TOP 20 분석", type="primary", key="horse_auto_top20_btn", use_container_width=True):
            with st.spinner("1단계: KOSPI 전체 후보군을 수집하고 있습니다…"):
                universe = cached_horse_kospi_universe(pages)
            if universe.data.empty:
                st.warning("KOSPI 후보군을 확보하지 못했습니다.")
            else:
                candidates = _horse_prefilter_kospi(universe.data, deep_count)
                st.caption(f"1차 수집 {len(universe.data)}종목 → 정밀 분석 후보 {len(candidates)}종목")
                progress = st.progress(0)
                progress.progress(20)
                with st.spinner(f"2단계: {len(candidates)}종목 일봉·모멘텀을 병렬 분석하고 있습니다…"):
                    rows = _deep_scan_horse_candidates(candidates, workers=6)
                progress.progress(75)
                if not rows:
                    progress.empty()
                    st.warning("정밀 분석 결과를 확보하지 못했습니다.")
                else:
                    raw = pd.DataFrame(rows).sort_values(
                        ["점수", "20일수익률(%)", "거래량배수"], ascending=[False, False, False]
                    ).reset_index(drop=True)
                    with st.spinner("3단계: 상위 20개 외국인·기관 수급을 확인하고 있습니다…"):
                        top20 = _enrich_top20_with_supply(raw)
                    progress.progress(100)
                    progress.empty()
                    st.session_state.horse_auto_raw = raw
                    st.session_state.horse_auto_top20_results = top20
                    st.session_state.horse_auto_meta = {
                        "universe": len(universe.data), "deep": len(candidates), "pages": pages,
                        "status": universe.status, "notes": universe.notes,
                    }

        top20 = st.session_state.get("horse_auto_top20_results")
        meta = st.session_state.get("horse_auto_meta")
        if isinstance(top20, pd.DataFrame) and not top20.empty:
            st.success(
                f"KOSPI {meta.get('universe', 0)}종목 1차 탐색 → "
                f"{meta.get('deep', 0)}종목 정밀 분석 → 최종 TOP 20"
            )
            _render_horse_leaderboard(top20.drop(columns=["_result"], errors="ignore"))

            st.markdown("##### 📋 TOP 20 종목별 전략 코멘트")
            for idx, row in top20.reset_index(drop=True).iterrows():
                rank = idx + 1
                badge = _horse_rank_badge(rank)
                with st.expander(
                    f"{badge} {rank}위 · {row['종목명']} ({row['코드']}) · {int(row['점수'])}점 · {row['수급']}"
                ):
                    sr = row["_result"]
                    inv = cached_horse_investors(str(row["코드"]))
                    commentary = build_horse_commentary(sr, inv)
                    st.info(commentary["view"])
                    cpos, crisk = st.columns(2)
                    with cpos:
                        st.markdown("**📈 상승 논거**")
                        for text in commentary["positives"]:
                            st.markdown(f"- {text}")
                    with crisk:
                        st.markdown("**⚠️ 리스크 요인**")
                        for text in commentary["risks"]:
                            st.markdown(f"- {text}")
                    supply = commentary["supply"]
                    st.caption(
                        "최근 5거래일 순매수 · "
                        f"외국인 {_fmt_shares(supply['foreign5'])} / 기관 {_fmt_shares(supply['institution5'])}"
                    )

            export = top20.drop(columns=["_result"], errors="ignore")
            st.download_button(
                "KOSPI 달리는 말 TOP20 CSV",
                export.to_csv(index=False).encode("utf-8-sig"),
                file_name="running_horse_KOSPI_TOP20.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.caption(
                "※ 전 종목을 동일 깊이로 전수 분석하는 방식이 아니라, KOSPI 시장 전체를 1차 경량 스크리닝한 뒤 "
                "상위 후보군에 기술·수급 분석을 집중하는 2단계 방식입니다. 속도와 시장 커버리지의 균형을 위한 설계입니다."
            )

    with rules:
        st.markdown("""
**추세 30점** — 현재가>20일선, 20>60>120일선, 20·60일선 상승, 60일 고점 근접/돌파  
**거래량 20점** — 20일 평균 대비 거래량 증가, 상승일 거래량 증가, 눌림 거래량 감소  
**모멘텀 20점** — RSI 55~70, MACD 양수·Signal 상회, ADX 20 이상 +DI 우위  
**돌파/위치 20점** — 20·60일 신고가, 20일선 이격 적정, 20거래일 상승  
**과열 방지 10점** — RSI 78 미만, 20일선 이격 10% 미만  
**패널티** — RSI 80 이상 -8, 20일선 +15% 이상 이격 -8, 5일 +20% 이상 급등 -5

- **80점 이상:** 강한 후보
- **70~79점:** 우선 관찰
- **60~69점:** 관심
- **45~59점:** 추세 확인 필요
- **45점 미만:** 우선순위 낮음
        """)

def render_original_workspace(market):
    resolve_pending(market.data)
    with st.expander("종목명 검색 목록 상태"):
        st.caption("종목명 목록과 시장 시세는 별도로 수집합니다. 기본 이름 별칭에는 가격 정보가 없습니다.")
        if st.button("종목명 목록 새로고침", key="refresh_kr_directory"):
            stock_directory.clear('KR')
            cached_search.clear()
            st.rerun()
    left, right = st.columns([7, 3])
    with right:
        render_rankings(market)
    with left:
        c1, c2, c3 = st.columns([4, 1, 1])
        c1.text_input("종목 검색", key="query", placeholder="종목명 또는 6자리 코드", on_change=request_search, label_visibility="collapsed")
        c2.button("정밀 분석", type="primary", on_click=request_search, use_container_width=True)
        c3.button("데이터 갱신", on_click=refresh_selected, use_container_width=True)
        if st.session_state.search_message:
            st.info(st.session_state.search_message)
        if st.session_state.candidates:
            st.caption("분석하실 종목을 선택해 주십시오.")
            for item in st.session_state.candidates:
                st.button(f"{item['Name']} ({item['Code']})", key=f"search_{item['Code']}",
                          on_click=select_stock, args=(item["Code"], item["Name"]))
        code = st.session_state.selected_code
        if not code:
            st.info("종목을 검색하시거나 오른쪽 랭킹에서 선택해 주십시오.")
            return
        with st.spinner("일봉·재무·수급을 조회하고 있습니다…"):
            bundle = cached_bundle(code)
        render_analysis(bundle, code, st.session_state.selected_name)


# ===== 5. 과대낙폭 유망주 탐지기 =====
def oversold_technical_profile(frame):
    """과대낙폭 + 초기 반등 신호만 평가합니다. 기술 예비점수 최대 55점."""
    try:
        completed = completed_history(frame)
        if len(completed) < 260:
            return None
        df = add_running_indicators(completed)
        df["HIGH252"] = df.High.rolling(252, min_periods=200).max()
        df["HIGH120"] = df.High.rolling(120, min_periods=100).max()
        df["DD52"] = (df.Close / df.HIGH252 - 1) * 100
        df["DD120"] = (df.Close / df.HIGH120 - 1) * 100
        df["DIST_MA60"] = (df.Close / df.MA60 - 1) * 100
        df["RSI5_MIN"] = df.RSI.rolling(5, min_periods=3).min()
        ready = df.dropna(subset=["MA120", "VOL_MA20", "HIGH252", "HIGH120", "RSI", "MACD", "MACD_SIGNAL"])
        if len(ready) < 3:
            return None
        r, p = ready.iloc[-1], ready.iloc[-2]
    except Exception:
        return None

    score = 0
    detail = []

    def add(cond, pts, label):
        nonlocal score
        ok = bool(cond)
        if ok:
            score += pts
        detail.append({"영역": "낙폭·반등", "판정": "✅" if ok else "➖", "점수": pts if ok else 0,
                       "조건": label, "근거": "충족" if ok else "미충족"})

    # 낙폭·가격 위치 30점
    add(r.DD52 <= -15, 8, "52주 고점 대비 -15% 이하")
    add(r.DD52 <= -25, 7, "52주 고점 대비 -25% 이하")
    add(r.DD120 <= -12, 5, "120일 고점 대비 -12% 이하")
    add(28 <= r.RSI <= 45, 5, "RSI 28~45의 과매도 탐색 구간")
    add(r.DIST_MA60 <= -5, 5, "60일선 대비 -5% 이하 이격")

    # 반등 모멘텀 25점
    add(r.RET5 > 0, 5, "최근 5거래일 수익률 플러스")
    add(r.MACD_HIST > p.MACD_HIST, 5, "MACD 히스토그램 개선")
    add(r.MACD > r.MACD_SIGNAL, 5, "MACD가 Signal 상회")
    add((r.RSI - r.RSI5_MIN) >= 4, 5, "RSI가 최근 저점에서 4p 이상 반등")
    add((r.Close > p.Close) and (r.VOL_RATIO >= 1.2), 5, "상승일 거래량 20일 평균 1.2배 이상")

    penalties = []
    if r.DD52 <= -55 and r.MA20_SLOPE < 0 and r.MA60_SLOPE < 0:
        score -= 8
        penalties.append("52주 -55% 이하 + 20·60일선 동반 하락 -8")
    if r.RSI < 25 and r.RET5 < 0:
        score -= 4
        penalties.append("RSI 25 미만 + 단기 하락 지속 -4")

    score = max(0, min(55, int(round(score))))
    return {
        "technical_score": score,
        "row": r,
        "df": ready,
        "detail": detail,
        "penalties": penalties,
        "high52": float(r.HIGH252),
        "dd52": float(r.DD52),
        "dd120": float(r.DD120),
    }


def _fundamental_outlook_points(fund, close):
    """실적·컨센서스 최대 30점. 미확인 데이터에는 점수를 부여하지 않습니다."""
    points = 0
    observed = 0
    details = []

    def add_known(known, pts, max_pts, label, evidence):
        nonlocal points, observed
        if known:
            observed += max_pts
            pts = int(pts)
            points += pts
            details.append({"영역": "실적·전망", "판정": "✅" if pts else "➖", "점수": pts,
                            "조건": label, "근거": evidence})
        else:
            details.append({"영역": "실적·전망", "판정": "❔", "점수": 0,
                            "조건": label, "근거": "데이터 미확인"})

    target = number(fund.get("Target"))
    upside = (target / close - 1) * 100 if target is not None and target > 0 and close > 0 else None
    add_known(upside is not None, 8 if upside >= 20 else 4 if upside >= 10 else 0, 8,
              "컨센서스 목표가 상승여력", f"종가 대비 {upside:+.1f}%" if upside is not None else "")

    roe = number(fund.get("ROE"))
    add_known(roe is not None, 5 if roe >= 10 else 2 if roe >= 5 else 0, 5,
              "최근 확정 연간 ROE", f"{roe:.1f}%" if roe is not None else "")

    op_growth = number(fund.get("ForwardOperatingProfitGrowth"))
    op_turn = bool(fund.get("ForwardOperatingProfitTurnaround"))
    op_known = op_growth is not None or op_turn
    op_pts = 8 if op_turn or (op_growth is not None and op_growth >= 10) else 4 if op_growth is not None and op_growth >= 0 else 0
    op_text = "비양수 → 흑자 추정" if op_turn else (f"{fund.get('ForwardPeriod') or ''} {op_growth:+.1f}%" if op_growth is not None else "")
    add_known(op_known, op_pts, 8, "향후 연간 영업이익 컨센서스", op_text)

    eps_growth = number(fund.get("ForwardEPSGrowth"))
    eps_turn = bool(fund.get("ForwardEPSTurnaround"))
    eps_known = eps_growth is not None or eps_turn
    eps_pts = 5 if eps_turn or (eps_growth is not None and eps_growth >= 10) else 3 if eps_growth is not None and eps_growth >= 0 else 0
    eps_text = "비양수 → 양수 추정" if eps_turn else (f"{fund.get('ForwardPeriod') or ''} {eps_growth:+.1f}%" if eps_growth is not None else "")
    add_known(eps_known, eps_pts, 5, "향후 연간 EPS 컨센서스", eps_text)

    per, industry = number(fund.get("PER")), number(fund.get("IndustryPER"))
    comparable = per is not None and industry is not None and per > 0 and industry > 0
    add_known(comparable, 4 if comparable and per <= industry else 0, 4,
              "업종 대비 PER", f"PER {per:.1f}배 / 업종 {industry:.1f}배" if comparable else "")

    return points, observed, details, upside


def _combine_oversold_score(tech, fund=None, investors=None):
    if tech is None:
        return None
    fund = fund or empty_fund()
    r = tech["row"]
    score = tech["technical_score"]
    detail = list(tech["detail"])

    f_points, f_observed, f_details, upside = _fundamental_outlook_points(fund, float(r.Close))
    score += f_points
    detail.extend(f_details)

    supply = _horse_supply_metrics(investors, tech["df"])
    supply_points = 0
    supply_observed = 0

    def supply_add(value, pts, label):
        nonlocal supply_points, supply_observed
        if value is None:
            detail.append({"영역": "수급", "판정": "❔", "점수": 0, "조건": label, "근거": "데이터 미확인"})
            return
        supply_observed += pts
        hit = value > 0
        if hit:
            supply_points += pts
        detail.append({"영역": "수급", "판정": "✅" if hit else "➖", "점수": pts if hit else 0,
                       "조건": label, "근거": _fmt_shares(value)})

    supply_add(supply["foreign5"], 4, "최근 5거래일 외국인 순매수")
    supply_add(supply["institution5"], 4, "최근 5거래일 기관 순매수")
    supply_add(supply["foreign20"], 3, "최근 20거래일 외국인 순매수")
    supply_add(supply["institution20"], 2, "최근 20거래일 기관 순매수")

    frc = supply["foreign_rate_change10"]
    if frc is None:
        detail.append({"영역": "수급", "판정": "❔", "점수": 0, "조건": "10거래일 외국인 보유율 증가", "근거": "데이터 미확인"})
    else:
        supply_observed += 2
        hit = frc > 0
        if hit:
            supply_points += 2
        detail.append({"영역": "수급", "판정": "✅" if hit else "➖", "점수": 2 if hit else 0,
                       "조건": "10거래일 외국인 보유율 증가", "근거": f"{frc:+.2f}%p"})

    score += supply_points
    score = max(0, min(100, int(round(score))))
    coverage = 55 + f_observed + supply_observed
    prev = tech["df"].iloc[-2]
    rebound = bool(r.RET5 > 0 and r.MACD_HIST > prev.MACD_HIST)
    outlook_ok = f_points >= 12
    supply_ok = supply_points >= 6

    if score >= 80 and tech["dd52"] <= -20 and rebound and outlook_ok:
        status = "💎 우선 관찰 — 실적 전망과 반등 신호 동반"
    elif score >= 70 and outlook_ok:
        status = "🟢 과대낙폭 유망 후보 — 펀더멘털 대비 가격 메리트"
    elif score >= 60:
        status = "🟡 관심 — 반등 또는 실적 확증 추가 필요"
    elif score >= 45:
        status = "🟠 신중한 관찰 — 하락 추세 지속 가능"
    else:
        status = "🔴 우선순위 낮음 — 낙폭보다 훼손 가능성 우세"

    return {
        **tech,
        "score": score,
        "status": status,
        "detail": pd.DataFrame(detail),
        "fund": fund,
        "supply": supply,
        "fund_points": f_points,
        "supply_points": supply_points,
        "coverage": coverage,
        "target_upside": upside,
        "rebound": rebound,
        "outlook_ok": outlook_ok,
        "supply_ok": supply_ok,
    }


def oversold_quality_score(frame, fund=None, investors=None):
    return _combine_oversold_score(oversold_technical_profile(frame), fund, investors)


def build_oversold_commentary(result):
    """관측된 낙폭·컨센서스·수급만으로 만드는 리서치 데스크형 코멘트."""
    r, fund, supply = result["row"], result["fund"], result["supply"]
    positives, risks = [], []

    dd = result["dd52"]
    if -50 <= dd <= -15:
        positives.append(f"52주 고점 대비 {dd:.1f}% 조정돼 과거 고점 대비 가격 부담이 상당 부분 완화된 구간입니다.")
    elif dd < -50:
        risks.append(f"52주 고점 대비 {dd:.1f}% 하락해 단순 가격 메리트보다 펀더멘털 훼손 여부를 우선 점검해야 하는 구간입니다.")
    else:
        risks.append(f"52주 고점 대비 낙폭이 {dd:.1f}%로 전형적인 과대낙폭 구간으로 보기에는 조정 폭이 제한적입니다.")

    op_growth = number(fund.get("ForwardOperatingProfitGrowth"))
    eps_growth = number(fund.get("ForwardEPSGrowth"))
    if fund.get("ForwardOperatingProfitTurnaround"):
        positives.append("연간 영업이익 컨센서스가 비양수 구간에서 흑자로 전환되는 추정 구조여서 실적 턴어라운드 모멘텀을 기대할 여지가 있습니다.")
    elif op_growth is not None and op_growth >= 10:
        positives.append(f"향후 연간 영업이익 컨센서스가 직전 확정치 대비 {op_growth:+.1f}% 개선되는 구조로 주가 낙폭과 실적 방향의 괴리가 존재합니다.")
    elif op_growth is not None and op_growth < 0:
        risks.append(f"향후 연간 영업이익 컨센서스가 {op_growth:+.1f}%로 감소 추정돼 낙폭만으로 저평가를 단정하기 어렵습니다.")

    if fund.get("ForwardEPSTurnaround"):
        positives.append("EPS 컨센서스도 양수 전환이 추정돼 이익 정상화 신호가 동반되고 있습니다.")
    elif eps_growth is not None and eps_growth >= 10:
        positives.append(f"향후 EPS 컨센서스가 {eps_growth:+.1f}% 개선 추정돼 이익 모멘텀은 우호적인 편입니다.")
    elif eps_growth is not None and eps_growth < 0:
        risks.append(f"향후 EPS 컨센서스가 {eps_growth:+.1f}%로 둔화돼 밸류에이션 리레이팅의 동력은 제한적일 수 있습니다.")

    upside = result.get("target_upside")
    if upside is not None and upside >= 20:
        positives.append(f"컨센서스 목표주가와 확정 종가의 괴리율이 {upside:+.1f}%로 시장 기대치 대비 가격 여력이 비교적 크게 남아 있습니다.")
    elif upside is not None and upside < 5:
        risks.append(f"컨센서스 목표주가 상승여력이 {upside:+.1f}%에 그쳐 가격 메리트에 대한 증권가 기대는 제한적입니다.")

    prev = result["df"].iloc[-2]
    if r.RET5 > 0 and r.MACD_HIST > prev.MACD_HIST:
        positives.append("최근 5거래일 수익률이 플러스로 전환되고 MACD 히스토그램도 개선돼 낙폭 이후 단기 모멘텀 회복 조짐이 확인됩니다.")
    elif r.MA20_SLOPE < 0 and r.MA60_SLOPE < 0:
        risks.append("20·60일 이동평균선의 기울기가 모두 하락 중이어서 주가는 조정되었지만 추세 전환의 기술적 확증은 아직 부족합니다.")
    else:
        risks.append("반등 신호가 일부 관찰되지만 이동평균선과 모멘텀 지표가 동시에 추세 전환을 확인한 단계는 아닙니다.")

    f5, i5 = supply.get("foreign5"), supply.get("institution5")
    if f5 is not None and i5 is not None:
        if f5 > 0 and i5 > 0:
            positives.append(f"최근 5거래일 외국인({_fmt_shares(f5)})·기관({_fmt_shares(i5)}) 동반 순매수가 확인돼 저가 매수 수급이 유입되고 있습니다.")
        elif f5 < 0 and i5 < 0:
            risks.append("최근 5거래일 외국인과 기관이 모두 순매도여서 가격 하락을 흡수하는 주도 수급은 아직 확인되지 않습니다.")
        else:
            risks.append("외국인과 기관의 수급 방향이 엇갈려 저점 매수의 주체가 명확하게 형성됐다고 보기 어렵습니다.")
    else:
        risks.append("기관·외국인 수급 데이터가 충분하지 않아 저가 매수 주체의 유입 여부는 별도 확인이 필요합니다.")

    if result["score"] >= 80 and result["outlook_ok"] and result["rebound"]:
        view = "가격 낙폭은 충분한 반면 이익 전망과 단기 반등 신호가 함께 개선되는 유형입니다. 다만 과대낙폭주는 추세 전환 전 재차 저점을 시험할 수 있어 20일선 회복 또는 저점 상향 확인 후 접근하는 전략이 합리적입니다."
    elif result["score"] >= 70:
        view = "가격 메리트와 실적 전망 중 상당 부분이 우호적입니다. 낙폭 자체보다 거래량을 동반한 20일선 회복과 외국인·기관 수급 개선을 함께 확인할 필요가 있습니다."
    elif result["score"] >= 55:
        view = "과대낙폭 후보로는 분류되지만 실적·수급·기술적 반등 중 한 축 이상이 아직 부족합니다. 현재는 선취매보다 펀더멘털 훼손이 멈췄는지 확인하는 관찰 구간에 가깝습니다."
    else:
        view = "현재 낙폭은 크지만 가격 하락을 정당화한 요인이 해소됐다는 증거가 부족합니다. 실적 추정치 개선과 추세 반전이 확인될 때까지 우선순위를 낮게 두는 편이 적절합니다."

    return {"view": view, "positives": positives[:5], "risks": risks[:5]}


@st.cache_data(ttl=900, max_entries=256, show_spinner=False)
def cached_oversold_technical(code):
    result = fetch_history(code)
    if result.status == "error" or result.data is None or result.data.empty:
        return None
    return oversold_technical_profile(result.data)


@st.cache_data(ttl=900, max_entries=256, show_spinner=False)
def cached_oversold_fund(code):
    result = fetch_main(code)
    if result.status == "error":
        return empty_fund()
    return result.data.get("fund", empty_fund())


@st.cache_data(ttl=900, max_entries=256, show_spinner=False)
def cached_oversold_score(code):
    history = fetch_history(code)
    if history.status == "error" or history.data is None or history.data.empty:
        return None
    fund = cached_oversold_fund(code)
    investors = cached_horse_investors(code)
    return oversold_quality_score(history.data, fund, investors)


def render_oversold_chart(result, name):
    df = result["df"].tail(180)
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.035,
                        row_heights=[0.52, 0.16, 0.16, 0.16])
    fig.add_trace(go.Candlestick(x=df.index, open=df.Open, high=df.High, low=df.Low,
                                 close=df.Close, name="Price"), row=1, col=1)
    for ma in [20, 60, 120]:
        fig.add_trace(go.Scatter(x=df.index, y=df[f"MA{ma}"], mode="lines", name=f"MA{ma}"), row=1, col=1)
    fig.add_hline(y=result["high52"], line_dash="dot", annotation_text="52주 고점", row=1, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df.Volume, name="Volume"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df.VOL_MA20, mode="lines", name="Vol MA20"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df.RSI, mode="lines", name="RSI14"), row=3, col=1)
    for level in [70, 50, 30]:
        fig.add_hline(y=level, line_dash="dot", row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df.MACD, mode="lines", name="MACD"), row=4, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df.MACD_SIGNAL, mode="lines", name="Signal"), row=4, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df.MACD_HIST, name="Histogram"), row=4, col=1)
    fig.update_layout(template="plotly_dark", title=f"{name} — 반등 프로파일", height=900,
                      xaxis_rangeslider_visible=False, legend={"orientation": "h"},
                      margin={"l": 10, "r": 10, "t": 55, "b": 10},
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)


def _oversold_prefilter_kospi(stocks, deep_count=80):
    """시총·유동성 중심 1차 압축. 실제 낙폭은 일봉 조회 후 계산합니다."""
    if stocks is None or stocks.empty:
        return stocks
    if 'AsOf' in stocks.columns and stocks.Marcap.isna().all():
        # 시장 전체 수집 장애 시 명시된 기본 종목 표본만 정밀 분석합니다.
        return stocks[(stocks.Close > 0) & (stocks.Volume > 0)].head(deep_count).copy()
    df = stocks.copy()
    for col in ["Close", "Chg", "Volume", "Marcap"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Close", "Chg", "Volume", "Marcap"])
    df = df[(df.Close > 0) & (df.Volume > 0) & (df.Marcap > 0)].copy()
    if df.empty:
        return df
    df["AmountEstimate"] = df.Close * df.Volume
    turn_rank = df.AmountEstimate.rank(pct=True)
    cap_rank = df.Marcap.rank(pct=True)
    liquid = df[(turn_rank >= 0.35) | (cap_rank >= 0.70)].copy()
    liquid["PreScore"] = (
        45 * liquid.Marcap.rank(pct=True)
        + 40 * liquid.AmountEstimate.rank(pct=True)
        + 15 * ((-liquid.Chg).clip(lower=-5, upper=10) + 5) / 15
    )
    parts = [
        liquid.sort_values("PreScore", ascending=False).head(deep_count),
        liquid.sort_values("Marcap", ascending=False).head(max(30, deep_count // 2)),
        liquid.sort_values("AmountEstimate", ascending=False).head(max(30, deep_count // 2)),
        liquid.sort_values("Chg", ascending=True).head(max(20, deep_count // 3)),
    ]
    merged = pd.concat(parts, ignore_index=True).drop_duplicates("Code")
    return merged.sort_values("PreScore", ascending=False).head(deep_count)


def _deep_scan_oversold_candidates(candidates, workers=6):
    if candidates is None or candidates.empty:
        return []
    records = candidates.to_dict("records")

    def analyze(rec):
        code = str(rec["Code"])
        tech = cached_oversold_technical(code)
        if tech is None:
            return None
        r = tech["row"]
        return {
            "종목명": rec.get("Name", code), "코드": code,
            "기술예비점수": tech["technical_score"], "종가": round(float(r.Close)),
            "52주고점대비(%)": round(float(tech["dd52"]), 2),
            "120일고점대비(%)": round(float(tech["dd120"]), 2),
            "5일수익률(%)": round(float(r.RET5), 2), "20일수익률(%)": round(float(r.RET20), 2),
            "RSI": round(float(r.RSI), 1), "거래량배수": round(float(r.VOL_RATIO), 2),
            "_tech": tech,
        }

    rows = []
    with ThreadPoolExecutor(max_workers=max(2, min(int(workers), 8))) as pool:
        futures = [pool.submit(analyze, rec) for rec in records]
        for future in futures:
            try:
                item = future.result()
                if item is not None:
                    rows.append(item)
            except Exception:
                pass
    return rows


def _finalize_oversold_candidates(raw, final_pool=35, workers=6):
    if raw is None or raw.empty:
        return pd.DataFrame()
    shortlist = raw.sort_values(["기술예비점수", "52주고점대비(%)"], ascending=[False, True]).head(final_pool).copy()
    records = shortlist.to_dict("records")

    def enrich(rec):
        code = str(rec["코드"])
        tech = rec["_tech"]
        fund = cached_oversold_fund(code)
        inv = cached_horse_investors(code)
        result = _combine_oversold_score(tech, fund, inv)
        if result is None:
            return None
        commentary = build_oversold_commentary(result)
        f5, i5 = result["supply"]["foreign5"], result["supply"]["institution5"]
        if f5 is not None and i5 is not None:
            flow = "외인·기관 동반매수" if f5 > 0 and i5 > 0 else "외국인 우위" if f5 > 0 else "기관 우위" if i5 > 0 else "동반매도"
        else:
            flow = "수급 미확인"
        op_growth = number(fund.get("ForwardOperatingProfitGrowth"))
        eps_growth = number(fund.get("ForwardEPSGrowth"))
        op_text = "흑자전환" if fund.get("ForwardOperatingProfitTurnaround") else (f"{op_growth:+.1f}%" if op_growth is not None else "미확인")
        eps_text = "양수전환" if fund.get("ForwardEPSTurnaround") else (f"{eps_growth:+.1f}%" if eps_growth is not None else "미확인")
        return {
            "종목명": rec["종목명"], "코드": code, "점수": result["score"], "상태": result["status"],
            "종가": rec["종가"], "52주고점대비(%)": rec["52주고점대비(%)"], "RSI": rec["RSI"],
            "5일수익률(%)": rec["5일수익률(%)"], "거래량배수": rec["거래량배수"],
            "목표가상승여력(%)": round(result["target_upside"], 1) if result["target_upside"] is not None else np.nan,
            "영업이익전망": op_text, "EPS전망": eps_text,
            "수급": flow, "데이터확보(%)": result["coverage"], "전략 코멘트": commentary["view"],
            "_result": result,
        }

    out = []
    with ThreadPoolExecutor(max_workers=max(2, min(int(workers), 8))) as pool:
        futures = [pool.submit(enrich, rec) for rec in records]
        for future in futures:
            try:
                item = future.result()
                if item is not None:
                    out.append(item)
            except Exception:
                pass
    if not out:
        return pd.DataFrame()
    return pd.DataFrame(out).sort_values(["점수", "52주고점대비(%)"], ascending=[False, True]).head(20).reset_index(drop=True)


def _render_oversold_leaderboard(out):
    if out is None or out.empty:
        return
    st.markdown("#### 🏆 과대낙폭 유망주 순위")
    st.caption("낙폭·반등 모멘텀·실적 컨센서스·외국인/기관 수급을 합산한 후보 순위입니다.")
    top = out.head(5).reset_index(drop=True)
    cols = st.columns(len(top))
    for i, row in top.iterrows():
        rank = i + 1
        with cols[i]:
            target_txt = "미확인" if pd.isna(row["목표가상승여력(%)"]) else f"{row['목표가상승여력(%)']:+.0f}%"
            html = (
                f'<div class="horse-rank-card horse-rank-{rank}">'
                f'<div class="horse-rank-badge">{_horse_rank_badge(rank)}</div>'
                f'<div class="horse-rank-name">{escape(str(row["종목명"]))}</div>'
                f'<div class="horse-rank-score">{int(row["점수"])}<span>/100</span></div>'
                f'<div class="horse-rank-meta">52주 {row["52주고점대비(%)"]:+.1f}% · RSI {row["RSI"]:.1f}</div>'
                f'<div class="horse-rank-meta">목표가 여력 {target_txt}</div>'
                '</div>'
            )
            st.markdown(html, unsafe_allow_html=True)
    st.markdown("##### 📋 전체 순위")
    display = out.drop(columns=["_result"], errors="ignore").copy().reset_index(drop=True)
    display.insert(0, "순위", np.arange(1, len(display) + 1))
    st.dataframe(
        display, hide_index=True, use_container_width=True,
        column_config={
            "순위": st.column_config.NumberColumn("순위", width="small", format="%d위"),
            "점수": st.column_config.ProgressColumn("유망 낙폭 점수", min_value=0, max_value=100, format="%d점"),
            "상태": st.column_config.TextColumn("판정", width="large"),
            "전략 코멘트": st.column_config.TextColumn("전략 코멘트", width="large"),
        },
    )


def render_oversold_hunter(market_result):
    st.markdown("### 💎 과대낙폭 유망주")
    st.caption("단순 낙폭이 아닌 가격 메리트, 반등 모멘텀, 실적 컨센서스와 외국인·기관 수급을 함께 평가해 펀더멘털 대비 과도하게 할인된 후보를 선별합니다.")
    single, scanner, rules = st.tabs(["🔎 개별 분석", "🏆 KOSPI TOP 20", "📐 모델 기준"])

    with single:
        default_query = st.session_state.get("selected_name") or st.session_state.get("selected_code") or "삼성전자"
        c1, c2 = st.columns([4, 1])
        query = c1.text_input("종목명 또는 종목코드", value=default_query, key="oversold_query",
                              placeholder="예: 삼성전자 또는 005930")
        run = c2.button("과대낙폭 분석", key="oversold_run", use_container_width=True, type="primary")
        if run:
            matches = _resolve_horse_query(query, market_result)
            st.session_state.oversold_matches = matches
            if len(matches) == 1:
                st.session_state.oversold_selected_code = matches[0]["Code"]
                st.session_state.oversold_selected_name = matches[0]["Name"]
            elif not matches:
                st.session_state.oversold_selected_code = ""
                st.session_state.oversold_selected_name = ""

        matches = st.session_state.get("oversold_matches", [])
        if run and not matches:
            st.warning("종목을 찾지 못했습니다. 정확한 종목명 또는 6자리 종목코드를 입력해 주십시오.")
        if len(matches) > 1:
            st.caption("검색 결과가 여러 개입니다. 분석할 종목을 선택해 주십시오.")
            labels = [f"{x['Name']} ({x['Code']})" for x in matches]
            chosen = st.selectbox("검색 결과", labels, key="oversold_match_select", label_visibility="collapsed")
            if st.button("선택 종목 분석", key="oversold_match_run", use_container_width=True):
                idx = labels.index(chosen)
                st.session_state.oversold_selected_code = matches[idx]["Code"]
                st.session_state.oversold_selected_name = matches[idx]["Name"]

        code = st.session_state.get("oversold_selected_code", "")
        name = st.session_state.get("oversold_selected_name", code)
        if code:
            with st.spinner(f"{name}의 낙폭·실적·수급을 분석하고 있습니다…"):
                result = cached_oversold_score(code)
            if result is None:
                st.warning("분석 가능한 데이터가 부족하거나 조회에 실패했습니다.")
            else:
                r, fund = result["row"], result["fund"]
                st.markdown(f"#### {escape(str(name))} ({code})")
                boxes = st.columns(6)
                boxes[0].metric("유망 낙폭 점수", f"{result['score']} / 100")
                boxes[1].metric("종가", f"{r.Close:,.0f}원")
                boxes[2].metric("52주 고점 대비", f"{result['dd52']:+.1f}%")
                boxes[3].metric("RSI", f"{r.RSI:.1f}")
                boxes[4].metric("목표가 여력", fmt(result["target_upside"], "%", 1, True))
                boxes[5].metric("데이터 확보", f"{result['coverage']}%")
                st.subheader(result["status"])

                op = number(fund.get("ForwardOperatingProfitGrowth"))
                eps = number(fund.get("ForwardEPSGrowth"))
                op_text = "흑자전환 추정" if fund.get("ForwardOperatingProfitTurnaround") else (f"{op:+.1f}%" if op is not None else "미확인")
                eps_text = "양수전환 추정" if fund.get("ForwardEPSTurnaround") else (f"{eps:+.1f}%" if eps is not None else "미확인")
                st.caption(f"향후 연간 영업이익 {op_text} · EPS {eps_text} · 추정 기준 {fund.get('ForwardPeriod') or '미확인'}")

                commentary = build_oversold_commentary(result)
                st.markdown("##### 🧠 리서치 데스크 코멘트")
                st.info(commentary["view"])
                cpos, crisk = st.columns(2)
                with cpos:
                    st.markdown("**💡 투자 포인트**")
                    if commentary["positives"]:
                        for item in commentary["positives"]:
                            st.markdown(f"- {item}")
                    else:
                        st.caption("현재 확인 가능한 강한 투자 포인트가 제한적입니다.")
                with crisk:
                    st.markdown("**⚠️ 리스크 요인**")
                    if commentary["risks"]:
                        for item in commentary["risks"]:
                            st.markdown(f"- {item}")
                    else:
                        st.caption("주요 위험 신호가 두드러지지 않습니다.")

                render_oversold_chart(result, name)
                st.dataframe(result["detail"], hide_index=True, use_container_width=True)
                if result["penalties"]:
                    st.warning(" / ".join(result["penalties"]))

    with scanner:
        st.markdown("#### 🏆 KOSPI 과대낙폭 유망주 · TOP 20")
        st.caption("KOSPI를 1차 경량 압축한 뒤 일봉 낙폭/반등을 분석하고, 상위 후보에만 실적 컨센서스와 외국인·기관 수급을 붙여 부하를 줄입니다.")
        c1, c2, c3 = st.columns(3)
        deep_count = c1.slider("일봉 정밀 후보 수", 50, 120, 80, 10, key="oversold_deep_count",
                               help="클수록 시장 커버리지는 넓어지지만 일봉 조회 시간이 증가합니다.")
        final_pool = c2.slider("실적·수급 정밀 후보", 20, 50, 35, 5, key="oversold_final_pool",
                               help="이 단계에서 컨센서스와 외국인·기관 데이터를 추가 조회합니다.")
        pages = c3.slider("KOSPI 시장 페이지", 10, 25, 20, 5, key="oversold_kospi_pages")

        if st.button("KOSPI 리바운드 TOP 20 분석", type="primary", key="oversold_auto_top20_btn", use_container_width=True):
            with st.spinner("1단계: KOSPI 후보군을 수집하고 있습니다…"):
                universe = cached_horse_kospi_universe(pages)
            if universe.data.empty:
                st.warning("KOSPI 후보군을 확보하지 못했습니다.")
            else:
                candidates = _oversold_prefilter_kospi(universe.data, deep_count)
                progress = st.progress(10)
                st.caption(f"1차 수집 {len(universe.data)}종목 → 일봉 정밀 후보 {len(candidates)}종목")
                with st.spinner(f"2단계: {len(candidates)}종목의 52주 낙폭과 반등 모멘텀을 분석하고 있습니다…"):
                    rows = _deep_scan_oversold_candidates(candidates, workers=6)
                progress.progress(65)
                if not rows:
                    progress.empty()
                    st.warning("일봉 정밀 분석 결과를 확보하지 못했습니다.")
                else:
                    raw = pd.DataFrame(rows)
                    with st.spinner(f"3단계: 상위 {min(final_pool, len(raw))}개 후보의 실적 컨센서스와 수급을 분석하고 있습니다…"):
                        top20 = _finalize_oversold_candidates(raw, final_pool=final_pool, workers=6)
                    progress.progress(100)
                    progress.empty()
                    st.session_state.oversold_auto_raw = raw
                    st.session_state.oversold_auto_top20_results = top20
                    st.session_state.oversold_auto_meta = {
                        "universe": len(universe.data), "deep": len(candidates),
                        "final_pool": min(final_pool, len(raw)), "pages": pages,
                    }

        top20 = st.session_state.get("oversold_auto_top20_results")
        meta = st.session_state.get("oversold_auto_meta") or {}
        if isinstance(top20, pd.DataFrame) and not top20.empty:
            st.success(
                f"KOSPI {meta.get('universe', 0)}종목 1차 탐색 → {meta.get('deep', 0)}종목 일봉 분석 → "
                f"{meta.get('final_pool', 0)}종목 실적·수급 분석 → TOP {len(top20)}"
            )
            _render_oversold_leaderboard(top20)
            st.markdown("##### 📋 TOP 20 종목별 전략 코멘트")
            for idx, row in top20.reset_index(drop=True).iterrows():
                rank = idx + 1
                with st.expander(
                    f"{_horse_rank_badge(rank)} {rank}위 · {row['종목명']} ({row['코드']}) · "
                    f"{int(row['점수'])}점 · 52주 {row['52주고점대비(%)']:+.1f}% · {row['수급']}"
                ):
                    result = row["_result"]
                    commentary = build_oversold_commentary(result)
                    st.info(commentary["view"])
                    cpos, crisk = st.columns(2)
                    with cpos:
                        st.markdown("**💡 투자 포인트**")
                        for item in commentary["positives"]:
                            st.markdown(f"- {item}")
                    with crisk:
                        st.markdown("**⚠️ 리스크 요인**")
                        for item in commentary["risks"]:
                            st.markdown(f"- {item}")
                    fund = result["fund"]
                    st.caption(
                        f"목표가 여력 {fmt(result['target_upside'], '%', 1, True)} · "
                        f"최근 확정 ROE {fmt(fund.get('ROE'), '%', 1)} · 데이터 확보 {result['coverage']}%"
                    )

            export = top20.drop(columns=["_result"], errors="ignore").copy()
            st.download_button(
                "KOSPI 과대낙폭 TOP20 CSV",
                export.to_csv(index=False).encode("utf-8-sig"),
                file_name="oversold_quality_KOSPI_TOP20.csv", mime="text/csv", use_container_width=True,
            )
            st.caption("※ 점수는 미래 수익률 예측값이 아닙니다. 실적 추정치 하향이 지속되는 종목은 낙폭이 커도 가치함정이 될 수 있어 추정치·반등·수급을 함께 평가합니다.")

    with rules:
        st.markdown("""
**낙폭·가격 위치 30점** — 52주/120일 고점 대비 하락폭, RSI 28~45, 60일선 하방 이격  
**반등 모멘텀 25점** — 5일 수익률, MACD 개선/Signal 상회, RSI 저점 반등, 상승일 거래량 증가  
**실적·전망 30점** — 컨센서스 목표가 여력, 확정 ROE, 향후 영업이익/EPS 추정 개선, 업종 대비 PER  
**수급 15점** — 외국인·기관 5/20거래일 순매수, 외국인 보유율 증가  
**가치함정 패널티** — 52주 -55% 이하이면서 20·60일선이 동반 하락하거나 RSI 25 미만에서 단기 하락이 지속되는 경우 감점

점수가 높더라도 '많이 빠졌으니 오른다'는 의미가 아닙니다. **실적 추정치 방향 + 저점 상향 + 거래량을 동반한 20일선 회복**을 최종 확인 신호로 보십시오.
""")

# ===== 6. 한국/미국 독립 종목 검색 및 공급원 장애 복구 =====
# 가격이 없는 내장 목록은 이름 검색만 돕습니다. 시세나 랭킹에 사용하지 않습니다.
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
EXCHANGE_TZ = {'KR': KST, 'US': ZoneInfo('America/New_York')}

# 자식 프로세스로 외부 라이브러리의 무기한 대기를 제한합니다.
_DATA_WORKER = r'''
import sys, json, contextlib
import FinanceDataReader as fdr
with contextlib.redirect_stdout(sys.stderr):
    df = fdr.StockListing(sys.argv[2]) if sys.argv[1] == 'listing' else fdr.DataReader(sys.argv[2], sys.argv[3], sys.argv[4])
print(df.to_json(orient='split', date_format='iso'))
'''


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


@st.cache_data(ttl=86400, max_entries=2, show_spinner=False)
def stock_directory(region):
    seeds = pd.DataFrame(SEED_KR if region == 'KR' else SEED_US, columns=['Code', 'Name'])
    frames, notes = [], []
    if region == 'KR':
        # KIND 상장회사 목록은 네이버 시장 랭킹과 독립된 이름/코드 공급원입니다.
        try:
            body = request_bytes('https://kind.krx.co.kr/corpgeneral/corpList.do',
                                 {'method': 'download', 'searchType': '13'})
            frame = pd.read_html(StringIO(body.decode('euc-kr', errors='replace')), converters={'종목코드': str})[0]
            frames.append(normalize_directory(frame, region))
            notes.append('KRX KIND 상장회사 목록')
        except Exception:
            try:
                frames.append(normalize_directory(fdr_table('listing', 'KRX-DESC'), region))
                notes.append('FinanceDataReader KRX-DESC')
            except Exception:
                notes.append('전체 한국 종목 목록 수집 실패')
    else:
        for filename, symbol_col in [('nasdaqlisted.txt', 'Symbol'), ('otherlisted.txt', 'ACT Symbol')]:
            try:
                body = request_bytes('https://www.nasdaqtrader.com/dynamic/SymDir/' + filename)
                df = pd.read_csv(StringIO(body.decode('utf-8-sig')), sep='|', dtype=str)
                df = df[df['Test Issue'].eq('N')]
                if 'ETF' in df:
                    df = df[df.ETF.eq('N')]
                if 'Exchange' in df:
                    df = df[df.Exchange.eq('N')]  # NYSE only; NASDAQ comes from first file.
                df = df.rename(columns={symbol_col: 'Code', 'Security Name': 'Name'})
                frames.append(normalize_directory(df, region))
                notes.append(filename)
            except Exception:
                notes.append(filename + ' 수집 실패')
        if not frames:
            for exchange in ['NASDAQ', 'NYSE']:
                try:
                    frames.append(normalize_directory(fdr_table('listing', exchange), region))
                    notes.append('FinanceDataReader ' + exchange)
                except Exception:
                    notes.append(exchange + ' 대체 목록 실패')
    count = sum(len(f) for f in frames)
    # 한글 별칭은 원래 회사명에 덧붙여 영문 검색도 보존합니다.
    df = pd.concat(frames + [seeds], ignore_index=True).drop_duplicates('Code')
    aliases = dict(seeds.values)
    df['Name'] = [str(n) + (' / ' + aliases[c] if c in aliases and aliases[c] != n else '') for c, n in zip(df.Code, df.Name)]
    return Result(df[['Code', 'Name']], ' · '.join(notes), 'ok' if count and not any('실패' in n for n in notes) else 'partial',
                  [f'검색 전용 목록 {len(df):,}개 · 가격 없음', '전체 목록 실패 시 내장 이름 별칭과 직접 코드/티커 입력을 지원합니다.'])




def local_search(query, stocks):
    # 세 한국 탭의 검색이 시장 스냅샷 실패와 무관하게 동작합니다.
    directory = stock_directory('KR').data
    merged = pd.concat([stocks[['Code', 'Name']], directory], ignore_index=True).drop_duplicates('Code')
    return _original_local_search(query, merged)


def us_search(query, directory):
    q = query.strip()
    if not q:
        return directory.iloc[:0]
    codes = directory.Code.str.upper()
    exact = codes.eq(q.upper())
    names = directory.Name.str.contains(q, case=False, regex=False, na=False)
    matches = directory[exact | codes.str.startswith(q.upper()) | names]
    return pd.concat([directory[exact], matches]).drop_duplicates('Code').head(15)


@st.cache_data(ttl=300, max_entries=2, show_spinner=False)
def kr_market_backup():
    try:
        df = normalize_directory(fdr_table('listing', 'KRX'), 'KR')
        df = df.rename(columns={'ChagesRatio': 'Chg', 'ChangeRatio': 'Chg'})
        if not set(MARKET_COLS).issubset(df):
            raise ValueError('KRX 시세 열 부족')
        for col in ['Close', 'Chg', 'Volume', 'Marcap']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=['Close', 'Chg', 'Volume', 'Marcap'])
        df = df[(df.Close > 0) & (df.Volume >= 0) & (df.Marcap > 0)]
        if df.empty:
            raise ValueError('KRX 시세 없음')
        return Result(df[MARKET_COLS], 'FinanceDataReader / KRX', notes=[f'KRX 시장 스냅샷 {len(df):,}종목 · 체결 기준시각 미제공'])
    except Exception as exc:
        return failure('FinanceDataReader / KRX', pd.DataFrame(columns=MARKET_COLS), exc)




def fetch_market(pages_per_market=2):
    primary = _naver_market(pages_per_market)
    if primary.status == 'ok':
        return primary
    backup = kr_market_backup()
    if not backup.data.empty:
        backup.notes += ['네이버 수집 장애로 KRX 대체 공급원 사용']
        return backup
    primary.notes += ['KRX 대체 공급원도 조회 실패. 이름 검색과 개별 일봉 조회는 별도로 이용할 수 있습니다.']
    return primary




@st.cache_data(ttl=600, max_entries=4, show_spinner=False)
def cached_horse_kospi_universe(max_pages=20):
    # 20페이지의 실패 요청을 반복하기 전에 현재 시장과 KRX 대체 공급원을 확인합니다.
    market = cached_market()
    if market.source == 'FinanceDataReader / KRX':
        df = market.data[market.data.Market.eq('KOSPI')].copy()
        return Result(df, market.source, notes=[f'KRX KOSPI {len(df)}종목'])
    if market.data.empty:
        return cached_kr_daily_sample()
    return _naver_horse_universe(max_pages)


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


@st.cache_data(ttl=600, max_entries=128, show_spinner=False)
def cached_us_history(code):
    return fetch_history(code, region='US')


@st.cache_data(ttl=900, max_entries=64, show_spinner=False)
def cached_us_news(code):
    source = 'https://news.google.com/rss/search'
    try:
        body = request_bytes(source, {'q': f'"{code}" stock', 'hl': 'en-US', 'gl': 'US', 'ceid': 'US:en'})
        return Result(parse_news(body), source)
    except Exception as exc:
        return failure(source, [], exc)


def us_profile(code):
    result = cached_us_history(code)
    if result.status == 'error':
        return {'Code': code, 'error': ' / '.join(result.notes)}
    try:
        df = add_indicators(completed_history(result.data))
        if len(df) < 61:
            raise ValueError('최소 61개 확정 일봉이 필요합니다.')
        horse = running_horse_score(result.data)
        oversold = oversold_technical_profile(result.data)
        score = evaluate_technical_score(df)
        return {'Code': code, 'df': df, 'horse': horse, 'oversold': oversold, 'score': score,
                'source': result.source, 'fetched_at': result.fetched_at}
    except Exception as exc:
        return {'Code': code, 'error': str(exc)}


@st.cache_data(ttl=1800, max_entries=1, show_spinner=False)
def cached_kr_daily_sample():
    """Verified completed bars only. Missing market caps stay missing."""
    def get(pair):
        code, name = pair
        result = fetch_history(code)
        if result.status == 'error':
            return None
        df = completed_history(result.data)
        if len(df) < 2 or (datetime.now(KST).date() - df.index[-1].date()).days > 7:
            return None
        return {'Code': code, 'Name': name, 'Market': 'KOSPI',
                'Close': float(df.Close.iloc[-1]), 'Chg': period_return(df, 1),
                'Volume': float(df.Volume.iloc[-1]), 'Marcap': np.nan,
                'AsOf': df.index[-1], 'AmountEstimate': float(df.Close.iloc[-1] * df.Volume.iloc[-1])}
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = [row for row in pool.map(get, SEED_KR) if row is not None]
    df = pd.DataFrame(rows)
    if not df.empty:
        # 서로 다른 날짜를 같은 일간 순위로 섞지 않습니다.
        df = df[df.AsOf.eq(df.AsOf.max())].reset_index(drop=True)
    else:
        df = pd.DataFrame(columns=MARKET_COLS + ['AsOf', 'AmountEstimate'])
    return Result(df, '기본 한국 종목의 확정 일봉 표본', 'partial' if len(df) else 'error',
                  [f'기본 {len(SEED_KR)}종목 중 동일 기준일 {len(df)}종목 · 전체 시장/당일 실시간 순위가 아닙니다.'])


def refresh_market_sources():
    cached_market.clear()
    kr_market_backup.clear()
    cached_kr_daily_sample.clear()
    cached_horse_kospi_universe.clear()


def render_kr_daily_sample():
    st.info('시장 전체 시세를 조회하지 못했습니다. 종목명 검색과 개별 분석은 계속 이용할 수 있습니다.')
    if st.button('확정 일봉으로 기본 표본 흐름 조회', key='load_kr_daily_sample'):
        st.session_state.show_kr_daily_sample = True
    if not st.session_state.get('show_kr_daily_sample'):
        return
    with st.spinner('기본 종목의 확정 일봉을 조회하고 있습니다…'):
        result = cached_kr_daily_sample()
    st.caption(result.notes[0])
    if result.data.empty:
        st.warning('개별 일봉 표본도 수집하지 못했습니다. 잠시 후 새로고침해 주십시오.')
        return
    st.caption(f"확정 일봉 기준: {result.data.AsOf.max():%Y-%m-%d} · 시가총액 주목도 점수는 산출하지 않습니다.")
    for row in result.data.sort_values('Chg', ascending=False).itertuples():
        st.button(f'{row.Name} · {row.Chg:+.2f}%', key='daily_sample_' + row.Code,
                  on_click=select_stock, args=(row.Code, row.Name), use_container_width=True)
        st.caption(f'{row.Close:,.0f}원 · 종가×거래량 추정 {row.AmountEstimate/1e8:,.1f}억원')


def render_us_workspace():
    st.caption('NASDAQ · NYSE | USD | 뉴욕 현지 당일 봉 제외 | 한국/미국 점수는 데이터 범위가 다릅니다.')
    directory = stock_directory('US')
    if directory.status != 'ok':
        st.info('미국 종목 목록의 일부 또는 전체를 수집하지 못했습니다. 기본 별칭 검색 또는 티커 직접 입력을 이용해 주십시오.')
    query = st.text_input('미국 종목 검색', placeholder='NVDA, 애플, Tesla, Palantir', key='us_query')
    matches = us_search(query, directory.data)
    options = {row.Code: row.Name for row in matches.itertuples()}
    raw = query.strip().upper()
    if re.fullmatch(r'[A-Z]{1,6}(?:[.-][A-Z]{1,2})?', raw):
        options.setdefault(raw, '직접 티커 조회')
    if options:
        chosen = st.selectbox('분석 종목', list(options), format_func=lambda c: f'{c} · {options[c]}', key='us_choice')
        if st.button('미국 종목 정밀 분석', type='primary'):
            st.session_state.us_selected = chosen
    elif query:
        st.info('검색 결과가 없습니다. 회사의 미국 상장 티커를 입력해 주십시오.')
    if st.button('미국 데이터 새로고침'):
        cached_us_history.clear()
        cached_us_news.clear()
        stock_directory.clear('US')
        st.session_state.pop('us_scan_horse', None)
        st.session_state.pop('us_scan_oversold', None)
        st.rerun()
    main_tab, horse_tab, over_tab = st.tabs(['📊 종합 분석', '🐎 달리는 말 탐지기', '💎 과대낙폭 유망주'])
    selected = st.session_state.get('us_selected', '')
    profile = None
    if selected:
        with st.spinner(f'{selected} 미국 일봉 조회 중…'):
            profile = us_profile(selected)
    with main_tab:
        if not profile:
            st.info('종목을 검색하고 정밀 분석을 눌러 주십시오.')
        elif 'error' in profile:
            st.error(profile['error'])
        else:
            df, score = profile['df'], profile['score']
            st.subheader(selected + ' · USD')
            st.caption(f"일봉 기준 {df.index[-1]:%Y-%m-%d} · {profile['source']} · 조회 {profile['fetched_at']}")
            if (datetime.now(EXCHANGE_TZ['US']).date() - df.index[-1].date()).days > 7:
                st.warning('최근 일봉이 7일 이상 경과했습니다. 거래정지 또는 수집 지연을 확인해 주십시오.')
            cols = st.columns(4)
            cols[0].metric('확정 종가', '$' + fmt(df.Close.iloc[-1], digits=2))
            cols[1].metric('5거래일', fmt(period_return(df, 5), '%', 2, True))
            cols[2].metric('20거래일', fmt(period_return(df, 20), '%', 2, True))
            cols[3].metric('RSI', fmt(df.RSI.iloc[-1], digits=1))
            st.metric('기술 점수 (100점 기준)', f"{score['points']} / 확인 배점 {score['possible']}")
            st.write(score['grade'])
            st.caption('추세 35 · 모멘텀 25 · 거래량 20 · 진입 부담 20. 한국과 같은 기술 규칙입니다. 기관 수급·재무 점수는 포함하지 않으며, 승률이나 상승 확률을 의미하지 않습니다.')
            scenario = price_scenario(df)
            if scenario:
                cols = st.columns(5)
                for box, label in zip(cols, ["1차 참고 진입가", "2차 참고 진입가", "1차 참고 목표가", "2차 참고 목표가", "참고 손절가"]):
                    box.metric(label, '$' + fmt(scenario[label], digits=2))
                st.caption('ATR 변동성으로 계산한 가격 시나리오입니다. 애널리스트 목표가나 주문 가격이 아닙니다.')
            research = build_general_research_commentary(df, pd.DataFrame(columns=INV_COLS), {}, score, None,
                stale=(datetime.now(EXCHANGE_TZ['US']).date()-df.index[-1].date()).days>7)
            render_research_cards(research)
            with st.expander('긍정 요인과 유의 사항'):
                for item in research['positives']:
                    st.write('• ' + item)
                for item in research['risks']:
                    st.write('• ' + item)
            render_chart(df)
            st.bar_chart(df.Volume.tail(100), height=150)
            with st.expander('채점 근거'):
                st.dataframe(score['logs'], hide_index=True, use_container_width=True)
            with st.expander('퀀트 백테스트'):
                render_backtest(df)
            with st.expander('뉴스 브리핑'):
                news = cached_us_news(selected)
                if not news.data:
                    st.caption('뉴스를 수집하지 못했습니다.')
                else:
                    for item in news.data:
                        st.markdown(f"- [{item.get('Title', item.get('title', '뉴스'))}]({item.get('Link', item.get('link', '#'))})")
    with horse_tab:
        st.subheader('🐎 미국 달리는 말 탐지기')
        if profile and 'error' not in profile:
            horse = profile['horse']
            if horse:
                st.metric(selected + ' 기술 모멘텀', f"{horse['score']} / 100")
                st.write(horse['status'])
                st.info(f"20일선 대비 이격은 {horse['row'].DIST_MA20:+.1f}%, 거래량은 20일 평균의 {horse['row'].VOL_RATIO:.2f}배입니다. 돌파가 이어지려면 가격 상승에 거래량이 동반되고, 조정 시 주요 이동평균선의 지지가 유지되는지 확인하실 필요가 있습니다.")
                st.dataframe(horse['detail'], hide_index=True)
            else:
                st.info('모멘텀 탐지에는 최소 130개 확정 일봉이 필요합니다.')
        render_us_scan('horse')
    with over_tab:
        st.subheader('💎 미국 과대낙폭 유망주 · 기술 예비 후보')
        st.caption('낙폭·반등 기술점수 최대 55점입니다. 재무 건전성·실적 전망을 검증한 유망주 확정 판정은 제공하지 않습니다.')
        if profile and 'error' not in profile:
            over = profile['oversold']
            if over:
                st.metric(selected + ' 낙폭·반등', f"{over['technical_score']} / 55", f"52주 고점 대비 {over['dd52']:.1f}%", delta_color='off')
                st.info(f"52주 고점 대비 {abs(over['dd52']):.1f}% 조정된 상태이며, RSI는 {over['row'].RSI:.1f}입니다. 낙폭만으로 저평가를 판단하기보다는 최근 저점의 지지, MACD 개선, 20일선 회복이 함께 나타나는지 살펴보시기를 권해드립니다. 이 점수는 기술적 반등 후보 평가이며 기업의 실적 회복을 확인한 결과는 아닙니다.")
                st.dataframe(pd.DataFrame(over['detail']), hide_index=True)
            else:
                st.info('과대낙폭 탐지에는 최소 260개 확정 일봉이 필요합니다.')
        render_us_scan('oversold')


def render_us_scan(mode):
    default = ', '.join(c for c, _ in SEED_US)
    text = st.text_area('스캔할 티커 목록 (최대 30개)', value=default, key='us_scan_input_' + mode)
    st.caption('입력한 표본 안에서만 비교합니다. 미국 시장 전체 순위가 아닙니다. 기본 표본에는 NYSE와 NASDAQ 종목이 함께 있습니다.')
    key = 'us_scan_' + mode
    if st.button('후보 스캔', key='start_' + mode):
        symbols = list(dict.fromkeys(x.upper() for x in re.split(r'[,\s]+', text.strip()) if x))
        if len(symbols) > 30 or any(not re.fullmatch(r'[A-Z]{1,6}(?:[.-][A-Z]{1,2})?', c) for c in symbols):
            st.error('유효한 티커를 30개 이하로 입력해 주십시오.')
            return
        rows, failures = [], []
        progress = st.progress(0, text='미국 후보 조회 중…')
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(us_profile, c): c for c in symbols}
            from concurrent.futures import as_completed
            for i, future in enumerate(as_completed(futures)):
                try:
                    item = future.result()
                    candidate = item.get(mode)
                    if 'error' in item or not candidate:
                        failures.append(futures[future] + ': ' + item.get('error', '지표 준비 일봉 부족'))
                    else:
                        df = item['df']
                        if (datetime.now(EXCHANGE_TZ['US']).date() - df.index[-1].date()).days > 7:
                            raise ValueError('일봉이 7일 이상 경과하여 순위 제외')
                        rows.append({'Ticker': item['Code'], '종가(USD)': round(df.Close.iloc[-1], 2),
                                     '점수': candidate['score'] if mode == 'horse' else candidate['technical_score'],
                                     '기준일': str(df.index[-1].date()), '5일(%)': period_return(df, 5)})
                except Exception as exc:
                    failures.append(futures[future] + ': ' + str(exc))
                progress.progress((i + 1) / len(symbols), text=f'{i+1}/{len(symbols)} 조회')
        st.session_state[key] = (rows, failures, len(symbols))
    if key in st.session_state:
        rows, failures, total = st.session_state[key]
        st.caption(f'요청 {total}개 · 평가 성공 {len(rows)}개 · 제외 {len(failures)}개')
        if rows:
            st.dataframe(pd.DataFrame(rows).sort_values('점수', ascending=False), hide_index=True, use_container_width=True)
        if failures:
            with st.expander('제외 종목과 사유'):
                st.write('\n\n'.join(failures))



# ===== v5.5: 공통 기술 평가와 독립 수급 해석 =====
def evaluate_technical_score(df):
    """Fixed 100-point technical rubric; missing evidence is never rescaled."""
    r, p = df.iloc[-1], df.iloc[-2]
    rows = []
    def add(group, label, weight, value, detail):
        pts = None if value is None else round(float(np.clip(value, 0, weight)), 1)
        rows.append({'영역': group, '항목': label, '배점': weight, '득점': pts,
                     '상태': '미확인' if pts is None else '확인', '근거': detail})
    ready = all(number(r.get(k)) is not None for k in ['Close','MA5','MA20','MA60'])
    add('추세', '이동평균 구조', 15,
        sum([5*(r.Close>r.MA20), 5*(r.MA5>r.MA20), 5*(r.MA20>r.MA60)]) if ready else None,
        '종가>20일선, 5일선>20일선, 20일선>60일선 각각 5점')
    slope = r.MA20/df.MA20.iloc[-6]-1 if len(df)>=26 and number(df.MA20.iloc[-6]) else None
    add('추세','20일선 방향',10, None if slope is None else 10 if slope>0.005 else 7 if slope>0 else 3 if slope>=-0.005 else 0,
        '5거래일 전 대비: +0.5% 초과 10 / 0% 초과 7 / -0.5% 이상 3 / 그 외 0')
    macd = number(r.get('MACD_HIST'))
    add('추세','MACD 방향',10, None if macd is None or number(p.get('MACD_HIST')) is None else
        (6 if macd>0 else 0)+(4 if macd>p.MACD_HIST else 0), '히스토그램 양수 6점 + 전일 대비 개선 4점')
    rsi = number(r.get('RSI'))
    add('모멘텀','RSI 위치',10,None if rsi is None else 10 if 50<=rsi<=65 else 7 if 45<=rsi<50 or 65<rsi<=70 else 4 if 35<=rsi<45 or 70<rsi<=75 else 0,
        '50~65:10 / 45~50·65~70:7 / 35~45·70~75:4 / 그 외 0')
    ret = period_return(df,20)
    add('모멘텀','20거래일 수익률',10,None if ret is None else 10 if 3<=ret<=15 else 7 if ret>0 else 3 if ret>=-5 else 0,
        '3~15%:10 / 그 밖의 양수:7 / -5~0%:3 / -5% 미만:0')
    bb = number(r.get('BB_pct'))
    add('모멘텀','볼린저 위치',5,None if bb is None else 5 if .5<=bb<=1 else 3 if .2<=bb<.5 or 1<bb<=1.1 else 0,
        '%b 0.5~1:5 / 0.2~0.5·1~1.1:3 / 그 외 0')
    active = df.Volume.tail(20).sum()>0 and r.Volume>0
    mfi=number(r.get('MFI'))
    add('거래량','MFI',10,None if not active or mfi is None else 10 if 50<=mfi<=75 else 6 if 40<=mfi<50 or 75<mfi<=80 else 2 if 20<=mfi<40 else 0,
        '50~75:10 / 40~50·75~80:6 / 20~40:2 / 그 외 0; 투자자 신원은 구분하지 않음')
    obv = number(r.get('OBV'))
    add('거래량','OBV 흐름',10,None if not active or obv is None else
        5*(obv>df.OBV.tail(20).mean())+5*(obv>df.OBV.iloc[-6]), '20일 평균 상회 5 + 5거래일 전 대비 증가 5')
    dist=(r.Close/r.MA20-1)*100 if ready else None
    add('진입 부담','20일선 이격',10,None if dist is None else 10 if 0<=dist<=5 else 6 if -3<=dist<0 or 5<dist<=10 else 2 if -8<=dist<-3 or 10<dist<=15 else 0,
        '0~5%:10 / -3~0·5~10%:6 / -8~-3·10~15%:2 / 그 외 0')
    atr = number(r.get('ATR14'))
    atr_pct=atr/r.Close*100 if atr is not None and r.Close>0 else None
    add('진입 부담','상대 변동성',10,None if not active or atr_pct is None else 10 if atr_pct<=2 else 7 if atr_pct<=4 else 3 if atr_pct<=6 else 0,
        'ATR/종가 2% 이하:10 / 4% 이하:7 / 6% 이하:3 / 그 외 0; 거래량 0이면 보류')
    logs=pd.DataFrame(rows)
    points=round(sum(x['득점'] for x in rows if x['득점'] is not None),1)
    possible=sum(x['배점'] for x in rows if x['득점'] is not None)
    grade='기술 지표 일부 미확인' if possible<100 else '추세 조건 우호' if points>=75 else '선별 관찰 구간' if points>=55 else '추세 회복 확인 구간' if points>=35 else '보수적 접근 구간'
    return {'points':points,'possible':possible,'coverage':possible,'score':points if possible==100 else None,
            'upper_bound':points+100-possible,'grade':grade,'logs':logs}


def read_investor_csv(content, code, history):
    required=['Code','Date','ForeignNet','InstitutionNet']
    table=pd.read_csv(StringIO(content.decode('utf-8-sig')),dtype={'Code':str})
    if not set(required).issubset(table):
        raise ValueError('Code, Date, ForeignNet, InstitutionNet 열이 필요합니다.')
    table=table[table.Code.astype(str).str.zfill(6).eq(code)].copy()
    if table.empty:
        raise ValueError('선택한 종목코드에 해당하는 수급이 없습니다.')
    table['Date']=pd.to_datetime(table.Date,errors='coerce').dt.normalize()
    if table.Date.isna().any() or table.Date.duplicated().any():
        raise ValueError('날짜 형식 또는 중복 날짜를 확인해 주십시오.')
    for column in ['ForeignNet','InstitutionNet']:
        table[column]=pd.to_numeric(table[column],errors='coerce')
        if not np.isfinite(table[column]).all() or (table[column]%1!=0).any():
            raise ValueError('순매수 수량은 결측 없는 정수(주)여야 합니다.')
    table['ForeignRate']=pd.to_numeric(table.get('ForeignRate',pd.Series(np.nan,index=table.index)),errors='coerce')
    if ((table.ForeignRate.dropna()<0)|(table.ForeignRate.dropna()>100)).any():
        raise ValueError('외국인 보유율은 0~100% 범위여야 합니다.')
    table=table[table.Date.isin(history.index)].copy()
    if table.empty:
        raise ValueError('확정 일봉과 일치하는 거래일이 없습니다.')
    table['Close']=table.Date.map(history.Close)
    table['ForeignAmountEstimate']=table.ForeignNet*table.Close/1e8
    table['InstitutionAmountEstimate']=table.InstitutionNet*table.Close/1e8
    return table[INV_COLS].sort_values('Date')


def assess_investor_flow(investors, df):
    columns=['ForeignNet','InstitutionNet']
    w=investor_window(investors,df,5,columns)
    dates=pd.to_datetime(investors.get('Date',pd.Series(dtype='datetime64[ns]')),errors='coerce')
    available=dates[dates<=df.index[-1]].max() if len(dates) else pd.NaT
    base={'score':None,'label':'수급 확인 대기','foreign':None,'institution':None,'ratio':None,
          'buy_days':None,'asof':available,'detail':[]}
    if w is None or df.Volume.tail(5).sum()<=0:
        base['text']='분석 기준일과 일치하는 최근 5거래일 수급이 충분하지 않아 외국인·기관의 매수 방향은 판단을 유보하겠습니다. 거래량 지표만으로 특정 투자자의 매집을 단정하지 않습니다.'
        return base
    volume=df.Volume.tail(5)
    # 서로 다른 거래 범위·수정주가 자료가 섞인 경우 강한 매수로 해석하지 않습니다.
    bad = (w.ForeignNet.abs()>volume) | (w.InstitutionNet.abs()>volume) | ((w.ForeignNet+w.InstitutionNet).abs()>volume)
    if bad.any():
        base['text']='순매수 수량과 일봉 거래량의 범위가 일치하지 않아 수급 강도 계산을 보류하겠습니다. 자료의 거래시장·수량 단위·기업행사 조정 여부를 먼저 확인할 필요가 있습니다.'
        return base
    foreign,institution=float(w.ForeignNet.sum()),float(w.InstitutionNet.sum())
    total=foreign+institution
    ratio=total/float(df.Volume.tail(5).sum())*100
    days=int(((w.ForeignNet+w.InstitutionNet)>0).sum())
    direction=20 if foreign>0 and institution>0 else 10 if total>0 else 0
    intensity=60 if ratio>=3 else 40 if ratio>=1 else 20 if ratio>=.2 else 5 if ratio>0 else 0
    persistence=days*4
    points=direction+intensity+persistence
    label='동반 매수 우위' if foreign>0 and institution>0 else '엇갈린 매매' if foreign*institution<0 else '합산 매수 우위' if total>0 else '동반 매도 우위' if foreign<0 and institution<0 else '방향성 제한'
    text=f'최근 5거래일 외국인은 {foreign:+,.0f}주, 기관은 {institution:+,.0f}주 순매수하여 {label}가 관측됩니다. 두 주체의 합산 순매수는 같은 기간 거래량의 {ratio:+.2f}%이며, 합산 순매수일은 5일 중 {days}일입니다.'
    w20=investor_window(investors,df,20,columns)
    if w20 is not None:
        total20=float(w20[columns].sum().sum())
        text+=f' 20거래일 합산 순매수는 {total20:+,.0f}주로, '+('단기·중기 방향이 일치합니다.' if total*total20>0 else '최근 흐름과 중기 방향을 함께 점검할 필요가 있습니다.')
    else:
        text+=' 20거래일 수급의 지속성은 자료가 부족하여 추가 확인이 필요합니다.'
    return dict(base,score=points,label=label,foreign=foreign,institution=institution,ratio=ratio,buy_days=days,text=text,
                detail=[{'항목':'매수 주체 방향','득점':direction,'배점':20,'기준':'동반 양수 20 / 합산 양수 10 / 그 외 0'},
                        {'항목':'거래량 대비 합산 순매수','득점':intensity,'배점':60,'기준':'3% 이상 60 / 1% 이상 40 / 0.2% 이상 20 / 양수 5 / 그 외 0'},
                        {'항목':'매수 지속성','득점':persistence,'배점':20,'기준':'최근 5일 합산 순매수일당 4점'}])


def build_general_research_commentary(df, investors, fund, score, news_result, short=None, stale=False):
    r,p=df.iloc[-1],df.iloc[-2]
    dist20=(r.Close/r.MA20-1)*100
    dist60=(r.Close/r.MA60-1)*100
    aligned=r.MA5>r.MA20>r.MA60
    support=r.Close>r.MA20
    trend='정배열 유지' if aligned and support else '단기 회복 시도' if support else '추세 회복 대기'
    improving=r.MACD_HIST>p.MACD_HIST
    momentum='개선 흐름' if improving else '탄력 둔화'
    trend_text=(f'5일·20일·60일 이동평균선이 정배열을 유지하고 있으며, 종가는 20일선보다 {dist20:.1f}% 높은 수준입니다.' if aligned and support else
                f'종가가 20일선을 {dist20:.1f}% 웃돌며 단기 회복을 시도하고 있습니다. 중기 상승 추세로 판단하려면 이동평균선의 정배열 전환이 함께 확인되어야 합니다.' if support else
                f'종가가 20일선을 {abs(dist20):.1f}% 밑돌고 있어, 현재는 상승 추세의 재개보다 지지력과 회복 신호를 확인할 구간으로 판단됩니다.')
    momentum_text=f'RSI는 {r.RSI:.1f}이며, MACD 히스토그램은 전일 대비 '+('개선되고 있습니다.' if improving else '둔화되고 있습니다.')
    flow=assess_investor_flow(investors,df)
    us=df.attrs.get('region')=='US'
    if us:
        vol_ratio=r.Volume/df.Volume.iloc[-21:-1].mean() if df.Volume.iloc[-21:-1].mean()>0 else None
        flow_label='거래 활발' if vol_ratio is not None and vol_ratio>=1.2 else '거래량 보통' if vol_ratio is not None else '거래량 미확인'
        flow_text=(f'최근 거래량은 직전 20거래일 평균의 {vol_ratio:.2f}배입니다. MFI {r.MFI:.1f}와 OBV를 함께 보면 가격 움직임에 거래가 동반되는지 살펴보실 수 있습니다. ' if vol_ratio is not None else '최근 거래량을 확인하기 어렵습니다. ')+ '해당 지표는 기관 매수의 직접적인 증거는 아닙니다.'
    else:
        flow_label,flow_text=flow['label'],flow['text']
    if stale:
        action='최근 가격 자료가 지연되어 신규 진입에 관한 판단은 보류하겠습니다. 최신 거래일 자료를 확보한 뒤 다시 평가하시는 편이 적절합니다.'
    elif score['score'] is None:
        action='기술 지표 일부가 부족하여 진입 판단은 유보하겠습니다. 우선 데이터 확보 후 추세와 거래량의 일치 여부를 확인하시기를 권해드립니다.'
    elif aligned and support and 0<=dist20<=6 and 45<=r.RSI<=70 and score['points']>=55 and (us or flow['score'] is None or flow['foreign']+flow['institution']>=0):
        action=('정배열과 외국인·기관의 동반 순매수가 함께 확인되어 관심 종목으로 우선 검토하실 만합니다. 20일선 부근의 지지와 매수 지속성을 확인한 뒤 분할 접근을 고려하시는 전략이 적절합니다.'
                if not us and flow['score'] is not None and flow['foreign']>0 and flow['institution']>0 and flow['buy_days']>=3 and flow['ratio']>=1 else
                '추세와 가격 이격을 고려하면 관심 종목으로 검토하실 만합니다. 조정 과정에서 20일선 지지와 거래량 회복을 확인한 뒤 분할 접근을 고려하시는 편이 적절합니다.')
    elif dist20>10 or r.RSI>75:
        action='상승 탄력이 이어지더라도 단기 진입 부담이 커진 구간입니다. 추격 매수보다는 가격 이격이 줄어드는 조정을 기다리며 지지력을 확인하시기를 권해드립니다.'
    elif not support and improving:
        action='단기 반등의 단서는 나타나고 있으나 추세 전환이 확인된 단계는 아닙니다. 20일선 회복과 후속 거래량을 확인한 뒤 접근 여부를 판단하시는 편이 적절합니다.'
    else:
        action='현재는 신규 비중 확대보다 추세 회복을 확인하는 접근을 권해드립니다. 20일선 회복 여부와 모멘텀 개선이 함께 나타나는지 관찰하실 필요가 있습니다.'
    if not us and flow['score'] is not None and flow['foreign']<0 and flow['institution']<0:
        action+=' 외국인과 기관의 동반 매도가 이어지고 있어, 기술적 반등만으로 비중을 늘리는 데에는 신중한 접근이 필요합니다.'
    positives=[];risks=[]
    (positives if support else risks).append(trend_text)
    (positives if improving else risks).append(momentum_text)
    if not us:
        (positives if flow['score'] is not None and flow['score']>=60 else risks).append(flow_text)
    else:
        positives.append(flow_text)
    if abs(dist20)>10: risks.append(f'20일선 대비 이격은 {dist20:+.1f}%로, 평소보다 가격 변동과 진입 위치를 세심하게 점검할 필요가 있습니다.')
    logs=score['logs']
    earned=logs[logs['득점'].fillna(0)>0].sort_values('득점',ascending=False)
    limited=logs[logs['득점'].isna() | (logs['득점']<logs['배점'])]
    headlines=[item.get('title','') for item in (news_result.data if news_result is not None else [])][:3]
    return {'view':trend_text+' '+momentum_text+' '+action,'action':action,'flow_text':flow_text,'flow':flow,
            'positives':positives,'risks':risks,'trend_label':trend,'flow_label':flow_label,'momentum_label':momentum,
            'news_label':'기사 확인' if headlines else '자료 대기','headlines':headlines,
            'dist20':dist20,'dist60':dist60,
            'contributors':[f"{x['항목']}: {x['득점']:g}/{x['배점']}점 — {x['근거']}" for x in earned.to_dict('records')],
            'limiters':[f"{x['항목']}: {'미확인' if pd.isna(x['득점']) else format(x['득점'],'g')} / {x['배점']}점 — {x['근거']}" for x in limited.to_dict('records')]}


def render_research_cards(research):
    st.markdown('#### 🧠 종합 리서치 코멘트')
    st.info(research['view'])
    cols=st.columns(3)
    for box,label,key in zip(cols,['추세','수급·거래량','모멘텀'],['trend_label','flow_label','momentum_label']):
        box.metric(label,research[key])
    st.markdown('**수급·거래량 해석**')
    st.write(research['flow_text'])
    st.caption('지표에 따른 조건부 해석입니다. 수익 확률이나 매수 적합성이 검증된 추천 모델은 아닙니다.')



# ===== 개인용 주식 터미널 V2 (확정 일봉, 공개 데이터 기반) =====
V2_MODEL = 'precursor-2.0'
V2_MENU = ['📊 종목 분석', '🧭 시장 상황판', '🚨 급등 전조', '🔥 섹터 순환',
           '🐋 수급 추적', '📰 뉴스·공시', '💼 내 종목', '🧪 백테스트·성과']
V2_SECTORS = {
    'KR': {'반도체': ['005930','000660','042700'], '전력기기': ['010120','267260','298040'],
           '바이오': ['068270','207940','326030'], '방산': ['012450','064350','047810'],
           '2차전지': ['373220','006400','051910']},
    'US': {'반도체': ['NVDA','AMD','AVGO','INTC'], '대형 기술주': ['AAPL','MSFT','GOOGL','META'],
           '헬스케어': ['LLY','JNJ','UNH'], '에너지': ['XOM','CVX'], '보안': ['CRWD','PANW','NET']},
}


def v2_symbol(code, region):
    code = str(code).strip().upper()
    return bool(re.fullmatch(r'\d{6}',code) if region=='KR' else re.fullmatch(r'[A-Z]{1,6}(?:[.-][A-Z]{1,2})?',code))


def v2_init():
    for key, value in {'v2_portfolio': [], 'v2_signals': [], 'v2_schedule_seen': [], 'v2_selected': {}}.items():
        if key not in st.session_state:
            st.session_state[key] = value


def v2_streak(values):
    count=0
    for value in reversed(list(values)):
        if number(value) is None or value<=0:
            break
        count+=1
    return count


def v2_precursor(df):
    """A fixed technical signal rubric. No future prices or missing flow proxies."""
    if len(df)<66:
        raise ValueError('최소 66개 확정 일봉이 필요합니다.')
    r,p=df.iloc[-1],df.iloc[-2]
    required=['MA5','MA20','MA60','RSI','MACD_HIST','ATR14','BB_Upper','BB_Lower']
    if any(number(r.get(c)) is None for c in required):
        raise ValueError('전조 점수 지표가 부족합니다.')
    avg=df.Volume.iloc[-21:-1].mean()
    if avg<=0 or r.Volume<=0:
        raise ValueError('거래량이 없어 신호 평가를 보류합니다.')
    ratio=r.Volume/avg
    dist=(r.Close/r.MA20-1)*100
    high=df.High.iloc[-61:-1].max()
    gap=(high/r.Close-1)*100
    aligned=r.MA5>r.MA20>r.MA60
    prev_aligned=p.MA5>p.MA20>p.MA60
    width=(df.BB_Upper-df.BB_Lower)/df.MA20
    # Compression baseline excludes the current bar.
    baseline=width.iloc[-61:-1].dropna()
    compressed=len(baseline)>=20 and width.iloc[-2]<=baseline.quantile(.25)
    rules=[('거래량',25,25 if ratio>=2 and r.Close>p.Close else 17 if ratio>=1.5 and r.Close>p.Close else 8 if ratio>=1.2 else 0,
            f'직전 20일 평균 대비 {ratio:.2f}배; 상승일 2배 25 / 상승일 1.5배 17 / 1.2배 8'),
           ('이동평균 구조',25,(15 if aligned else 7 if r.Close>r.MA20 else 0)+(10 if r.MA20>df.MA20.iloc[-6] else 0),
            '정배열 15 또는 종가>20일선 7 + 20일선 5일간 상승 10'),
           ('모멘텀',20,(10 if 50<=r.RSI<=68 else 5 if 40<=r.RSI<50 or 68<r.RSI<=75 else 0)+(10 if r.MACD_HIST>p.MACD_HIST else 0),
            'RSI 50~68 10 또는 40~50·68~75 5 + MACD 히스토그램 개선 10'),
           ('변동성 수축·돌파',15,(7 if compressed else 0)+(8 if r.Close>high else 4 if 0<=gap<=5 else 0),
            '전일 밴드폭 하위 25% 7 + 직전 60일 고점 돌파 8 또는 고점까지 5% 이내 4'),
           ('가격 이격',15,15 if 0<=dist<=5 else 8 if -3<=dist<0 or 5<dist<=10 else 0,
            f'20일선 이격 {dist:+.1f}%; 0~5% 15 / -3~0·5~10% 8')]
    penalty=10 if r.RSI>78 or dist>15 else 0
    score=max(0,sum(x[2] for x in rules)-penalty)
    status='관심 조건 충족' if score>=75 and penalty==0 else '변화 관찰' if score>=55 else '조건 확인 대기'
    text=(f'거래량은 직전 20일 평균의 {ratio:.2f}배이며, '+
          ('이동평균선이 정배열로 전환된 초기 구간입니다. ' if aligned and not prev_aligned else '이동평균선 정배열이 유지되고 있습니다. ' if aligned else '이동평균선의 정배열은 아직 확인되지 않았습니다. ')+
          f'20일선 이격은 {dist:+.1f}%, RSI는 {r.RSI:.1f}입니다. '+
          ('단기 과열 부담이 있어 가격 이격이 줄어드는지 먼저 확인하시기를 권해드립니다.' if penalty else
           '가격 상승에 거래가 동반되는지, 조정 시 20일선 지지가 유지되는지 관찰하실 만합니다.' if score>=55 else
           '현재는 신호가 충분히 모이지 않아 거래량 회복과 추세 개선을 확인하는 접근이 적절합니다.'))
    return {'score':score,'status':status,'comment':text,'volume_ratio':float(ratio),'dist20':float(dist),
            'gap_high':float(gap),'rsi':float(r.RSI),'aligned_new':bool(aligned and not prev_aligned),
            'penalty':penalty,'rules':[{'항목':x[0],'배점':x[1],'득점':x[2],'근거':x[3]} for x in rules]}


@st.cache_data(ttl=600,max_entries=512,show_spinner=False)
def v2_history(region,code):
    return fetch_history(code,region=region)


@st.cache_data(ttl=600,max_entries=256,show_spinner=False)
def v2_flow(code):
    return fetch_investors(code)


def v2_profile(region,code,name=''):
    result=v2_history(region,code)
    if result.status=='error':
        raise ValueError('일봉 수집 실패: '+' / '.join(result.notes))
    df=add_indicators(completed_history(result.data))
    if len(df)<71:
        raise ValueError('전일·5일 전 점수 비교에 필요한 확정 일봉이 부족합니다.')
    if (datetime.now(EXCHANGE_TZ[region]).date()-df.index[-1].date()).days>7:
        raise ValueError('일봉이 7일 이상 경과하여 스캔에서 제외합니다.')
    signal=v2_precursor(df)
    previous=v2_precursor(df.iloc[:-1])
    past5=v2_precursor(df.iloc[:-5])
    tech=evaluate_technical_score(df)
    trajectory=[{'date':str(df.index[end-1].date()),'score':v2_precursor(df.iloc[:end])['score']} for end in range(max(66,len(df)-19),len(df)+1)]
    return {'region':region,'code':code,'name':name or code,'asof':str(df.index[-1].date()),
            'close':float(df.Close.iloc[-1]),'ret1':period_return(df,1),'ret5':period_return(df,5),
            'signal':signal,'delta':signal['score']-previous['score'],
            'delta5':signal['score']-past5['score'],'technical':tech['score'],'trajectory':trajectory,'df':df,'source':result.source}


def v2_scan_batch(region,records):
    def get(item):
        try:
            profile=v2_profile(region,item['Code'],item.get('Name',''))
            profile.pop('df',None)  # Keep large full-universe runs bounded in session memory.
            return profile,None
        except Exception as exc:
            return None,{'code':item['Code'],'reason':str(exc)}
    with ThreadPoolExecutor(max_workers=4) as pool:
        result=list(pool.map(get,records))
    return [row for row,err in result if row is not None],[err for row,err in result if err is not None]


def v2_record_signal(profile,now=None,origin="manual"):
    now=now or datetime.now(EXCHANGE_TZ[profile['region']])
    return {'id':'|'.join([V2_MODEL,profile['region'],profile['code'],profile['asof']]),
            'model':V2_MODEL,'origin':origin,'region':profile['region'],'code':profile['code'],'name':profile['name'],
            'signal_date':profile['asof'],'recorded_at':now.isoformat(),'score':profile['signal']['score'],
            'signal_close':profile['close'],'comment':profile['signal']['comment']}


def v2_record_batch(profiles):
    existing={row['id'] for row in st.session_state.v2_signals}
    for profile in profiles:
        if profile['signal']['score']>=75 and profile['signal']['penalty']==0:
            record=v2_record_signal(profile,origin="automatic")
            if record['id'] not in existing:
                st.session_state.v2_signals.append(record)
                existing.add(record['id'])
    st.session_state.v2_signals=sorted(st.session_state.v2_signals,key=lambda r:pd.Timestamp(r['recorded_at']))[-5000:]


def v2_signal_outcome(record,frame,horizon=5,cost_bps=10):
    """First tradable open AFTER the observation date; excludes retrospective entries."""
    tz=EXCHANGE_TZ[record['region']]
    recorded=pd.Timestamp(record['recorded_at'])
    if recorded.tzinfo is None:
        raise ValueError('신호 기록 시각에는 시간대가 필요합니다.')
    after=max(pd.Timestamp(record['signal_date']),pd.Timestamp(recorded.tz_convert(tz).date()))
    completed=completed_history(frame)
    future=completed[completed.index>after]
    tradable=future[(future.Volume>0)&(future.Open>0)]
    if tradable.empty:
        return {'status':'진입 대기','return_pct':None}
    entry=tradable.index[0]
    holding=future.loc[entry:]
    if len(holding)<horizon:
        return {'status':f'{horizon}거래일 경과 대기','return_pct':None,'entry_date':str(entry.date())}
    cost=float(cost_bps)/10000
    if not 0<=cost<1:
        raise ValueError('비용 범위 오류')
    end=holding.iloc[horizon-1]
    ret=(float(end.Close)*(1-cost)/(float(tradable.Open.iloc[0])*(1+cost))-1)*100
    return {'status':'평가 완료','return_pct':ret,'entry_date':str(entry.date()),
            'exit_date':str(holding.index[horizon-1].date())}


def v2_sector_table(profiles,region):
    if not profiles:
        return pd.DataFrame()
    date=max(p['asof'] for p in profiles)
    rows=[]
    for name,symbols in V2_SECTORS[region].items():
        members=[p for p in profiles if p['code'] in symbols and p['asof']==date]
        if len(members)<2:
            rows.append({'섹터':name,'확인 종목':len(members),'정의 종목':len(symbols),'강도':None,'5일 변화':None,'상태':'최소 2종목 필요','기준일':date})
            continue
        rows.append({'섹터':name,'확인 종목':len(members),'정의 종목':len(symbols),
                     '강도':round(float(np.mean([p['signal']['score'] for p in members])),1),
                     '5일 변화':round(float(np.mean([p['delta5'] for p in members])),1),
                     '거래량 배수':round(float(np.median([p['signal']['volume_ratio'] for p in members])),2),
                     '상태':'표본 강도 비교','기준일':date})
    return pd.DataFrame(rows).sort_values('강도',ascending=False,na_position='last')


def v2_portfolio_value(positions,prices):
    rows=[]
    for item in positions:
        price=prices.get((item['region'],item['code']))
        current=number(price.get('close')) if price else None
        cost=item['quantity']*item['average']
        value=item['quantity']*current if current is not None else None
        rows.append(dict(item,currency='KRW' if item['region']=='KR' else 'USD',cost=cost,value=value,
                         pnl=value-cost if value is not None else None,
                         return_pct=(current/item['average']-1)*100 if current is not None else None,
                         asof=price.get('asof') if price else None,
                         alert='시세 확인 대기' if current is None else '손절 기준 이하' if item.get('stop') and current<=item['stop'] else '목표 기준 이상' if item.get('target') and current>=item['target'] else '관찰'))
    return rows


def v2_validate_backup(content):
    if len(content)>5_000_000:
        raise ValueError('백업은 5MB 이하로 입력해 주십시오.')
    doc=json.loads(content)
    if not isinstance(doc,dict) or doc.get('schema')!=2:
        raise ValueError('V2 JSON 백업 형식이 아닙니다.')
    positions,signals=doc.get('portfolio',[]),doc.get('signals',[])
    if not isinstance(positions,list) or len(positions)>500 or not isinstance(signals,list) or len(signals)>5000:
        raise ValueError('보유 종목 또는 기록 개수가 허용 범위를 넘습니다.')
    clean_positions=[]; clean_signals=[]
    keys=set()
    for item in positions:
        region,code=item.get('region'),str(item.get('code','')).upper()
        if region not in EXCHANGE_TZ or not v2_symbol(code,region) or (region,code) in keys:
            raise ValueError('보유 종목의 시장·코드·중복 여부를 확인해 주십시오.')
        keys.add((region,code))
        numbers={key:number(item.get(key)) for key in ['quantity','average','stop','target']}
        if any(numbers[key] is None or numbers[key]<=0 for key in ['quantity','average']):
            raise ValueError('수량과 평균 매수가는 양수여야 합니다.')
        if any(item.get(key) is not None and (numbers[key] is None or numbers[key]<=0) for key in ['stop','target']):
            raise ValueError('손절·목표가는 미입력 또는 양수여야 합니다.')
        if numbers['stop'] and numbers['target'] and numbers['stop']>=numbers['target']:
            raise ValueError('손절가는 목표가보다 낮아야 합니다.')
        clean_positions.append(dict(region=region,code=code,name=str(item.get('name',code))[:100],**numbers))
    ids=set()
    for item in signals:
        region,code=item.get('region'),str(item.get('code','')).upper()
        if region not in EXCHANGE_TZ or not v2_symbol(code,region) or item.get('model')!=V2_MODEL:
            raise ValueError('신호의 시장·종목·모델 버전을 확인해 주십시오.')
        day=pd.Timestamp(item['signal_date'])
        timestamp=pd.Timestamp(item['recorded_at'])
        if day.tzinfo is not None or timestamp.tzinfo is None or pd.isna(day) or pd.isna(timestamp):
            raise ValueError('신호 날짜와 기록 시각 형식 오류')
        if day.date()>timestamp.tz_convert(EXCHANGE_TZ[region]).date():
            raise ValueError('기록일보다 미래의 신호는 복원할 수 없습니다.')
        score,price=number(item.get('score')),number(item.get('signal_close'))
        identity='|'.join([V2_MODEL,region,code,str(day.date())])
        if identity in ids or score is None or not 0<=score<=100 or price is None or price<=0:
            raise ValueError('신호의 중복·점수·가격을 확인해 주십시오.')
        if item.get('origin','manual') not in {'automatic','manual'}:
            raise ValueError('신호 기록 종류 오류')
        ids.add(identity)
        clean_signals.append({'id':identity,'model':V2_MODEL,'region':region,'code':code,'name':str(item.get('name',code))[:100],
                              'signal_date':str(day.date()),'recorded_at':timestamp.isoformat(),'origin':item.get('origin','manual'),'score':score,
                              'signal_close':price,'comment':str(item.get('comment',''))[:3000]})
    return {'schema':2,'portfolio':clean_positions,'signals':clean_signals}


def v2_secret(name):
    import os
    try:
        return str(st.secrets.get(name,os.environ.get(name,'')))
    except Exception:
        return os.environ.get(name,'')


@st.cache_data(ttl=86400,max_entries=2,show_spinner=False)
def v2_dart_codes(_key):
    import zipfile
    from io import BytesIO
    body=request_bytes('https://opendart.fss.or.kr/api/corpCode.xml',{'crtfc_key':_key})
    with zipfile.ZipFile(BytesIO(body)) as archive:
        info=archive.getinfo('CORPCODE.xml')
        if info.file_size>60_000_000:
            raise ValueError('공시 회사 목록 크기 초과')
        root=ET.fromstring(archive.read(info))
    return {row.findtext('stock_code','').strip():row.findtext('corp_code','') for row in root.findall('list') if row.findtext('stock_code','').strip()}


def v2_sec_json(url,agent):
    response=requests.get(url,headers={'User-Agent':agent,'Accept':'application/json'},timeout=(3.05,10))
    response.raise_for_status()
    if len(response.content)>20_000_000:
        raise ValueError('SEC 응답 크기 초과')
    return response.json()


@st.cache_data(ttl=86400,max_entries=2,show_spinner=False)
def v2_sec_tickers(_agent):
    data=v2_sec_json('https://www.sec.gov/files/company_tickers.json',_agent)
    return {row['ticker'].upper():int(row['cik_str']) for row in data.values()}


@st.cache_data(ttl=900,max_entries=100,show_spinner=False)
def v2_disclosures(region,code,_credential):
    try:
        if region=='KR':
            corp=v2_dart_codes(_credential).get(code)
            if not corp:
                raise ValueError('공시 회사 고유번호를 찾지 못했습니다.')
            data=json.loads(request_bytes('https://opendart.fss.or.kr/api/list.json',
                {'crtfc_key':_credential,'corp_code':corp,'bgn_de':(datetime.now(KST)-timedelta(days=90)).strftime('%Y%m%d'),
                 'page_count':30,'sort':'date','sort_mth':'desc'}))
            if data.get('status')=='013':
                return Result([],'OpenDART',notes=['최근 90일 공시 없음'])
            if data.get('status')!='000':
                raise ValueError('OpenDART 응답 상태 '+str(data.get('status')))
            rows=[{'title':r['report_nm'],'date':r['rcept_dt'],'link':'https://dart.fss.or.kr/dsaf001/main.do?rcpNo='+r['rcept_no']} for r in data.get('list',[])]
            return Result(rows,'OpenDART')
        cik=v2_sec_tickers(_credential).get(code.replace('.','-'))
        if not cik:
            raise ValueError('SEC CIK를 찾지 못했습니다.')
        data=v2_sec_json(f'https://data.sec.gov/submissions/CIK{cik:010d}.json',_credential)
        recent=data.get('filings',{}).get('recent',{})
        from urllib.parse import quote
        rows=[{'title':form,'date':date,'link':f'https://www.sec.gov/Archives/edgar/data/{cik}/{acc.replace("-","")}/{quote(doc)}'}
              for form,date,acc,doc in zip(recent.get('form',[]),recent.get('filingDate',[]),recent.get('accessionNumber',[]),recent.get('primaryDocument',[]))][:30]
        return Result(rows,'SEC EDGAR')
    except Exception as exc:
        # Do not include exception URLs: DART query strings contain the credential.
        return Result([],'OpenDART' if region=='KR' else 'SEC EDGAR','error',[f'공시 조회 실패 ({type(exc).__name__}); 인증 설정과 공급원 접근 상태를 확인해 주십시오.'])


def v2_news_clues(title):
    text=title.casefold()
    pos=[word for word in ['수주','흑자전환','상향','신기록','upgrade','record revenue','beats estimates'] if word in text]
    neg=[word for word in ['적자','하향','소송','리콜','downgrade','recall','lawsuit','misses estimates'] if word in text]
    label='혼재·원문 확인' if pos and neg else '긍정 표현 단서' if pos else '주의 표현 단서' if neg else '방향 판단 보류'
    return label,', '.join(pos+neg) or '단서 없음'


def v2_go_stock(region,code,name):
    st.session_state.market_region='🇰🇷 한국주식' if region=='KR' else '🇺🇸 미국주식'
    st.session_state.v2_menu='📊 종목 분석'
    if region=='KR':
        select_stock(code,name)
    else:
        st.session_state.us_selected=code
        st.session_state.us_query=code


def v2_pick(region,prefix):
    query=st.text_input('종목명 또는 코드/티커',key=prefix+'_query')
    if not query.strip():
        return None
    directory=stock_directory(region)
    matches=local_search(query,pd.DataFrame(columns=MARKET_COLS)) if region=='KR' else us_search(query,directory.data)
    options={r.Code:r.Name for r in matches.itertuples()}
    code=query.strip().upper()
    if v2_symbol(code,region):
        options.setdefault(code,code)
    if not options:
        st.info('검색 결과가 없습니다. 종목코드 또는 티커를 입력해 주십시오.')
        return None
    selected=st.selectbox('종목 선택',list(options),format_func=lambda c:c+' · '+options[c],key=prefix+'_choice')
    return selected,options[selected]


@st.cache_data(ttl=1800,max_entries=3,show_spinner=False)
def v2_index(symbol,region):
    try:
        df=fdr_table('history',symbol,400)
        df.attrs['region']=region
        df=add_indicators(completed_history(df))
        if len(df)<61 or (datetime.now(EXCHANGE_TZ[region]).date()-df.index[-1].date()).days>7:
            raise ValueError('최근 지수 일봉 부족')
        r=df.iloc[-1]
        temperature=20*sum([r.Close>r.MA20,r.Close>r.MA60,r.MA20>df.MA20.iloc[-6],period_return(df,5)>0,period_return(df,20)>0])
        return {'close':float(r.Close),'ret':period_return(df,1),'temperature':int(temperature),'asof':str(df.index[-1].date())}
    except Exception:
        return None


def v2_render_market(region):
    st.subheader('🧭 시장 상황판')
    st.caption('확정 일봉 기준입니다. 시장 온도는 20·60일선 상회, 20일선 상승, 5·20일 수익률 양수 각 20점이며 투자 확률이 아닙니다.')
    if st.button('지수 상황 조회',key='v2_indices'):
        with st.spinner('지수 일봉 확인 중…'):
            st.session_state['v2_index_results']={name:v2_index(code,reg) for name,code,reg in [('KOSPI','KS11','KR'),('KOSDAQ','KQ11','KR'),('NASDAQ','IXIC','US')]}
    values=st.session_state.get('v2_index_results',{})
    for box,name in zip(st.columns(3),['KOSPI','KOSDAQ','NASDAQ']):
        item=values.get(name)
        box.metric(name,fmt(item['close'],digits=2) if item else '조회 대기·미확인',fmt(item['ret'],'%',2,True) if item else None)
        if item:
            box.caption(f"시장 온도 {item['temperature']} / 100 · {item['asof']}")
    st.caption('시장 전체 외국인·기관·개인 순매수 금액은 검증된 공급원이 연결되지 않아 표시하지 않습니다.')
    profiles=st.session_state.get('v2_profiles_'+region,[])
    if profiles:
        date=max(p['asof'] for p in profiles)
        same=[p for p in profiles if p['asof']==date]
        st.write(f'최근 스캔 표본 {len(same)}종목 중 상승 종목 {sum(p["ret1"]>0 for p in same)}개 · 기준일 {date}')
        st.caption('전체 시장의 상승 종목 비율이 아닙니다.')
        st.dataframe(v2_sector_table(same,region),hide_index=True,use_container_width=True)
    else:
        st.info('급등 전조나 섹터 순환에서 스캔하시면 표본 강도도 함께 표시합니다.')


def v2_start_job(region,records,scope):
    st.session_state['v2_job_'+region]={'records':records,'cursor':0,'results':[],'errors':[],'scope':scope,'started_at':datetime.now(EXCHANGE_TZ[region]).isoformat()}


def v2_advance_job(region):
    job=st.session_state.get('v2_job_'+region)
    if not job or job['cursor']>=len(job['records']):
        return
    batch=job['records'][job['cursor']:job['cursor']+10]
    rows,errors=v2_scan_batch(region,batch)
    job['results'].extend(rows);job['errors'].extend(errors);job['cursor']+=len(batch)
    st.session_state['v2_profiles_'+region]=job['results']
    v2_record_batch(rows)


def v2_due_slot(now,slots,seen):
    if now.weekday()>=5:
        return None
    today=now.date().isoformat()
    for slot in sorted(slots,reverse=True):
        scheduled=datetime.strptime(today+' '+slot,'%Y-%m-%d %H:%M').replace(tzinfo=now.tzinfo)
        elapsed=(now-scheduled).total_seconds()
        identity=today+' '+slot
        if 0<=elapsed<300 and identity not in seen:
            return identity
    return None


@st.fragment(run_every='60s')
def v2_scheduler(region):
    enabled=st.checkbox('이 화면이 열려 있는 동안 자동 진행',key='v2_auto_'+region)
    slots=st.multiselect('예약 시각 (거래소 현지 시각)', ['09:30','11:00','14:00','15:30' if region=='KR' else '16:00'],key='v2_slots_'+region)
    st.caption('이 화면·세션이 활성화된 동안만 1분마다 확인합니다. 닫힌 브라우저의 백그라운드 실행은 지원하지 않습니다. 예약은 평일 해당 시각부터 5분 이내에 실행하며 휴장일을 별도로 판별하지 않습니다. 확정 일봉이 같으면 점수도 같습니다.')
    if enabled:
        job=st.session_state.get('v2_job_'+region)
        seen=st.session_state.get('v2_seen_'+region,[])
        due=v2_due_slot(datetime.now(EXCHANGE_TZ[region]),slots,seen)
        if due and job and job['cursor']>=len(job['records']):
            v2_start_job(region,job['records'],job['scope'])
            st.session_state['v2_seen_'+region]=(seen+[due])[-100:]
        v2_advance_job(region)
    job=st.session_state.get('v2_job_'+region)
    if job:
        total=len(job['records'])
        st.progress(job['cursor']/total if total else 0,text=f"{job['cursor']}/{total} 처리 · 성공 {len(job['results'])} · 실패 {len(job['errors'])}")
        st.caption('아래 결과를 최신 상태로 보시려면 결과 갱신을 눌러 주십시오.')


def v2_render_scan(region):
    st.subheader('🚨 급등 전조 · 종목 레이더')
    st.caption('급등을 예측하는 확률 모델이 아닙니다. 기술적 전조 조건의 충족도를 평가하며, 수급은 선택 종목에서 별도 확인합니다. 75점 이상·과열 감점 없음 조건을 충족하면 신호 이력에 자동 기록합니다.')
    scope=st.radio('스캔 범위',['직접 입력 표본','전체 목록 순차 스캔'],horizontal=True,key='v2_scope_'+region)
    default=', '.join(code for code,_ in (SEED_KR if region=='KR' else SEED_US))
    query=st.text_area('종목코드/티커 (쉼표 구분)',value=default,key='v2_symbols_'+region)
    mincap=st.number_input('한국 전체 스캔 최소 시가총액 (억원)',min_value=0,value=1000,step=100,key='v2_mincap_'+region) if region=='KR' else 0
    if st.button('스캔 시작',type='primary',key='v2_start_scan'):
        try:
            if scope=='직접 입력 표본':
                symbols=list(dict.fromkeys(x.upper() for x in re.split(r'[,\s]+',query.strip()) if x))
                if not symbols or len(symbols)>100 or any(not v2_symbol(c,region) for c in symbols):
                    raise ValueError('올바른 종목코드/티커를 1~100개 입력해 주십시오.')
                names=dict((SEED_KR if region=='KR' else SEED_US))
                records=[{'Code':c,'Name':names.get(c,c)} for c in symbols]
            elif region=='KR':
                market=kr_market_backup()
                if market.data.empty:
                    raise ValueError('전체 시장 시가총액 수집에 실패했습니다. 직접 입력 표본을 이용해 주십시오.')
                df=market.data[market.data.Market.isin(['KOSPI','KOSDAQ'])&(market.data.Marcap>=mincap*1e8)]
                records=df[['Code','Name']].to_dict('records')
            else:
                directory=stock_directory(region)
                if directory.status!='ok':
                    raise ValueError('전체 미국 목록을 확보하지 못했습니다. 직접 입력 표본을 이용해 주십시오.')
                records=directory.data[['Code','Name']].to_dict('records')
            if not records:
                raise ValueError('조건에 맞는 종목이 없습니다.')
            v2_start_job(region,records,scope)
            with st.spinner('첫 10종목을 분석하고 있습니다…'):
                v2_advance_job(region)
        except ValueError as exc:
            st.error(str(exc))
    a,b=st.columns(2)
    if a.button('다음 10종목 진행',key='v2_next_batch'):
        with st.spinner('일봉 분석 중…'):
            v2_advance_job(region)
    b.button('결과 갱신',key='v2_refresh_scan')
    v2_scheduler(region)
    job=st.session_state.get('v2_job_'+region)
    if job:
        st.caption(f"범위: {job['scope']} · 시작: {job['started_at']}")
        if job['errors']:
            with st.expander('제외 종목과 사유'):
                st.dataframe(pd.DataFrame(job['errors']),hide_index=True,use_container_width=True)
    profiles=st.session_state.get('v2_profiles_'+region,[])
    if not profiles:
        return
    latest=max(p['asof'] for p in profiles)
    minimum=st.slider('최소 전조 점수',0,100,50,key='v2_min_score')
    rise=st.slider('최소 전일 대비 변화',-100,100,0,key='v2_min_delta')
    rows=[{'종목':p['name'],'코드':p['code'],'전조 점수':p['signal']['score'],'전일 대비':p['delta'],
           '기술 점수':p['technical'],'거래량 배수':round(p['signal']['volume_ratio'],2),'기준일':p['asof'],'판정':p['signal']['status']}
          for p in profiles if p['asof']==latest and p['signal']['score']>=minimum and p['delta']>=rise]
    st.caption(f'동일 기준일 {latest}만 비교합니다. 점수 변화는 수집 시각의 차이가 아닌 확정 거래일 간 변화입니다.')
    if rows:
        st.dataframe(pd.DataFrame(rows).sort_values(['전조 점수','전일 대비'],ascending=False),hide_index=True,use_container_width=True)
    else:
        st.info('현재 필터를 충족한 종목이 없습니다.')
    v2_render_profile_choice(profiles,region,'radar')


def v2_render_profile_choice(profiles,region,prefix):
    if not profiles:
        return
    options={p['code']:p for p in profiles}
    code=st.selectbox('종목 상세',list(options),format_func=lambda c:options[c]['name']+' ('+c+')',key='v2_detail_'+prefix+'_'+region)
    p=options[code]
    st.info(p['signal']['comment'])
    st.caption(f"전조 {p['signal']['score']}점 · 전일 대비 {p['delta']:+.0f}점 · 과열 감점 {p['signal']['penalty']}점 · {p['source']} · {p['asof']}")
    if p.get('trajectory'):
        trend=pd.DataFrame(p['trajectory']).set_index('date')
        st.line_chart(trend.rename(columns={'score':'전조 점수'}),height=180)
        st.caption('각 거래일 당시까지의 가격으로 다시 계산한 점수 변화입니다. 실제 과거 관찰 기록과는 구분합니다.')
    if region=='KR':
        if st.button('이 종목 수급 함께 확인',key='v2_detail_flow_'+prefix):
            st.session_state['v2_detail_flow_open_'+prefix]=code
        if st.session_state.get('v2_detail_flow_open_'+prefix)==code:
            supply=v2_flow(code)
            history=v2_history(region,code)
            try:
                df=add_indicators(completed_history(history.data))
                flow=assess_investor_flow(supply.data,df)
                st.info(flow['text'])
                if flow['score'] is not None:
                    st.metric('별도 수급 점수',f"{flow['score']} / 100")
                if p['signal']['score']>=75 and flow['score'] is not None and flow['score']>=60 and flow['ratio']>=1:
                    st.write('기술적 전조와 외국인·기관 합산 매수 강도가 함께 관측됩니다. 가격 이격과 매수 지속성을 확인하며 관심 종목으로 검토하실 만합니다.')
            except (ValueError,AttributeError):
                st.info('수급과 일봉을 함께 확인하지 못해 수급 해석을 보류합니다.')
    with st.expander('전조 점수 근거'):
        st.dataframe(pd.DataFrame(p['signal']['rules']),hide_index=True,use_container_width=True)
    if st.button('관심 신호 기록',key='v2_manual_record_'+prefix):
        record=v2_record_signal(p)
        if record['id'] not in {r['id'] for r in st.session_state.v2_signals}:
            st.session_state.v2_signals.append(record)
        st.success('기록했습니다. 성과 화면에서 확인하실 수 있습니다.')
    st.button('기존 종목 분석으로 이동',key='v2_open_'+prefix,on_click=v2_go_stock,args=(region,code,p['name']))


def v2_render_sectors(region):
    st.subheader('🔥 섹터 순환 · 표본 비교')
    st.caption('수동으로 정의한 테마 표본입니다. 공식 업종 전체 구성이나 실제 자금 유입액을 의미하지 않습니다. 같은 종목들의 전조 점수 5일 변화를 비교합니다.')
    with st.expander('테마 구성 종목'):
        st.json(V2_SECTORS[region])
    if st.button('테마 표본 전체 스캔',key='v2_sector_scan'):
        records=[{'Code':c,'Name':c} for c in dict.fromkeys(c for codes in V2_SECTORS[region].values() for c in codes)]
        with st.spinner('테마 표본 일봉 분석 중…'):
            rows,errors=v2_scan_batch(region,records)
        st.session_state['v2_profiles_'+region]=rows
        st.session_state['v2_sector_errors_'+region]=errors
        v2_record_batch(rows)
    profiles=st.session_state.get('v2_profiles_'+region,[])
    table=v2_sector_table(profiles,region)
    if not table.empty:
        st.dataframe(table,hide_index=True,use_container_width=True)
        valid=table.dropna(subset=['강도'])
        if not valid.empty:
            st.bar_chart(valid.set_index('섹터')[['강도']],height=260)
        chosen=st.selectbox('섹터 종목 보기',list(V2_SECTORS[region]),key='v2_sector_choice')
        v2_render_profile_choice([p for p in profiles if p['code'] in V2_SECTORS[region][chosen]],region,'sector')
    errors=st.session_state.get('v2_sector_errors_'+region,[])
    if errors:
        with st.expander('수집하지 못한 종목'):
            st.dataframe(pd.DataFrame(errors),hide_index=True)


def v2_render_flow(region):
    st.subheader('🐋 외국인·기관 수급 추적')
    if region=='US':
        st.info('미국 기관 보유 공시는 일별 순매수 자료가 아닙니다. 한국식 연속 순매수를 추정하지 않습니다. 뉴스·공시에서 SEC 공시를 조회하실 수 있습니다.')
        return
    selected=v2_pick(region,'v2_flow')
    if not selected:
        return
    if st.button('수급 추적',key='v2_load_flow'):
        st.session_state.v2_flow_selected=selected
    if st.session_state.get('v2_flow_selected')!=selected:
        return
    code,name=selected
    try:
        profile=v2_profile(region,code,name)
        result=v2_flow(code)
        inv=result.data
        upload=st.file_uploader('수급 CSV 대체 입력',type=['csv'],key='v2_flow_csv')
        if upload:
            inv=read_investor_csv(upload.getvalue(),code,profile['df'])
        flow=assess_investor_flow(inv,profile['df'])
        st.info(flow['text'])
        st.metric('수급 점수',f"{flow['score']} / 100" if flow['score'] is not None else '확인 대기')
        w=investor_window(inv,profile['df'],20,['ForeignNet','InstitutionNet'])
        if w is not None:
            f,i=v2_streak(w.ForeignNet),v2_streak(w.InstitutionNet)
            c1,c2=st.columns(2)
            c1.metric('외국인 연속 순매수',f'{f}일'+(' 이상' if f==20 else ''))
            c2.metric('기관 연속 순매수',f'{i}일'+(' 이상' if i==20 else ''))
            turn=w.InstitutionNet.iloc[-1]>0 and w.InstitutionNet.iloc[-2]<=0
            st.write('기관은 분석 기준일에 순매수로 전환했습니다.' if turn else '기관의 당일 순매수 전환 조건은 충족하지 않았습니다.')
        else:
            st.caption('연속 순매수 일수는 최근 20거래일 자료가 모두 있을 때 표시합니다.')
        render_supply(profile['df'],inv)
        st.caption(result.source+' · '+' / '.join(result.notes))
    except ValueError as exc:
        st.warning(str(exc))


def v2_render_news(region):
    st.subheader('📰 뉴스·공시')
    selected=v2_pick(region,'v2_news')
    if not selected:
        return
    code,name=selected
    if st.button('뉴스·공시 조회',key='v2_load_news'):
        st.session_state.v2_news_selected=(region,code)
    if st.session_state.get('v2_news_selected')!=(region,code):
        return
    news=cached_news(name) if region=='KR' else cached_us_news(code)
    st.caption('제목에 나타난 표현 단서를 표시합니다. 원문을 읽은 AI 분석이나 가격 영향도 예측은 아닙니다. 발표 시점·공시 원문을 확인해 주십시오.')
    if not news.data:
        st.info('뉴스를 수집하지 못했거나 검색 결과가 없습니다.')
    for item in news.data:
        label,clues=v2_news_clues(item['title'])
        st.link_button(item['title'],item['link'])
        st.caption(label+' · '+clues+' · '+item.get('date','날짜 미제공'))
    credential=v2_secret('DART_API_KEY' if region=='KR' else 'SEC_USER_AGENT')
    if not credential:
        st.info('공시 자동 수집에는 Streamlit Secrets의 DART_API_KEY가 필요합니다.' if region=='KR' else 'SEC 공시 자동 수집에는 Streamlit Secrets에 SEC_USER_AGENT를 앱명과 연락 이메일 형식으로 설정해 주십시오.')
        st.link_button('공식 공시 사이트 열기','https://dart.fss.or.kr/' if region=='KR' else 'https://www.sec.gov/edgar/search/')
        return
    reports=v2_disclosures(region,code,credential)
    st.caption(reports.source+' · '+' / '.join(reports.notes))
    for report in reports.data:
        st.link_button(report['date']+' · '+report['title'],report['link'])


def v2_render_backup():
    st.caption('보유 종목과 신호 기록은 현재 브라우저 세션에 보관합니다. 새 세션·서버 재시작 시 사라질 수 있으므로 JSON을 내려받아 보관해 주십시오. GitHub나 다른 사용자에게 저장하지 않습니다.')
    doc={'schema':2,'portfolio':st.session_state.v2_portfolio,'signals':st.session_state.v2_signals}
    st.download_button('보유 종목·신호 JSON 백업',json.dumps(doc,ensure_ascii=False,indent=2).encode('utf-8'),file_name='stock_terminal_v2_backup.json',mime='application/json',key='v2_backup_download')
    upload=st.file_uploader('V2 JSON 복원',type=['json'],key='v2_backup_upload')
    if upload and st.button('검증 후 현재 기록과 병합',key='v2_restore'):
        try:
            parsed=v2_validate_backup(upload.getvalue())
            # Existing records are immutable; restored copies cannot rewrite observed scores.
            positions={(p['region'],p['code']):p for p in parsed['portfolio']}
            positions.update({(p['region'],p['code']):p for p in st.session_state.v2_portfolio})
            signals={p['id']:p for p in parsed['signals']}
            signals.update({p['id']:p for p in st.session_state.v2_signals})
            if len(positions)>500 or len(signals)>5000:
                raise ValueError('병합 후 기록 개수가 허용 범위를 넘습니다.')
            st.session_state.v2_portfolio=list(positions.values())
            st.session_state.v2_signals=list(signals.values())
            st.success('복원했습니다. 중복 항목은 현재 세션의 기록을 유지했습니다.')
        except (ValueError,TypeError,KeyError,AttributeError) as exc:
            st.error('복원 실패: '+str(exc))


def v2_render_portfolio(region):
    st.subheader('💼 내 종목')
    with st.form('v2_position_form'):
        code=st.text_input('종목코드 / 티커',key='v2_position_code').strip().upper()
        name=st.text_input('표시 이름 (선택)',key='v2_position_name')
        quantity=st.number_input('보유 수량',min_value=0.0,value=1.0,step=1.0,key='v2_position_quantity')
        average=st.number_input('평균 매수가 (KRW)' if region=='KR' else '평균 매수가 (USD)',min_value=0.0,value=1.0,step=1.0,key='v2_position_average')
        stop=st.number_input('손절 참고가 (0: 미설정)',min_value=0.0,value=0.0,key='v2_position_stop')
        target=st.number_input('목표 참고가 (0: 미설정)',min_value=0.0,value=0.0,key='v2_position_target')
        if st.form_submit_button('보유 종목 추가·수정'):
            try:
                entry={'region':region,'code':code,'name':name or code,'quantity':quantity,'average':average,'stop':stop or None,'target':target or None}
                clean=v2_validate_backup(json.dumps({'schema':2,'portfolio':[entry]}).encode())['portfolio'][0]
                rows=[p for p in st.session_state.v2_portfolio if (p['region'],p['code'])!=(region,code)]
                if len(rows)>=500:
                    raise ValueError('보유 종목은 최대 500개까지 보관합니다.')
                st.session_state.v2_portfolio=rows+[clean]
                st.session_state.pop('v2_portfolio_values',None)
            except ValueError as exc:
                st.error(str(exc))
    if st.button('보유 종목 평가 갱신',key='v2_value_portfolio'):
        prices={}
        for item in st.session_state.v2_portfolio:
            result=v2_history(item['region'],item['code'])
            try:
                df=completed_history(result.data)
                if not df.empty and (datetime.now(EXCHANGE_TZ[item['region']]).date()-df.index[-1].date()).days<=7:
                    prices[(item['region'],item['code'])]={'close':df.Close.iloc[-1],'asof':str(df.index[-1].date())}
            except (ValueError,AttributeError):
                pass
        st.session_state.v2_portfolio_values=v2_portfolio_value(st.session_state.v2_portfolio,prices)
    rows=st.session_state.get('v2_portfolio_values',v2_portfolio_value(st.session_state.v2_portfolio,{}))
    if rows:
        table=pd.DataFrame(rows)
        for currency in ['KRW','USD']:
            part=table[table.currency==currency].copy()
            if not part.empty:
                valid=part.dropna(subset=['value'])
                total=valid.value.sum()
                part['weight_pct']=part.value/total*100 if total>0 else np.nan
                st.markdown('**'+currency+' 보유 종목**')
                st.dataframe(part.rename(columns={'name':'종목','quantity':'수량','average':'평단','cost':'매수원금','value':'평가액','pnl':'평가손익','return_pct':'수익률(%)','weight_pct':'평가가능 종목 내 비중(%)','asof':'시세 기준일','alert':'기준가 상태'}),hide_index=True,use_container_width=True)
                st.caption(f'평가 가능한 {len(valid)}/{len(part)}종목 합계 {total:,.2f} {currency} · 수수료·세금·배당·환율 미반영. 기준일을 확인해 주십시오.')
        options={p['region']+':'+p['code']:p for p in st.session_state.v2_portfolio}
        selected=st.selectbox('삭제할 보유 종목',list(options),key='v2_remove_position')
        if st.button('선택한 보유 종목 삭제',key='v2_delete_position'):
            old=options[selected]
            st.session_state.v2_portfolio=[p for p in st.session_state.v2_portfolio if (p['region'],p['code'])!=(old['region'],old['code'])]
            st.session_state.pop('v2_portfolio_values',None)
            st.rerun()
    v2_render_backup()


def v2_render_performance(region):
    st.subheader('🧪 전략 백테스트 · 관찰 신호 성과')
    strategy_tab,signal_tab=st.tabs(['전략 백테스트','기록 이후 성과'])
    with strategy_tab:
        selected=v2_pick(region,'v2_bt')
        if selected and st.button('전략 검증',key='v2_run_bt'):
            st.session_state.v2_bt_selected=(region,selected[0])
        if selected and st.session_state.get('v2_bt_selected')==(region,selected[0]):
            try:
                p=v2_profile(region,selected[0],selected[1])
                render_backtest(p['df'])
            except ValueError as exc:
                st.warning(str(exc))
    with signal_tab:
        st.caption('75점 이상 자동 기록과 직접 기록한 관심 신호를 평가합니다. 실제 매매 성적이 아닙니다. 복원된 JSON은 외부 검증된 감사 기록이 아닙니다.')
        cohort=st.selectbox('평가 대상',['자동 조건 신호','전체 관심 신호','직접 기록 신호'],key='v2_cohort')
        horizon=st.selectbox('보유 거래일', [5,10,20],key='v2_horizon')
        cost=st.number_input('편도 비용 (수수료·슬리피지 합계, bp)',min_value=0.0,max_value=1000.0,value=10.0,key='v2_cost')
        st.caption('기록 시각의 거래소 현지 날짜 다음 거래일 이후 첫 거래 가능한 시가에 진입하고, 보유 N번째 거래일 종가로 평가합니다. 청산 비용도 반영하며 배당·환율·세금은 제외합니다.')
        if st.button('신호 성과 계산',key='v2_evaluate_signals'):
            rows=[]
            now=datetime.now(EXCHANGE_TZ[region])
            records=[r for r in st.session_state.v2_signals if r['region']==region and pd.Timestamp(r['recorded_at'])>=pd.Timestamp(now)-pd.Timedelta(days=30)]
            if cohort!='전체 관심 신호':
                origin='automatic' if cohort=='자동 조건 신호' else 'manual'
                records=[r for r in records if r.get('origin','manual')==origin]
            for record in records:
                result=v2_history(record['region'],record['code'])
                try:
                    outcome=v2_signal_outcome(record,result.data,horizon,cost)
                except (ValueError,KeyError,AttributeError):
                    outcome={'status':'시세 확인 대기','return_pct':None}
                rows.append(dict(record,**outcome))
            st.session_state['v2_performance_'+region]={'rows':rows,'horizon':horizon,'cost':cost,'cohort':cohort}
        result=st.session_state.get('v2_performance_'+region)
        if result and (result['horizon'],result['cost'],result['cohort'])==(horizon,cost,cohort):
            table=pd.DataFrame(result['rows'])
            if table.empty:
                st.info('최근 30일에 기록한 해당 시장 신호가 없습니다.')
            else:
                done=table.dropna(subset=['return_pct'])
                c1,c2,c3=st.columns(3)
                c1.metric('최근 30일 기록 / 평가 완료',f'{len(table)} / {len(done)}')
                c2.metric(f'{horizon}거래일 평균 수익률',fmt(done.return_pct.mean() if len(done) else None,'%',2,True))
                c3.metric('양의 수익 비율',fmt((done.return_pct>0).mean()*100 if len(done) else None,'%',1))
                st.dataframe(table,hide_index=True,use_container_width=True)
                st.caption('동일 종목의 중복 기간 신호는 서로 독립적이지 않습니다. 생존편향과 선택편향을 제거한 전 종목 전략 검증이 아닙니다.')
        st.caption(f'현재 세션에 보관된 전체 신호 {len(st.session_state.v2_signals)}개')
        v2_render_backup()


def render_v2_terminal(region):
    v2_init()
    menu=st.radio('V2 메뉴',V2_MENU,horizontal=True,key='v2_menu')
    if menu=='📊 종목 분석':
        return False
    handlers={'🧭 시장 상황판':v2_render_market,'🚨 급등 전조':v2_render_scan,'🔥 섹터 순환':v2_render_sectors,
              '🐋 수급 추적':v2_render_flow,'📰 뉴스·공시':v2_render_news,'💼 내 종목':v2_render_portfolio,'🧪 백테스트·성과':v2_render_performance}
    handlers[menu](region)
    return True


def main():
    st.set_page_config(page_title="퀀트 투자전략실", page_icon="📊", layout="wide")
    st.markdown("""<style>
    :root {
        --bg:#070b12;
        --panel:rgba(17,24,39,.72);
        --panel-strong:rgba(17,24,39,.92);
        --line:rgba(148,163,184,.14);
        --line-strong:rgba(125,211,252,.26);
        --text:#f4f7fb;
        --muted:#8fa0b7;
        --cyan:#67e8f9;
        --blue:#60a5fa;
        --violet:#a78bfa;
        --green:#34d399;
    }

    html, body, [class*="css"] {
        font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans KR",sans-serif;
    }
    .stApp {
        background:
          radial-gradient(900px 520px at 8% -8%, rgba(59,130,246,.19), transparent 60%),
          radial-gradient(760px 480px at 94% 3%, rgba(139,92,246,.16), transparent 58%),
          radial-gradient(620px 420px at 58% 110%, rgba(6,182,212,.08), transparent 64%),
          linear-gradient(180deg,#080c14 0%,#070a10 50%,#06090f 100%);
        color:var(--text);
        background-attachment:fixed;
    }
    .block-container {
        padding-top:1.35rem;
        padding-bottom:3.2rem;
        max-width:1540px;
    }

    /* 상단 브랜드 카드 */
    .hero {
        position:relative;
        overflow:hidden;
        padding:30px 32px 27px;
        margin-bottom:22px;
        border:1px solid rgba(148,163,184,.16);
        border-radius:24px;
        background:
          linear-gradient(120deg,rgba(15,23,42,.90),rgba(17,24,39,.72)),
          radial-gradient(circle at 90% 10%,rgba(96,165,250,.15),transparent 35%);
        box-shadow:0 24px 70px rgba(0,0,0,.28), inset 0 1px 0 rgba(255,255,255,.035);
        backdrop-filter:blur(18px);
    }
    .hero:before {
        content:'';
        position:absolute;
        left:0; top:0; right:0;
        height:2px;
        background:linear-gradient(90deg,var(--cyan),var(--blue),var(--violet),transparent 88%);
        opacity:.9;
    }
    .hero:after {
        content:'';
        position:absolute;
        width:360px; height:360px;
        right:-140px; top:-220px;
        border-radius:50%;
        background:radial-gradient(circle,rgba(103,232,249,.14),transparent 66%);
        pointer-events:none;
    }
    .hero-badge {
        display:inline-flex;
        align-items:center;
        gap:7px;
        padding:6px 10px;
        border:1px solid rgba(103,232,249,.18);
        border-radius:999px;
        background:rgba(8,145,178,.08);
        color:#a5f3fc;
        font-size:11px;
        font-weight:750;
        letter-spacing:.04em;
        margin-bottom:12px;
    }
    .hero-title {
        font-size:34px;
        line-height:1.15;
        font-weight:900;
        letter-spacing:-.035em;
        color:#f8fafc;
        margin:0;
    }
    .hero-subtitle {
        color:#b0bfd2;
        margin-top:9px;
        font-size:14px;
        font-weight:550;
    }
    .hero-meta {
        display:flex;
        gap:9px;
        flex-wrap:wrap;
        margin-top:17px;
    }
    .hero-chip {
        display:inline-block;
        padding:5px 9px;
        border-radius:999px;
        border:1px solid rgba(148,163,184,.13);
        background:rgba(15,23,42,.48);
        color:#7f91a9;
        font-size:11px;
    }

    /* 상단 메인 탭: 카드형 내비게이션 */
    [data-baseweb='tab-list'] {
        gap:8px;
        padding:6px;
        margin-bottom:15px;
        border:1px solid rgba(148,163,184,.11);
        border-radius:15px;
        background:rgba(15,23,42,.48);
        backdrop-filter:blur(14px);
        box-shadow:0 8px 28px rgba(0,0,0,.12);
    }
    [data-baseweb='tab'] {
        height:42px;
        padding:0 16px;
        border-radius:10px;
        color:#8fa0b7;
        font-weight:720;
        transition:all .18s ease;
    }
    [data-baseweb='tab']:hover {
        color:#e8eef7;
        background:rgba(51,65,85,.34);
    }
    [aria-selected='true'][data-baseweb='tab'] {
        color:#f8fafc !important;
        background:linear-gradient(110deg,rgba(14,165,233,.18),rgba(99,102,241,.16));
        box-shadow:inset 0 0 0 1px rgba(125,211,252,.20),0 5px 18px rgba(2,132,199,.08);
    }

    /* 메트릭 카드 */
    [data-testid='stMetric'] {
        border:1px solid var(--line);
        border-radius:16px;
        padding:15px 16px;
        background:linear-gradient(145deg,rgba(17,24,39,.78),rgba(10,15,25,.68));
        box-shadow:0 10px 30px rgba(0,0,0,.14),inset 0 1px 0 rgba(255,255,255,.025);
        backdrop-filter:blur(12px);
        transition:border-color .18s ease,transform .18s ease;
    }
    [data-testid='stMetric']:hover {
        border-color:rgba(103,232,249,.22);
        transform:translateY(-1px);
    }
    [data-testid='stMetricLabel'] { color:#899bb2; font-weight:650; }
    [data-testid='stMetricValue'] { font-size:24px; font-weight:850; color:#f3f7fb; letter-spacing:-.02em; }

    /* 입력 / 선택 */
    [data-baseweb='input'] > div,
    [data-baseweb='select'] > div,
    [data-testid='stNumberInput'] input {
        background:rgba(15,23,42,.72) !important;
        border-color:rgba(148,163,184,.16) !important;
        border-radius:12px !important;
    }
    [data-baseweb='input'] > div:focus-within,
    [data-baseweb='select'] > div:focus-within {
        border-color:rgba(103,232,249,.36) !important;
        box-shadow:0 0 0 3px rgba(34,211,238,.05) !important;
    }

    /* 버튼 */
    .stButton > button,
    .stDownloadButton > button,
    .stLinkButton > a {
        border-radius:12px !important;
        border:1px solid rgba(148,163,184,.16) !important;
        background:linear-gradient(145deg,rgba(30,41,59,.82),rgba(15,23,42,.82)) !important;
        color:#eaf1f8 !important;
        font-weight:750 !important;
        min-height:42px;
        box-shadow:0 6px 18px rgba(0,0,0,.12);
        transition:transform .16s ease,border-color .16s ease,box-shadow .16s ease;
    }
    .stButton > button:hover,
    .stDownloadButton > button:hover,
    .stLinkButton > a:hover {
        border-color:rgba(103,232,249,.32) !important;
        transform:translateY(-1px);
        box-shadow:0 9px 24px rgba(0,0,0,.18);
    }
    [data-testid='stBaseButton-primary'] {
        background:linear-gradient(105deg,#0f7fa5 0%,#2563a9 52%,#5b4ab2 100%) !important;
        border-color:rgba(125,211,252,.34) !important;
        box-shadow:0 10px 26px rgba(37,99,235,.16) !important;
    }

    /* 알림 카드 */
    [data-testid='stAlert'] {
        border-radius:14px;
        border:1px solid rgba(148,163,184,.12);
        background:rgba(15,23,42,.60);
        backdrop-filter:blur(10px);
    }

    /* 익스팬더 / 데이터프레임 */
    [data-testid='stExpander'] {
        border:1px solid rgba(148,163,184,.12) !important;
        border-radius:14px !important;
        background:rgba(15,23,42,.42) !important;
        overflow:hidden;
    }
    [data-testid='stDataFrame'] {
        border:1px solid rgba(148,163,184,.10);
        border-radius:14px;
        overflow:hidden;
        box-shadow:0 8px 24px rgba(0,0,0,.10);
    }

    /* 랭킹 카드 */
    .horse-rank-card {
        position:relative;
        overflow:hidden;
        border:1px solid rgba(148,163,184,.15);
        border-radius:18px;
        padding:18px 13px;
        min-height:174px;
        text-align:center;
        background:
          radial-gradient(circle at 50% -20%,rgba(96,165,250,.12),transparent 45%),
          linear-gradient(150deg,rgba(20,29,44,.92),rgba(9,14,23,.92));
        box-shadow:0 14px 38px rgba(0,0,0,.18),inset 0 1px 0 rgba(255,255,255,.025);
        transition:transform .18s ease,border-color .18s ease;
    }
    .horse-rank-card:hover { transform:translateY(-2px); border-color:rgba(103,232,249,.26); }
    .horse-rank-1 { border-color:rgba(250,204,21,.42); background:radial-gradient(circle at 50% -10%,rgba(250,204,21,.10),transparent 45%),linear-gradient(150deg,rgba(25,30,39,.94),rgba(10,14,22,.94)); }
    .horse-rank-2 { border-color:rgba(203,213,225,.30); }
    .horse-rank-3 { border-color:rgba(251,146,60,.30); }
    .horse-rank-badge { font-size:25px; margin-bottom:6px; }
    .horse-rank-name { font-size:16px; font-weight:850; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; color:#f3f6fa; }
    .horse-rank-score { font-size:31px; font-weight:900; line-height:1.2; margin:7px 0; letter-spacing:-.03em; }
    .horse-rank-score span { font-size:11px; color:#708298; margin-left:3px; }
    .horse-rank-meta { font-size:12px; color:#91a0b4; margin-top:4px; }

    /* 제목 */
    h1,h2,h3,h4,h5 { letter-spacing:-.025em; }
    h3 { margin-top:.4rem !important; }
    hr { border-color:rgba(148,163,184,.12); }

    /* 스크롤바 */
    ::-webkit-scrollbar { width:10px; height:10px; }
    ::-webkit-scrollbar-track { background:#080c13; }
    ::-webkit-scrollbar-thumb { background:#263449; border-radius:999px; border:2px solid #080c13; }
    ::-webkit-scrollbar-thumb:hover { background:#344761; }

    @media (max-width: 900px) {
        .block-container { padding-left:1rem; padding-right:1rem; }
        .hero { padding:24px 22px; border-radius:19px; }
        .hero-title { font-size:29px; }
        [data-baseweb='tab'] { padding:0 10px; font-size:13px; }
    }
    </style>""", unsafe_allow_html=True)
    st.markdown("""<div class="hero">
      <div class="hero-badge">● 데이터 기반 투자 리서치</div>
      <div class="hero-title">퀀트 투자전략실 V2</div>
      <div class="hero-subtitle">차트 · 수급 · 실적 · 전략 검증을 한 화면에서 분석합니다.</div>
      <div class="hero-meta">
        <span class="hero-chip">KOSPI · KOSDAQ · NASDAQ · NYSE</span>
        <span class="hero-chip">모멘텀 스크리닝</span>
        <span class="hero-chip">과대낙폭 탐색</span>
        <span class="hero-chip">외국인 · 기관 수급</span>
      </div>
    </div>""", unsafe_allow_html=True)
    for key, value in {"query": "", "selected_code": "", "selected_name": "", "needs_search": False,
                       "candidates": [], "search_message": "", "horse_matches": [],
                       "horse_selected_code": "", "horse_selected_name": "",
                       "horse_scan_raw": None, "horse_scan_out": None, "horse_scan_meta": None,
                       "horse_auto_raw": None, "horse_auto_top20_results": None, "horse_auto_meta": None,
                       "oversold_matches": [], "oversold_selected_code": "", "oversold_selected_name": "",
                       "oversold_auto_raw": None, "oversold_auto_top20_results": None, "oversold_auto_meta": None}.items():
        if key not in st.session_state:
            st.session_state[key] = value
    region = st.radio("분석 시장", ["🇰🇷 한국주식", "🇺🇸 미국주식"], horizontal=True, key="market_region")
    if render_v2_terminal("US" if region == "🇺🇸 미국주식" else "KR"):
        return
    if region == "🇺🇸 미국주식":
        render_us_workspace()
        return
    with st.spinner("시장 표본을 조회하고 있습니다…"):
        market = cached_market()
    main_tab, horse_tab, oversold_tab = st.tabs(["📊 종합 분석", "🐎 달리는 말 탐지기", "💎 과대낙폭 유망주"])
    with main_tab:
        render_original_workspace(market)
    with horse_tab:
        render_running_horse(market)
    with oversold_tab:
        render_oversold_hunter(market)


if __name__ == "__main__":
    main()

