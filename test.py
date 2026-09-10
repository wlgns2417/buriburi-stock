# -*- coding: utf-8 -*-
"""부리부리 종합 주식 작전실 — 단일 Python 파일 수정본

GitHub에 있는 기존 실행 .py 파일의 내용을 이 파일 전체로 교체하십시오.
기존 파일명과 실행 설정은 유지하실 수 있습니다.
core.py, providers.py, fdr_worker.py, 테마 설정 파일을 별도로 올릴 필요가 없습니다.

실행: python -m streamlit run app.py
필요 패키지(기존 앱과 동일):
    streamlit, finance-datareader, numpy, pandas, plotly, requests, beautifulsoup4
권장 Python: 3.12. 기존 requirements.txt는 설치 목록이므로 유지하십시오.

데이터 수집 실패는 미확인으로 표시하며 임의 가격으로 대체하지 않습니다.
공매도 자동 수집은 미제공이며 선택적 CSV 입력을 사용합니다.
이 파일로 합치는 과정에서 이전 수정본의 계산·데이터 처리 규칙은 유지했습니다.
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


def completed_history(frame, now=None):
    now = now or datetime.now(KST)
    if now.tzinfo is None:
        now = now.replace(tzinfo=KST)
    today = pd.Timestamp(now.astimezone(KST).date())
    # Deliberately exclude the entire current KST date, even after the close.
    # No holiday calendar, exchange closing-time assumption, or snapshot merge.
    df = clean_history(frame)
    return df.loc[df.index < today].copy()


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
_FDR_WORKER_CODE = '"""A killable FDR worker; prevents upstream requests without timeouts hanging UI."""\nimport contextlib\nimport re\nimport sys\n\n\ndef main():\n    if len(sys.argv) != 4 or not re.fullmatch(r"\\d{6}", sys.argv[1]):\n        raise ValueError("Expected code, start, end")\n    # Keep stdout machine-readable even if FDR writes progress text.\n    with contextlib.redirect_stdout(sys.stderr):\n        import FinanceDataReader as fdr\n        frame = fdr.DataReader(f"NAVER:{sys.argv[1]}", sys.argv[2], sys.argv[3])\n    if frame is None or frame.empty:\n        raise ValueError("No daily prices returned")\n    print(frame.to_json(orient="split", date_format="iso"))\n\n\nif __name__ == "__main__":\n    main()\n'

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


def fetch_market(pages_per_market=2):
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


def local_search(query, stocks):
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


def fetch_history(code, days=800):
    _require_code(code)
    today = datetime.now(KST).date()
    source = f"FinanceDataReader / NAVER:{code}"
    try:
        process = subprocess.run(
            [sys.executable, "-c", _FDR_WORKER_CODE, code,
             str(today - timedelta(days=days)), str(today)],
            capture_output=True, text=True, encoding="utf-8", timeout=25, check=True,
        )
        data = json.loads(process.stdout)
        frame = pd.DataFrame(data["data"], columns=data["columns"], index=pd.to_datetime(data["index"]))
        return Result(frame, source)
    except Exception as exc:
        return failure(source, pd.DataFrame(), exc)


def empty_fund():
    return {"PER": None, "PBR": None, "DividendYield": None, "IndustryPER": None,
            "Target": None, "ROE": None, "ROEPeriod": None, "Summary": None}


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
        # Derive annual span from headers, then explicitly exclude all (E) cells.
        annual = next((th for th in table.select("thead th") if "최근 연간 실적" in th.get_text(" ", strip=True)), None)
        count = int(annual.get("colspan", "0")) if annual else 0
        periods = [compact(th.get_text()) for th in table.select("thead th") if re.search(r"\d{4}\.\d{2}", th.get_text())]
        for tr in table.select("tbody tr"):
            th = tr.select_one("th")
            if not th or "ROE" not in th.get_text():
                continue
            cells = tr.find_all("td", recursive=False)
            if count <= 0 or len(periods) != len(cells):
                break
            for i in reversed(range(min(count, len(periods)))):
                period, value = periods[i], numeric(cells[i].get_text())
                if "(E)" not in period and value is not None:
                    fund["ROE"], fund["ROEPeriod"] = value, period
                    break
            break
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
        if not all(key in labels for key in ["전일비", "등락률", "거래량", "순매매량", "보유주수"]):
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
    st.markdown("#### 🏆 표본 종목 비교")
    st.caption(result.notes[0])
    st.button("랭킹 새로 조회", key="refresh_market", on_click=cached_market.clear, use_container_width=True)
    if result.status != "ok":
        st.warning("일부 또는 전체 페이지를 수집하지 못했습니다. 확보된 종목만 표시합니다.")
    ranked = market_rankings(result.data)
    if ranked.empty:
        st.info("랭킹을 계산할 유효한 시세가 없습니다. 종목코드로 개별 분석을 진행하실 수 있습니다.")
        return
    st.caption("랭킹 점수 = 당일 등락률 50 + 표본 내 시가총액 25 + 추정 거래대금 25. 개별 분석 점수와 다른 지표입니다.")
    top, bottom, lead = st.tabs(["점수 상위", "점수 하위", "모멘텀"])
    groups = [(top, ranked.head(10), "top"),
              (bottom, ranked.sort_values("ScreenScore").head(10), "bottom"),
              (lead, ranked.sort_values("Momentum", ascending=False).head(10), "lead")]
    for tab, frame, prefix in groups:
        with tab:
            for i, row in enumerate(frame.itertuples()):
                st.button(f"{i + 1}. {row.Name} · {row.ScreenScore:.1f}점", key=f"{prefix}_{row.Code}",
                          use_container_width=True, on_click=select_stock, args=(row.Code, row.Name))
                st.caption(f"{row.Close:,.0f}원 · {row.Chg:+.2f}% · 거래대금 추정 {row.AmountEstimate / 1e8:,.0f}억원")
    st.caption("전체 시장 순위가 아닙니다. 가격×거래량은 실제 거래대금·자금 유입액과 다릅니다.")


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
                       file_name=f"backtest_{st.session_state.selected_code}.csv", mime="text/csv")


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
    score = evaluate_score(df, investors, {} if stale else fund, short)
    cols = st.columns(3)
    cols[0].metric("확인된 항목의 득점", f"{score['points']} / {score['possible']}")
    cols[1].metric("전체 배점 중 데이터 확보", f"{score['coverage']}%")
    cols[2].metric("공매도 거래량 비중", fmt((short or {}).get("ShortRatio"), "%", 2))
    st.write(score["grade"])
    st.caption("설명 가능한 규칙 점수입니다. 승률·상승 확률이 아닙니다. 미확인 항목에는 점수를 주지 않고 100점으로 환산하지도 않습니다.")
    if score["score"] is None:
        st.caption(f"미확인 항목까지 확보했을 때의 산술적 점수 범위: {score['points']}~{score['upper_bound']} / 100. 신뢰구간이나 전망 범위가 아닙니다.")
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


def main():
    st.set_page_config(page_title="부리부리 종합 주식 작전실", page_icon="🐽", layout="wide")
    st.markdown("""<style>
    .stApp { background-color: #0c0f17; color: #e1e7f0; }
    .hero { padding: 22px; border: 1px solid #243249; border-radius: 16px;
            background: linear-gradient(120deg,#152235,#111826); margin-bottom:20px; }
    .hero h1 { font-size:26px; margin:0; padding:0; }
    [data-testid='stMetric'] { border:1px solid #243249; border-radius:12px; padding:12px; }
    [data-testid='stMetricValue'] { font-size:23px; }
    </style><div class='hero'><h1>🐽 부리부리 종합 주식 작전실</h1>
    <p style='color:#94a3b8;margin-bottom:0'>종목 분석 · 수급 확인 · 거래비용을 반영한 전략 검증</p></div>""", unsafe_allow_html=True)
    for key, value in {"query": "", "selected_code": "", "selected_name": "", "needs_search": False,
                       "candidates": [], "search_message": ""}.items():
        if key not in st.session_state:
            st.session_state[key] = value
    with st.spinner("시장 표본을 조회하고 있습니다…"):
        market = cached_market()
    resolve_pending(market.data)
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


if __name__ == "__main__":
    main()
