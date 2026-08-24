import urllib.parse
from datetime import datetime, timedelta
import FinanceDataReader as fdr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from bs4 import BeautifulSoup
import streamlit as st

st.set_page_config(
    page_title="부리부리 퀀트 작전실",
    page_icon="🐽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');
    * { font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, sans-serif; }
    .stApp { background-color: #0c0f17; color: #e1e7f0; }
    
    .header-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 18px 24px;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .score-container {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        margin-bottom: 20px;
    }
    
    .target-grid {
        display: flex;
        justify-content: space-between;
        gap: 10px;
        margin-top: 16px;
        flex-wrap: wrap;
    }
    .target-item {
        flex: 1;
        min-width: 110px;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 10px;
        padding: 10px 6px;
        text-align: center;
    }
    .target-title { font-size: 11px; color: #8b9bb4; margin-bottom: 4px; }
    .target-val { font-size: 15px; font-weight: 700; color: #38bdf8; }

    .insight-card {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        line-height: 1.6;
        font-size: 14px;
        color: #cbd5e1;
    }

    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        padding: 12px !important;
        border-radius: 12px !important;
    }
    [data-testid="stMetricLabel"] { font-size: 12px !important; color: #94a3b8 !important; }
    [data-testid="stMetricValue"] { font-size: 19px !important; font-weight: 700 !important; }
    </style>
""",
    unsafe_allow_html=True,
)

if "selected_stock" not in st.session_state:
  st.session_state.selected_stock = ""


@st.cache_data(ttl=3600)
def get_krx_listing():
  return fdr.StockListing("KRX")


def resolve_stock_code(query):
  query = query.strip()
  krx = get_krx_listing()
  if query.isdigit():
    matched = krx[krx["Code"] == query]
    if not matched.empty:
      return query, matched.iloc[0]["Name"]
    return query, query
  matched = krx[krx["Name"] == query]
  if not matched.empty:
    return matched.iloc[0]["Code"], query
  matched_part = krx[krx["Name"].str.contains(query, case=False)]
  if not matched_part.empty:
    return matched_part.iloc[0]["Code"], matched_part.iloc[0]["Name"]
  return None, None


def fetch_investor_naver(code):
  url = f"https://finance.naver.com/item/frgn.naver?code={code}"
  headers = {
      "User-Agent": (
          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
      )
  }
  res = requests.get(url, headers=headers, timeout=5)
  soup = BeautifulSoup(res.text, "html.parser")
  rows = []
  tables = soup.select("table.type2")
  if len(tables) >= 2:
    for tr in tables[1].select("tr"):
      tds = tr.select("td")
      if len(tds) >= 9 and tds[0].text.strip().replace(".", "").isdigit():
        try:
          date = tds[0].text.strip()
          close = int(tds[1].text.strip().replace(",", ""))
          inst_net = int(tds[5].text.strip().replace(",", ""))
          for_net = int(tds[6].text.strip().replace(",", ""))
          for_rate = float(
              tds[8].text.strip().replace("%", "").replace(",", "")
          )
          rows.append({
              "날짜": date,
              "종가": close,
              "기관순매수": inst_net,
              "외인순매수": for_net,
              "기관순매수금액": (inst_net * close) / 100000000,
              "외인순매수금액": (for_net * close) / 100000000,
              "외인보유율": for_rate,
          })
        except Exception:
          continue
  return pd.DataFrame(rows)


def fetch_fundamental_and_consensus(code):
  url = f"https://finance.naver.com/item/main.naver?code={code}"
  headers = {
      "User-Agent": (
          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
      )
  }
  data = {
      "PER": None,
      "PBR": None,
      "배당수익률": None,
      "업종PER": None,
      "목표주가": None,
      "ROE": None,
      "기업개요": "기업 정보 준비 중",
      "리포트_목록": [],
  }
  try:
    res = requests.get(url, headers=headers, timeout=5)
    soup = BeautifulSoup(res.text, "html.parser")

    per_tag = soup.select_one("#_per")
    pbr_tag = soup.select_one("#_pbr")
    dvr_tag = soup.select_one("#_dvr")
    c_per_tag = soup.select_one("#_cper")
    target_tag = soup.select_one("em#_target_money")

    if per_tag and per_tag.text.strip():
      data["PER"] = float(per_tag.text.replace(",", ""))
    if pbr_tag and pbr_tag.text.strip():
      data["PBR"] = float(pbr_tag.text.replace(",", ""))
    if dvr_tag and dvr_tag.text.strip():
      data["배당수익률"] = float(dvr_tag.text.replace(",", ""))
    if c_per_tag and c_per_tag.text.strip():
      data["업종PER"] = float(c_per_tag.text.replace(",", ""))
    if target_tag and target_tag.text.strip():
      data["목표주가"] = float(target_tag.text.replace(",", ""))

    summary_tag = soup.select_one("div.summary_info p")
    if summary_tag:
      data["기업개요"] = summary_tag.text.strip()

    cop_table = soup.select_one("div.section.cop_analysis table")
    if cop_table:
      for tr in cop_table.select("tbody tr"):
        th = tr.select_one("th")
        if th and "ROE" in th.text:
          tds = tr.select("td")
          for td in reversed(tds):
            val = td.text.strip().replace(",", "")
            if val and val != "-":
              try:
                data["ROE"] = float(val)
                break
              except Exception:
                continue
          break

    report_url = (
        f"https://finance.naver.com/item/research.naver?code={code}"
    )
    res_rep = requests.get(report_url, headers=headers, timeout=5)
    soup_rep = BeautifulSoup(res_rep.text, "html.parser")
    for tr in soup_rep.select("table.type2 tr")[2:7]:
      tds = tr.select("td")
      if len(tds) >= 4 and tds[0].text.strip():
        title = tds[0].text.strip()
        broker = tds[2].text.strip()
        date = tds[3].text.strip()
        link_tag = tds[0].select_one("a")
        link = (
            "https://finance.naver.com" + link_tag["href"]
            if link_tag
            else "#"
        )
        data["리포트_목록"].append({
            "title": title,
            "broker": broker,
            "date": date,
            "link": link,
        })
  except Exception:
    pass
  return data


def fetch_short_selling(code):
  url = f"https://finance.naver.com/item/short_selling.naver?code={code}"
  headers = {
      "User-Agent": (
          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
      )
  }
  short_data = {"공매도비중": 0.0, "공매도거래량": 0}
  try:
    res = requests.get(url, headers=headers, timeout=5)
    soup = BeautifulSoup(res.text, "html.parser")
    table = soup.select("table.type2 tbody tr")
    for tr in table:
      tds = tr.select("td")
      if len(tds) >= 5 and tds[0].text.strip():
        short_data["공매도비중"] = float(
            tds[4].text.strip().replace("%", "").replace(",", "")
        )
        short_data["공매도거래량"] = int(
            tds[1].text.strip().replace(",", "")
        )
        break
  except Exception:
    pass
  return short_data


def fetch_news(keyword):
  url = f"https://news.google.com/rss/search?q={urllib.parse.quote(keyword)}+주식&hl=ko&gl=KR&ceid=KR:ko"
  news_list = []
  try:
    res = requests.get(url, timeout=5)
    soup = BeautifulSoup(res.content, "html.parser")
    for item in soup.find_all("item")[:5]:
      t = item.title.text if item.title else ""
      l = item.link.text if item.link else "#"
      if " - " in t:
        t = t.rsplit(" - ", 1)[0]
      news_list.append({"title": t, "link": l})
  except Exception:
    pass
  return news_list


# ====================================================
# [차티스트 정통 추세선 작도 알고리즘]
# ====================================================
def calculate_chartist_trendlines(df):
  """
  사용자가 제시한 정통 차티스트 방식:
  - 상승추세선: 주요 최저점(Swing Low)과 다음 높은 저점(Higher Low)을 연결하여 우상향 연장
  - 하락추세선: 주요 최고점(Swing High)과 다음 낮은 고점(Lower High)을 연결하여 우하향 연장
  """
  highs = df["High"].values
  lows = df["Low"].values
  closes = df["Close"].values
  n = len(df)

  # 최근 80영업일 기준 (단기/중기 핵심 추세)
  lookback = min(80, n)
  start_idx = n - lookback

  sub_highs = highs[start_idx:]
  sub_lows = lows[start_idx:]

  # 1. 상승 추세선 계산 (저점 연결)
  # 최근 구간 최저점 위치
  min_idx_rel = int(np.argmin(sub_lows[: lookback - 10]))
  p1_low_idx = start_idx + min_idx_rel
  p1_low_val = lows[p1_low_idx]

  # 최저점 이후 형성된 유의미한 눌림목 저점(Higher Low) 탐색
  best_up_slope = None
  best_p2_low_idx = None

  for i in range(p1_low_idx + 8, n - 2):
    # 로컬 저점인지 확인
    if lows[i] <= lows[i - 1] and lows[i] <= lows[i + 1]:
      slope = (lows[i] - p1_low_val) / (i - p1_low_idx)
      if slope > 0:  # 우상향하는 기울기
        # 캔들이 추세선을 심하게 하향 이탈하지 않는 가장 탄탄한 지지선 선택
        valid = True
        for k in range(p1_low_idx, n):
          line_val = p1_low_val + slope * (k - p1_low_idx)
          if closes[k] < line_val * 0.96:  # 4% 이상 이탈 시 무효
            valid = False
            break
        if valid:
          best_up_slope = slope
          best_p2_low_idx = i
          break

  # 2. 하락 추세선 계산 (고점 연결)
  # 최근 구간 최고점 위치
  max_idx_rel = int(np.argmax(sub_highs[: lookback - 10]))
  p1_high_idx = start_idx + max_idx_rel
  p1_high_val = highs[p1_high_idx]

  best_down_slope = None
  best_p2_high_idx = None

  for i in range(p1_high_idx + 8, n - 2):
    if highs[i] >= highs[i - 1] and highs[i] >= highs[i + 1]:
      slope = (highs[i] - p1_high_val) / (i - p1_high_idx)
      if slope < 0:  # 우하향하는 기울기
        valid = True
        for k in range(p1_high_idx, n):
          line_val = p1_high_val + slope * (k - p1_high_idx)
          if closes[k] > line_val * 1.04:  # 4% 이상 돌파 시 무효
            valid = False
            break
        if valid:
          best_down_slope = slope
          best_p2_high_idx = i
          break

  # 전체 배열에 선 생성
  up_line = [None] * n
  if best_up_slope is not None:
    for k in range(p1_low_idx, n):
      up_line[k] = p1_low_val + best_up_slope * (k - p1_low_idx)

  down_line = [None] * n
  if best_down_slope is not None:
    for k in range(p1_high_idx, n):
      down_line[k] = p1_high_val + best_down_slope * (k - p1_high_idx)

  return up_line, down_line


# ====================================================
# [정밀 퀀트 채점 엔진]
# ====================================================
def evaluate_pro_quant_score(df, df_inv, fund, short):
  score = 0
  logs = []
  latest = df.iloc[-1]
  prev = df.iloc[-2]

  # 1. 기술적 추세 (20점)
  tech_score = 0
  if latest["MA5"] > latest["MA20"] > latest["MA60"]:
    tech_score += 10
    logs.append(
        ("이평선 완전 정배열 (5>20>60)", "+10점", "단기·중기 완벽한 상승 추세")
    )
  elif latest["Close"] > latest["MA20"]:
    tech_score += 5
    logs.append(("20일선 지지 안착", "+5점", "단기 지지선 반등 흐름 유지"))
  else:
    tech_score -= 5
    logs.append(("20일선 하회", "-5점", "단기 하락 추세 지속"))

  bb_b = latest["BB_%b"]
  if 0.8 <= bb_b <= 1.1:
    tech_score += 5
    logs.append(("볼린저 상단 확장", "+5점", "강한 모멘텀 밴드라이딩 구간"))
  elif bb_b < 0.2:
    tech_score -= 3
    logs.append(("볼린저 하단 이탈 경계", "-3점", "하방 압력 과도"))

  if latest["MACD_HIST"] > 0 and prev["MACD_HIST"] <= 0:
    tech_score += 5
    logs.append(
        ("MACD 골든크로스 발생", "+5점", "상승 모멘텀 진입 신호 포착")
    )
  elif latest["MACD"] > latest["MACD_SIGNAL"]:
    tech_score += 3
    logs.append(("MACD 시그널 상회", "+3점", "매수 우위 흐름 지속"))
  score += max(0, min(20, tech_score))

  # 2. 수급 & MFI (25점)
  supply_score = 0
  if not df_inv.empty:
    for_5d_amt = df_inv["외인순매수금액"].head(5).sum()
    inst_5d_amt = df_inv["기관순매수금액"].head(5).sum()
    if for_5d_amt > 0 and inst_5d_amt > 0:
      supply_score += 12
      logs.append((
          "외인·기관 동반 순매수",
          "+12점",
          f"5일 외인({for_5d_amt:+.1f}억), 기관({inst_5d_amt:+.1f}억) 유입",
      ))
    elif for_5d_amt > 0 or inst_5d_amt > 0:
      supply_score += 6
      logs.append((
          "주포 수급 유입",
          "+6점",
          f"외인({for_5d_amt:+.1f}억) 또는 기관({inst_5d_amt:+.1f}억) 순매수",
      ))
    else:
      supply_score -= 5
      logs.append((
          "외인·기관 동반 매도세",
          "-5점",
          f"5일 외인({for_5d_amt:+.1f}억), 기관({inst_5d_amt:+.1f}억) 이탈",
      ))

    if len(df_inv) >= 10 and df_inv["외인보유율"].iloc[0] > df_inv[
        "외인보유율"
    ].iloc[9]:
      supply_score += 5
      logs.append((
          "외국인 지분율 확대",
          "+5점",
          f"현재 외인 지분율 {df_inv['외인보유율'].iloc[0]:.2f}%",
      ))

  mfi = latest["MFI"]
  if 50 <= mfi <= 75:
    supply_score += 5
    logs.append(
        ("MFI 스마트 머니 유입", "+5점", f"MFI {mfi:.1f} (대량 매집 진행)")
    )
  elif mfi > 80:
    supply_score -= 3
    logs.append(("MFI 단기 과열 경계", "-3점", f"MFI {mfi:.1f}"))

  if latest["OBV"] > df["OBV"].tail(20).mean():
    supply_score += 3
    logs.append(("OBV 매집 추세 유지", "+3점", "거래량 기반 매집 에너지 양호"))
  score += max(0, min(25, supply_score))

  # 3. 밸류 & 퀄리티 (25점)
  analyst_score = 0
  if fund["목표주가"] and fund["목표주가"] > 0:
    upside = ((fund["목표주가"] - latest["Close"]) / latest["Close"]) * 100
    if upside >= 25.0:
      analyst_score += 10
      logs.append((
          "목표가 괴리율 매력",
          "+10점",
          f"목표가 {fund['목표주가']:,.0f}원 (상승여력 {upside:+.1f}%)",
      ))
    elif upside >= 10.0:
      analyst_score += 6
      logs.append((
          "상승여력 유효",
          "+6점",
          f"목표가 {fund['목표주가']:,.0f}원 (상승여력 {upside:+.1f}%)",
      ))
    elif upside < 0:
      analyst_score -= 5
      logs.append((
          "목표가 초과 고평가",
          "-5점",
          f"현재가가 목표주가({fund['목표주가']:,.0f}원) 상회",
      ))

  if fund["ROE"] is not None:
    if fund["ROE"] >= 15.0:
      analyst_score += 7
      logs.append((
          "고수익 퀄리티 (ROE 15%↑)",
          "+7점",
          f"ROE {fund['ROE']:.2f}% (우수한 자본 효율성)",
      ))
    elif fund["ROE"] >= 8.0:
      analyst_score += 4
      logs.append((
          "안정적 자본 효율성",
          "+4점",
          f"ROE {fund['ROE']:.2f}% (안정적 이익 창출)",
      ))
    elif fund["ROE"] < 0:
      analyst_score -= 6
      logs.append(("ROE 마이너스 적자", "-6점", f"ROE {fund['ROE']:.2f}%"))

  if fund["PER"] is not None:
    if fund["PER"] < 0:
      analyst_score -= 5
      logs.append(("실적 적자", "-5점", "PER 음수"))
    elif fund["업종PER"] and fund["PER"] <= fund["업종PER"] * 0.7:
      analyst_score += 5
      logs.append((
          "업종 대비 저평가",
          "+5점",
          f"PER {fund['PER']}배 (업종 {fund['업종PER']}배)",
      ))

  if fund["PBR"] and fund["PBR"] < 0.9:
    analyst_score += 3
    logs.append(
        ("저PBR 자산 가치주", "+3점", f"PBR {fund['PBR']}배로 장부가치 하회")
    )
  score += max(0, min(25, analyst_score))

  # 4. 모멘텀 & 신고가 (20점)
  m_score = 0
  rsi = latest["RSI"]
  if 45 <= rsi <= 65:
    m_score += 10
    logs.append(
        ("이상적 RSI 구간", "+10점", f"RSI {rsi:.1f} (과열 없는 안정적 추세)")
    )
  elif 30 <= rsi < 45:
    m_score += 5
    logs.append(("바닥권 반등 구간", "+5점", f"RSI {rsi:.1f}"))
  elif rsi > 75:
    m_score -= 5
    logs.append(("단기 과열 경계", "-5점", f"RSI {rsi:.1f}"))

  high_52w = df["High"].max()
  dist_high = ((latest["Close"] - high_52w) / high_52w) * 100
  if dist_high >= -7.0:
    m_score += 10
    logs.append((
        "52주 신고가 근접 (12M 모멘텀)",
        "+10점",
        f"최고가 대비 {dist_high:.1f}% 위치 (강한 주도력)",
    ))
  elif dist_high <= -35.0:
    m_score -= 5
    logs.append(
        ("장기 낙폭 과대 역배열", "-5점", f"최고가 대비 {dist_high:.1f}% 하락")
    )
  score += max(0, min(20, m_score))

  # 5. 공매도 리스크 (10점)
  risk_score = 10
  if short["공매도비중"] >= 15.0:
    risk_score -= 8
    logs.append((
        "공매도 폭탄 경보",
        "-8점",
        f"공매도 비중 {short['공매도비중']:.2f}% (하방 압력 과도)",
    ))
  elif short["공매도비중"] >= 7.0:
    risk_score -= 4
    logs.append((
        "공매도 경계",
        "-4점",
        f"공매도 비중 {short['공매도비중']:.2f}%",
    ))
  score += max(0, min(10, risk_score))

  final_score = int(max(10, min(100, score)))
  if final_score >= 85:
    grade, stars = "🌟 슈퍼 주도주 (강력 매수)", "★★★★★"
  elif final_score >= 70:
    grade, stars = "👍 우량 상승주 (분할 매수)", "★★★★☆"
  elif final_score >= 50:
    grade, stars = "⚖️ 중립 관망 (추세 확인)", "★★★☆☆"
  elif final_score >= 35:
    grade, stars = "⚠️ 약세 지속 (비중 축소)", "★★☆☆☆"
  else:
    grade, stars = "🚨 고위험 종목 (진입 금지)", "★☆☆☆☆"

  return final_score, grade, stars, logs


# ====================================================
# [신뢰도 100% 랭킹 엔진] 주도주 후보 퀀트 정밀 채점
# ====================================================
@st.cache_data(ttl=1800)
def generate_accurate_market_ranking():
  krx = get_krx_listing()
  # 거래대금 50억 이상 & 시총 5000억 이상 우량 유동성 종목 선별
  candidates = (
      krx[(krx["Volume"] > 0) & (krx["Amount"] >= 5000000000)]
      .sort_values(by="Amount", ascending=False)
      .head(80)
  )

  results = []
  for _, row in candidates.iterrows():
    c_code = row["Code"]
    c_name = row["Name"]
    c_close = row["Close"]
    c_chg = row["ChagesRatio"]

    try:
      # 최근 60일 데이터로 퀀트 핵심 스코어링
      c_df = fdr.DataReader(
          c_code, (datetime.today() - timedelta(days=100)).strftime("%Y-%m-%d")
      )
      if len(c_df) < 30:
        continue

      c_df["MA5"] = c_df["Close"].rolling(5).mean()
      c_df["MA20"] = c_df["Close"].rolling(20).mean()
      c_df["MA60"] = c_df["Close"].rolling(60).mean()

      delta = c_df["Close"].diff()
      gain = (delta.where(delta > 0, 0)).rolling(14).mean()
      loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
      rsi = 100 - (100 / (1 + (gain / (loss + 1e-9)))).iloc[-1]

      c_latest = c_df.iloc[-1]
      high_52 = c_df["High"].max()
      dist_high = ((c_latest["Close"] - high_52) / high_52) * 100

      # 정밀 계산과 일치하는 팩터 점수 산출
      score = 50
      # 1. 정배열 점수
      if c_latest["MA5"] > c_latest["MA20"] > c_latest["MA60"]:
        score += 20
      elif c_latest["Close"] > c_latest["MA20"]:
        score += 8
      else:
        score -= 15  # 20일선 아래 역배열 시 대폭 감점

      # 2. RSI 모멘텀
      if 45 <= rsi <= 65:
        score += 12
      elif rsi < 35:
        score -= 8

      # 3. 52주 신고가 이격도
      if dist_high >= -7:
        score += 15
      elif dist_high <= -25:
        score -= 12

      # 4. 거래대금 규모
      amt_억 = row["Amount"] / 100000000
      if amt_억 >= 1000:
        score += 10
      elif amt_억 >= 300:
        score += 5

      final_sc = int(max(15, min(95, score)))
      results.append({
          "Code": c_code,
          "Name": c_name,
          "Close": c_close,
          "등락률": f"{c_chg:+.2f}%",
          "RawChg": c_chg,
          "점수": final_sc,
          "Amount": row["Amount"],
      })
    except Exception:
      continue

  df_res = pd.DataFrame(results)
  if df_res.empty:
    return pd.DataFrame(), pd.DataFrame()

  top20 = (
      df_res.sort_values(by=["점수", "Amount"], ascending=[False, False])
      .head(20)
      .reset_index(drop=True)
  )
  bot20 = (
      df_res.sort_values(by=["점수", "RawChg"], ascending=[True, True])
      .head(20)
      .reset_index(drop=True)
  )
  return top20, bot20


# ====================================================
# 메인 헤더 & 레이아웃
# ====================================================
st.markdown(
    """
    <div class="header-card">
        <div>
            <div style="font-size: 13px; color: #38bdf8; font-weight:600; margin-bottom:2px;">QUANT INSIGHT ENGINE</div>
            <h2 style="margin:0; font-size:22px; font-weight:800; color:#f8fafc;">부리부리 종합 주식 작전실</h2>
        </div>
        <div style="font-size: 26px;">🐽📊</div>
    </div>
""",
    unsafe_allow_html=True,
)

main_col, rank_col = st.columns([7, 3])

# 1. 오른쪽 시장 랭킹 (동기화된 신뢰도 100% 퀀트 랭킹)
with rank_col:
  st.markdown(
      "<div style='font-size:15px; font-weight:700; margin-bottom:10px;'"
      ">🏆 퀀트 검증 시장 랭킹 (클릭 시 분석)</div>",
      unsafe_allow_html=True,
  )
  with st.spinner("퀀트 팩터 기반 랭킹 검증 중..."):
    top_20, bottom_20 = generate_accurate_market_ranking()

  rank_tab1, rank_tab2 = st.tabs(
      ["🔥 상위 TOP 20", "❄️ 하위 TOP 20"]
  )

  def render_rank_buttons(df_rank, prefix):
    if df_rank.empty:
      st.write("데이터 수집 중...")
      return
    for i, row in df_rank.iterrows():
      cols = st.columns([5, 3, 2])
      if cols[0].button(
          f"{i+1}. {row['Name']}",
          key=f"{prefix}_{row['Code']}",
          use_container_width=True,
      ):
        st.session_state.selected_stock = row["Name"]
        st.rerun()
      cols[1].markdown(
          f"<div style='text-align:right; font-size:13px; padding-top:6px;'>"
          f" {row['Close']:,}원 ({row['등락률']})</div>",
          unsafe_allow_html=True,
      )
      cols[2].markdown(
          f"<div style='text-align:center; font-weight:700; font-size:13px;"
          f" color:#38bdf8; padding-top:6px;'>{row['점수']}점</div>",
          unsafe_allow_html=True,
      )

  with rank_tab1:
    render_rank_buttons(top_20, "top")
  with rank_tab2:
    render_rank_buttons(bottom_20, "bot")


# 2. 왼쪽 메인 정밀 분석
with main_col:
  col_s1, col_s2 = st.columns([4, 1])
  default_search = st.session_state.selected_stock
  search_input = col_s1.text_input(
      "종목 검색",
      value=default_search,
      placeholder="종목명(예: 삼성전자, 현대차, SK하이닉스) 또는 6자리 코드 입력",
      label_visibility="collapsed",
  )
  analyze_btn = col_s2.button(
      "🚀 정밀 분석", type="primary", use_container_width=True
  )

  if search_input != st.session_state.selected_stock:
    st.session_state.selected_stock = search_input

  if search_input.strip():
    code, stock_name = resolve_stock_code(search_input)

    if not code:
      st.error(f"'{search_input}' 종목을 찾을 수 없습니다.")
    else:
      with st.spinner(f"[{stock_name}] 정밀 퀀트 분석 중..."):
        end_dt = datetime.today()
        start_dt = end_dt - timedelta(days=365)
        df = fdr.DataReader(code, start_dt.strftime("%Y-%m-%d"))

        if df.empty or len(df) < 60:
          st.error("데이터 수집에 실패했거나 거래일 데이터가 부족합니다.")
        else:
          latest_price = df["Close"].iloc[-1]
          prev_price = df["Close"].iloc[-2]

          df["MA5"] = df["Close"].rolling(5).mean()
          df["MA20"] = df["Close"].rolling(20).mean()
          df["MA60"] = df["Close"].rolling(60).mean()
          df["STD20"] = df["Close"].rolling(20).std()
          df["BB_Upper"] = df["MA20"] + (df["STD20"] * 2)
          df["BB_Lower"] = df["MA20"] - (df["STD20"] * 2)
          df["BB_%b"] = (df["Close"] - df["BB_Lower"]) / (
              df["BB_Upper"] - df["BB_Lower"] + 1e-9
          )

          exp12 = df["Close"].ewm(span=12, adjust=False).mean()
          exp26 = df["Close"].ewm(span=26, adjust=False).mean()
          df["MACD"] = exp12 - exp26
          df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()
          df["MACD_HIST"] = df["MACD"] - df["MACD_SIGNAL"]

          delta = df["Close"].diff()
          gain = (delta.where(delta > 0, 0)).rolling(14).mean()
          loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
          df["RSI"] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
          df["OBV"] = (
              np.sign(df["Close"].diff()).fillna(0) * df["Volume"]
          ).cumsum()

          tp = (df["High"] + df["Low"] + df["Close"]) / 3
          rmf = tp * df["Volume"]
          pos_mf = (rmf.where(tp > tp.shift(1), 0)).rolling(14).sum()
          neg_mf = (rmf.where(tp < tp.shift(1), 0)).rolling(14).sum()
          mfr = pos_mf / (neg_mf + 1e-9)
          df["MFI"] = 100 - (100 / (1 + mfr))

          ret_1d = ((latest_price - prev_price) / prev_price) * 100
          ret_1w = (
              (latest_price - df["Close"].iloc[-5]) / df["Close"].iloc[-5]
          ) * 100
          ret_1m = (
              (latest_price - df["Close"].iloc[-20]) / df["Close"].iloc[-20]
          ) * 100
          ret_1y = (
              (latest_price - df["Close"].iloc[0]) / df["Close"].iloc[0]
          ) * 100

          high_20 = df["High"].tail(20).max()
          low_20 = df["Low"].tail(20).min()

          # 매물대 프로파일
          price_min, price_max = df["Low"].min(), df["High"].max()
          bins = np.linspace(price_min, price_max, 13)
          v_counts, _ = np.histogram(
              df["Close"], bins=bins, weights=df["Volume"]
          )

          # 차티스트 정통 상승/하락 추세선 작도
          up_trend, down_trend = calculate_chartist_trendlines(df)

          df_inv = fetch_investor_naver(code)
          fund = fetch_fundamental_and_consensus(code)
          short = fetch_short_selling(code)
          news_items = fetch_news(stock_name)

          for_5d = (
              df_inv["외인순매수금액"].head(5).sum()
              if not df_inv.empty
              else 0
          )
          inst_5d = (
              df_inv["기관순매수금액"].head(5).sum()
              if not df_inv.empty
              else 0
          )
          for_20d = (
              df_inv["외인순매수금액"].head(20).sum()
              if not df_inv.empty
              else 0
          )
          for_rate = (
              df_inv["외인보유율"].iloc[0] if not df_inv.empty else 0.0
          )

          total_score, grade_text, stars, logs = evaluate_pro_quant_score(
              df, df_inv, fund, short
          )

          entry_1 = round(latest_price * 0.99, -2)
          entry_2 = round(df["MA20"].iloc[-1], -2)
          t1_calc = max(high_20 * 1.02, latest_price * 1.07)
          target_1 = round(t1_calc, -2)

          if fund["목표주가"] and fund["목표주가"] > target_1 * 1.05:
            target_2 = round(fund["목표주가"], -2)
          else:
            target_2 = round(target_1 * 1.10, -2)

          stop_loss = round(min(low_20 * 0.98, latest_price * 0.94), -2)

          target_grid_html = (
              f'<div class="score-container">'
              f'<div style="font-size: 13px; color: #94a3b8; font-weight:600;">{stock_name} ({code})</div>'
              f'<div style="font-size: 44px; color: #38bdf8; font-weight: 800; margin: 2px 0;">{total_score}<span style="font-size:18px; color:#64748b;"> / 100</span></div>'
              f'<div style="font-size: 15px; font-weight: 600; color: #f1f5f9; margin-bottom: 12px;">{grade_text} <span style="color:#eab308;">{stars}</span></div>'
              f'<div class="target-grid">'
              f'<div class="target-item"><div class="target-title">1차 진입 (현재가)</div><div class="target-val">{entry_1:,.0f}원</div></div>'
              f'<div class="target-item"><div class="target-title">2차 진입 (눌림목)</div><div class="target-val">{entry_2:,.0f}원</div></div>'
              f'<div class="target-item"><div class="target-title">1차 목표 (단기 저항)</div><div class="target-val">{target_1:,.0f}원</div></div>'
              f'<div class="target-item"><div class="target-title">2차 목표 (추세 확장)</div><div class="target-val">{target_2:,.0f}원</div></div>'
              f'<div class="target-item"><div class="target-title">손절 기준선</div><div class="target-val" style="color:#ef4444;">{stop_loss:,.0f}원</div></div>'
              f"</div>"
              f"</div>"
          )
          st.markdown(target_grid_html, unsafe_allow_html=True)

          t1, t2, t3, t4, t5 = st.tabs([
              "차트 & 정통 추세선 & 매물대",
              "외인/기관 수급",
              "전망 & 애널리스트",
              "퀀트 채점표",
              "뉴스 브리핑",
          ])

          with t1:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("현재가", f"{latest_price:,.0f}원", f"{ret_1d:+.2f}%")
            c2.metric("1주일", f"{ret_1w:+.2f}%")
            c3.metric("1개월", f"{ret_1m:+.2f}%")
            c4.metric("1년(모멘텀)", f"{ret_1y:+.2f}%")

            fig = make_subplots(
                rows=2,
                cols=2,
                shared_xaxes=True,
                row_heights=[0.72, 0.28],
                column_widths=[0.85, 0.15],
                horizontal_spacing=0.01,
                vertical_spacing=0.04,
                specs=[[{}, {}], [{}, None]],
            )

            # 1. 캔들스틱 차트
            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df["Open"],
                    high=df["High"],
                    low=df["Low"],
                    close=df["Close"],
                    name="주가",
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["MA20"],
                    line=dict(color="#38bdf8", width=1.3),
                    name="20일선",
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["MA60"],
                    line=dict(color="#10b981", width=1.3),
                    name="60일선",
                ),
                row=1,
                col=1,
            )

            # 2. 정통 차티스트 빗금 추세선 (고점-저점 실선 연결)
            if any(v is not None for v in up_trend):
              fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=up_trend,
                    mode="lines",
                    line=dict(color="#22c55e", width=2.2),
                    name="상승 지지선",
                ),
                row=1,
                col=1,
              )
            if any(v is not None for v in down_trend):
              fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=down_trend,
                    mode="lines",
                    line=dict(color="#f43f5e", width=2.2),
                    name="하락 저항선",
                ),
                row=1,
                col=1,
              )

            # 3. 우측 매물대 프로파일 바
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            fig.add_trace(
                go.Bar(
                    y=bin_centers,
                    x=v_counts,
                    orientation="h",
                    marker_color="rgba(56, 189, 248, 0.35)",
                    showlegend=False,
                    hoverinfo="none",
                ),
                row=1,
                col=2,
            )

            # 4. 하단 MACD 서브 차트
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["MACD"],
                    line=dict(color="#f43f5e", width=1.2),
                    name="MACD",
                ),
                row=2,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["MACD_SIGNAL"],
                    line=dict(color="#fbbf24", width=1.2),
                    name="Signal",
                ),
                row=2,
                col=1,
            )

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=450,
                margin=dict(l=5, r=5, t=5, b=5),
                xaxis_rangeslider_visible=False,
            )
            fig.update_xaxes(showticklabels=False, row=1, col=2)
            fig.update_yaxes(showticklabels=False, row=1, col=2)
            st.plotly_chart(fig, use_container_width=True)

          with t2:
            s1, s2, s3, s4, s5 = st.columns(5)
            s1.metric("5일 외국인", f"{for_5d:+.1f}억원")
            s2.metric("5일 기관", f"{inst_5d:+.1f}억원")
            s3.metric("20일 외국인", f"{for_20d:+.1f}억원")
            s4.metric("외인 지분율", f"{for_rate:.2f}%")
            s5.metric("MFI(자금유입)", f"{df['MFI'].iloc[-1]:.1f} pt")

            if not df_inv.empty:
              show_inv = df_inv.head(10)[
                  ["날짜", "종가", "기관순매수금액", "외인순매수금액", "외인보유율"]
              ].copy()
              show_inv["기관순매수금액"] = show_inv[
                  "기관순매수금액"
              ].apply(lambda x: f"{x:+.1f}억원")
              show_inv["외인순매수금액"] = show_inv[
                  "외인순매수금액"
              ].apply(lambda x: f"{x:+.1f}억원")
              show_inv["외인보유율"] = show_inv["외인보유율"].apply(
                  lambda x: f"{x:.2f}%"
              )
              st.dataframe(show_inv, use_container_width=True)

          with t3:
            st.markdown("#### 🏢 기업 핵심 개요 & 사업 방향")
            st.markdown(
                f'<div class="insight-card">{fund["기업개요"]}</div>',
                unsafe_allow_html=True,
            )

            st.markdown("#### 📊 펀더멘털 & 애널리스트 컨센서스")
            v1, v2, v3, v4, v5 = st.columns(5)
            v1.metric(
                "목표주가",
                (
                    f"{fund['목표주가']:,.0f}원"
                    if fund["목표주가"]
                    else "미제공"
                ),
                (
                    f"상승여력"
                    f" {((fund['목표주가']-latest_price)/latest_price)*100:+.1f}%"
                    if fund["목표주가"]
                    else ""
                ),
            )
            v2.metric("ROE (퀄리티)", f"{fund['ROE'] or '-'}%")
            v3.metric(
                "PER / 업종",
                f"{fund['PER'] or '-'}배",
                f"업종 {fund['업종PER'] or '-'}배",
            )
            v4.metric("PBR", f"{fund['PBR'] or '-'}배")
            v5.metric("공매도비중", f"{short['공매도비중']:.2f}%")

            st.markdown("#### 📑 최신 증권사 애널리스트 리포트")
            if fund["리포트_목록"]:
              for rep in fund["리포트_목록"]:
                st.markdown(
                    f"- **[{rep['broker']}]** [{rep['title']}]({rep['link']})"
                    f" <span style='color:#64748b; font-size:12px;'>({rep['date']})</span>",
                    unsafe_allow_html=True,
                )
            else:
              st.write("최근 등록된 증권사 분석 리포트가 없습니다.")

          with t4:
            st.dataframe(
                pd.DataFrame(
                    logs, columns=["평가 항목", "가감점", "상세 내용"]
                ),
                use_container_width=True,
            )

          with t5:
            for item in news_items:
              st.markdown(f"- [{item['title']}]({item['link']})")
  else:
    st.info(
        "💡 상단 검색창에 **종목명**을 입력하시거나, 우측 **실시간 시장 퀀트"
        " 랭킹의 종목을 클릭**하시면 즉시 정밀 퀀트 분석이 시작됩니다부리!"
    )
