from datetime import datetime, timedelta
import urllib.parse
from bs4 import BeautifulSoup
import FinanceDataReader as fdr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st

st.set_page_config(
    page_title="부리부리 7대 종합 주식 작전실",
    page_icon="🐷",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(135deg, #0d1117 0%, #161b22 100%); color: #c9d1d9; }
    .hero-banner { 
        background: linear-gradient(90deg, #ff758c 0%, #ff7eb3 100%); 
        padding: 18px 24px; 
        border-radius: 14px; 
        color: white; 
        margin-bottom: 20px; 
        display: flex; 
        justify-content: space-between; 
        align-items: center; 
        box-shadow: 0 4px 20px rgba(255,117,140,0.2); 
    }
    .score-box { 
        background: rgba(255, 255, 255, 0.04); 
        border: 2px solid #ff7eb3; 
        border-radius: 14px; 
        padding: 20px; 
        text-align: center; 
        margin-bottom: 15px;
    }
    .strategy-card { 
        background: rgba(56, 139, 253, 0.08); 
        border-left: 4px solid #58a6ff; 
        padding: 15px 20px; 
        border-radius: 8px; 
        margin-top: 15px; 
    }
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 10px;
        border-radius: 10px;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# 종목코드 매핑
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


# 수급 크롤러
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


# 펀더멘털 & 컨센서스 크롤러
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
  except Exception:
    pass
  return data


# 공매도 크롤러
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


# 뉴스 크롤러
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


# 단일 종목 고속 스캔 퀀트 채점 (사이드 랭킹용)
def calculate_quick_score(row):
  score = 50
  chg = row.get("ChagesRatio", 0)
  if chg >= 5.0:
    score += 15
  elif chg > 0:
    score += 8
  elif chg <= -5.0:
    score -= 15
  else:
    score -= 8

  amt = row.get("Amount", 0) / 100000000
  if amt >= 1000:
    score += 20
  elif amt >= 300:
    score += 10
  elif amt < 50:
    score -= 10

  marcap = row.get("Marcap", 0) / 100000000
  if marcap >= 50000:
    score += 10
  elif marcap >= 10000:
    score += 5

  return int(max(10, min(98, score)))


# 메인 상세 퀀트 채점
def evaluate_pro_quant_score(df, df_inv, fund, short):
  score = 0
  logs = []
  latest = df.iloc[-1]
  prev = df.iloc[-2]

  # 1. 기술적 지표 (20점)
  tech_score = 0
  if latest["MA5"] > latest["MA20"] > latest["MA60"]:
    tech_score += 12
    logs.append(
        ("이평선 완전 정배열 (5>20>60)", "+12점", "단기·중기 완벽한 상승 추세")
    )
  elif latest["Close"] > latest["MA20"]:
    tech_score += 6
    logs.append(("20일 이동평균선 안착", "+6점", "단기 지지선 반등 흐름 유지"))
  else:
    tech_score -= 5
    logs.append(("20일 이동평균선 하회", "-5점", "단기 하락 추세 지속"))

  if latest["MACD_HIST"] > 0 and prev["MACD_HIST"] <= 0:
    tech_score += 8
    logs.append(
        ("MACD 골든크로스 발생", "+8점", "상승 모멘텀 진입 신호 포착")
    )
  elif latest["MACD"] > latest["MACD_SIGNAL"]:
    tech_score += 4
    logs.append(("MACD 시그널 상회", "+4점", "매수 우위 흐름 지속"))
  score += max(0, min(20, tech_score))

  # 2. 수급 에너지 (25점)
  supply_score = 0
  if not df_inv.empty:
    for_5d_amt = df_inv["외인순매수금액"].head(5).sum()
    inst_5d_amt = df_inv["기관순매수금액"].head(5).sum()
    if for_5d_amt > 0 and inst_5d_amt > 0:
      supply_score += 15
      logs.append((
          "외인·기관 쌍끌이 동반 순매수",
          "+15점",
          f"5일 외인({for_5d_amt:+.1f}억), 기관({inst_5d_amt:+.1f}억) 유입",
      ))
    elif for_5d_amt > 0 or inst_5d_amt > 0:
      supply_score += 8
      logs.append((
          "메이저 수급 유입",
          "+8점",
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
          "외국인 지분율 상승세",
          "+5점",
          f"현재 외인 지분율 {df_inv['외인보유율'].iloc[0]:.2f}%",
      ))

  if latest["OBV"] > df["OBV"].tail(20).mean():
    supply_score += 5
    logs.append(("OBV 매집 시그널", "+5점", "거래량 기반 매집 에너지 지속"))
  score += max(0, min(25, supply_score))

  # 3. 밸류 & 컨센서스 (25점)
  analyst_score = 0
  if fund["목표주가"] and fund["목표주가"] > 0:
    upside = ((fund["목표주가"] - latest["Close"]) / latest["Close"]) * 100
    if upside >= 25.0:
      analyst_score += 12
      logs.append((
          "목표가 괴리율 매력",
          "+12점",
          f"목표가 {fund['목표주가']:,.0f}원 (상승여력 {upside:+.1f}%)",
      ))
    elif upside >= 10.0:
      analyst_score += 7
      logs.append((
          "상승여력 유효",
          "+7점",
          f"목표가 {fund['목표주가']:,.0f}원 (상승여력 {upside:+.1f}%)",
      ))
    elif upside < 0:
      analyst_score -= 5
      logs.append((
          "목표가 초과 고평가",
          "-5점",
          f"현재가가 목표주가({fund['목표주가']:,.0f}원) 상회",
      ))

  if fund["PER"] is not None:
    if fund["PER"] < 0:
      analyst_score -= 8
      logs.append(("실적 적자 지속", "-8점", "PER 음수 기업"))
    elif fund["업종PER"] and fund["PER"] <= fund["업종PER"] * 0.7:
      analyst_score += 8
      logs.append((
          "업종 대비 저평가",
          "+8점",
          f"PER {fund['PER']}배 (업종 {fund['업종PER']}배)",
      ))
  if fund["PBR"] and fund["PBR"] < 0.9:
    analyst_score += 5
    logs.append(
        ("저PBR 자산주", "+5점", f"PBR {fund['PBR']}배로 청산가치 하회")
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
    logs.append(
        ("단기 과매수 주의", "-5점", f"RSI {rsi:.1f} (차익 실현 매물 경계)")
    )

  high_52w = df["High"].max()
  dist_high = ((latest["Close"] - high_52w) / high_52w) * 100
  if dist_high >= -7.0:
    m_score += 10
    logs.append((
        "52주 신고가 근접",
        "+10점",
        f"최고가 대비 {dist_high:.1f}% 위치 (강한 주도력)",
    ))
  elif dist_high <= -35.0:
    m_score -= 5
    logs.append(
        ("장기 낙폭 과대", "-5점", f"최고가 대비 {dist_high:.1f}% 하락")
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
# 메인 상단 헤더
# ====================================================
st.markdown(
    """
    <div class="hero-banner">
        <div>
            <h2 style="margin:0; font-weight:800;">🐷 부리부리 7차원 종목 정밀 진단 & 랭킹 시스템</h2>
            <p style="margin:4px 0 0 0; font-size:13px; opacity:0.95;">정밀 퀀트 분석과 실시간 시장 점수 TOP 20 / 하위 20 랭킹 대시보드</p>
        </div>
        <div style="font-size: 38px;">🐽📊</div>
    </div>
""",
    unsafe_allow_html=True,
)

main_col, rank_col = st.columns([7, 3])

# 1. 오른쪽 랭킹
with rank_col:
  st.markdown("### 🏆 시장 퀀트 랭킹")
  with st.spinner("시장 주도주 스캔 중..."):
    krx_all = get_krx_listing()
    active_krx = krx_all[
        (krx_all["Volume"] > 0) & (krx_all["Amount"] >= 5000000000)
    ].copy()
    active_krx["점수"] = active_krx.apply(calculate_quick_score, axis=1)
    active_krx["등락률"] = active_krx["ChagesRatio"].apply(
        lambda x: f"{x:+.2f}%"
    )

    top_20 = (
        active_krx.sort_values(
            by=["점수", "Amount"], ascending=[False, False]
        )
        .head(20)[["Name", "Close", "등락률", "점수"]]
        .reset_index(drop=True)
    )
    bottom_20 = (
        active_krx.sort_values(by=["점수", "ChagesRatio"], ascending=[True, True])
        .head(20)[["Name", "Close", "등락률", "점수"]]
        .reset_index(drop=True)
    )

  rank_tab1, rank_tab2 = st.tabs(
      ["🔥 상위 TOP 20", "❄️ 하위 TOP 20"]
  )
  with rank_tab1:
    st.dataframe(
        top_20.rename(
            columns={"Name": "종목명", "Close": "현재가", "등락률": "등락"}
        ),
        use_container_width=True,
        height=520,
    )
  with rank_tab2:
    st.dataframe(
        bottom_20.rename(
            columns={"Name": "종목명", "Close": "현재가", "등락률": "등락"}
        ),
        use_container_width=True,
        height=520,
    )

# 2. 왼쪽 메인 정밀 분석
with main_col:
  col_s1, col_s2 = st.columns([4, 1])
  search_input = col_s1.text_input(
      "🔍 분석할 종목명 또는 종목코드",
      value="",  # 기본값을 빈칸으로 설정!
      placeholder="예: 삼성전자, 현대차, SK하이닉스, 000660 등 입력",
  )
  analyze_btn = col_s2.button(
      "🚀 정밀 분석", type="primary", use_container_width=True
  )

  if search_input.strip():
    code, stock_name = resolve_stock_code(search_input)

    if not code:
      st.error(f"'{search_input}' 종목을 찾을 수 없습니다.")
    else:
      with st.spinner(
          f"부리부리 대마왕이 [{stock_name}]의 데이터를 정밀 채점 중..."
      ):
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

          exp12 = df["Close"].ewm(span=12, adjust=False).mean()
          exp26 = df["Close"].ewm(span=26, adjust=False).mean()
          df["MACD"] = exp12 - exp26
          df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()
          df["MACD_HIST"] = df["MACD"] - df["MACD_SIGNAL"]

          delta = df["Close"].diff()
          gain = (delta.where(delta > 0, 0)).rolling(14).mean()
          loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
          df["RSI"] = 100 - (100 / (1 + (gain / loss)))
          df["OBV"] = (
              np.sign(df["Close"].diff()).fillna(0) * df["Volume"]
          ).cumsum()

          ret_1d = ((latest_price - prev_price) / prev_price) * 100
          ret_1w = (
              (latest_price - df["Close"].iloc[-5]) / df["Close"].iloc[-5]
          ) * 100
          ret_1m = (
              (latest_price - df["Close"].iloc[-20]) / df["Close"].iloc[-20]
          ) * 100
          ret_3m = (
              (latest_price - df["Close"].iloc[-60]) / df["Close"].iloc[-60]
        ) * 100
          ret_1y = (
              (latest_price - df["Close"].iloc[0]) / df["Close"].iloc[0]
          ) * 100

          high_20 = df["High"].tail(20).max()
          low_20 = df["Low"].tail(20).min()

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
          stop_loss = round(low_20 * 0.97, -2)
          target_1 = (
              fund["목표주가"]
              if fund["목표주가"]
              else round(high_20 * 1.05, -2)
          )
          target_2 = round(latest_price * 1.15, -2)

          st.markdown(
              f"""
                <div class="score-box">
                    <h3 style="color:#ff7eb3; margin:0;">[{stock_name} ({code})] 종합 매력도 진단 리포트</h3>
                    <h1 style="font-size: 52px; color: #ffd166; margin: 6px 0; font-weight:800;">{total_score}점 / 100점</h1>
                    <h3 style="margin:0;">판정: {grade_text} ({stars})</h3>
                </div>
                """,
              unsafe_allow_html=True,
          )

          st.markdown(
              f"""
            <div class="strategy-card">
                <h4 style="margin:0 0 6px 0; color:#58a6ff;">🎯 부리부리 매매 가격표</h4>
                <div style="display: flex; gap: 15px; flex-wrap: wrap; font-size:13px;">
                    <div>🔹 <b>1차 진입:</b> {entry_1:,.0f}원</div>
                    <div>🔹 <b>2차 진입:</b> {entry_2:,.0f}원</div>
                    <div>🎯 <b>1차 목표가:</b> {target_1:,.0f}원</div>
                    <div>🚀 <b>2차 목표가:</b> {target_2:,.0f}원</div>
                    <div>🚨 <b>손절선:</b> {stop_loss:,.0f}원</div>
                </div>
            </div>
            """,
              unsafe_allow_html=True,
          )

          st.divider()

          t1, t2, t3, t4, t5 = st.tabs([
              "① 주가 & 차트",
              "② 외인/기관 수급",
              "③ 밸류/컨센서스",
              "④ 채점 근거표",
              "⑤ 뉴스/재료",
          ])

          with t1:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("현재가", f"{latest_price:,.0f}원", f"{ret_1d:+.2f}%")
            c2.metric("1주일", f"{ret_1w:+.2f}%")
            c3.metric("1개월", f"{ret_1m:+.2f}%")
            c4.metric("1년", f"{ret_1y:+.2f}%")

            fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                row_heights=[0.7, 0.3],
                vertical_spacing=0.04,
            )
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
                    line=dict(color="#58a6ff", width=1.5),
                    name="20일선",
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["MA60"],
                    line=dict(color="#238636", width=1.5),
                    name="60일선",
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["MACD"],
                    line=dict(color="#ff758c", width=1.2),
                    name="MACD",
                ),
                row=2,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["MACD_SIGNAL"],
                    line=dict(color="#ffd166", width=1.2),
                    name="Signal",
                ),
                row=2,
                col=1,
            )
            fig.update_layout(
                template="plotly_dark",
                height=450,
                margin=dict(l=5, r=5, t=5, b=5),
                xaxis_rangeslider_visible=False,
            )
            st.plotly_chart(fig, use_container_width=True)

          with t2:
            st.subheader("🏦 외인 & 기관 실시간 수급")
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("5일 외국인", f"{for_5d:+.1f}억원")
            s2.metric("5일 기관", f"{inst_5d:+.1f}억원")
            s3.metric("20일 외국인", f"{for_20d:+.1f}억원")
            s4.metric("외인 지분율", f"{for_rate:.2f}%")

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
            v1, v2, v3, v4 = st.columns(4)
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
            v2.metric(
                "PER / 업종",
                f"{fund['PER'] or '-'}배",
                f"업종 {fund['업종PER'] or '-'}배",
            )
            v3.metric("PBR", f"{fund['PBR'] or '-'}배")
            v4.metric("공매도비중", f"{short['공매도비중']:.2f}%")

          with t4:
            st.markdown("### 📋 퀀트 채점 가감점 내역")
            st.dataframe(
                pd.DataFrame(
                    logs, columns=["평가 항목", "가감점", "상세 내용"]
                ),
                use_container_width=True,
            )

          with t5:
            st.subheader(f"📰 [{stock_name}] 주요 뉴스")
            for item in news_items:
              st.markdown(f"- 🐽 [{item['title']}]({item['link']})")
  else:
    # 검색어가 없을 때 뜨는 초기 대기 카드
    st.info(
        "💡 상단 검색창에 분석하고자 하는 **종목명(예: 삼성전자, 현대차)** 또는"
        " **6자리 종목코드**를 입력하신 후 엔터를 누르시거나 [🚀 정밀 분석]"
        " 버튼을 눌러주세요부리!"
    )
