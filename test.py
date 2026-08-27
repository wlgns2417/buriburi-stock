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
    page_title="부리부리 종합 주식 작전실",
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

    .autocomplete-box {
        background: rgba(22, 27, 34, 0.95);
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 14px;
        padding: 12px 16px;
        margin-bottom: 15px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.4);
    }
    .autocomplete-header {
        font-size: 12px;
        color: #94a3b8;
        font-weight: 600;
        margin-bottom: 8px;
        display: flex;
        justify-content: space-between;
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
    matched_part = krx[krx["Name"].str.contains(query, case=False, na=False)]
    if not matched_part.empty:
        return matched_part.iloc[0]["Code"], matched_part.iloc[0]["Name"]
    return None, None


def search_similar_stocks(query):
    query = query.strip()
    if not query:
        return pd.DataFrame()
    krx = get_krx_listing()
    if query.isdigit():
        matched = krx[krx["Code"].str.startswith(query)].copy()
    else:
        matched = krx[krx["Name"].str.contains(query, case=False, na=False)].copy()
    
    if matched.empty:
        return pd.DataFrame()
    
    return matched.sort_values(by="Amount", ascending=False).head(5)


def fetch_investor_naver(code):
    url = f"https://finance.naver.com/item/frgn.naver?code={code}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
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
                    for_rate = float(tds[8].text.strip().replace("%", "").replace(",", ""))
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
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    data = {
        "PER": None, "PBR": None, "배당수익률": None, "업종PER": None,
        "목표주가": None, "ROE": None, "기업개요": "기업 정보 준비 중", "리포트_목록": []
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

        report_url = f"https://finance.naver.com/item/research.naver?code={code}"
        res_rep = requests.get(report_url, headers=headers, timeout=5)
        soup_rep = BeautifulSoup(res_rep.text, "html.parser")
        for tr in soup_rep.select("table.type2 tr")[2:7]:
            tds = tr.select("td")
            if len(tds) >= 4 and tds[0].text.strip():
                title = tds[0].text.strip()
                broker = tds[2].text.strip()
                date = tds[3].text.strip()
                link_tag = tds[0].select_one("a")
                link = "https://finance.naver.com" + link_tag["href"] if link_tag else "#"
                data["리포트_목록"].append({"title": title, "broker": broker, "date": date, "link": link})
    except Exception:
        pass
    return data


def fetch_short_selling(code):
    url = f"https://finance.naver.com/item/short_selling.naver?code={code}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    short_data = {"공매도비중": 0.0, "공매도거래량": 0}
    try:
        res = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(res.text, "html.parser")
        table = soup.select("table.type2 tbody tr")
        for tr in table:
            tds = tr.select("td")
            if len(tds) >= 5 and tds[0].text.strip():
                short_data["공매도비중"] = float(tds[4].text.strip().replace("%", "").replace(",", ""))
                short_data["공매도거래량"] = int(tds[1].text.strip().replace(",", ""))
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
# [사진 첨부 4대 다중 팩터 + 추세 추종 정밀 퀀트 채점 엔진]
# ====================================================
def evaluate_pro_quant_score(df, df_inv, fund, short):
    score = 0
    logs = []
    latest = df.iloc[-1]
    prev = df.iloc[-2]

    # 1. 모멘텀 및 추세 추종 팩터 (25점)
    tech_score = 0
    if latest["MA5"] > latest["MA20"] > latest["MA60"]:
        tech_score += 12
        logs.append(("이평선 완전 정배열 (5>20>60)", "+12점", "상승 추세 추종 적합 (골든 상태)"))
    elif latest["Close"] > latest["MA20"]:
        tech_score += 6
        logs.append(("20일선 지지 안착", "+6점", "단기 지지선 반등 흐름 유지"))
    else:
        tech_score -= 6
        logs.append(("20일선 하회 역배열", "-6점", "단기 하락 추세 지속"))

    bb_b = latest["BB_%b"]
    if 0.75 <= bb_b <= 1.05:
        tech_score += 7
        logs.append(("볼린저 밴드라이딩 모멘텀", "+7점", "강한 추세 확장 구간 진입"))
    elif bb_b < 0.2:
        tech_score -= 4
        logs.append(("볼린저 하단 이탈 경계", "-4점", "하방 압력 과도"))

    if latest["MACD_HIST"] > 0 and prev["MACD_HIST"] <= 0:
        tech_score += 6
        logs.append(("MACD 골든크로스 발생", "+6점", "상승 모멘텀 전환 신호 포착"))
    elif latest["MACD"] > latest["MACD_SIGNAL"]:
        tech_score += 3
        logs.append(("MACD 시그널 상회", "+3점", "매수 우위 흐름 지속"))
    score += max(0, min(25, tech_score))

    # 2. 수급 및 저변동성 팩터 (25점)
    supply_score = 0
    if not df_inv.empty:
        for_5d_amt = df_inv["외인순매수금액"].head(5).sum()
        inst_5d_amt = df_inv["기관순매수금액"].head(5).sum()
        if for_5d_amt > 0 and inst_5d_amt > 0:
            supply_score += 12
            logs.append(("외인·기관 쌍끌이 동반 순매수", "+12점", f"5일 외인({for_5d_amt:+.1f}억), 기관({inst_5d_amt:+.1f}억) 유입"))
        elif for_5d_amt > 0 or inst_5d_amt > 0:
            supply_score += 6
            logs.append(("메이저 주포 수급 유입", "+6점", f"외인({for_5d_amt:+.1f}억) 또는 기관({inst_5d_amt:+.1f}억) 순매수"))
        else:
            supply_score -= 5
            logs.append(("외인·기관 동반 매도세", "-5점", f"5일 외인({for_5d_amt:+.1f}억), 기관({inst_5d_amt:+.1f}억) 이탈"))

        if len(df_inv) >= 10 and df_inv["외인보유율"].iloc[0] > df_inv["외인보유율"].iloc[9]:
            supply_score += 5
            logs.append(("외국인 지분율 확대", "+5점", f"현재 외인 지분율 {df_inv['외인보유율'].iloc[0]:.2f}%"))

    mfi = latest["MFI"]
    if 50 <= mfi <= 75:
        supply_score += 5
        logs.append(("MFI 스마트 머니 유입", "+5점", f"MFI {mfi:.1f} (대량 매집 진행)"))
    elif mfi > 80:
        supply_score -= 4
        logs.append(("MFI 단기 과열 경계", "-4점", f"MFI {mfi:.1f}"))

    if latest["OBV"] > df["OBV"].tail(20).mean():
        supply_score += 3
        logs.append(("OBV 매집 추세 유지", "+3점", "거래량 기반 매집 에너지 양호"))
    score += max(0, min(25, supply_score))

    # 3. 밸류 & 퀄리티 팩터 (PER, PBR, ROE, 25점)
    analyst_score = 0
    if fund["목표주가"] and fund["목표주가"] > 0:
        upside = ((fund["목표주가"] - latest["Close"]) / latest["Close"]) * 100
        if upside >= 25.0:
            analyst_score += 10
            logs.append(("목표가 괴리율 매력", "+10점", f"목표가 {fund['목표주가']:,.0f}원 (상승여력 {upside:+.1f}%)"))
        elif upside >= 10.0:
            analyst_score += 6
            logs.append(("상승여력 유효", "+6점", f"목표가 {fund['목표주가']:,.0f}원 (상승여력 {upside:+.1f}%)"))
        elif upside < 0:
            analyst_score -= 6
            logs.append(("목표가 초과 고평가", "-6점", f"현재가가 목표주가({fund['목표주가']:,.0f}원) 상회"))

    if fund["ROE"] is not None:
        if fund["ROE"] >= 15.0:
            analyst_score += 8
            logs.append(("고수익 퀄리티 (ROE 15%↑)", "+8점", f"ROE {fund['ROE']:.2f}% (우수한 자본 효율성)"))
        elif fund["ROE"] >= 8.0:
            analyst_score += 4
            logs.append(("안정적 자본 효율성", "+4점", f"ROE {fund['ROE']:.2f}% (안정적 이익 창출)"))
        elif fund["ROE"] < 0:
            analyst_score -= 8
            logs.append(("ROE 마이너스 적자 기업", "-8점", f"ROE {fund['ROE']:.2f}%"))

    if fund["PER"] is not None:
        if fund["PER"] < 0:
            analyst_score -= 5
            logs.append(("실적 적자", "-5점", "PER 음수"))
        elif fund["업종PER"] and fund["PER"] <= fund["업종PER"] * 0.7:
            analyst_score += 4
            logs.append(("업종 대비 저평가", "+4점", f"PER {fund['PER']}배 (업종 {fund['업종PER']}배)"))

    if fund["PBR"] and fund["PBR"] < 0.9:
        analyst_score += 3
        logs.append(("저PBR 자산 가치주", "+3점", f"PBR {fund['PBR']}배로 장부가치 하회"))
    score += max(0, min(25, analyst_score))

    # 4. 가격 모멘텀 & 과열 리스크 관리 (25점)
    m_score = 0
    rsi = latest["RSI"]
    if 45 <= rsi <= 65:
        m_score += 10
        logs.append(("이상적 RSI 구간", "+10점", f"RSI {rsi:.1f} (과열 없는 안정적 추세)"))
    elif 30 <= rsi < 45:
        m_score += 5
        logs.append(("바닥권 반등 구간", "+5점", f"RSI {rsi:.1f}"))
    elif rsi > 75:
        m_score -= 6
        logs.append(("단기 과매수 과열 경계", "-6점", f"RSI {rsi:.1f}"))

    high_52w = df["High"].max()
    dist_high = ((latest["Close"] - high_52w) / high_52w) * 100
    if dist_high >= -7.0:
        m_score += 10
        logs.append(("52주 신고가 근접 (12M 모멘텀)", "+10점", f"최고가 대비 {dist_high:.1f}% 위치 (강한 주도력)"))
    elif dist_high <= -35.0:
        m_score -= 6
        logs.append(("장기 낙폭 과대 역배열", "-6점", f"최고가 대비 {dist_high:.1f}% 하락"))

    if short["공매도비중"] >= 15.0:
        m_score -= 8
        logs.append(("공매도 폭탄 경보", "-8점", f"공매도 비중 {short['공매도비중']:.2f}% (하방 압력 과도)"))
    elif short["공매도비중"] >= 7.0:
        m_score -= 4
        logs.append(("공매도 경계", "-4점", f"공매도 비중 {short['공매도비중']:.2f}%"))
    else:
        m_score += 5

    score += max(0, min(25, m_score))

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
# [우측 상단 퀀트 종합 점수 & 우측 하단 시장 주도주 연산 엔진]
# ====================================================
@st.cache_data(ttl=600)
def generate_dual_market_rankings():
    krx = get_krx_listing()
    df_active = krx[(krx["Volume"] > 0) & (krx["Amount"] >= 5000000000)].copy()

    # 1. 우측 상단: 퀀트 다중 팩터 종합 점수 랭킹
    marcap_log = np.log10(df_active["Marcap"].clip(lower=1e10))
    amount_log = np.log10(df_active["Amount"].clip(lower=1e8))
    chg = df_active["ChagesRatio"]

    trend_factor = np.where(chg > 20, 15 - (chg - 20) * 1.5, np.where(chg > 0, 22 + chg * 1.2, 18 + chg * 2.0))
    quality_factor = ((marcap_log - 10.5) * 6).clip(lower=5, upper=25)
    liquidity_factor = ((amount_log - 8.5) * 6).clip(lower=5, upper=25)
    
    calc_score = trend_factor + quality_factor + liquidity_factor + 15
    df_active["종합점수"] = calc_score.clip(lower=18, upper=95).astype(int)
    df_active["등락률표시"] = df_active["ChagesRatio"].apply(lambda x: f"{x:+.2f}%")
    df_active["거래대금_억"] = (df_active["Amount"] / 100000000).astype(int)

    top10_score = df_active.sort_values(by=["종합점수", "Amount"], ascending=[False, False]).head(10).reset_index(drop=True)
    bot10_score = df_active.sort_values(by=["종합점수", "ChagesRatio"], ascending=[True, True]).head(10).reset_index(drop=True)

    # 2. 우측 하단: 시장 자금 주도주 랭킹
    df_active["모멘텀"] = (chg * 2.5) + (amount_log * 5)
    top10_lead = df_active.sort_values(by="모멘텀", ascending=False).head(10).reset_index(drop=True)
    bot10_lead = df_active.sort_values(by="모멘텀", ascending=True).head(10).reset_index(drop=True)

    return top10_score, bot10_score, top10_lead, bot10_lead


# ====================================================
# 메인 헤더 & 레이아웃
# ====================================================
st.markdown(
    """
    <div class="header-card">
        <div>
            <div style="font-size: 13px; color: #38bdf8; font-weight:600; margin-bottom:2px;">MULTI-FACTOR & TREND FOLLOWING ENGINE</div>
            <h2 style="margin:0; font-size:22px; font-weight:800; color:#f8fafc;">부리부리 종합 주식 작전실</h2>
        </div>
        <div style="font-size: 26px;">🐽📊</div>
    </div>
""",
    unsafe_allow_html=True,
)

main_col, rank_col = st.columns([7, 3])

# 1. 오른쪽 시장 랭킹 (상단: 종합점수 TOP10 / 하단: 주도주 TOP10)
with rank_col:
    top10_score, bot10_score, top10_lead, bot10_lead = generate_dual_market_rankings()

    # [우측 상단] 퀀트 종합 점수 랭킹
    st.markdown("<div style='font-size:14px; font-weight:700; color:#38bdf8; margin-bottom:6px;'>🏆 퀀트 팩터 종합 점수 랭킹 TOP 10</div>", unsafe_allow_html=True)
    score_tab1, score_tab2 = st.tabs(["🌟 상위 우량주", "🚨 하위 주의주"])

    def render_score_buttons(df_rank, prefix):
        if df_rank.empty:
            st.write("데이터 준비 중...")
            return
        for i, row in df_rank.iterrows():
            cols = st.columns([5, 3, 2])
            if cols[0].button(f"{i+1}. {row['Name']}", key=f"{prefix}_{row['Code']}", use_container_width=True):
                st.session_state.selected_stock = row["Name"]
                st.rerun()
            cols[1].markdown(f"<div style='text-align:right; font-size:12px; padding-top:6px;'>{row['Close']:,}원 ({row['등락률표시']})</div>", unsafe_allow_html=True)
            cols[2].markdown(f"<div style='text-align:center; font-weight:700; font-size:13px; color:#38bdf8; padding-top:6px;'>{row['종합점수']}점</div>", unsafe_allow_html=True)

    with score_tab1:
        render_score_buttons(top10_score, "score_top")
    with score_tab2:
        render_score_buttons(bot10_score, "score_bot")

    st.write("")
    st.divider()

    # [우측 하단] 시장 자금 주도주 랭킹
    st.markdown("<div style='font-size:14px; font-weight:700; color:#f59e0b; margin-bottom:6px;'>🔥 시장 자금 주도주 랭킹 TOP 10</div>", unsafe_allow_html=True)
    lead_tab1, lead_tab2 = st.tabs(["🚀 상승 주도주", "📉 하락 소외주"])

    def render_lead_buttons(df_rank, prefix):
        if df_rank.empty:
            st.write("데이터 준비 중...")
            return
        for i, row in df_rank.iterrows():
            cols = st.columns([5, 3, 2])
            if cols[0].button(f"{i+1}. {row['Name']}", key=f"{prefix}_{row['Code']}", use_container_width=True):
                st.session_state.selected_stock = row["Name"]
                st.rerun()
            cols[1].markdown(f"<div style='text-align:right; font-size:12px; padding-top:6px;'>{row['Close']:,}원 ({row['등락률표시']})</div>", unsafe_allow_html=True)
            cols[2].markdown(f"<div style='text-align:center; font-size:12px; color:#94a3b8; padding-top:6px;'>{row['거래대금_억']:,}억</div>", unsafe_allow_html=True)

    with lead_tab1:
        render_lead_buttons(top10_lead, "lead_top")
    with lead_tab2:
        render_lead_buttons(bot10_lead, "lead_bot")


# 2. 왼쪽 메인 정밀 분석 (연관 종목 드롭다운 및 실전 타점)
with main_col:
    col_s1, col_s2 = st.columns([4, 1])
    default_search = st.session_state.selected_stock
    search_input = col_s1.text_input(
        "종목 검색",
        value=default_search,
        placeholder="종목명(예: 현대, 삼성, 카카오) 또는 6자리 코드 입력",
        label_visibility="collapsed"
    )
    analyze_btn = col_s2.button("🚀 정밀 분석", type="primary", use_container_width=True)

    # 검색어 입력 시 연관 종목 드롭다운 박스 노출 (토스 스타일)
    if search_input.strip() and search_input != st.session_state.selected_stock:
        sim_df = search_similar_stocks(search_input)
        if not sim_df.empty:
            st.markdown(
                """
                <div class="autocomplete-box">
                    <div class="autocomplete-header">
                        <span>🔍 연관 종목 검색 결과 (클릭 시 즉시 분석)</span>
                        <span>실시간 현재가 / 등락률</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            for idx, srow in sim_df.iterrows():
                ac_cols = st.columns([5, 3, 2])
                if ac_cols[0].button(f"🏢 {srow['Name']} ({srow['Code']})", key=f"ac_{srow['Code']}", use_container_width=True):
                    st.session_state.selected_stock = srow["Name"]
                    st.rerun()
                
                chg_val = srow.get("ChagesRatio", 0.0)
                chg_color = "#ef4444" if chg_val > 0 else ("#38bdf8" if chg_val < 0 else "#94a3b8")
                ac_cols[1].markdown(f"<div style='text-align:right; font-weight:700; font-size:13px; padding-top:6px;'>{srow['Close']:,}원</div>", unsafe_allow_html=True)
                ac_cols[2].markdown(f"<div style='text-align:right; font-weight:700; font-size:13px; color:{chg_color}; padding-top:6px;'>{chg_val:+.2f}%</div>", unsafe_allow_html=True)

    if search_input != st.session_state.selected_stock:
        st.session_state.selected_stock = search_input

    current_target = st.session_state.selected_stock.strip() or search_input.strip()

    if current_target:
        code, stock_name = resolve_stock_code(current_target)

        if not code:
            st.error(f"'{current_target}' 종목을 찾을 수 없습니다.")
        else:
            with st.spinner(f"[{stock_name}] 다중 팩터 정밀 분석 중..."):
                end_dt = datetime.today()
                start_dt = end_dt - timedelta(days=365)
                df = fdr.DataReader(code, start_dt.strftime("%Y-%m-%d"))

                if df.empty or len(df) < 60:
                    st.error("데이터 수집에 실패했거나 거래일 데이터가 부족합니다.")
                else:
                    latest_price = df["Close"].iloc[-1]
                    prev_price = df["Close"].iloc[-2]
                    today_open = df["Open"].iloc[-1]

                    df["MA5"] = df["Close"].rolling(5).mean()
                    df["MA20"] = df["Close"].rolling(20).mean()
                    df["MA60"] = df["Close"].rolling(60).mean()
                    df["STD20"] = df["Close"].rolling(20).std()
                    df["BB_Upper"] = df["MA20"] + (df["STD20"] * 2)
                    df["BB_Lower"] = df["MA20"] - (df["STD20"] * 2)
                    df["BB_%b"] = (df["Close"] - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"] + 1e-9)

                    tr1 = df["High"] - df["Low"]
                    tr2 = (df["High"] - df["Close"].shift(1)).abs()
                    tr3 = (df["Low"] - df["Close"].shift(1)).abs()
                    df["TR"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                    df["ATR14"] = df["TR"].rolling(14).mean()
                    atr_val = df["ATR14"].iloc[-1] if not pd.isna(df["ATR14"].iloc[-1]) else latest_price * 0.03

                    exp12 = df["Close"].ewm(span=12, adjust=False).mean()
                    exp26 = df["Close"].ewm(span=26, adjust=False).mean()
                    df["MACD"] = exp12 - exp26
                    df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()
                    df["MACD_HIST"] = df["MACD"] - df["MACD_SIGNAL"]

                    delta = df["Close"].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                    df["RSI"] = 100 - (100 / (1 + (gain / (loss + 1e-9))))
                    df["OBV"] = (np.sign(df["Close"].diff()).fillna(0) * df["Volume"]).cumsum()

                    tp = (df["High"] + df["Low"] + df["Close"]) / 3
                    rmf = tp * df["Volume"]
                    pos_mf = (rmf.where(tp > tp.shift(1), 0)).rolling(14).sum()
                    neg_mf = (rmf.where(tp < tp.shift(1), 0)).rolling(14).sum()
                    mfr = pos_mf / (neg_mf + 1e-9)
                    df["MFI"] = 100 - (100 / (1 + mfr))

                    ret_1d = ((latest_price - prev_price) / prev_price) * 100
                    ret_1w = ((latest_price - df["Close"].iloc[-5]) / df["Close"].iloc[-5]) * 100
                    ret_1m = ((latest_price - df["Close"].iloc[-20]) / df["Close"].iloc[-20]) * 100
                    ret_1y = ((latest_price - df["Close"].iloc[0]) / df["Close"].iloc[0]) * 100

                    high_20 = df["High"].tail(20).max()
                    low_20 = df["Low"].tail(20).min()

                    # 매물대 프로파일
                    price_min, price_max = df["Low"].min(), df["High"].max()
                    bins = np.linspace(price_min, price_max, 13)
                    v_counts, _ = np.histogram(df["Close"], bins=bins, weights=df["Volume"])

                    df_inv = fetch_investor_naver(code)
                    fund = fetch_fundamental_and_consensus(code)
                    short = fetch_short_selling(code)
                    news_items = fetch_news(stock_name)

                    for_5d = df_inv["외인순매수금액"].head(5).sum() if not df_inv.empty else 0
                    inst_5d = df_inv["기관순매수금액"].head(5).sum() if not df_inv.empty else 0
                    for_20d = df_inv["외인순매수금액"].head(20).sum() if not df_inv.empty else 0
                    for_rate = df_inv["외인보유율"].iloc[0] if not df_inv.empty else 0.0

                    total_score, grade_text, stars, logs = evaluate_pro_quant_score(df, df_inv, fund, short)

                    # =========================================================
                    # [개선된 실전 매매 타점 알고리즘 (현재가 -1% 단순 공식 완전 폐기)]
                    # =========================================================
                    ma5_val = df["MA5"].iloc[-1]
                    ma20_val = df["MA20"].iloc[-1]

                    # 1차 진입가: 5일선 눌림목 또는 당일 시가 기준 지지선 (-2.5% ~ -4% 실질 눌림목)
                    if latest_price > ma5_val:
                        entry_1 = round(max(ma5_val, latest_price * 0.97), -2)
                    else:
                        entry_1 = round(latest_price * 0.965, -2)

                    # 2차 진입가: 20일 이동평균선 또는 주요 지지선
                    if (latest_price - ma20_val) / ma20_val > 0.15:
                        entry_2 = round(entry_1 * 0.95, -2)
                    else:
                        entry_2 = round(ma20_val, -2)
                    
                    if entry_2 >= entry_1:
                        entry_2 = round(entry_1 * 0.95, -2)

                    # 목표가
                    t1_calc = max(high_20 * 1.02, latest_price * 1.06)
                    target_1 = round(t1_calc, -2)

                    if fund["목표주가"] and fund["목표주가"] > target_1 * 1.05:
                        target_2 = round(fund["목표주가"], -2)
                    else:
                        target_2 = round(target_1 * 1.08, -2)

                    # 손절 기준선
                    if (latest_price - low_20) / low_20 > 0.20:
                        stop_loss = round(min(today_open * 0.95, latest_price * 0.92), -2)
                    else:
                        stop_loss = round(max(low_20 * 0.98, latest_price - (atr_val * 1.8)), -2)

                    target_grid_html = (
                        f'<div class="score-container">'
                        f'<div style="font-size: 13px; color: #94a3b8; font-weight:600;">{stock_name} ({code})</div>'
                        f'<div style="font-size: 44px; color: #38bdf8; font-weight: 800; margin: 2px 0;">{total_score}<span style="font-size:18px; color:#64748b;"> / 100</span></div>'
                        f'<div style="font-size: 15px; font-weight: 600; color: #f1f5f9; margin-bottom: 12px;">{grade_text} <span style="color:#eab308;">{stars}</span></div>'
                        f'<div class="target-grid">'
                        f'<div class="target-item"><div class="target-title">1차 진입 (눌림목 지지)</div><div class="target-val">{entry_1:,.0f}원</div></div>'
                        f'<div class="target-item"><div class="target-title">2차 진입 (추세 지지선)</div><div class="target-val">{entry_2:,.0f}원</div></div>'
                        f'<div class="target-item"><div class="target-title">1차 목표 (단기 저항)</div><div class="target-val">{target_1:,.0f}원</div></div>'
                        f'<div class="target-item"><div class="target-title">2차 목표 (추세 확장)</div><div class="target-val">{target_2:,.0f}원</div></div>'
                        f'<div class="target-item"><div class="target-title">정밀 손절선</div><div class="target-val" style="color:#ef4444;">{stop_loss:,.0f}원</div></div>'
                        f'</div>'
                        f'</div>'
                    )
                    st.markdown(target_grid_html, unsafe_allow_html=True)

                    t1, t2, t3, t4, t5 = st.tabs(["차트 & 매물대 프로파일", "외인/기관 수급", "전망 & 애널리스트", "종합 팩터 채점표", "뉴스 브리핑"])

                    with t1:
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("현재가", f"{latest_price:,.0f}원", f"{ret_1d:+.2f}%")
                        c2.metric("1주일", f"{ret_1w:+.2f}%")
                        c3.metric("1개월", f"{ret_1m:+.2f}%")
                        c4.metric("1년(모멘텀)", f"{ret_1y:+.2f}%")

                        fig = make_subplots(
                            rows=2, cols=2,
                            shared_xaxes=True,
                            row_heights=[0.72, 0.28],
                            column_widths=[0.85, 0.15],
                            horizontal_spacing=0.01,
                            vertical_spacing=0.04,
                            specs=[[{}, {}], [{}, None]]
                        )

                        fig.add_trace(go.Candlestick(
                            x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="주가"
                        ), row=1, col=1)
                        fig.add_trace(go.Scatter(
                            x=df.index, y=df['MA20'], line=dict(color='#38bdf8', width=1.3), name="20일선"
                        ), row=1, col=1)
                        fig.add_trace(go.Scatter(
                            x=df.index, y=df['MA60'], line=dict(color='#10b981', width=1.3), name="60일선"
                        ), row=1, col=1)

                        bin_centers = 0.5 * (bins[:-1] + bins[1:])
                        fig.add_trace(go.Bar(
                            y=bin_centers, x=v_counts, orientation='h',
                            marker_color='rgba(56, 189, 248, 0.35)', showlegend=False, hoverinfo='none'
                        ), row=1, col=2)

                        fig.add_trace(go.Scatter(
                            x=df.index, y=df['MACD'], line=dict(color='#f43f5e', width=1.2), name="MACD"
                        ), row=2, col=1)
                        fig.add_trace(go.Scatter(
                            x=df.index, y=df['MACD_SIGNAL'], line=dict(color='#fbbf24', width=1.2), name="Signal"
                        ), row=2, col=1)

                        fig.update_layout(
                            template="plotly_dark",
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            height=450,
                            margin=dict(l=5, r=5, t=5, b=5),
                            xaxis_rangeslider_visible=False
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
                            show_inv = df_inv.head(10)[["날짜", "종가", "기관순매수금액", "외인순매수금액", "외인보유율"]].copy()
                            show_inv["기관순매수금액"] = show_inv["기관순매수금액"].apply(lambda x: f"{x:+.1f}억원")
                            show_inv["외인순매수금액"] = show_inv["외인순매수금액"].apply(lambda x: f"{x:+.1f}억원")
                            show_inv["외인보유율"] = show_inv["외인보유율"].apply(lambda x: f"{x:.2f}%")
                            st.dataframe(show_inv, use_container_width=True)

                    with t3:
                        st.markdown("#### 🏢 기업 핵심 개요 & 사업 방향")
                        st.markdown(f'<div class="insight-card">{fund["기업개요"]}</div>', unsafe_allow_html=True)

                        st.markdown("#### 📊 펀더멘털 & 애널리스트 컨센서스")
                        v1, v2, v3, v4, v5 = st.columns(5)
                        v1.metric("목표주가", f"{fund['목표주가']:,.0f}원" if fund["목표주가"] else "미제공",
                                  f"상승여력 {((fund['목표주가']-latest_price)/latest_price)*100:+.1f}%" if fund["목표주가"] else "")
                        v2.metric("ROE (퀄리티)", f"{fund['ROE'] or '-'}%")
                        v3.metric("PER / 업종", f"{fund['PER'] or '-'}배", f"업종 {fund['업종PER'] or '-'}배")
                        v4.metric("PBR", f"{fund['PBR'] or '-'}배")
                        v5.metric("공매도비중", f"{short['공매도비중']:.2f}%")

                        st.markdown("#### 📑 최신 증권사 애널리스트 리포트")
                        if fund["리포트_목록"]:
                            for rep in fund["리포트_목록"]:
                                st.markdown(f"- **[{rep['broker']}]** [{rep['title']}]({rep['link']}) <span style='color:#64748b; font-size:12px;'>({rep['date']})</span>", unsafe_allow_html=True)
                        else:
                            st.write("최근 등록된 증권사 분석 리포트가 없습니다.")

                    with t4:
                        st.dataframe(pd.DataFrame(logs, columns=["평가 항목", "가감점", "상세 내용"]), use_container_width=True)

                    with t5:
                        for item in news_items:
                            st.markdown(f"- [{item['title']}]({item['link']})")
    else:
        st.info("💡 상단 검색창에 **종목명**을 입력하시거나, 우측 **점수 랭킹 또는 주도주 랭킹의 종목을 클릭**하시면 즉시 정밀 퀀트 분석이 시작됩니다부리!")
