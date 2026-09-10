# -*- coding: utf-8 -*-
"""
달리는 말 탐지기 v1.0
- KRX 종목 단일 분석
- KOSPI / KOSDAQ 모멘텀 후보 스캔
- 추세 / 거래량 / RSI / MACD / ADX / 돌파 / 눌림 점수화

실행:
    streamlit run app.py

필수 패키지:
    pip install streamlit FinanceDataReader pandas numpy plotly
"""

from datetime import datetime, timedelta
import time
import numpy as np
import pandas as pd
import FinanceDataReader as fdr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(
    page_title="🐎 달리는 말 탐지기",
    page_icon="🐎",
    layout="wide",
)

# -----------------------------
# 기본 설정
# -----------------------------
TODAY = datetime.today().date()
START_DEFAULT = TODAY - timedelta(days=450)


# -----------------------------
# 기술적 지표
# -----------------------------
def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def calc_macd(close: pd.Series):
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist


def calc_adx(df: pd.DataFrame, period: int = 14):
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=df.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=df.index,
    )

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    plus_di = 100 * (
        plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
        / atr.replace(0, np.nan)
    )
    minus_di = 100 * (
        minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
        / atr.replace(0, np.nan)
    )

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    return adx.fillna(0), plus_di.fillna(0), minus_di.fillna(0)


def prepare_indicators(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()

    for ma in [5, 20, 60, 120]:
        x[f"MA{ma}"] = x["Close"].rolling(ma).mean()

    x["VOL_MA20"] = x["Volume"].rolling(20).mean()
    x["RSI14"] = calc_rsi(x["Close"], 14)

    x["MACD"], x["MACD_SIGNAL"], x["MACD_HIST"] = calc_macd(x["Close"])
    x["ADX14"], x["PLUS_DI"], x["MINUS_DI"] = calc_adx(x, 14)

    x["HIGH20_PREV"] = x["High"].shift(1).rolling(20).max()
    x["HIGH60_PREV"] = x["High"].shift(1).rolling(60).max()
    x["HIGH120_PREV"] = x["High"].shift(1).rolling(120).max()

    x["RET_5"] = x["Close"].pct_change(5) * 100
    x["RET_20"] = x["Close"].pct_change(20) * 100
    x["RET_60"] = x["Close"].pct_change(60) * 100

    x["DIST_MA20"] = (x["Close"] / x["MA20"] - 1) * 100
    x["DIST_MA60"] = (x["Close"] / x["MA60"] - 1) * 100
    x["VOL_RATIO"] = x["Volume"] / x["VOL_MA20"].replace(0, np.nan)

    # 이동평균선 방향
    x["MA20_SLOPE"] = x["MA20"].pct_change(5) * 100
    x["MA60_SLOPE"] = x["MA60"].pct_change(10) * 100

    return x


# -----------------------------
# 점수 시스템
# -----------------------------
def score_stock(df: pd.DataFrame):
    if len(df) < 130:
        return None

    x = prepare_indicators(df).dropna(subset=["MA120", "VOL_MA20"])
    if len(x) < 5:
        return None

    r = x.iloc[-1]
    prev = x.iloc[-2]

    score = 0
    details = []

    def add(cond, pts, text):
        nonlocal score
        if bool(cond):
            score += pts
            details.append(("✅", pts, text))
        else:
            details.append(("➖", 0, text))

    # 1. 추세 30점
    add(r["Close"] > r["MA20"], 5, "현재가가 20일선 위")
    add(r["MA20"] > r["MA60"], 5, "20일선이 60일선 위")
    add(r["MA60"] > r["MA120"], 5, "60일선이 120일선 위")
    add(r["MA20_SLOPE"] > 0, 5, "20일 이동평균선 상승")
    add(r["MA60_SLOPE"] > 0, 5, "60일 이동평균선 상승")
    add(r["Close"] >= r["HIGH60_PREV"] * 0.99, 5, "60일 고점 돌파/근접")

    # 2. 거래량 20점
    add(r["VOL_RATIO"] >= 1.2, 5, "거래량이 20일 평균의 1.2배 이상")
    add(r["VOL_RATIO"] >= 2.0, 5, "거래량이 20일 평균의 2배 이상")
    add((r["Close"] > prev["Close"]) and (r["Volume"] > prev["Volume"]), 5,
        "상승일에 거래량 증가")

    # 눌림 구간에서 거래량 감소
    recent = x.tail(5)
    pullback_volume = (
        (recent["Close"].pct_change() < 0).any()
        and recent["Volume"].iloc[-1] < recent["VOL_MA20"].iloc[-1]
    )
    add(pullback_volume, 5, "최근 눌림 구간에서 거래량 감소")

    # 3. 모멘텀 20점
    add(55 <= r["RSI14"] <= 70, 7, "RSI 55~70의 건강한 강세")
    add((r["MACD"] > r["MACD_SIGNAL"]) and (r["MACD"] > 0), 7,
        "MACD가 Signal 및 0선 위")
    add((r["ADX14"] >= 20) and (r["PLUS_DI"] > r["MINUS_DI"]), 6,
        "ADX 추세 강도 + 상승 방향 우위")

    # 4. 돌파 / 위치 20점
    add(r["Close"] > r["HIGH20_PREV"], 5, "20일 신고가 돌파")
    add(r["Close"] > r["HIGH60_PREV"], 5, "60일 신고가 돌파")
    add(-1 <= r["DIST_MA20"] <= 6, 5, "20일선과의 이격도가 적정")
    add(r["RET_20"] > 0, 5, "최근 20거래일 수익률 플러스")

    # 5. 과열 방지 10점
    add(r["RSI14"] < 78, 5, "RSI 극단적 과열 아님")
    add(r["DIST_MA20"] < 10, 5, "20일선 대비 과도한 이격 아님")

    # 과열 패널티
    penalties = []
    if r["RSI14"] >= 80:
        score -= 8
        penalties.append("RSI 80 이상 과열 -8")
    if r["DIST_MA20"] >= 15:
        score -= 8
        penalties.append("20일선 대비 +15% 이상 이격 -8")
    if r["RET_5"] >= 20:
        score -= 5
        penalties.append("5거래일 +20% 이상 급등 -5")

    score = max(0, min(100, int(round(score))))

    # 진입 상태 판정
    breakout = r["Close"] > r["HIGH60_PREV"]
    near_breakout = r["Close"] >= r["HIGH60_PREV"] * 0.97
    healthy_pullback = (
        r["Close"] > r["MA20"]
        and -1 <= r["DIST_MA20"] <= 4
        and 50 <= r["RSI14"] <= 68
    )

    if score >= 80 and healthy_pullback:
        status = "🔥 최우선 관찰 — 강한 추세 + 좋은 눌림"
    elif score >= 80 and breakout and r["RSI14"] < 75:
        status = "🚀 강한 돌파 — 추격보다 눌림 대기"
    elif score >= 70 and near_breakout:
        status = "🟢 달리는 말 후보 — 돌파/지지 확인"
    elif score >= 60:
        status = "🟡 관심 종목 — 조건 일부 미충족"
    elif score >= 45:
        status = "🟠 애매 — 추세 확인 필요"
    else:
        status = "🔴 우선순위 낮음"

    # 매매 참고 구간
    support1 = float(r["MA20"])
    support2 = float(r["MA60"])
    resistance = float(r["HIGH60_PREV"])

    return {
        "score": score,
        "status": status,
        "details": details,
        "penalties": penalties,
        "row": r,
        "df": x,
        "support1": support1,
        "support2": support2,
        "resistance": resistance,
    }


# -----------------------------
# 데이터
# -----------------------------
@st.cache_data(ttl=1800)
def load_price(code: str, start=START_DEFAULT, end=TODAY):
    df = fdr.DataReader(code, start, end)
    if df is None or df.empty:
        raise ValueError("가격 데이터를 불러오지 못했습니다.")
    return df


@st.cache_data(ttl=3600)
def load_listing(market: str):
    return fdr.StockListing(market)


def find_name(code: str, listing: pd.DataFrame):
    code = str(code).zfill(6)
    candidates = ["Code", "Symbol"]
    code_col = next((c for c in candidates if c in listing.columns), None)
    name_col = "Name" if "Name" in listing.columns else None

    if code_col and name_col:
        hit = listing[listing[code_col].astype(str).str.zfill(6) == code]
        if not hit.empty:
            return str(hit.iloc[0][name_col])
    return code


# -----------------------------
# 차트
# -----------------------------
def make_chart(result, name):
    df = result["df"].tail(150)

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.035,
        row_heights=[0.52, 0.16, 0.16, 0.16],
        specs=[
            [{"secondary_y": False}],
            [{"secondary_y": False}],
            [{"secondary_y": False}],
            [{"secondary_y": False}],
        ],
    )

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Price",
        ),
        row=1, col=1,
    )

    for ma in [20, 60, 120]:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[f"MA{ma}"],
                mode="lines",
                name=f"MA{ma}",
                line=dict(width=1.4),
            ),
            row=1, col=1,
        )

    # 60일 전고점
    fig.add_hline(
        y=result["resistance"],
        line_dash="dot",
        annotation_text="60일 전고점",
        row=1, col=1,
    )

    # 거래량
    fig.add_trace(
        go.Bar(x=df.index, y=df["Volume"], name="Volume"),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["VOL_MA20"],
            mode="lines",
            name="Vol MA20",
            line=dict(width=1.2),
        ),
        row=2, col=1,
    )

    # RSI
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["RSI14"],
            mode="lines",
            name="RSI14",
        ),
        row=3, col=1,
    )
    fig.add_hline(y=70, line_dash="dot", row=3, col=1)
    fig.add_hline(y=50, line_dash="dot", row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", row=3, col=1)

    # MACD
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["MACD"],
            mode="lines",
            name="MACD",
        ),
        row=4, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["MACD_SIGNAL"],
            mode="lines",
            name="Signal",
        ),
        row=4, col=1,
    )
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["MACD_HIST"],
            name="Histogram",
        ),
        row=4, col=1,
    )

    fig.update_layout(
        title=f"{name} — 달리는 말 기술적 분석",
        height=980,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h"),
        margin=dict(l=20, r=20, t=60, b=20),
    )

    return fig


# -----------------------------
# UI
# -----------------------------
st.title("🐎 달리는 말 탐지기")
st.caption(
    "상승 추세 + 거래량 + RSI + MACD + ADX + 돌파 + 이격도를 종합해 "
    "'달리는 말' 후보를 점수화합니다. 투자 판단 보조용입니다."
)

with st.sidebar:
    st.header("⚙️ 설정")
    market = st.selectbox("시장", ["KOSPI", "KOSDAQ"])
    scan_count = st.slider(
        "시장 스캔 종목 수",
        min_value=30,
        max_value=300,
        value=100,
        step=10,
        help="상위 시가총액 기준으로 우선 스캔합니다. 많을수록 오래 걸립니다.",
    )
    min_score = st.slider("표시 최소 점수", 40, 90, 65, 5)

listing = load_listing(market)

tab1, tab2, tab3 = st.tabs(
    ["🔎 종목 분석", "🏇 시장 스캐너", "📖 점수 기준"]
)

# -----------------------------
# TAB 1
# -----------------------------
with tab1:
    col_input, col_btn = st.columns([4, 1])

    with col_input:
        code = st.text_input(
            "종목코드",
            value="005930",
            placeholder="예: 삼성전자 005930",
        ).strip()

    with col_btn:
        st.write("")
        analyze = st.button("분석 실행", use_container_width=True)

    if analyze or code:
        try:
            code = code.zfill(6)
            name = find_name(code, listing)
            df = load_price(code)
            result = score_stock(df)

            if result is None:
                st.warning("분석에 필요한 데이터가 부족합니다.")
            else:
                r = result["row"]

                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("종합 점수", f'{result["score"]} / 100')
                c2.metric("현재가", f'{r["Close"]:,.0f}원')
                c3.metric("RSI(14)", f'{r["RSI14"]:.1f}')
                c4.metric("ADX(14)", f'{r["ADX14"]:.1f}')
                c5.metric("거래량 배수", f'{r["VOL_RATIO"]:.2f}x')

                st.subheader(result["status"])

                a1, a2, a3, a4 = st.columns(4)
                a1.metric("20일선", f'{r["MA20"]:,.0f}원')
                a2.metric("60일선", f'{r["MA60"]:,.0f}원')
                a3.metric("60일 전고점", f'{result["resistance"]:,.0f}원')
                a4.metric("20일선 이격", f'{r["DIST_MA20"]:+.1f}%')

                st.plotly_chart(
                    make_chart(result, name),
                    use_container_width=True,
                )

                left, right = st.columns([3, 2])

                with left:
                    st.subheader("🧮 점수 상세")
                    detail_df = pd.DataFrame(
                        result["details"],
                        columns=["판정", "점수", "조건"],
                    )
                    st.dataframe(
                        detail_df,
                        use_container_width=True,
                        hide_index=True,
                    )

                with right:
                    st.subheader("🎯 차트 위치 해석")
                    st.write(f"**1차 지지:** {result['support1']:,.0f}원 (20일선)")
                    st.write(f"**2차 지지:** {result['support2']:,.0f}원 (60일선)")
                    st.write(f"**주요 돌파선:** {result['resistance']:,.0f}원")

                    if result["penalties"]:
                        st.warning(" / ".join(result["penalties"]))

                    if r["DIST_MA20"] > 10:
                        st.error("현재가는 20일선에서 많이 이격되어 있습니다. 신규 추격 진입은 주의 구간입니다.")
                    elif -1 <= r["DIST_MA20"] <= 4 and r["Close"] > r["MA60"]:
                        st.success("20일선 근처의 건강한 눌림 후보입니다. 지지 확인 여부를 우선 보십시오.")
                    elif r["Close"] > result["resistance"]:
                        st.info("전고점을 돌파했습니다. 돌파 유지 또는 재눌림에서 지지 전환 여부가 중요합니다.")
                    else:
                        st.info("추세는 유지하되 전고점 돌파 확인이 추가로 필요합니다.")

        except Exception as e:
            st.error(f"분석 중 오류가 발생했습니다: {e}")


# -----------------------------
# TAB 2
# -----------------------------
with tab2:
    st.subheader(f"🏇 {market} 달리는 말 후보 스캐너")

    st.write(
        "시가총액 상위 종목을 대상으로 최근 추세를 분석합니다. "
        "네트워크 상태에 따라 일부 종목 데이터 조회가 실패할 수 있습니다."
    )

    if st.button("시장 스캔 시작", type="primary"):
        work = listing.copy()

        # 컬럼 대응
        code_col = next(
            (c for c in ["Code", "Symbol"] if c in work.columns),
            None,
        )

        if code_col is None:
            st.error("종목코드 컬럼을 찾지 못했습니다.")
            st.stop()

        # 시가총액 기준 정렬 가능하면 적용
        marcap_col = next(
            (c for c in ["Marcap", "MarketCap", "Marcap(억원)"] if c in work.columns),
            None,
        )

        if marcap_col:
            work = work.sort_values(marcap_col, ascending=False)

        work = work.head(scan_count)

        progress = st.progress(0)
        status_box = st.empty()

        rows = []
        failed = 0

        for i, (_, stock) in enumerate(work.iterrows(), start=1):
            stock_code = str(stock[code_col]).zfill(6)
            stock_name = str(stock.get("Name", stock_code))

            status_box.text(
                f"[{i}/{len(work)}] {stock_name} ({stock_code}) 분석 중..."
            )

            try:
                sdf = load_price(stock_code)
                sr = score_stock(sdf)

                if sr is not None:
                    rr = sr["row"]

                    rows.append({
                        "종목명": stock_name,
                        "코드": stock_code,
                        "점수": sr["score"],
                        "상태": sr["status"],
                        "현재가": round(float(rr["Close"])),
                        "5일수익률(%)": round(float(rr["RET_5"]), 2),
                        "20일수익률(%)": round(float(rr["RET_20"]), 2),
                        "RSI": round(float(rr["RSI14"]), 1),
                        "ADX": round(float(rr["ADX14"]), 1),
                        "거래량배수": round(float(rr["VOL_RATIO"]), 2),
                        "20일선이격(%)": round(float(rr["DIST_MA20"]), 2),
                        "60일고점대비(%)": round(
                            (float(rr["Close"]) / float(sr["resistance"]) - 1) * 100,
                            2,
                        ),
                    })

            except Exception:
                failed += 1

            progress.progress(i / len(work))

            # 너무 빠른 연속 요청 방지
            time.sleep(0.03)

        status_box.empty()
        progress.empty()

        if rows:
            result_df = pd.DataFrame(rows)
            result_df = result_df[result_df["점수"] >= min_score]
            result_df = result_df.sort_values(
                ["점수", "20일수익률(%)"],
                ascending=[False, False],
            ).reset_index(drop=True)

            result_df.index = result_df.index + 1

            st.success(
                f"스캔 완료: {len(work)}종목 중 "
                f"{len(result_df)}종목이 {min_score}점 이상입니다."
                + (f" / 조회 실패 {failed}종목" if failed else "")
            )

            st.dataframe(
                result_df,
                use_container_width=True,
            )

            csv = result_df.to_csv(
                index=False,
                encoding="utf-8-sig",
            ).encode("utf-8-sig")

            st.download_button(
                "CSV 다운로드",
                csv,
                file_name=f"running_horse_{market}_{TODAY}.csv",
                mime="text/csv",
            )

            if not result_df.empty:
                st.subheader("🥇 상위 후보")
                top = result_df.head(10)

                for _, row in top.iterrows():
                    with st.expander(
                        f'{row["종목명"]} ({row["코드"]}) — {row["점수"]}점'
                    ):
                        cc1, cc2, cc3, cc4 = st.columns(4)
                        cc1.metric("현재가", f'{row["현재가"]:,.0f}원')
                        cc2.metric("RSI", row["RSI"])
                        cc3.metric("ADX", row["ADX"])
                        cc4.metric("거래량", f'{row["거래량배수"]}x')

                        st.write(row["상태"])
        else:
            st.warning("분석 가능한 종목을 찾지 못했습니다.")


# -----------------------------
# TAB 3
# -----------------------------
with tab3:
    st.subheader("📖 달리는 말 점수 기준")

    st.markdown(
        """
### 1. 추세 — 30점
- 현재가 > 20일선
- 20일선 > 60일선
- 60일선 > 120일선
- 20일선 상승
- 60일선 상승
- 60일 고점 돌파/근접

### 2. 거래량 — 20점
- 현재 거래량 > 20일 평균 1.2배
- 현재 거래량 > 20일 평균 2배
- 상승일 거래량 증가
- 눌림에서 거래량 감소

### 3. 모멘텀 — 20점
- RSI 55~70
- MACD > Signal, MACD > 0
- ADX 20 이상 + +DI 우위

### 4. 돌파 / 위치 — 20점
- 20일 신고가
- 60일 신고가
- 20일선 이격 적정
- 최근 20거래일 상승

### 5. 과열 방지 — 10점
- RSI 78 미만
- 20일선 이격 10% 미만

### 과열 패널티
- RSI 80 이상: -8점
- 20일선 대비 +15% 이상 이격: -8점
- 최근 5거래일 +20% 이상 급등: -5점

---

### 점수 해석
- **80점 이상:** 강한 달리는 말 후보
- **70~79점:** 우선 관찰
- **60~69점:** 관심
- **45~59점:** 애매
- **45점 미만:** 우선순위 낮음

> 점수가 높다는 이유만으로 매수하는 시스템이 아닙니다.
> 최종 진입은 전고점 돌파 후 지지, 20일선 눌림, 거래량 감소 등을 함께 확인하는 용도입니다.
"""
    )
