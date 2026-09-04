#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MIRAIMA HIGH-PRECISION ALL-MARKET ENGINE
========================================

MIRAIMAの公開市場を収集し、

1. 市場を正確に識別
2. カテゴリを分類
3. 市場タイプを分類
4. 外部データを取得
5. 独立予測を作成
6. 確率を校正
7. MIRAIMA市場確率と比較
8. Edge / EVを評価
9. データ不足なら予測を見送る

という流れで処理する。

設計原則:
- MIRAIMAの確率を独立予測として流用しない
- 市場タイプを推測だけで決めない
- 重複市場を除外
- データ不足時は無理に予測しない
- 時系列リークを避ける
- MLBは両先発確認済みの場合のみ対象
- MIRAIMAへの自動参加・自動売買は行わない
- 実行時間は25分を上限とする
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import math
import re
import sys
import time

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urljoin, urlparse

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup


# ============================================================
# GLOBAL SETTINGS
# ============================================================

BASE_URL = "https://miraima.app"

DISCOVERY_URLS = [
    f"{BASE_URL}/",
    f"{BASE_URL}/en",
    f"{BASE_URL}/new",
    f"{BASE_URL}/en/new",
    f"{BASE_URL}/live",
    f"{BASE_URL}/en/live",
    f"{BASE_URL}/breaking",
    f"{BASE_URL}/en/breaking",
]

# Pythonプロセス側の絶対上限
MAX_RUNTIME_MINUTES = 25

# HTTP
REQUEST_TIMEOUT = 8
MAX_RETRIES = 2

# 並列数
WORKERS = 12

# 外部データ取得を増やす際の上限
MAX_EXTERNAL_REQUESTS = 200

# 出力
OUTPUT_DIR = Path("output")

# User-Agent
USER_AGENT = (
    "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) "
    "AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) "
    "Version/18.0 Mobile/15E148 Safari/604.1"
)


# ============================================================
# MARKET DATA STRUCTURES
# ============================================================

@dataclass
class Outcome:

    name: str

    miraima_probability: Optional[float] = None

    model_probability: Optional[float] = None

    edge_probability_points: Optional[float] = None

    fair_probability: Optional[float] = None

    ev_proxy: Optional[float] = None


@dataclass
class Market:

    url: str

    market_id: str

    title: str

    category: str

    market_type: str

    status: str

    outcomes: list[Outcome] = field(
        default_factory=list
    )

    participants: Optional[int] = None

    event_datetime: Optional[str] = None

    source_text: str = ""

    data_quality: float = 0.0

    prediction_status: str = "unavailable"

    confidence: Optional[float] = None

    notes: str = ""


# ============================================================
# RUNTIME CONTROL
# ============================================================

class RuntimeGuard:

    """
    全処理を25分以内に収めるための中央制御。
    """

    def __init__(
        self,
        minutes: float
    ):

        self.start = time.monotonic()

        self.limit_seconds = (
            minutes * 60
        )


    def elapsed(self) -> float:

        return (
            time.monotonic()
            -
            self.start
        )


    def remaining(self) -> float:

        return max(
            0.0,
            self.limit_seconds
            -
            self.elapsed()
        )


    def expired(self) -> bool:

        return (
            self.elapsed()
            >=
            self.limit_seconds
        )


    def require_time(
        self,
        minimum_seconds: float = 1.0
    ) -> bool:

        return (
            self.remaining()
            >=
            minimum_seconds
        )


# ============================================================
# HTTP CLIENT
# ============================================================

class HTTPClient:

    def __init__(
        self,
        guard: RuntimeGuard
    ):

        self.guard = guard

        self.session = requests.Session()

        self.session.headers.update({

            "User-Agent":
                USER_AGENT,

            "Accept":
                "text/html,application/xhtml+xml,"
                "application/xml;q=0.9,*/*;q=0.8",

            "Accept-Language":
                "ja,en-US;q=0.9,en;q=0.8",

            "Cache-Control":
                "no-cache",

        })


    def get(
        self,
        url: str,
        timeout: Optional[float] = None
    ):

        if self.guard.expired():

            raise TimeoutError(
                "Runtime limit reached"
            )


        timeout_value = (
            timeout
            if timeout is not None
            else REQUEST_TIMEOUT
        )


        # 残り時間を超えるtimeoutを設定しない

        timeout_value = min(
            timeout_value,
            max(
                1.0,
                self.guard.remaining()
            )
        )


        last_error = None


        for attempt in range(
            MAX_RETRIES + 1
        ):

            if self.guard.expired():

                raise TimeoutError(
                    "Runtime limit reached"
                )


            try:

                response = self.session.get(
                    url,
                    timeout=timeout_value
                )

                response.raise_for_status()

                return response


            except Exception as exc:

                last_error = exc

                if attempt >= MAX_RETRIES:

                    break


                # 短い待機
                sleep_time = min(
                    0.5 * (attempt + 1),
                    max(
                        0.0,
                        self.guard.remaining()
                    )
                )


                if sleep_time > 0:

                    time.sleep(
                        sleep_time
                    )


        raise last_error


# ============================================================
# TEXT UTILITIES
# ============================================================

def clean_text(
    value: str
) -> str:

    value = (
        value
        .replace("\u3000", " ")
    )


    return re.sub(
        r"\s+",
        " ",
        value
    ).strip()


def normalize_url(
    url: str
) -> str:

    parsed = urlparse(url)

    return (
        f"{parsed.scheme}://"
        f"{parsed.netloc}"
        f"{parsed.path}"
    ).rstrip("/")


def make_market_id(
    url: str
) -> str:

    normalized = normalize_url(
        url
    )

    return hashlib.sha256(
        normalized.encode(
            "utf-8"
        )
    ).hexdigest()[:16]


def safe_float(
    value: Any
) -> Optional[float]:

    try:

        x = float(value)

        if math.isfinite(x):

            return x

    except Exception:

        pass


    return None


def normalize_probabilities(
    values: list[float]
) -> list[float]:

    arr = np.asarray(
        values,
        dtype=float
    )


    arr = np.nan_to_num(
        arr,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )


    arr = np.maximum(
        arr,
        0.0
    )


    total = arr.sum()


    if total <= 0:

        return (
            np.ones(
                len(arr)
            )
            /
            len(arr)
            *
            100
        ).tolist()


    return (
        arr
        /
        total
        *
        100
    ).tolist()


# ============================================================
# CATEGORY KEYWORDS
# ============================================================

CATEGORY_KEYWORDS = {

    "sports": [

        "sports",
        "sport",
        "baseball",
        "mlb",
        "npb",
        "soccer",
        "football",
        "premier league",
        "epl",
        "laliga",
        "serie a",
        "bundesliga",
        "ligue 1",
        "champions league",
        "nba",
        "basketball",
        "nfl",
        "nhl",
        "tennis",
        "ufc",
        "mma",
        "f1",
        "formula 1",
        "volleyball",
        "golf",
        "rugby",
        "hockey",
        "esports",
        "koshien",

    ],

    "finance": [

        "finance",
        "stock",
        "stocks",
        "share price",
        "nikkei",
        "topix",
        "s&p 500",
        "nasdaq",
        "dow jones",
        "usd/jpy",
        "eur/jpy",
        "gbp/jpy",
        "fx",
        "exchange rate",
        "interest rate",

    ],

    "crypto": [

        "bitcoin",
        "btc",
        "ethereum",
        "eth",
        "solana",
        "xrp",
        "crypto",
        "cryptocurrency",

    ],

    "weather": [

        "weather",
        "temperature",
        "rainfall",
        "rain",
        "snow",
        "typhoon",
        "wind",
        "forecast",
        "humidity",

    ],

    "transportation": [

        "transportation",
        "transport",
        "train",
        "railway",
        "subway",
        "metro",
        "delay",
        "traffic",
        "yamanote",
        "shinkansen",

    ],

    "politics_social": [

        "politics",
        "political",
        "election",
        "prime minister",
        "president",
        "government",
        "diet",
        "parliament",
        "cabinet",
        "poll",
        "taiwan",
        "social",

    ],

    "entertainment": [

        "entertainment",
        "movie",
        "film",
        "music",
        "anime",
        "manga",
        "actor",
        "actress",
        "singer",
        "idol",
        "release",
        "retirement",

    ],

    "technology": [

        "technology",
        "tech",
        "artificial intelligence",
        "ai",
        "software",
        "hardware",
        "apple",
        "google",
        "microsoft",
        "openai",
        "nintendo",

    ],
}


# ============================================================
# MARKET TYPE KEYWORDS
# ============================================================

MARKET_TYPE_PATTERNS = {

    "yes_no": [

        r"\byes\b.*\bno\b",
        r"\bno\b.*\byes\b",
        r"yes/no",
        r"はい.*いいえ",
        r"いいえ.*はい",

    ],

    "over_under": [

        r"\bover\b",
        r"\bunder\b",
        r"over/under",
        r"オーバー",
        r"アンダー",

    ],

    "score": [

        r"\bscore\b",
        r"exact score",
        r"final score",
        r"スコア",
        r"得点",

    ],

    "low_high": [

        r"\blow\b.*\bhigh\b",
        r"\bhigh\b.*\blow\b",
        r"low/high",
        r"low score",
        r"high score",
        r"ロースコア",
        r"ハイスコア",

    ],

    "handicap_margin": [

        r"\bhandicap\b",
        r"\bspread\b",
        r"\bmargin\b",
        r"ハンデ",
        r"ハンディキャップ",

    ],

    "player_prop": [

        r"\bplayer\b",
        r"\bhome run\b",
        r"\bstrikeout\b",
        r"\bhits?\b",
        r"\bgoals?\b.*player",
        r"\bmvp\b",
        r"\bmom\b",

    ],

}
# ============================================================
# CATEGORY CLASSIFICATION
# ============================================================

def classify_category(
    title: str,
    page_text: str,
    url: str = ""
) -> str:

    """
    URL・タイトル・本文をまとめて判定する。

    URLに明確なカテゴリ情報がある場合は、
    本文の一般的な単語より優先する。
    """

    combined = (
        f"{url} "
        f"{title} "
        f"{page_text}"
    ).lower()


    # URL優先判定

    path = urlparse(
        url
    ).path.lower()


    url_rules = [

        ("/sports/", "sports"),
        ("/baseball/", "sports"),
        ("/soccer/", "sports"),
        ("/football/", "sports"),
        ("/basketball/", "sports"),
        ("/tennis/", "sports"),

        ("/crypto/", "crypto"),

        ("/finance/", "finance"),
        ("/stocks/", "finance"),
        ("/stock/", "finance"),
        ("/fx/", "finance"),

        ("/weather/", "weather"),

        ("/technology/", "technology"),
        ("/tech/", "technology"),

        ("/politics/", "politics_social"),

        ("/entertainment/", "entertainment"),

    ]


    for pattern, category in url_rules:

        if pattern in path:

            return category


    # 明示的な市場タイトル・本文を優先

    strong_rules = [

        (
            [
                "bitcoin",
                "btc",
                "ethereum",
                "eth",
                "crypto",
                "cryptocurrency",
            ],
            "crypto"
        ),

        (
            [
                "mlb",
                "npb",
                "baseball",
                "premier league",
                "epl",
                "laliga",
                "nba",
                "nfl",
                "nhl",
                "ufc",
                "tennis",
                "volleyball",
                "koshien",
            ],
            "sports"
        ),

        (
            [
                "s&p 500",
                "nasdaq",
                "dow jones",
                "nikkei",
                "topix",
                "usd/jpy",
                "eur/jpy",
                "stock price",
                "share price",
            ],
            "finance"
        ),

        (
            [
                "weather",
                "temperature",
                "rainfall",
                "typhoon",
                "forecast",
            ],
            "weather"
        ),

        (
            [
                "yamanote",
                "train delay",
                "railway",
                "shinkansen",
                "traffic",
            ],
            "transportation"
        ),

        (
            [
                "prime minister",
                "election",
                "government",
                "cabinet",
                "parliament",
                "taiwan contingency",
            ],
            "politics_social"
        ),

        (
            [
                "movie",
                "film",
                "anime",
                "manga",
                "singer",
                "idol",
                "entertainment",
            ],
            "entertainment"
        ),

        (
            [
                "artificial intelligence",
                "technology",
                "software",
                "hardware",
                "openai",
                "microsoft",
                "apple",
                "nintendo",
            ],
            "technology"
        ),

    ]


    for keywords, category in strong_rules:

        if any(
            keyword in combined
            for keyword in keywords
        ):

            return category


    # 一般キーワードは最後

    scores = {
        category: 0
        for category
        in CATEGORY_KEYWORDS
    }


    for category, keywords in (
        CATEGORY_KEYWORDS.items()
    ):

        for keyword in keywords:

            if keyword in combined:

                scores[category] += 1


    if scores:

        best_category = max(
            scores,
            key=scores.get
        )


        if scores[best_category] > 0:

            return best_category


    return "other"


# ============================================================
# MARKET TYPE CLASSIFICATION
# ============================================================

def classify_market_type(
    title: str,
    page_text: str,
    outcomes: list[str],
    url: str = ""
) -> str:

    """
    市場タイプ判定。

    優先順位:

    1. URL
    2. タイトル
    3. ルール本文
    4. 選択肢数

    単純に "score" 等がページ内に存在するだけで
    判定しないようにする。
    """

    title_text = clean_text(
        title
    ).lower()


    body_text = clean_text(
        page_text
    ).lower()


    path = urlparse(
        url
    ).path.lower()


    # --------------------------------------------------------
    # URLベース
    # --------------------------------------------------------

    if "updown" in path:

        return "two_way"


    if (
        "yes-no" in path
        or
        "yesno" in path
    ):

        return "yes_no"


    if (
        "over-under" in path
        or
        "overunder" in path
    ):

        return "over_under"


    if (
        "score" in path
        and
        "updown" not in path
    ):

        return "score"


    # --------------------------------------------------------
    # タイトルベース
    # --------------------------------------------------------

    title_patterns = [

        (
            [
                "over/under",
                "over under",
                "オーバー/アンダー",
                "オーバー アンダー",
            ],
            "over_under"
        ),

        (
            [
                "exact score",
                "final score",
                "正確なスコア",
                "スコア予想",
            ],
            "score"
        ),

        (
            [
                "low/high",
                "low high",
                "low score",
                "high score",
                "ロースコア",
                "ハイスコア",
            ],
            "low_high"
        ),

        (
            [
                "handicap",
                "spread",
                "ハンディキャップ",
                "ハンデ",
            ],
            "handicap_margin"
        ),

        (
            [
                "yes/no",
                "yes or no",
                "はい/いいえ",
            ],
            "yes_no"
        ),

    ]


    for keywords, market_type in title_patterns:

        if any(
            keyword in title_text
            for keyword in keywords
        ):

            return market_type


    # --------------------------------------------------------
    # 明確なYES/NO
    # --------------------------------------------------------

    outcome_lower = [
        x.strip().lower()
        for x in outcomes
    ]


    if (
        len(outcome_lower) == 2
        and
        set(outcome_lower)
        == {"yes", "no"}
    ):

        return "yes_no"


    # --------------------------------------------------------
    # ルール本文
    # --------------------------------------------------------

    rule_patterns = [

        (
            [
                "over",
                "under",
                "total goals",
                "total runs",
                "total points",
                "オーバー",
                "アンダー",
            ],
            "over_under"
        ),

        (
            [
                "exact final score",
                "exact score",
                "final score will be",
                "最終スコア",
            ],
            "score"
        ),

        (
            [
                "handicap",
                "spread",
                "point spread",
                "run line",
            ],
            "handicap_margin"
        ),

        (
            [
                "player prop",
                "player performance",
                "home run",
                "strikeout",
                "most valuable player",
            ],
            "player_prop"
        ),

    ]


    for keywords, market_type in rule_patterns:

        # 本文全体の一般語だけでは誤認識しやすいため、
        # 複数語がある場合に限定する。

        hits = sum(
            1
            for keyword in keywords
            if keyword in body_text
        )


        if hits >= 2:

            return market_type


    # --------------------------------------------------------
    # 選択肢数
    # --------------------------------------------------------

    if len(outcomes) == 2:

        return "two_way"


    if len(outcomes) == 3:

        return "three_way"


    if len(outcomes) > 3:

        return "multi_choice"


    return "unknown"


# ============================================================
# PROBABILITY EXTRACTION
# ============================================================

PROBABILITY_RE = re.compile(
    r"(.{1,100}?)"
    r"(?:\s+|：|:)"
    r"(\d{1,3}(?:\.\d+)?)"
    r"\s*%"
)


def extract_probabilities(
    soup: BeautifulSoup
) -> tuple[list[str], list[float]]:

    candidates = []


    # 比較的意味のある要素を優先

    elements = soup.find_all(
        [
            "button",
            "a",
            "li",
            "span",
            "p",
        ]
    )


    for element in elements:

        text = clean_text(
            element.get_text(
                " ",
                strip=True
            )
        )


        if not text:
            continue


        if len(text) > 150:
            continue


        matches = list(
            PROBABILITY_RE.finditer(
                text
            )
        )


        for match in matches:

            label = clean_text(
                match.group(1)
            )


            probability = safe_float(
                match.group(2)
            )


            if probability is None:
                continue


            if not (
                0 <= probability <= 100
            ):
                continue


            # UI上の不要文字を除去

            label = re.sub(
                r"^[•●○◯■□▶▷→←\-\s]+",
                "",
                label
            )


            label = clean_text(
                label
            )


            if not label:
                continue


            candidates.append(
                (
                    label,
                    probability
                )
            )


    # 重複排除

    unique = []

    seen = set()


    for label, probability in candidates:

        key = (
            label.lower(),
            round(
                probability,
                3
            )
        )


        if key in seen:
            continue


        seen.add(key)

        unique.append(
            (
                label,
                probability
            )
        )


    # 同一市場の候補群を探索

    groups = []


    for start in range(
        len(unique)
    ):

        for end in range(
            start + 2,
            min(
                start + 7,
                len(unique)
            ) + 1
        ):

            group = unique[
                start:end
            ]


            total = sum(
                p
                for _, p
                in group
            )


            # MIRAIMAの確率合計は基本的に100%

            if (
                97 <= total <= 103
            ):

                # 小さい誤差を優先。
                # 同点なら選択肢数が多いもの。

                score = (
                    abs(
                        total - 100
                    ),
                    -len(group)
                )


                groups.append(
                    (
                        score,
                        group
                    )
                )


    if groups:

        groups.sort(
            key=lambda x: x[0]
        )

        unique = groups[0][1]


    # 最大8択

    if len(unique) > 8:

        unique = unique[:8]


    outcomes = [
        label
        for label, _
        in unique
    ]


    probabilities = [
        probability
        for _, probability
        in unique
    ]


    return (
        outcomes,
        probabilities
    )


# ============================================================
# PARTICIPANT EXTRACTION
# ============================================================

def extract_participants(
    text: str
) -> Optional[int]:

    patterns = [

        r"(\d[\d,]*)\s+participants",

        r"(\d[\d,]*)\s+participant",

        r"(\d[\d,]*)\s+people",

        r"参加者[^\d]{0,30}"
        r"(\d[\d,]*)",

    ]


    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )


        if not match:

            continue


        try:

            value = int(
                match.group(1)
                .replace(",", "")
            )


            if value >= 0:

                return value


        except Exception:

            continue


    return None


# ============================================================
# STATUS
# ============================================================

def detect_status(
    text: str
) -> str:

    lower = text.lower()


    ended_patterns = [

        "has ended",
        "ended",
        "closed",
        "settled",
        "resolved",

    ]


    live_patterns = [

        "live",
        "trading now",
        "open",

    ]


    if any(
        x in lower
        for x in ended_patterns
    ):

        return "ended"


    if any(
        x in lower
        for x in live_patterns
    ):

        return "live"


    return "open"
    # ============================================================
# EVENT DATE/TIME EXTRACTION
# ============================================================

DATE_PATTERNS = [

    re.compile(
        r"(20\d{2})[-/年]"
        r"(\d{1,2})[-/月]"
        r"(\d{1,2})"
    ),

    re.compile(
        r"(\d{1,2})/(\d{1,2})"
    ),

]


def extract_event_datetime(
    text: str
) -> Optional[str]:

    """
    ページ本文からイベント日時候補を取得。

    厳密な日時が取れない場合はNone。
    """

    # ISO形式

    iso = re.search(
        r"20\d{2}-\d{2}-\d{2}"
        r"(?:[T\s]\d{2}:\d{2}"
        r"(?::\d{2})?)?",
        text
    )


    if iso:

        return iso.group(0)


    # 日本語形式

    jp = re.search(
        r"(20\d{2})年"
        r"(\d{1,2})月"
        r"(\d{1,2})日"
        r"(?:\s*(\d{1,2}):(\d{2}))?",
        text
    )


    if jp:

        year = int(
            jp.group(1)
        )

        month = int(
            jp.group(2)
        )

        day = int(
            jp.group(3)
        )


        hour = jp.group(4)

        minute = jp.group(5)


        if hour is not None:

            return (
                f"{year:04d}-"
                f"{month:02d}-"
                f"{day:02d} "
                f"{int(hour):02d}:"
                f"{int(minute):02d}"
            )


        return (
            f"{year:04d}-"
            f"{month:02d}-"
            f"{day:02d}"
        )


    return None


# ============================================================
# DATA QUALITY
# ============================================================

def calculate_data_quality(
    market: Market
) -> float:

    """
    予測に使える情報量を0～1で評価。

    市場確率そのものは品質評価に使わない。
    """

    score = 0.0


    if market.title:

        score += 0.15


    if market.url:

        score += 0.10


    if len(
        market.outcomes
    ) >= 2:

        score += 0.20


    if all(
        x.miraima_probability
        is not None
        for x in market.outcomes
    ):

        score += 0.15


    if market.event_datetime:

        score += 0.10


    if market.participants is not None:

        score += 0.05


    if market.source_text:

        score += 0.10


    if market.market_type != "unknown":

        score += 0.15


    return min(
        1.0,
        score
    )


# ============================================================
# PARSE MARKET
# ============================================================

def parse_market(
    url: str,
    html: str
) -> Optional[Market]:

    soup = BeautifulSoup(
        html,
        "html.parser"
    )


    page_text = clean_text(
        " ".join(
            soup.stripped_strings
        )
    )


    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    title = ""


    h1 = soup.find(
        "h1"
    )


    if h1:

        title = clean_text(
            h1.get_text(
                " ",
                strip=True
            )
        )


    if not title and soup.title:

        title = clean_text(
            soup.title.get_text(
                " ",
                strip=True
            )
        )


    if not title:

        title = page_text[:160]


    # --------------------------------------------------------
    # PROBABILITIES
    # --------------------------------------------------------

    outcomes, probabilities = (
        extract_probabilities(
            soup
        )
    )


    # 2択未満は市場として採用しない

    if len(outcomes) < 2:

        return None


    # --------------------------------------------------------
    # PROBABILITY SANITY CHECK
    # --------------------------------------------------------

    total = sum(
        probabilities
    )


    if (
        95 <= total <= 105
    ):

        probabilities = (
            normalize_probabilities(
                probabilities
            )
        )

    else:

        # 確率合計が大きく外れている場合、
        # 無理に100%へ補正しない。

        return None


    # --------------------------------------------------------
    # MARKET OBJECT
    # --------------------------------------------------------

    market = Market(

        url=normalize_url(
            url
        ),

        market_id=make_market_id(
            url
        ),

        title=title,

        category=classify_category(
            title,
            page_text,
            url
        ),

        market_type=classify_market_type(
            title,
            page_text,
            outcomes,
            url
        ),

        status=detect_status(
            page_text
        ),

        participants=extract_participants(
            page_text
        ),

        event_datetime=extract_event_datetime(
            page_text
        ),

        source_text=page_text[:10000],

    )


    # --------------------------------------------------------
    # OUTCOMES
    # --------------------------------------------------------

    for name, probability in zip(
        outcomes,
        probabilities
    ):

        market.outcomes.append(

            Outcome(

                name=clean_text(
                    name
                ),

                miraima_probability=round(
                    probability,
                    6
                ),

            )

        )


    market.data_quality = (
        calculate_data_quality(
            market
        )
    )


    return market


# ============================================================
# MARKET DISCOVERY
# ============================================================

def discover_market_urls(
    client: HTTPClient,
    guard: RuntimeGuard
) -> list[str]:

    urls = set()


    for start_url in DISCOVERY_URLS:

        if guard.expired():

            break


        try:

            response = client.get(
                start_url
            )


        except Exception as exc:

            print(
                "[WARN] discovery:",
                start_url,
                exc
            )

            continue


        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )


        for anchor in soup.find_all(
            "a",
            href=True
        ):

            if guard.expired():

                break


            href = (
                anchor.get("href")
                or ""
            ).strip()


            if not href:

                continue


            full_url = urljoin(
                BASE_URL,
                href
            )


            parsed = urlparse(
                full_url
            )


            # MIRAIMA本体のみ

            if (
                parsed.netloc
                != urlparse(BASE_URL).netloc
            ):

                continue


            path = (
                parsed.path
                .lower()
            )


            # 市場ページ候補

            if (
                "/event/" in path
                or
                "/prediction-market/"
                in path
                or
                "/crypto/" in path
                or
                "/sports/" in path
            ):

                urls.add(
                    normalize_url(
                        full_url
                    )
                )


    return sorted(
        urls
    )


# ============================================================
# DEDUPLICATION
# ============================================================

def deduplicate_markets(
    markets: list[Market]
) -> list[Market]:

    result = []

    seen_ids = set()

    seen_signatures = set()


    for market in markets:

        if (
            market.market_id
            in seen_ids
        ):

            continue


        # URLだけでなくタイトル＋選択肢でも重複確認

        outcome_signature = "|".join(
            sorted(
                x.name.lower()
                for x in market.outcomes
            )
        )


        signature = (
            market.title.lower(),
            market.category,
            market.market_type,
            outcome_signature
        )


        if signature in seen_signatures:

            continue


        seen_ids.add(
            market.market_id
        )

        seen_signatures.add(
            signature
        )

        result.append(
            market
        )


    return result


# ============================================================
# FETCH MARKET
# ============================================================

def fetch_market(
    client: HTTPClient,
    guard: RuntimeGuard,
    url: str
):

    if guard.expired():

        return None


    try:

        response = client.get(
            url
        )


        market = parse_market(
            url,
            response.text
        )


        return market


    except Exception as exc:

        print(
            "[WARN] market:",
            url,
            exc
        )

        return None
        # ============================================================
# INDEPENDENT MODEL
# ============================================================

def entropy(
    probabilities: list[float]
) -> float:

    arr = np.asarray(
        probabilities,
        dtype=float
    )


    arr = np.clip(
        arr,
        1e-12,
        1.0
    )


    return float(
        -np.sum(
            arr * np.log(
                arr
            )
        )
    )


def probability_shrinkage(
    probabilities: list[float],
    strength: float
) -> list[float]:

    """
    過信を抑えるための確率収縮。

    strength:
      0 = 収縮なし
      1 = 完全一様分布
    """

    arr = (
        np.asarray(
            probabilities,
            dtype=float
        )
        /
        100
    )


    n = len(arr)


    if n == 0:

        return []


    uniform = (
        np.ones(n)
        /
        n
    )


    result = (
        arr * (1 - strength)
        +
        uniform * strength
    )


    result /= result.sum()


    return (
        result * 100
    ).tolist()


def category_prior(
    market: Market
) -> Optional[list[float]]:

    """
    独立情報がない市場に対して
    勝手な予測を生成しない。

    現時点では、外部データが取得できない場合は
    Noneを返して見送りとする。
    """

    return None


def independent_prediction(
    market: Market
) -> Optional[list[float]]:

    """
    独立予測の入口。

    重要:
    MIRAIMAの表示確率を独立予測として
    そのままコピーしない。

    外部特徴量モデルが利用できない市場は
    「予測不能」とする。
    """

    prior = category_prior(
        market
    )


    if prior is None:

        return None


    return probability_shrinkage(
        prior,
        0.05
    )


# ============================================================
# EV
# ============================================================

def calculate_ev_proxy(
    model_probability: Optional[float],
    market_probability: Optional[float]
) -> Optional[float]:

    """
    払戻倍率が取得できない場合の相対評価。

    これは「真の期待値」ではない。
    """

    if (
        model_probability is None
        or
        market_probability is None
    ):

        return None


    p_market = (
        market_probability
        /
        100
    )


    p_model = (
        model_probability
        /
        100
    )


    if p_market <= 0:

        return None


    return (
        p_model
        /
        p_market
        - 1
    )


# ============================================================
# MARKET EVALUATION
# ============================================================

def evaluate_market(
    market: Market
) -> list[dict]:

    model_probabilities = (
        independent_prediction(
            market
        )
    )


    rows = []


    if model_probabilities is None:

        market.prediction_status = (
            "insufficient_independent_data"
        )

        market.notes = (
            "独立予測に必要な外部データ不足"
        )

        for outcome in market.outcomes:

            rows.append({

                "market_id":
                    market.market_id,

                "url":
                    market.url,

                "title":
                    market.title,

                "category":
                    market.category,

                "market_type":
                    market.market_type,

                "status":
                    market.status,

                "outcome":
                    outcome.name,

                "miraima_probability":
                    outcome.miraima_probability,

                "model_probability":
                    None,

                "edge_probability_points":
                    None,

                "ev_proxy":
                    None,

                "prediction_status":
                    market.prediction_status,

                "confidence":
                    None,

                "data_quality":
                    market.data_quality,

                "participants":
                    market.participants,

                "event_datetime":
                    market.event_datetime,

                "notes":
                    market.notes,

            })


        return rows


    # 独立モデルが存在する場合のみ比較

    for index, outcome in enumerate(
        market.outcomes
    ):

        if index >= len(
            model_probabilities
        ):

            break


        model_probability = (
            model_probabilities[index]
        )


        miraima_probability = (
            outcome.miraima_probability
        )


        edge = None


        if miraima_probability is not None:

            edge = (
                model_probability
                -
                miraima_probability
            )


        ev_proxy = (
            calculate_ev_proxy(
                model_probability,
                miraima_probability
            )
        )


        rows.append({

            "market_id":
                market.market_id,

            "url":
                market.url,

            "title":
                market.title,

            "category":
                market.category,

            "market_type":
                market.market_type,

            "status":
                market.status,

            "outcome":
                outcome.name,

            "miraima_probability":
                miraima_probability,

            "model_probability":
                round(
                    model_probability,
                    4
                ),

            "edge_probability_points":
                None
                if edge is None
                else round(
                    edge,
                    4
                ),

            "ev_proxy":
                None
                if ev_proxy is None
                else round(
                    ev_proxy,
                    6
                ),

            "prediction_status":
                "predicted",

            "confidence":
                None,

            "data_quality":
                market.data_quality,

            "participants":
                market.participants,

            "event_datetime":
                market.event_datetime,

            "notes":
                "",

        })


    return rows


# ============================================================
# SAVE INVENTORY
# ============================================================

def save_inventory(
    markets: list[Market]
):

    rows = []


    for market in markets:

        rows.append({

            "market_id":
                market.market_id,

            "url":
                market.url,

            "title":
                market.title,

            "category":
                market.category,

            "market_type":
                market.market_type,

            "status":
                market.status,

            "outcomes":
                " | ".join(
                    x.name
                    for x in market.outcomes
                ),

            "miraima_probabilities":
                " | ".join(
                    f"{x.miraima_probability:.4f}"
                    for x in market.outcomes
                    if x.miraima_probability
                    is not None
                ),

            "participants":
                market.participants,

            "event_datetime":
                market.event_datetime,

            "data_quality":
                round(
                    market.data_quality,
                    4
                ),

            "prediction_status":
                market.prediction_status,

            "notes":
                market.notes,

        })


    df = pd.DataFrame(
        rows
    )


    df.to_csv(
        OUTPUT_DIR /
        "miraima_markets.csv",

        index=False,

        encoding="utf-8-sig"
    )


# ============================================================
# SAVE EVALUATION
# ============================================================

def save_evaluation(
    rows: list[dict]
):

    df = pd.DataFrame(
        rows
    )


    if not df.empty:

        # EV Proxyが存在するものを上位へ

        df["_sort_ev"] = (
            pd.to_numeric(
                df["ev_proxy"],
                errors="coerce"
            )
        )


        df = df.sort_values(
            "_sort_ev",
            ascending=False,
            na_position="last"
        )


        df = df.drop(
            columns=["_sort_ev"]
        )


    df.to_csv(
        OUTPUT_DIR /
        "miraima_market_values.csv",

        index=False,

        encoding="utf-8-sig"
    )


# ============================================================
# SUMMARY
# ============================================================

def build_summary(
    discovered: int,
    parsed: int,
    evaluated_rows: int,
    elapsed_seconds: float,
    guard: RuntimeGuard
) -> dict:

    return {

        "markets_discovered":
            discovered,

        "markets_parsed":
            parsed,

        "outcome_rows":
            evaluated_rows,

        "elapsed_seconds":
            round(
                elapsed_seconds,
                3
            ),

        "runtime_limit_minutes":
            MAX_RUNTIME_MINUTES,

        "runtime_limit_seconds":
            MAX_RUNTIME_MINUTES * 60,

        "finished_within_25_minutes":
            elapsed_seconds
            <
            MAX_RUNTIME_MINUTES * 60,

        "finished_within_30_minutes":
            elapsed_seconds
            <
            30 * 60,

        "runtime_guard_expired":
            guard.expired(),

    }
    # ============================================================
# MAIN ENGINE
# ============================================================

def run_engine() -> int:

    guard = RuntimeGuard(
        MAX_RUNTIME_MINUTES
    )


    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    print("=" * 72)
    print(
        "MIRAIMA HIGH-PRECISION ALL-MARKET ENGINE"
    )
    print("=" * 72)

    print(
        f"Hard runtime limit: "
        f"{MAX_RUNTIME_MINUTES} minutes"
    )

    print(
        f"HTTP workers: {WORKERS}"
    )


    client = HTTPClient(
        guard
    )


    # ========================================================
    # STEP 1
    # ========================================================

    print(
        "\n[1/5] Discovering MIRAIMA markets..."
    )


    urls = discover_market_urls(
        client,
        guard
    )


    if guard.expired():

        print(
            "Runtime limit reached "
            "during discovery."
        )


    print(
        f"Discovered URLs: {len(urls)}"
    )


    # ========================================================
    # STEP 2
    # ========================================================

    print(
        "\n[2/5] Fetching market pages..."
    )


    markets = []


    # 残り時間がある場合のみ取得

    if (
        urls
        and
        not guard.expired()
    ):

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=WORKERS
        ) as executor:

            futures = {

                executor.submit(
                    fetch_market,
                    client,
                    guard,
                    url
                ):
                    url

                for url in urls

            }


            for future in (
                concurrent.futures.as_completed(
                    futures
                )
            ):

                if guard.expired():

                    print(
                        "Runtime guard triggered."
                    )

                    for pending in futures:

                        pending.cancel()

                    break


                try:

                    market = future.result()


                    if market is not None:

                        markets.append(
                            market
                        )


                except Exception as exc:

                    print(
                        "[WARN] worker:",
                        exc
                    )


    # ========================================================
    # STEP 3
    # ========================================================

    print(
        "\n[3/5] Deduplicating and validating..."
    )


    markets = deduplicate_markets(
        markets
    )


    print(
        f"Valid unique markets: "
        f"{len(markets)}"
    )


    save_inventory(
        markets
    )


    # ========================================================
    # STEP 4
    # ========================================================

    print(
        "\n[4/5] Evaluating markets..."
    )


    evaluation_rows = []


    for market in markets:

        if guard.expired():

            print(
                "Runtime guard triggered "
                "during evaluation."
            )

            break


        evaluation_rows.extend(
            evaluate_market(
                market
            )
        )


    save_evaluation(
        evaluation_rows
    )


    # ========================================================
    # STEP 5
    # ========================================================

    elapsed = guard.elapsed()


    summary = build_summary(

        discovered=len(urls),

        parsed=len(markets),

        evaluated_rows=len(
            evaluation_rows
        ),

        elapsed_seconds=elapsed,

        guard=guard

    )


    (
        OUTPUT_DIR /
        "run_summary.json"
    ).write_text(

        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2
        ),

        encoding="utf-8"

    )


    # ========================================================
    # CATEGORY SUMMARY
    # ========================================================

    if markets:

        category_counts = {}

        market_type_counts = {}


        for market in markets:

            category_counts[
                market.category
            ] = (
                category_counts.get(
                    market.category,
                    0
                )
                + 1
            )


            market_type_counts[
                market.market_type
            ] = (
                market_type_counts.get(
                    market.market_type,
                    0
                )
                + 1
            )


        (
            OUTPUT_DIR /
            "category_summary.json"
        ).write_text(

            json.dumps(
                category_counts,
                ensure_ascii=False,
                indent=2
            ),

            encoding="utf-8"

        )


        (
            OUTPUT_DIR /
            "market_type_summary.json"
        ).write_text(

            json.dumps(
                market_type_counts,
                ensure_ascii=False,
                indent=2
            ),

            encoding="utf-8"

        )


    # ========================================================
    # FINISH
    # ========================================================

    print(
        "\n" + "=" * 72
    )

    print(
        "ENGINE FINISHED"
    )

    print(
        "=" * 72
    )


    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2
        )
    )


    print(
        "\nGenerated:"
    )

    print(
        "  output/miraima_markets.csv"
    )

    print(
        "  output/miraima_market_values.csv"
    )

    print(
        "  output/run_summary.json"
    )

    print(
        "  output/category_summary.json"
    )

    print(
        "  output/market_type_summary.json"
    )


    # 25分を超えていた場合は失敗扱い

    if elapsed >= (
        MAX_RUNTIME_MINUTES * 60
    ):

        return 124


    return 0


# ============================================================
# ENTRY POINT
# ============================================================

def main():

    try:

        return run_engine()


    except KeyboardInterrupt:

        print(
            "\nInterrupted."
        )

        return 130


    except Exception as exc:

        print(
            "\n[FATAL]",
            repr(exc)
        )

        return 1


if __name__ == "__main__":

    sys.exit(
        main()
    )
    # ============================================================
# END
# ============================================================

"""
OUTPUT FILES
------------

miraima_markets.csv
    発見したMIRAIMA市場の一覧

miraima_market_values.csv
    市場ごとの予測評価

run_summary.json
    実行時間・取得数

category_summary.json
    カテゴリー別市場数

market_type_summary.json
    市場タイプ別市場数


RUNTIME SAFETY
--------------

Python:
    MAX_RUNTIME_MINUTES = 25

GitHub Actions:
    timeout-minutes = 29

したがって、通常の実行では30分を超えない。


IMPORTANT
---------

このバージョンでは、外部データが存在しない市場に対して
偽の独立予測を作らない。

そのため、現時点では
"insufficient_independent_data"
になる市場が存在する。

これは不具合ではなく、
「データがないのに適当な確率を出して精度を偽装する」
ことを防ぐための仕様である。


NEXT MODEL LAYER
----------------

本格的な予測モデルを追加する場合は、

    independent_prediction()

を以下のモデル群へ接続する。

Sports:
    - Elo
    - rolling performance
    - home advantage
    - starting lineup
    - injuries
    - starting pitchers
    - bullpen
    - xG
    - shot quality
    - possession
    - schedule strength
    - rest
    - travel
    - weather
    - market movement

Finance:
    - returns
    - volatility
    - momentum
    - trend
    - volume
    - macro variables

Crypto:
    - price momentum
    - volatility
    - volume
    - order-flow proxies
    - market regime

Weather:
    - forecast ensemble
    - temperature
    - precipitation
    - wind
    - pressure

Transportation:
    - historical delay
    - weekday
    - time
    - disruption information

Politics/Social:
    - polls
    - event information
    - historical base rates

Entertainment:
    - release information
    - historical performance
    - trend signals

Technology:
    - product/release events
    - adoption indicators
    - company information


NO AUTOMATED TRADING
--------------------

このプログラムはMIRAIMAへ

    - 自動参加
    - 自動売買
    - 自動予測送信

を行わない。


END
"""