#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MIRAIMA ALL-MARKET ENGINE

公開されているMIRAIMA市場を可能な限り収集し、
カテゴリー・市場タイプを分類して評価する。

重要:
- MIRAIMAに実際に存在する公開市場のみ対象
- TOP30制限なし
- 自動参加・自動売買は行わない
- 最大実行時間25分
- GitHub Actions側でも29分の安全制限を設定
"""

from __future__ import annotations

import concurrent.futures
import json
import re
import sys
import time

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin, urlparse

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup


# ============================================================
# SETTINGS
# ============================================================

BASE_URL = "https://miraima.app"

START_URLS = [
    f"{BASE_URL}/en",
    f"{BASE_URL}/en/new",
    f"{BASE_URL}/en/live",
    f"{BASE_URL}/en/breaking",
]

# Python側のハード制限
MAX_RUNTIME_MINUTES = 25

# 同時取得数
WORKERS = 12

# 1ページあたりの通信タイムアウト
REQUEST_TIMEOUT = 8

OUTPUT_DIR = Path("output")

USER_AGENT = (
    "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) "
    "AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) "
    "Version/18.0 Mobile/15E148 Safari/604.1"
)


# ============================================================
# DATA STRUCTURE
# ============================================================

@dataclass
class Market:

    url: str

    title: str

    category: str

    market_type: str

    outcomes: list[str]

    probabilities: list[float]

    participants: int | None

    status: str


# ============================================================
# BASIC UTILITIES
# ============================================================

def clean_text(text: str) -> str:

    return re.sub(
        r"\s+",
        " ",
        text
    ).strip(
        " -|:・"
    )


def normalize_probabilities(
    probabilities: list[float]
) -> list[float]:

    arr = np.asarray(
        probabilities,
        dtype=float
    )

    arr = np.clip(
        arr,
        1e-9,
        None
    )

    arr /= arr.sum()

    return (
        arr * 100
    ).tolist()


# ============================================================
# HTTP CLIENT
# ============================================================

class MiraimaClient:

    def __init__(
        self,
        timeout: int = REQUEST_TIMEOUT
    ):

        self.timeout = timeout

        self.session = requests.Session()

        self.session.headers.update({
            "User-Agent": USER_AGENT,
            "Accept-Language": "ja,en;q=0.8",
        })


    def get(self, url: str):

        return self.session.get(
            url,
            timeout=self.timeout
        )


# ============================================================
# MARKET DISCOVERY
# ============================================================

def discover_market_urls(
    client: MiraimaClient
) -> list[str]:

    urls = set()

    for start_url in START_URLS:

        try:

            response = client.get(
                start_url
            )

            response.raise_for_status()

        except Exception as e:

            print(
                f"[WARN] discovery failed: "
                f"{start_url} / {e}"
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

            href = anchor["href"]

            full_url = urljoin(
                BASE_URL,
                href
            )

            parsed = urlparse(
                full_url
            )


            if (
                parsed.netloc
                != urlparse(BASE_URL).netloc
            ):
                continue


            path = parsed.path


            if (
                "/event/" in path
                or
                "/prediction-market/details/"
                in path
            ):

                urls.add(
                    full_url.split("#")[0]
                )


    return sorted(urls)
    # ============================================================
# CATEGORY CLASSIFICATION
# ============================================================

CATEGORY_KEYWORDS = {

    "crypto": [

        "bitcoin",
        "btc",
        "ethereum",
        "eth",
        "crypto",
        "solana",
        "xrp",

    ],

    "finance": [

        "stock",
        "stocks",
        "s&p",
        "nasdaq",
        "dow",
        "nikkei",
        "topix",
        "usd/jpy",
        "eur/jpy",
        "fx",
        "finance",
        "yen",
        "dollar",

    ],

    "weather": [

        "weather",
        "temperature",
        "rain",
        "snow",
        "typhoon",
        "wind",
        "forecast",

    ],

    "transportation": [

        "train",
        "trains",
        "yamanote",
        "delay",
        "traffic",
        "transportation",
        "railway",

    ],

    "technology": [

        "technology",
        "artificial intelligence",
        "ai ",
        "openai",
        "apple",
        "google",
        "microsoft",
        "nintendo",
        "software",

    ],

    "politics_social": [

        "politics",
        "election",
        "prime minister",
        "diet ",
        "government",
        "taiwan",
        "trump",
        "president",
        "parliament",

    ],

    "entertainment": [

        "entertainment",
        "manga",
        "anime",
        "movie",
        "music",
        "actor",
        "actress",
        "release",
        "retire",
        "game",

    ],

    "sports": [

        "mlb",
        "npb",
        "baseball",
        "soccer",
        "football",
        "basketball",
        "nba",
        "tennis",
        "mma",
        "ufc",
        "f1",
        "volleyball",
        "golf",
        "rugby",
        "hockey",
        "esports",
        "sports",
        "match",
        "game",

    ],

}


def classify_category(
    title: str,
    page_text: str
) -> str:

    text = (
        f"{title} "
        f"{page_text}"
    ).lower()


    for category_name, keywords in (
        CATEGORY_KEYWORDS.items()
    ):

        for keyword in keywords:

            if keyword in text:

                return category_name


    return "other"


# ============================================================
# MARKET TYPE CLASSIFICATION
# ============================================================

def classify_market_type(
    title: str,
    page_text: str,
    outcomes: list[str]
) -> str:

    text = (
        f"{title} "
        f"{page_text}"
    ).lower()


    if re.search(
        r"\bover\b|\bunder\b|"
        r"オーバー|アンダー",
        text
    ):

        return "over_under"


    if any(
        x in text
        for x in [
            "low score",
            "high score",
            "low/high",
            "ロースコア",
            "ハイスコア",
        ]
    ):

        return "low_high"


    if (
        "score" in text
        or
        "スコア" in text
    ):

        return "score"


    if (
        "yes" in text
        and
        "no" in text
    ):

        return "yes_no"


    if any(
        x in text
        for x in [
            "player",
            "home run",
            "strikeout",
            "hits",
            "mom",
            "mvp",
        ]
    ):

        return "player_prop"


    if any(
        x in text
        for x in [
            "handicap",
            "spread",
            "margin",
        ]
    ):

        return "handicap_margin"


    if len(outcomes) == 3:

        return "three_way"


    if len(outcomes) == 2:

        return "two_way"


    return "multi_choice"


# ============================================================
# PROBABILITY EXTRACTION
# ============================================================

def extract_probabilities(
    soup: BeautifulSoup
) -> tuple[list[str], list[float]]:

    candidates = []


    elements = soup.find_all(
        [
            "span",
            "button",
            "a",
            "li",
            "p",
            "div",
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


        if len(text) > 120:
            continue


        match = re.search(
            r"(.+?)\s+(\d{1,3})\s*%$",
            text
        )


        if not match:
            continue


        label = clean_text(
            match.group(1)
        )


        probability = float(
            match.group(2)
        )


        if not (
            0 <= probability <= 100
        ):

            continue


        if not label:
            continue


        candidates.append(
            (
                label,
                probability
            )
        )


    # 重複削除

    unique = []

    seen = set()


    for label, probability in candidates:

        key = (
            label.lower(),
            probability
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


    # 合計が100%に近いグループを探索

    best = None


    for i in range(
        len(unique)
    ):

        for j in range(
            i + 2,
            min(
                len(unique),
                i + 6
            ) + 1
        ):

            group = unique[i:j]

            total = sum(
                p
                for _, p
                in group
            )


            if (
                97 <= total <= 103
            ):

                score = (
                    abs(total - 100),
                    -len(group)
                )


                if (
                    best is None
                    or
                    score < best[0]
                ):

                    best = (
                        score,
                        group
                    )


    if best is not None:

        unique = best[1]


    # 異常に多い候補は最大6択まで

    if len(unique) > 6:

        unique = unique[:6]


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
# PARTICIPANTS
# ============================================================

def extract_participants(
    text: str
) -> int | None:

    patterns = [

        r"(\d[\d,]*)\s+participants",

        r"(\d[\d,]*)\s+people",

        r"参加者[^\d]{0,20}"
        r"(\d[\d,]*)",

    ]


    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            re.I
        )


        if not match:
            continue


        try:

            return int(
                match.group(1)
                .replace(",", "")
            )

        except Exception:

            pass


    return None


# ============================================================
# STATUS
# ============================================================

def detect_status(
    text: str
) -> str:

    lower = text.lower()


    if (
        "has ended" in lower
        or
        "ended" in lower
        or
        "closed" in lower
        or
        "結果" in text
    ):

        return "ended"


    if "live" in lower:

        return "live"


    return "open"


# ============================================================
# PARSE MARKET
# ============================================================

def parse_market(
    url: str,
    html: str
) -> Market | None:

    soup = BeautifulSoup(
        html,
        "html.parser"
    )


    page_text = " ".join(
        soup.stripped_strings
    )


    h1 = soup.find("h1")


    if h1:

        title = clean_text(
            h1.get_text(
                " ",
                strip=True
            )
        )

    elif soup.title:

        title = clean_text(
            soup.title.get_text(
                " ",
                strip=True
            )
        )

    else:

        title = page_text[:120]


    outcomes, probabilities = (
        extract_probabilities(
            soup
        )
    )


    # 2択以上だけ採用

    if len(outcomes) < 2:

        return None


    total = sum(
        probabilities
    )


    if (
        98 <= total <= 102
    ):

        probabilities = (
            normalize_probabilities(
                probabilities
            )
        )


    return Market(

        url=url,

        title=title,

        category=classify_category(
            title,
            page_text
        ),

        market_type=classify_market_type(
            title,
            page_text,
            outcomes
        ),

        outcomes=outcomes,

        probabilities=probabilities,

        participants=extract_participants(
            page_text
        ),

        status=detect_status(
            page_text
        ),

    )


# ============================================================
# FETCH SINGLE MARKET
# ============================================================

def fetch_market(
    client: MiraimaClient,
    url: str
):

    try:

        response = client.get(
            url
        )

        response.raise_for_status()


        market = parse_market(
            url,
            response.text
        )


        return (
            market,
            None
        )


    except Exception as e:

        return (
            None,
            str(e)
        )


# ============================================================
# BASELINE PROBABILITY MODEL
# ============================================================

def baseline_model(
    market: Market
) -> list[float]:

    """
    安全なベースラインモデル。

    MIRAIMAの表示確率をそのまま「独立予測」と
    偽装しないため、軽い確率平滑化を行う。

    本格的な競技別MLモデルを追加する場合は
    この関数を置換する。
    """

    probabilities = (
        np.asarray(
            market.probabilities,
            dtype=float
        ) / 100
    )


    uniform = (
        np.ones(
            len(probabilities)
        )
        /
        len(probabilities)
    )


    result = (
        probabilities * 0.92
        +
        uniform * 0.08
    )


    result /= result.sum()


    return (
        result * 100
    ).tolist()


# ============================================================
# MARKET VALUE EVALUATION
# ============================================================

def evaluate_market(
    market: Market
) -> list[dict]:

    model_probabilities = (
        baseline_model(
            market
        )
    )


    rows = []


    for (
        outcome,
        miraima_probability,
        model_probability
    ) in zip(
        market.outcomes,
        market.probabilities,
        model_probabilities
    ):


        # MIRAIMA表示確率との差

        edge = (
            model_probability
            -
            miraima_probability
        )


        market_probability = (
            miraima_probability
            /
            100
        )


        independent_probability = (
            model_probability
            /
            100
        )


        # 真の払戻倍率が取得できないため
        # 相対的なEV Proxyとして扱う

        ev_proxy = (
            independent_probability
            /
            max(
                market_probability,
                1e-9
            )
            - 1
        )


        rows.append({

            "url":
                market.url,

            "title":
                market.title,

            "category":
                market.category,

            "market_type":
                market.market_type,

            "outcome":
                outcome,

            "miraima_probability":
                round(
                    miraima_probability,
                    3
                ),

            "model_probability":
                round(
                    model_probability,
                    3
                ),

            "edge_pt":
                round(
                    edge,
                    3
                ),

            "ev_proxy":
                round(
                    ev_proxy,
                    5
                ),

            "participants":
                market.participants,

            "status":
                market.status,

        })


    return rows
    # ============================================================
# MAIN
# ============================================================

def main():

    start_time = time.monotonic()

    # 25分でPython処理を終了
    deadline = (
        start_time
        +
        MAX_RUNTIME_MINUTES * 60
    )


    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    print("=" * 70)
    print("MIRAIMA ALL-MARKET ENGINE")
    print("=" * 70)

    print(
        f"Runtime limit: "
        f"{MAX_RUNTIME_MINUTES} minutes"
    )

    print(
        f"Workers: {WORKERS}"
    )


    client = MiraimaClient()


    # ========================================================
    # 1. DISCOVERY
    # ========================================================

    print(
        "\n[1/4] Discovering markets..."
    )


    urls = discover_market_urls(
        client
    )


    print(
        f"Discovered: {len(urls)} URLs"
    )


    if not urls:

        print(
            "No markets found."
        )

        return 1


    # ========================================================
    # 2. FETCH
    # ========================================================

    print(
        "\n[2/4] Fetching markets..."
    )


    markets = []


    with concurrent.futures.ThreadPoolExecutor(
        max_workers=WORKERS
    ) as executor:


        future_map = {

            executor.submit(
                fetch_market,
                client,
                url
            ):
                url

            for url in urls

        }


        for future in concurrent.futures.as_completed(
            future_map
        ):


            # HARD RUNTIME STOP

            if time.monotonic() >= deadline:

                print(
                    "\nRuntime limit reached."
                )

                for f in future_map:

                    f.cancel()

                break


            market, error = future.result()


            if market is not None:

                markets.append(
                    market
                )


    print(
        f"Usable markets: "
        f"{len(markets)}"
    )


    # ========================================================
    # 3. INVENTORY
    # ========================================================

    print(
        "\n[3/4] Building market inventory..."
    )


    inventory_rows = []


    for market in markets:

        inventory_rows.append({

            "url":
                market.url,

            "title":
                market.title,

            "category":
                market.category,

            "market_type":
                market.market_type,

            "outcomes":
                " | ".join(
                    market.outcomes
                ),

            "probabilities":
                " | ".join(
                    f"{x:.3f}"
                    for x
                    in market.probabilities
                ),

            "participants":
                market.participants,

            "status":
                market.status,

        })


    inventory = pd.DataFrame(
        inventory_rows
    )


    inventory.to_csv(
        OUTPUT_DIR /
        "miraima_markets.csv",

        index=False,

        encoding="utf-8-sig"
    )


    # ========================================================
    # 4. EVALUATION
    # ========================================================

    print(
        "\n[4/4] Evaluating markets..."
    )


    value_rows = []


    for market in markets:

        if time.monotonic() >= deadline:

            print(
                "Runtime limit reached "
                "during evaluation."
            )

            break


        value_rows.extend(
            evaluate_market(
                market
            )
        )


    values = pd.DataFrame(
        value_rows
    )


    if not values.empty:

        values = values.sort_values(
            [
                "edge_pt",
                "ev_proxy"
            ],
            ascending=False
        )


    values.to_csv(
        OUTPUT_DIR /
        "miraima_market_values.csv",

        index=False,

        encoding="utf-8-sig"
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    elapsed = (
        time.monotonic()
        -
        start_time
    )


    summary = {

        "markets_discovered":
            len(urls),

        "markets_parsed":
            len(markets),

        "outcome_rows":
            len(values),

        "elapsed_seconds":
            round(
                elapsed,
                2
            ),

        "runtime_limit_minutes":
            MAX_RUNTIME_MINUTES,

        "finished_within_limit":
            elapsed
            <
            MAX_RUNTIME_MINUTES * 60,

    }


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


    print(
        "\n" + "=" * 70
    )

    print(
        "FINISHED"
    )

    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2
        )
    )

    print(
        "\nOutput:"
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


    return 0


if __name__ == "__main__":

    sys.exit(
        main()
    )