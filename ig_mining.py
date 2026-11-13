#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IG Mining(login + 分批輪流 + 放慢)— 本地執行版

用法:
    pip install -r requirements.txt
    playwright install chromium            # Linux 如缺系統庫再跑: playwright install-deps
    # 設定 key(三揀一):
    #   1) 同一個資料夾放 Key.env(或 .env),內容見下面
    #   2) 設定環境變數 IG_SESSIONID / DEEPSEEK_API_KEY
    #   3) 直接跑,程式會提示你輸入(唔會顯示喺畫面)
    python ig_mining.py

Key.env 內容例子(唔好上傳去 GitHub):
    IG_SESSIONID=你嘅sessionid
    DEEPSEEK_API_KEY=sk-xxxx
    IG_COOKIES={"csrftoken": "...", "ds_user_id": "..."}   # 可選,一行 JSON

停止:Ctrl+C。重開會由 ig_state.json 記住嘅位置接住。
所有輸出檔(ig_posts_db.json / final_file.json / ig_state.json / covers/)都放喺本檔案同一個資料夾。
"""
import asyncio
import json
import os
import random
import re
import sys
import time
import requests
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from getpass import getpass
from urllib.parse import quote_plus

import pandas as pd
from openai import OpenAI
from playwright.async_api import async_playwright
from selectolax.lexbor import LexborHTMLParser

# 一律用本檔案所在資料夾做工作目錄,唔會因為喺邊度開 terminal 而生出第二份資料庫
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# ==============================================================================
# 0. Key 讀取(.env → 環境變數 → 手動輸入)
# ==============================================================================
def load_dotenv_simple(paths=("Key.env", ".env")):
    """簡單讀 Key.env / .env(KEY=VALUE,# 開頭當註解)。已存在嘅環境變數唔會被蓋"""
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8-sig") as f:      # utf-8-sig:Windows 記事本存檔有時會加 BOM,唔處理會令第一個變數名讀錯
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                v = v.strip()
                if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'" and not v.startswith('{'):
                    v = v[1:-1]                                  # 去掉包住成個值嘅引號(JSON 唔郁)
                os.environ.setdefault(k.strip(), v)


def _need_secret(env_name: str, prompt: str) -> str:
    val = os.environ.get(env_name, "").strip()
    if not val:
        if not sys.stdin or not sys.stdin.isatty():
            raise RuntimeError(f"冇 {env_name}:背景執行時請用 .env 或環境變數提供")
        val = getpass(prompt).strip()
    if not val:
        raise RuntimeError(f"冇 {env_name}")
    os.environ[env_name] = val                         # 記住,唔會重複問
    return val


def get_sessionid() -> str:
    """IG sessionid(建議用後備帳號)"""
    return _need_secret("IG_SESSIONID", "IG sessionid:")


def get_deepseek_key() -> str:
    return _need_secret("DEEPSEEK_API_KEY", "DeepSeek API key:")

def build_cookies(sessionid: str) -> list:
    """sessionid + 可選嘅其他 cookie(Key.env 入面 IG_COOKIES= 一行 JSON)"""
    try:
        extra = json.loads(os.environ.get("IG_COOKIES", "") or "{}")
    except Exception:
        print("⚠️ IG_COOKIES 唔係有效 JSON,忽略")
        extra = {}
    d = {**extra, "sessionid": sessionid}
    return [{"name": k, "value": str(v).strip(), "domain": ".instagram.com", "path": "/",
             "secure": True, "httpOnly": k == "sessionid"} for k, v in d.items() if v]

# ==============================================================================
# 1. 設定 + 工具函式
# ==============================================================================

DB_FILE = "ig_posts_db.json"             # 唯一資料庫:{ url: 貼文資料(包埋 history、llm 分析)}
# ---- 分批輪流 + 放慢 設定 ----
LOOP_MINUTES = 20                        # 每輪最少隔幾耐(AI 分析喺呢段等待時間內跑)
REFRESH_AFTER_MINUTES = 360              # 已入庫貼文隔幾耐先再更新數字(0 = 每輪都更新)
MAX_HISTORY = 168                        # 每篇貼文最多保留幾多個歷史快照
MAX_CONCURRENT = 2                       # 同時開幾多個貼文分頁(login 帳號建議 1,最多 2)
MAX_NEW_PER_TAG = 3                      # 每個 hashtag 每輪:最多 3 篇新貼文
MAX_OLD_PER_TAG = 0                      # 每個 hashtag 每輪:最多幾篇舊貼文(更新數字)。0 = 唔更新,最慳請求(但 growth_per_hour 會一直係空)
TAGS_PER_ROUND = 5                       # 每輪只做幾多個 hashtag,其餘留俾下一輪(輪流循環)
TAG_DELAY = (10, 30)                     # 兩個 hashtag 之間隨機停頓(秒)
PAGE_DELAY =  (6, 15)                    # 兩篇貼文之間隨機停頓(秒)
BACKOFF_MINUTES = 45                     # 被擋(429 / 跳去 login / checkpoint)後停幾耐;連續被擋會倍增
MAX_CONSECUTIVE_BLOCKS = 1               # 連續被擋幾次就停低,唔再撞
EMPTY_TAG_LIMIT = 3                      # 連續幾個 hashtag「頁面連一條貼文連結都冇」,當作被限流(唔係睇有冇新貼文)
STATE_FILE = "ig_state.json"             # 記住輪到邊個 hashtag(斷線重跑都會接住)
COVER_DIR = "covers"                     # 封面圖儲存資料夾
os.makedirs(COVER_DIR, exist_ok=True)

QUIET_HOURS = (1, 8)                     # 夜間暫停:凌晨 1 點到 8 點唔爬(用部機本地時間);設成 None 就 24 小時跑
INCLUDE_ART_PLACES = False               # True = 連 art_place(藝術場地 + Bloger)都爬;False = 只爬下面 HASHTAGS 嘅 15 個

# ---- 經緯度:由 AI 喺分析時一併填(lat / lng),唔另外查地圖服務 ----
HK_BOUNDS = (22.10, 22.62, 113.80, 114.45)           # (lat 最細, lat 最大, lng 最細, lng 最大):AI 填嘅座標跌出香港範圍就當錯,轉成 null

# 只做 show / event 兩類,每個貼文會自動帶上 category
HASHTAGS = {
    "show": ["香港演唱會", "香港展覽", "香港藝術", "香港音樂會", "香港話劇", "hkconcert", "hkexhibition", "hkart"],
    "event": ["香港活動", "香港市集", "香港節慶", "香港好去處", "假日好去處", "hkevent", "hkmarket"]
}

art_place = {"art_venues":[
    # 商場 / 活化文創地標
    "AIRSIDE", "AIRSIDEHK", "K11MUSEA", "K11MUSEAHK", "TheMills", "南豐紗廠",
    "大館", "TaiKwun", "PMQ", "PMQhk", "JCCAC", "JCCACHK",
    # 博物館 & 文化區
    "mplus", "mplusmuseum", "香港故宮文化博物館", "HKPM", "香港藝術館", "HKMoA", "香港藝術中心", "HongKongArtsCentre", "展藝館",
    # 畫廊大廈 & 拍賣行 "no 畢打行"
    "HQueens", "HQueensHK", "HKADC", "SHOWCASE", "蘇富比旗艦藝廊", "蘇富比", "Christies", "佳士得全新亞太區總部",
    # 大型展覽場地 & 文化區
    "會展", "HKCEC", "香港會議展覽中心", "中環海濱活動空間", "centralharbourfronteventspace", "WestK", "西九文化區"],
    "Bloger":["美紙", "artandpiecehk"]}

if INCLUDE_ART_PLACES:
    HASHTAGS.update(art_place)

# ---- 瀏覽器環境(盡量同你日常登入 IG 嘅瀏覽器一致;數值請用你 Console 嘅結果)----
BROWSER_PROFILE = dict(
    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/155.0.0.0 Safari/537.36 Edg/155.0.0.0",
    viewport={"width": 716, "height": 698},            # innerWidth × innerHeight(最大化、100% 縮放時)
    screen={"width": 1528, "height": 698},             # screen.width × screen.height
    device_scale_factor=1.25,                              # devicePixelRatio(Mac Retina 通常係 2)
    locale="zh-HK",                                     # navigator.language
    timezone_id="Asia/Hong_Kong",                       # Intl...timeZone
)

def load_db() -> dict:
    """讀取資料庫;檔案唔存在就由空白開始。順便一次性併入舊版分析結果,避免重複付費分析"""
    db = {}
    if os.path.exists(DB_FILE):
        try:
            with open(DB_FILE, encoding="utf-8") as f:
                db = json.load(f)
        except Exception as e:
            print(f"⚠️ 讀取 {DB_FILE} 失敗: {e}")
    return db


def save_db(db: dict):
    """先寫暫存檔再替換,避免中途停止令檔案壞咗"""
    tmp = DB_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)
    os.replace(tmp, DB_FILE)


def parse_stat_number(val: str) -> int:
    """K / M / 萬 / 逗號 字串 → 整數"""
    if not val:
        return 0
    s = str(val).replace(",", "").strip().upper()
    try:
        for unit, mult in (("K", 1_000), ("M", 1_000_000), ("萬", 10_000)):
            if unit in s:
                return int(float(s.replace(unit, "")) * mult)
        m = re.search(r"\d+(\.\d+)?", s)
        return int(float(m.group(0))) if m else 0
    except Exception:
        return 0


async def block_resources(route):
    """唔載入圖片/影片/字型/CSS,加快速度"""
    if route.request.resource_type in ("image", "media", "font", "stylesheet"):
        await route.abort()
    else:
        await route.continue_()


def snapshot(post: dict) -> dict:
    """某一刻嘅互動數字(歷史紀錄用)"""
    return {"t": post.get("scraped_at"), "likes": post.get("likes", 0), "comments": post.get("comments", 0)}


def in_final(post: dict) -> bool:
    """呢篇有冇出現喺 final_file.json(AI 分析成功、有地點、係景點貼文)"""
    llm = post.get("llm") or {}
    return bool(llm.get("place_name")) and str(llm.get("is_hit_point")).lower() != "false"


def is_due(post) -> bool:
    """新貼文 → 要抓;已入庫但唔喺 final → 唔再更新;喺 final 而且超過 REFRESH_AFTER_MINUTES → 更新"""
    if not post:
        return True
    if not in_final(post):
        return False
    try:
        last = datetime.fromisoformat(post["scraped_at"])
        return datetime.now(timezone.utc) - last >= timedelta(minutes=REFRESH_AFTER_MINUTES)
    except Exception:
        return True


def upsert(db: dict, r: dict) -> str:
    """新貼文 → 入庫;已入庫 → 只更新 likes/comments/scraped_at 並加一個歷史快照。回傳 'new' / 'updated'"""
    old = db.get(r["url"])
    if old is None:
        r["likes"], r["comments"] = r["likes"] or 0, r["comments"] or 0
        r["history"] = [snapshot(r)]
        db[r["url"]] = r
        return "new"

    old.setdefault("history", [snapshot(old)])      # 舊版資料冇 history,用舊數字補返第一個快照
    for k in ("likes", "comments"):
        if r[k] is not None:                        # 今次抓唔到數字(例如 IG 隱藏讚數)就保留舊值
            old[k] = r[k]
    for k in ("cover_url", "cover_path"):           # 之前封面失敗嘅,今次補返
        if not old.get(k) and r.get(k):
            old[k] = r[k]
    old["scraped_at"] = r["scraped_at"]
    old["history"].append(snapshot(old))
    del old["history"][:-MAX_HISTORY]
    return "updated"


# ==============================================================================
# Login(sessionid)+ 被擋偵測 + 輪流批次
# ==============================================================================
CONSENT_BUTTONS = ["Allow all cookies", "Allow essential and optional cookies", "Accept all", "Allow all",
                   "允許所有", "接受所有", "全部接受", "同意"]


class BlockedError(Exception):
    """IG 限流 / 要求重新登入 / checkpoint:要停一陣"""


async def accept_consent(page) -> bool:
    """遇到 cookie 同意頁就嘗試撳「允許」"""
    for label in CONSENT_BUTTONS:
        btn = page.get_by_role("button", name=re.compile(label, re.I))
        if await btn.count():
            await btn.first.click()
            await page.wait_for_timeout(3000)
            return True
    return False


def check_blocked(page, resp=None):
    """被導去 login / challenge / 429 → 拋 BlockedError"""
    if resp is not None and resp.status == 429:
        raise BlockedError("HTTP 429 限流")
    u = page.url
    if "/accounts/login" in u:
        raise BlockedError("被導去 login(sessionid 失效或被限流)")
    if "/challenge" in u or "/suspended" in u:
        raise BlockedError(f"需要驗證/帳號受限: {u}")


async def goto_checked(page, url: str, timeout: int = 15000):
    """goto + 自動處理 cookie 同意頁 + 偵測被擋"""
    resp = await page.goto(url, wait_until="domcontentloaded", timeout=timeout)
    await page.wait_for_timeout(random.randint(2000, 3500))
    if "/consent" in page.url:
        await accept_consent(page)
        resp = await page.goto(url, wait_until="domcontentloaded", timeout=timeout)
        await page.wait_for_timeout(random.randint(2000, 3500))
    check_blocked(page, resp)
    return resp


# ---- 輪流批次:由 STATE_FILE 記住 cursor ----
def all_jobs() -> list:
    """固定次序嘅 (category, tag) 清單(唔 shuffle,先可以公平輪流)"""
    return [(cat, tag) for cat, tags in HASHTAGS.items() for tag in tags]


def load_cursor() -> int:
    try:
        with open(STATE_FILE, encoding="utf-8") as f:
            return int(json.load(f).get("cursor", 0))
    except Exception:
        return 0


def save_cursor(cursor: int):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump({"cursor": cursor, "updated": datetime.now().isoformat()}, f)


def pick_batch(cursor: int):
    """由 cursor 起攞 TAGS_PER_ROUND 個 tag(到尾就由頭接上),回傳 (batch, 新 cursor)"""
    jobs = all_jobs()
    n = min(TAGS_PER_ROUND, len(jobs))
    batch = [jobs[(cursor + i) % len(jobs)] for i in range(n)]
    return batch, (cursor + n) % len(jobs)


# ==============================================================================
# 2. 解析函式(全部由同一份 HTML 解析,唔使逐個 selector 等 30 秒)
# ==============================================================================

def get_meta(tree, prop: str) -> str:
    node = tree.css_first(f'meta[property="{prop}"]') or tree.css_first(f'meta[name="{prop}"]')
    return node.attributes.get("content", "") if node else ""


def extract_date(tree, html: str):
    """發布日期:ld+json → 內嵌 timestamp → <time> 標籤"""
    for node in tree.css('script[type="application/ld+json"]'):
        try:
            data = json.loads(node.text())
        except Exception:
            continue
        for d in (data if isinstance(data, list) else [data]):
            if isinstance(d, dict):
                date = d.get("uploadDate") or d.get("datePublished")
                if date:
                    return date

    m = re.search(r'"taken_at(?:_timestamp)?":\s*(\d{10})', html)
    if m:
        return datetime.fromtimestamp(int(m.group(1)), tz=timezone.utc).isoformat()

    t = tree.css_first("time[datetime]")
    return t.attributes.get("datetime") if t else None


def extract_caption(tree, meta_desc: str) -> str:
    """內文:og:description → og:title → h1(全部都係穩定標籤,唔用 IG 自動生成嘅 class)"""
    # og:description 格式:「123 likes, 4 comments - user on Date: "內文"」
    caption = re.sub(r"^.*?comments?\s*-\s*.*?:\s*", "", meta_desc, count=1, flags=re.I | re.S)
    caption = caption.strip(' "“”')

    if len(caption) < 5:
        title = get_meta(tree, "og:title")  # 格式:「user on Instagram: "內文"」
        caption = title.split(": ", 1)[1].strip(' "“”') if ": " in title else ""

    if len(caption) < 5:
        h1 = tree.css_first("h1")
        caption = h1.text(strip=True) if h1 else ""

    return caption or "（無內文或無法提取）"


async def download_cover(context, tree, post_url: str):
    """下載封面(Post = 第一張圖,Reel = 影片封面)。回傳 (cover_url, 本地路徑)
    IG 圖片連結幾日後會過期,所以要真係落載存檔,淨係存 URL 之後會裂圖"""
    cover_url = get_meta(tree, "og:image")
    if not cover_url:
        return None, None

    shortcode = post_url.rstrip("/").split("/")[-1]
    path = os.path.join(COVER_DIR, f"{shortcode}.jpg")
    if os.path.exists(path):
        return cover_url, path
    try:
        resp = await context.request.get(cover_url, timeout=10000)
        if resp.ok:
            with open(path, "wb") as f:
                f.write(await resp.body())
            return cover_url, path
    except Exception as e:
        print(f"  ⚠️ 封面下載失敗 {shortcode}: {e}")
    return cover_url, None  # 封面失敗唔影響整篇貼文


# ==============================================================================
# 3. 爬蟲:Part 1(搜 hashtag)+ Part 2(抓貼文詳情)
# ==============================================================================

async def search_hashtag(page, tag: str, category: str, claimed: set, db: dict) -> tuple:
    """Part 1:喺同一個分頁搜一個 hashtag,回傳 (要抓嘅貼文清單, 頁面上嘅貼文連結總數)
    連結總數只用嚟判斷有冇被限流(頁面完全冇連結才可疑),唔受「全部都已入庫」影響
    - 新貼文(未入庫)最多 MAX_NEW_PER_TAG 篇;舊貼文(只限已喺 final_file.json 嘅,用嚟更新數字)最多 MAX_OLD_PER_TAG 篇
    - claimed = 本輪其他 hashtag 已搵到嘅連結,避免重複"""
    print(f"🔍 搜尋 #{tag} ({category})")
    found, count = [], {True: 0, False: 0}   # count[是否新貼文]
    n_links = 0
    try:
        await goto_checked(page, f"https://www.instagram.com/explore/search/keyword/?q={tag}")
        for _ in range(3):
            await page.evaluate("window.scrollBy(0, 1000)")
            await page.wait_for_timeout(1000)

        hrefs = await page.eval_on_selector_all(
            'a[href*="/p/"], a[href*="/reel/"]', "els => els.map(e => e.getAttribute('href'))"
        )
        n_links = len(hrefs)
        for href in hrefs:
            if not href:
                continue
            post_type = "Reel" if "/reel/" in href else "Post"
            url = "https://www.instagram.com" + href.split("?")[0].replace("/reel/", "/p/")
            if url in claimed:
                continue
            is_new = url not in db
            if not is_new and not in_final(db[url]):
                continue                          # 已入庫但唔喺 final → 唔使更新,亦唔佔名額
            if count[is_new] >= (MAX_NEW_PER_TAG if is_new else MAX_OLD_PER_TAG):
                continue
            count[is_new] += 1
            claimed.add(url)
            found.append({"hashtag": tag, "category": category, "url": url, "type": post_type})
    except BlockedError:
        raise                                    # 交俾 run_pipeline 處理(停低 + backoff)
    except Exception as e:
        print(f"❌ 搜尋 #{tag} 失敗: {e}")
    return found, n_links


async def fetch_details(context, post: dict, sem: asyncio.Semaphore, state: dict):
    """Part 2:打開單一貼文攞 likes / comments / 內文 / 發布日期。失敗回傳 None
    likes / comments 抓唔到會係 None(更新舊貼文時就保留舊數字)"""
    async with sem:
        if state["blocked"]:                      # 已經被擋,後面嘅唔好再撞
            return None
        page = await context.new_page()
        await page.route("**/*", block_resources)
        try:
            print(f"  [抓取] {post['url']}")
            resp = await page.goto(post["url"], wait_until="domcontentloaded", timeout=15000)
            await page.wait_for_timeout(random.randint(1200, 1800))
            check_blocked(page, resp)

            html = await page.content()  # 只攞一次 HTML,之後全部喺本地解析
            tree = LexborHTMLParser(html)

            meta_desc = get_meta(tree, "og:description") or get_meta(tree, "description")
            if not meta_desc:
                raise ValueError("搵唔到 meta 資料(可能被 login wall 擋住)")

            likes = re.search(r"([\d,KkMm\.]+)\s+(likes|讚|個讚)", meta_desc, re.I)
            comments = re.search(r"([\d,KkMm\.]+)\s+(comments|則留言|留言)", meta_desc, re.I)

            cover_url, cover_path = await download_cover(context, tree, post["url"])

            return {
                **post,
                "cover_url": cover_url,
                "cover_path": cover_path,
                "caption": extract_caption(tree, meta_desc),
                "likes": parse_stat_number(likes.group(1)) if likes else None,
                "comments": parse_stat_number(comments.group(1)) if comments else None,
                "posted_at": extract_date(tree, html),
                "scraped_at": datetime.now(timezone.utc).isoformat(),
            }
        except BlockedError as e:
            state["blocked"] = True
            print(f"  🚫 被擋: {e}")
            return None
        except Exception as e:
            print(f"  ❌ 讀取失敗 {post['url']}: {e}")
            return None
        finally:
            await page.close()
            await asyncio.sleep(random.uniform(*PAGE_DELAY))  # 每篇之間隨機停頓(仍然佔住 semaphore)


async def run_pipeline(db: dict, batch: list, sessionid: str) -> str:
    """一輪爬蟲(只做 batch 入面嗰幾個 hashtag):搜 hashtag → 抓詳情 → 存檔。回傳 'ok' 或 'blocked'"""
    state = {"blocked": False}
    results, targets = [], []
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-setuid-sandbox"])
        context = await browser.new_context(**BROWSER_PROFILE)       # 環境固定,唔好每輪轉
        await context.add_cookies(build_cookies(sessionid))

        try:
            # ---- Part 1:搜今輪嗰批 hashtag(批內次序打亂)----
            jobs = batch[:]
            random.shuffle(jobs)
            print(f"🎯 今輪 {len(jobs)} 個 tag: " + ", ".join(f"#{t}" for _, t in jobs))

            search_page = await context.new_page()
            found_all, claimed, empty_streak = [], set(), 0
            try:
                for cat, tag in jobs:
                    found, n_links = await search_hashtag(search_page, tag, cat, claimed, db)
                    found_all.extend(found)
                    print(f"   頁面貼文連結 {n_links} 條 | 可抓 {len(found)} 篇")
                    empty_streak = 0 if n_links else empty_streak + 1
                    if empty_streak >= EMPTY_TAG_LIMIT:
                        raise BlockedError(f"連續 {EMPTY_TAG_LIMIT} 個 hashtag 頁面完全冇貼文連結,疑似被限流")
                    await asyncio.sleep(random.uniform(*TAG_DELAY))
            except BlockedError as e:
                print(f"🚫 Part 1 被擋: {e}")
                return "blocked"
            finally:
                await search_page.close()

            targets = [x for x in found_all if is_due(db.get(x["url"]))]
            n_new = sum(x["url"] not in db for x in targets)
            print(f"\n📋 搵到 {len(found_all)} 篇 | 新貼文 {n_new} | 到期更新 {len(targets) - n_new} | "
                  f"未到期跳過 {len(found_all) - len(targets)}")
            if not targets:
                return "ok"

            # ---- Part 2:抓詳情(MAX_CONCURRENT 個分頁)----
            sem = asyncio.Semaphore(MAX_CONCURRENT)
            results = await asyncio.gather(*(fetch_details(context, post, sem, state) for post in targets))
        finally:
            await browser.close()

    # 就算中途被擋,已抓到嘅都照入庫
    stats = Counter(upsert(db, r) for r in results if r)
    save_db(db)
    print(f"✅ 本輪新增 {stats['new']} | 更新 {stats['updated']} | 失敗/跳過 {len(targets) - sum(stats.values())} "
          f"| 資料庫共 {len(db)} 筆")
    return "blocked" if state["blocked"] else "ok"


# ==============================================================================
# 4. AI 分析 + 輸出 final_file.json(每輪喺等待時間內自動跑)
# ==============================================================================
ANALYZE_SINCE_DAYS = 45        # 只分析最近幾日發布嘅貼文(取代固定日期)
MIN_CAPTION_LEN = 30           # 剔走 hashtag / @ / 連結後,內文少過幾多字就唔分析
FINAL_FILE = "final_file.json"

GEOCODE_CACHE_FILE = "geocode_cache.json"

def load_geocode_cache() -> dict:
    if os.path.exists(GEOCODE_CACHE_FILE):
        try:
            with open(GEOCODE_CACHE_FILE, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_geocode_cache(cache: dict):
    with open(GEOCODE_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

def geocode_with_google(place_name: str, district: str = "") -> dict | None:
    api_key = os.environ.get("GOOGLE_MAPS_API_KEY", "").strip()
    if not api_key:
        return None
    if "hong kong" in place_name.lower():
        query = f"{place_name} {district}".strip()
    else:
        query = f"{place_name} {district} Hong Kong".strip()
    try:
        resp = requests.get(
            "https://maps.googleapis.com/maps/api/geocode/json",
            params={"address": query, "key": api_key, "components": "country:HK"},
            timeout=10,
        )
        data = resp.json()
        if data.get("status") == "OK" and data.get("results"):
            loc = data["results"][0]["geometry"]["location"]
            return {"lat": loc["lat"], "lng": loc["lng"]}
        print(f"  ❌ Google geocode {place_name}: {data.get('status')}")
    except Exception as e:
        print(f"  ❌ Google geocode {place_name} 失敗: {e}")
    return None

def fill_all_coords(records: list):
    """對所有有 place_name 嘅記錄用 Google 查座標；AI 唔再負責估座標"""
    cache = load_geocode_cache()
    updated, failed = 0, 0
    for rec in records:
        place = rec.get("llm.place_name")
        if not place:
            rec["lat"], rec["lng"] = None, None
            continue

        # 有 cache 直接用（同一地點只查一次）
        if place in cache:
            rec["lat"], rec["lng"] = cache[place]["lat"], cache[place]["lng"]
            updated += 1
            continue

        coords = geocode_with_google(place, rec.get("llm.district") or "")
        if coords:
            # 用 clean_coord 驗證係咪喺香港範圍
            lat, lng = clean_coord(coords["lat"], coords["lng"])
            if lat is not None:
                cache[place] = {"lat": lat, "lng": lng}
                rec["lat"], rec["lng"] = lat, lng
                updated += 1
            else:
                rec["lat"], rec["lng"] = None, None
                failed += 1
                print(f"  ⚠️ Google 回傳座標跌出香港範圍：{place} → {coords}")
        else:
            rec["lat"], rec["lng"] = None, None
            failed += 1

        time.sleep(0.1)

    save_geocode_cache(cache)
    print(f"🗺️ Google 座標：成功 {updated} | 失敗 {failed} | cache {len(cache)} 筆")
    return updated

client = None        # 喺 main() 入面用 DEEPSEEK_API_KEY 建立

PROMPT = """分析香港 IG 貼文內文,只回傳 json:
{"is_hit_point": "true 如果貼文係介紹一個可以去嘅地點或活動;自拍、日常、純廣告抽獎就 false",
 "place_name": "最主要嘅一個地點或店名(貼文提到多個地點,只填最主要嗰個,唔好用頓號列晒),或null",
 "district": "所在地區(例如旺角、中環)或null",
 "category": "show/event/other",
 "sentiment": "positive/neutral/negative",
 "summary": "20字內廣東話總結",
 "event_name": "活動或展覽名稱或null",
 "start_date": "活動開始日期 YYYY-MM-DD 或null",
 "end_date": "活動結束日期 YYYY-MM-DD 或null",
 "price": "票價或消費(例如 $180、免費)或null",
 "opening_hours": "營業或開放時間或null"}
 """

def clean_len(caption: str) -> int:
    """剔走 hashtag、@、連結、空白之後嘅字數"""
    s = re.sub(r"#\S+|@\w+|https?://\S+", "", caption or "")
    return len(re.sub(r"\s+", "", s))

def clean_coord(lat, lng):
    """AI 填嘅座標 → (lat, lng)。唔係數字、或者跌出香港範圍就回傳 (None, None)"""
    try:
        lat, lng = float(lat), float(lng)
    except (TypeError, ValueError):
        return None, None
    a, b, c, d = HK_BOUNDS
    if not (a <= lat <= b and c <= lng <= d):          # NaN 都會喺呢度被剔走
        return None, None
    return round(lat, 6), round(lng, 6)


def analyze(caption: str) -> dict:
    caption = re.sub(r"@\w+", "@user", caption)[:600]
    res = client.chat.completions.create(
        model="deepseek-chat", temperature=0.1, max_tokens=500,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": PROMPT}, {"role": "user", "content": caption}])
    return json.loads(res.choices[0].message.content)


def gmap_url(llm: dict):
    name = llm.get("place_name")
    if not name:
        return None
    return "https://www.google.com/maps/search/?api=1&query=" + quote_plus(f"{name} {llm.get('district') or ''} Hong Kong")


def growth_per_hour(history):
    """用歷史快照計:由第一個快照到最新快照,每粒鐘增加幾多互動(讚 + 3×留言)。少過 2 個快照就係 None"""
    if not isinstance(history, list) or len(history) < 2:
        return None
    try:
        a, b = history[0], history[-1]
        hours = (datetime.fromisoformat(b["t"]) - datetime.fromisoformat(a["t"])).total_seconds() / 3600
        if hours <= 0:
            return None
        gain = (b["likes"] + 3 * b["comments"]) - (a["likes"] + 3 * a["comments"])
        return round(gain / hours, 2)
    except Exception:
        return None


def run_analysis(db: dict):
    """AI 分析(只做未分析過、喺日期範圍內、有內文嘅貼文)→ 存檔 → 輸出 final_file.json"""
    if not db:
        return
    posted = pd.to_datetime(pd.Series({u: p.get("posted_at") for u, p in db.items()}),
                            utc=True, errors="coerce", format="ISO8601")
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=ANALYZE_SINCE_DAYS)
    recent = set(posted[posted >= cutoff].index)

    candidates = [u for u, p in db.items()
                  if u in recent and not p.get("llm") and not p.get("caption", "").startswith("（無內文")]
    todo = [u for u in candidates if clean_len(db[u].get("caption", "")) >= MIN_CAPTION_LEN]
    print(f"🤖 已分析 {sum(1 for p in db.values() if p.get('llm'))} 篇 | 今次要分析 {len(todo)} 篇 ")

    if todo:
        with ThreadPoolExecutor(max_workers=5) as ex:                       # 5 個同時跑
            futures = {ex.submit(analyze, db[u]["caption"]): u for u in todo}
            for i, fut in enumerate(as_completed(futures), 1):
                url = futures[fut]
                try:
                    llm = fut.result()
                    db[url]["llm"], db[url]["gmap_url"] = llm, gmap_url(llm)
                except Exception as e:
                    print("  失敗:", url, e)
                if i % 10 == 0 or i == len(todo):
                    print(f"  進度 {i}/{len(todo)}")
        save_db(db)

    export_final(db)


def export_final(db: dict):
    """Spike Velocity 排行 → final_file.json（座標全部由 Google 填）"""
    result = pd.json_normalize(list(db.values()))
    if "llm.place_name" not in result.columns:
        return
    result = result[result["llm.place_name"].notna()]
    result = result[result["llm.is_hit_point"].astype(str).str.lower() != "false"]
    if result.empty:
        return

    posted  = pd.to_datetime(result["posted_at"],  utc=True, errors="coerce", format="ISO8601")
    scraped = pd.to_datetime(result["scraped_at"], utc=True, errors="coerce", format="ISO8601")
    age_hours = ((scraped - posted).dt.total_seconds() / 3600).clip(lower=1)

    likes    = pd.to_numeric(result["likes"],    errors="coerce").fillna(0)
    comments = pd.to_numeric(result["comments"], errors="coerce").fillna(0)

    result["engagement"]      = likes + 3 * comments
    result["spike_velocity"]  = (result["engagement"] / age_hours).round(2)
    result["spike_score"]     = (result["spike_velocity"].rank(pct=True) * 100).round(1)
    result["growth_per_hour"] = [growth_per_hour(h) for h in result.get("history", [None] * len(result))]

    # 移除 AI 座標相關欄位（如果有）
    result = result.drop(columns=["llm.lat", "llm.lng"], errors="ignore")

    result = result.sort_values("spike_velocity", ascending=False)
    records = json.loads(result.drop(columns=["history"], errors="ignore")
                         .to_json(orient="records", force_ascii=False, date_format="iso"))

    # 全部座標由 Google 查
    fill_all_coords(records)

    with open(FINAL_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"📊 {FINAL_FILE} 已更新:{len(records)} 個地點。Top 3:")
    for r in records[:3]:
        print(f"   {r.get('llm.place_name')} | {r.get('lat')}, {r.get('lng')} | "
              f"likes={r.get('likes')} comments={r.get('comments')} velocity={r.get('spike_velocity')}")

# ==============================================================================
# 5. 自動循環:爬蟲(Part 1+2)→ 等待時間內跑 AI 分析 → 兩邊都完咗先開下一輪
#    要停就喺 terminal 按 Ctrl+C
# ==============================================================================

async def safe_analysis(db: dict):
    """AI 分析搬去背景 thread(唔阻住 event loop);出錯只印訊息,唔會搞死循環"""
    try:
        await asyncio.to_thread(run_analysis, db)
    except Exception as e:
        print(f"❌ AI 分析出錯,下一輪再試: {e}")


def in_quiet_hours(now: datetime) -> bool:
    if not QUIET_HOURS:
        return False
    a, b = QUIET_HOURS
    return a <= now.hour < b if a < b else (now.hour >= a or now.hour < b)    # 支援跨午夜,例如 (23, 7)


def wake_time(now: datetime) -> datetime:
    """下一個 QUIET_HOURS 結束時間,加 0–20 分鐘隨機"""
    wake = now.replace(hour=QUIET_HOURS[1], minute=random.randint(0, 20), second=0, microsecond=0)
    return wake if wake > now else wake + timedelta(days=1)


async def main_loop():
    sessionid = get_sessionid()
    db = load_db()
    cursor = load_cursor()
    n_tags = len(all_jobs())
    print(f"共 {n_tags} 個 hashtag,每輪 {TAGS_PER_ROUND} 個,約 {-(-n_tags // TAGS_PER_ROUND)} 輪行完一圈 "
          f"(每輪最少 {LOOP_MINUTES} 分鐘)")
    if QUIET_HOURS:
        print(f"🌙 夜間暫停時段:{QUIET_HOURS[0]:02d}:00–{QUIET_HOURS[1]:02d}:00")
    round_no, blocks, errors = 0, 0, 0
    while True:
        now = datetime.now()
        if in_quiet_hours(now):
            wake = wake_time(now)
            print(f"\n🌙 夜間暫停,睡到 {wake:%m-%d %H:%M}(期間先補做未分析嘅貼文)")
            await safe_analysis(db)
            await asyncio.sleep(max((wake - datetime.now()).total_seconds(), 60))
            continue

        round_no += 1
        batch, next_cursor = pick_batch(cursor)
        print(f"\n===== 第 {round_no} 輪 | {datetime.now():%H:%M:%S} | cursor {cursor}→{next_cursor} "
              f"| 資料庫現有 {len(db)} 筆 =====")
        try:
            status = await run_pipeline(db, batch, sessionid)
        except Exception as e:
            print(f"❌ 本輪爬蟲出錯: {e}")
            status = "error"

        if status == "blocked":
            blocks += 1
            if blocks >= MAX_CONSECUTIVE_BLOCKS:
                print(f"🛑 連續被擋 {blocks} 次,停低。請檢查帳號(手機/網頁登入睇下有冇驗證),"
                      f"換新 sessionid 或者等幾個鐘再跑。")
                await safe_analysis(db)
                return
            wait_min = BACKOFF_MINUTES * blocks
            print(f"🚫 被擋(第 {blocks} 次),今輪 cursor 唔前進,停 {wait_min} 分鐘後再試同一批")
        elif status == "ok":
            blocks = errors = 0
            cursor = next_cursor                      # 成功先前進
            save_cursor(cursor)
            wait_min = LOOP_MINUTES
        else:                                         # error:重試同一批,連續 3 次先跳過
            errors += 1
            if errors >= 3:
                print("⚠️ 同一批連續出錯 3 次,跳過去下一批")
                cursor, errors = next_cursor, 0
                save_cursor(cursor)
            wait_min = LOOP_MINUTES

        # 等待時間 = 同時跑 AI。AI 快過 wait_min 就補夠時間;慢過就 AI 一完即刻開下一輪
        wait_min += random.uniform(0, 3)              # 加少少隨機,唔好太似機器
        print(f"😴 等 {wait_min:.0f} 分鐘,同時跑 AI 分析...")
        await asyncio.gather(asyncio.sleep(wait_min * 60), safe_analysis(db))


# ==============================================================================
# 入口
# ==============================================================================
def main():
    global client
    load_dotenv_simple()
    get_sessionid()                                   # 先確認有 key,冇就喺度即刻報錯,唔使行到一半先死
    client = OpenAI(api_key=get_deepseek_key(), base_url="https://api.deepseek.com",
                    timeout=30, max_retries=1)        # 卡住 30 秒就放棄,最多重試 1 次
    try:
        asyncio.run(main_loop())
    except KeyboardInterrupt:
        print("\n🛑 已手動停止")


if __name__ == "__main__":
    main()
