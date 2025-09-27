# -*- coding: utf-8 -*-
"""
KBBI ELO RATING Discord Bot (skeleton, spec-aligned)
====================================================
- Runs from current working directory (no absolute paths).
- Reads item banks from CSVs in the current folder:
    * new_difficulty_sheet_sorted.csv (required)
    * C4_KBBI_FREQ_ULTIMATE.csv (required)
    * indexes_kbbi_v2.4_english_nerf_v6.csv (optional, shown in admin)
- Reads Discord token from environment variable DISCORD_TOKEN, or token.txt in current folder.
- Persists user state to user_elo_data.json in current folder.
- Writes admin logs to diff_adjustments.csv, approved_reports.csv, approved_complaints.csv.
- Implements reports/complaints → blanks, global exclusion for reports, rebalance, undo by id, list_approvals, recalcelo, hide toggle, per-question info display.

NOTE: This is a full-featured skeleton with working command flows and a basic selection engine.
      The IRT sampler and SBERT similarity are scaffolded with clean hooks (see TODO sections).
      You can ship and iterate features without touching storage or user-facing behavior.
"""
from textwrap import dedent
import os
import io
import csv
import json
import math
import time
import random
import asyncio
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional, Set, Any

import discord
from discord.ext import commands

from typing import List
from sentence_transformers import SentenceTransformer, util

# ------------------------------
# Constants / Config
# ------------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")

INTENTS = discord.Intents.default()
INTENTS.message_content = True
INTENTS.members = True

COMMAND_PREFIX = "!"
BOT_DESCRIPTION = "KBBI ELO RATING Bot"

USER_DATA_FILE = "user_elo_data.json"
DIFF_FILE = "new_difficulty_sheet_sorted.csv"
FREQ_FILE = "C4_KBBI_FREQ_ULTIMATE.csv"
ENGLISH_NERF_FILE = "indexes_kbbi_v2.4_english_nerf_v6.csv"

DIFF_ADJUSTMENTS_CSV = "diff_adjustments.csv"
APPROVED_REPORTS_CSV = "approved_reports.csv"
APPROVED_COMPLAINTS_CSV = "approved_complaints.csv"

# Admins (hardcoded list of Discord user IDs)
ADMIN_ALLOWLIST = {209145376375046145, 495842018803712001}

# Elo / IRT knobs
SCALE_A = 1.2            # a_scaled = SCALE_A * a (global scale)
C_GUESS = 0.25           # 4-choice
THR1 = 0.6               # distractor sim (main threshold)
THR2 = 0.53              # distractor sim (secondary, pairwise loosening)
LEVEN_MAX = 2            # type2 word levenshtein screen
JACCARD_MAX = 0.65       # def jaccard screen

# Quiz
DEFAULT_QUIZ_LEN = 25
SESSION_IDLE_TIMEOUT_SEC = 20 * 60     # 20 minutes
LAST200_LIMIT = 200
WINDOW12_LIMIT = 12
MINE_MAX_KEYS = 100
MINE_MAX_CHARS = 2000

# Elo visibility
MIN_NONBLANK_FOR_ELO = 100

# Report / complaint approvals (global id counter)
APPROVAL_COUNTER_FILE = "approval_counter.json"  # stores {"next_id": int}

# Selection policy (easy bucket sense-aware threshold)
EASY_BUCKET_B_THRESH = -2.5

FREQ_COUNT_OF_KEY: Dict[str, float] = {}  # key -> raw count from C4_KBBI_FREQ_ULTIMATE.csv

# ------------------------------
# Data Models
# ------------------------------

@dataclass
class HistoryEvent:
    qid: int
    row_index: str
    key: str
    correct: bool
    date: str               # YYYY-MM-DD
    qtype: int              # 1 or 2
    distractors: List[str]  # row_index of distractors

@dataclass
class UserProfile:
    user_id: int
    mode: str = "freedom"  # "freedom" or "anki"
    reviewmode: str = "manual"  # "manual", "all", "mistakes"
    hide_item_info: bool = False
    history_list: List[HistoryEvent] = field(default_factory=list)  # append-only
    blanked_qids: Set[int] = field(default_factory=set)             # per-user blanks
    window_12: List[Dict[str, Any]] = field(default_factory=list)   # deque-of-12 items
    last_200_rowidx: List[str] = field(default_factory=list)        # last 200 row_index (order)
    seen_keys_all_senses: Set[str] = field(default_factory=set)     # for anki mode (keys)
    session_mine_list: List[str] = field(default_factory=list)      # keys for mine print

    # Opt-in flag for public leaderboard
    leaderboard_opt_in: bool = False

    def to_json(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "mode": self.mode,
            "reviewmode": self.reviewmode,
            "hide_item_info": self.hide_item_info,
            "history_list": [asdict(e) for e in self.history_list],
            "blanked_qids": list(self.blanked_qids),
            "window_12": self.window_12,
            "last_200_rowidx": self.last_200_rowidx,
            "seen_keys_all_senses": list(self.seen_keys_all_senses),
            "session_mine_list": self.session_mine_list,

            # persist opt-in flag
            "leaderboard_opt_in": self.leaderboard_opt_in,
        
        }

    @staticmethod
    def from_json(d: Dict[str, Any]) -> "UserProfile":
        up = UserProfile(user_id=d["user_id"])
        up.mode = d.get("mode", "freedom")
        up.reviewmode = d.get("reviewmode", "manual")
        up.hide_item_info = bool(d.get("hide_item_info", False))
        up.history_list = [HistoryEvent(**e) for e in d.get("history_list", [])]
        up.blanked_qids = set(d.get("blanked_qids", []))
        up.window_12 = d.get("window_12", [])
        up.last_200_rowidx = d.get("last_200_rowidx", [])
        up.seen_keys_all_senses = set(d.get("seen_keys_all_senses", []))
        up.session_mine_list = d.get("session_mine_list", [])
        # read opt-in flag safely (defaults to False if missing)
        up.leaderboard_opt_in = bool(d.get(\"leaderboard_opt_in\", False))
        return up

@dataclass
class ItemSense:
    row_index: str
    key: str
    definition: str
    a: float
    b: float

# ------------------------------
# Global State
# ------------------------------

bot = commands.Bot(command_prefix=COMMAND_PREFIX, intents=INTENTS, description=BOT_DESCRIPTION)
bot.help_command = None  # we provide custom help #added by Yuusei

USERS: Dict[int, UserProfile] = {}
ITEMS: Dict[str, ItemSense] = {}                 # row_index -> ItemSense
SENSES_BY_KEY: Dict[str, List[str]] = {}         # key -> list of row_index
FREQ_RANK_OF_KEY: Dict[str, int] = {}            # key -> rank (1-based)
B_OVERRIDES: Dict[str, float] = {}               # row_index -> new b
ENGLISH_NERF: Dict[str, Any] = {}                # optional, key or row_index mapping (display only)
GLOBAL_EXCLUDED_ROWIDX: Set[str] = set()         # globally ineligible as correct (reports)

REPORTED_ROWIDX_PENDING: Set[str] = set()
REPORTED_ROWIDX_APPROVED: Dict[str, int] = {}    # row_index -> refcount (approved)

NEXT_QID = 1
NEXT_APPROVAL_ID = 1
ADMIN_LOCK: Optional[int] = None                 # user_id of admin in admin mode, else None

# For quiz sessions (per-user)
ACTIVE_QUIZ: Dict[int, Dict[str, Any]] = {}      # user_id -> session dict

# ------------------------------
# Utility: Files / Persistence
# ------------------------------

def load_json_file(path: str, default):
    if not os.path.isfile(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except Exception:
            return default

def save_json_file(path: str, data) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def load_user_store():
    global USERS
    data = load_json_file(USER_DATA_FILE, {})
    out = {}
    for k, v in data.items():
        try:
            up = UserProfile.from_json(v)
            out[int(k)] = up
        except Exception:
            pass
    USERS = out

def save_user_store():
    data = {str(uid): up.to_json() for uid, up in USERS.items()}
    save_json_file(USER_DATA_FILE, data)

def ensure_csv_headers(path: str, headers: List[str]):
    exists = os.path.isfile(path)
    if not exists:
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(headers)

def append_csv(path: str, row: List[Any]):
    with open(path, "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(row)

def load_counter():
    global NEXT_QID, NEXT_APPROVAL_ID
    d = load_json_file(APPROVAL_COUNTER_FILE, {"next_qid": 1, "next_approval_id": 1})
    NEXT_QID = int(d.get("next_qid", 1))
    NEXT_APPROVAL_ID = int(d.get("next_approval_id", 1))

def save_counter():
    save_json_file(APPROVAL_COUNTER_FILE, {"next_qid": NEXT_QID, "next_approval_id": NEXT_APPROVAL_ID})

# ------------------------------
# Load Items / Frequencies
# ------------------------------
# ------------------------------
# Key Pools & Distributions (per spec)
# ------------------------------
DIFF_KEYS: List[str] = []            # all keys present in DIFF
DEFS_BY_KEY: Dict[str, List[str]] = {}   # key -> [row_index,...]
B_BY_KEY: Dict[str, float] = {}      # key -> mean b across its senses
FREQ_MAP: Dict[str, float] = {}      # key -> positive count
KEYS_POSFREQ: List[str] = []         # DIFF keys with count>0
KEYS_RANKED: List[str] = []          # KEYS_POSFREQ sorted by count desc
KEY_TO_RANKIDX: Dict[str, int] = {}  # key -> index (0-based) in KEYS_RANKED
CDF_15: List[float] = []
CDF_25: List[float] = []
CDF_40: List[float] = []
KEYS_ALL: List[str] = []             # all DIFF keys (even if count==0)
KEYS_HIGH_B: List[str] = []          # mean b > 1.0
KEYS_A: List[str] = []               # top 1000 by freq
KEYS_B: List[str] = []               # top 10000 by freq AND b < -2.5 and not in A

RANK_CAP = 400  # used in log-normalization window

def build_key_pools():
    # DIFF_KEYS, DEFS_BY_KEY, B_BY_KEY
    DIFF_KEYS.clear(); DEFS_BY_KEY.clear(); B_BY_KEY.clear()
    for key, rids in SENSES_BY_KEY.items():
        DEFS_BY_KEY[key] = list(rids)
        if rids:
            B_BY_KEY[key] = sum(ITEMS[r].b for r in rids) / float(len(rids))
        else:
            B_BY_KEY[key] = 0.0
    DIFF_KEYS.extend(DEFS_BY_KEY.keys())

    # FREQ_MAP from actual counts; keep only positives
    FREQ_MAP.clear()
    for k in DIFF_KEYS:
        c = float(FREQ_COUNT_OF_KEY.get(k, 0.0))
        if c > 0:
            FREQ_MAP[k] = c

    # Ranked keys = intersection DIFF ∩ positive freq
    KEYS_POSFREQ[:] = [k for k in DIFF_KEYS if k in FREQ_MAP]
    KEYS_RANKED[:] = sorted(KEYS_POSFREQ, key=lambda k: (-FREQ_MAP[k], k))
    KEY_TO_RANKIDX.clear()
    for i, k in enumerate(KEYS_RANKED):
        KEY_TO_RANKIDX[k] = i

    # Build capped, log-normalized weights then CDFs for k∈{1.5,2.5,4.0}
    N = len(KEYS_RANKED)
    if N:
        freq = [FREQ_MAP[k] for k in KEYS_RANKED]
        cap_idx = min(RANK_CAP, N) - 1
        cap_val = max(freq[cap_idx], 1e-9)
        freq_capped = [min(x, cap_val) for x in freq]
        log_max = math.log(cap_val)
        x = [max(min((math.log(v)/log_max) if v > 0 else 1e-12, 1.0), 1e-12) for v in freq_capped]

        def make_cdf(kval: float) -> List[float]:
            # smoothstep style: (x^(2k))*(3 - 2*x^k)
            xk = [t**kval for t in x]
            w = [(t**(2*kval)) * (3.0 - 2.0*s) for t, s in zip(x, xk)]
            tot = sum(w) or 1.0
            c, acc = [], 0.0
            for wi in w:
                acc += wi / tot
                c.append(acc)
            c[-1] = 1.0
            return c

        CDF_15[:] = make_cdf(1.5)
        CDF_25[:] = make_cdf(2.5)
        CDF_40[:] = make_cdf(4.0)
    else:
        CDF_15[:] = CDF_25[:] = CDF_40[:] = []

    # All DIFF keys
    KEYS_ALL[:] = list(DIFF_KEYS)

    # High-b (>1)
    KEYS_HIGH_B[:] = [k for k in KEYS_ALL if B_BY_KEY.get(k, 0.0) > 1.0]

    # Easy buckets
    top1000 = min(1000, len(KEYS_RANKED))
    top10000 = min(10000, len(KEYS_RANKED))
    Aset = set(KEYS_RANKED[:top1000])
    KEYS_A[:] = list(KEYS_RANKED[:top1000])
    KEYS_B[:] = [k for k in KEYS_RANKED[:top10000] if k not in Aset and B_BY_KEY.get(k, 0.0) < -2.5]

def route_policy_from_window12(up: UserProfile) -> str:
    # Count last-12 non-blank corrects
    cnt = corr = 0
    for ev in reversed(up.history_list):
        if ev.qid in up.blanked_qids:
            continue
        cnt += 1
        corr += 1 if ev.correct else 0
        if cnt >= 12: break
    if corr >= 12: return "uniform_bgt1"
    if 10 <= corr <= 11: return "uniform_all"
    if corr in (9, 8): return "h15"
    if corr in (7, 6): return "h25"
    if corr in (5, 4): return "h40"
    return "easy"

def draw_key_by_cdf(cdf: List[float]) -> Optional[str]:
    if not cdf or not KEYS_RANKED: return None
    u = random.random()
    lo, hi, ans = 0, len(cdf)-1, len(cdf)-1
    while lo <= hi:
        mid = (lo+hi)//2
        if u <= cdf[mid]:
            ans = mid; hi = mid-1
        else:
            lo = mid+1
    return KEYS_RANKED[ans]

def draw_key_uniform(pool: List[str]) -> Optional[str]:
    return random.choice(pool) if pool else None

def choose_key_and_sense_by_policy(up: UserProfile, mode: str) -> Optional[str]:
    if mode == "uniform_bgt1":
        key = draw_key_uniform(KEYS_HIGH_B)
    elif mode == "uniform_all":
        key = draw_key_uniform(KEYS_ALL)
    elif mode == "h15":
        key = draw_key_by_cdf(CDF_15)
    elif mode == "h25":
        key = draw_key_by_cdf(CDF_25)
    elif mode == "h40":
        key = draw_key_by_cdf(CDF_40)
    else:  # easy
        u = random.random()
        pool = KEYS_A if (u < 5/8 and KEYS_A) or not KEYS_B else (KEYS_B if KEYS_B else KEYS_A)
        key = draw_key_uniform(pool)
    if not key:
        key = draw_key_uniform(KEYS_ALL)
    if not key:
        return None

    senses = list(DEFS_BY_KEY.get(key, []))
    if not senses:
        return None

    # Sense-aware for Easy-B: enforce per-sense b <-2.5 if key is in B
    if mode == "easy" and key in set(KEYS_B):
        senses = [rid for rid in senses if effective_sense(rid).b < -2.5]
        if not senses:
            return None  # signal collision upstream

    random.shuffle(senses)
    return senses[0]

def next_item_via_collision(up: UserProfile, seed_key: str, mode: str) -> Optional[str]:
    def sense_ok(rid: str) -> bool:
        if is_globally_excluded_rowidx(rid): return False
        if up.mode == "freedom":
            if rid in up.last_200_rowidx: return False
        else:  # anki
            if ITEMS[rid].key in up.seen_keys_all_senses: return False
        if mode == "easy" and ITEMS[rid].key in set(KEYS_B):
            if effective_sense(rid).b >= -2.5: return False
        return True

    # try senses of seed key
    for rid in random.sample(DEFS_BY_KEY.get(seed_key, []), k=len(DEFS_BY_KEY.get(seed_key, []))):
        if sense_ok(rid): return rid

    # rank-adjacent wave (no cap)
    N = len(KEYS_RANKED)
    if N == 0: return None
    start = KEY_TO_RANKIDX.get(seed_key, N-1)
    visited = set(); step = 0
    while True:
        idx = start if step == 0 else (start + ((step+1)//2) if step % 2 else start - ((step+1)//2))
        step += 1
        if idx < 0 or idx >= N: 
            if step > 4*N: return None
            continue
        if idx in visited: continue
        visited.add(idx)
        k2 = KEYS_RANKED[idx]
        senses2 = DEFS_BY_KEY.get(k2, [])
        for rid in random.sample(senses2, k=len(senses2)):
            if sense_ok(rid): return rid
        if step > 4*N: return None


def load_items():
    """
    Load new_difficulty_sheet_sorted.csv
    Required columns: key, row_index, a, b, googletranslate/definition
    """
    required_cols = {"key", "row_index", "a", "b"}
    if not os.path.isfile(DIFF_FILE):
        raise RuntimeError(f"Missing {DIFF_FILE} in current folder.")
    with open(DIFF_FILE, "r", encoding="utf-8-sig", newline="") as f:
        rdr = csv.DictReader(f)
        cols = set([c.strip().lower() for c in rdr.fieldnames or []])
        missing = [c for c in required_cols if c not in cols]
        if missing:
            raise RuntimeError(f"{DIFF_FILE} is missing columns: {missing}")
        for r in rdr:
            key = r.get("key","").strip()
            row_index = r.get("row_index","").strip()
            if not key or not row_index:
                continue
            a = float(r.get("a","1") or 1.0)
            b = float(r.get("b","0") or 0.0)
            definition = (r.get("googletranslate") or r.get("definition") or "").strip()
            sense = ItemSense(row_index=row_index, key=key, definition=definition, a=a, b=b)
            ITEMS[row_index] = sense
            SENSES_BY_KEY.setdefault(key, []).append(row_index)

def load_freqs():
    """
    Load C4_KBBI_FREQ_ULTIMATE.csv with columns at least: key, count
    Produce unique ranks (1-based). Assume unique keys in file.
    """
    if not os.path.isfile(FREQ_FILE):
        raise RuntimeError(f"Missing {FREQ_FILE} in current folder.")
    rows = []
    with open(FREQ_FILE, "r", encoding="utf-8-sig", newline="") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            key = r.get("key","").strip()
            if not key:
                continue
            try:
                cnt = float(r.get("count","0") or 0.0)
            except:
                cnt = 0.0
            rows.append((key, cnt))
    # sort by count desc, assign rank 1..N
    rows.sort(key=lambda x: (-x[1], x[0]))

    FREQ_COUNT_OF_KEY.clear()
    for i, (key, cnt) in enumerate(rows, start=1):
        FREQ_RANK_OF_KEY[key] = i
        FREQ_COUNT_OF_KEY[key] = cnt


def load_english_nerf():
    if not os.path.isfile(ENGLISH_NERF_FILE):
        return
    with open(ENGLISH_NERF_FILE, "r", encoding="utf-8-sig", newline="") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            # We just keep the entire row keyed by 'key' or 'row_index' if present
            rid = r.get("row_index") or ""
            ky = r.get("key") or ""
            if rid:
                ENGLISH_NERF[rid] = r
            if ky:
                ENGLISH_NERF[ky] = r

# ------------------------------
# Similarity & token helpers (fast, pure-Python)
# ------------------------------

def norm_tokens(s: str) -> List[str]:
    s = (s or "").lower()
    out = []
    tok = []
    for ch in s:
        if "a" <= ch <= "z":
            tok.append(ch)
        else:
            if tok:
                out.append("".join(tok))
                tok = []
    if tok:
        out.append("".join(tok))
    return out

def cosine_tokens(ta: List[str], tb: List[str]) -> float:
    if not ta or not tb:
        return 0.0

    # join tokens back into a string (since ST expects sentences)
    text_a = " ".join(ta)
    text_b = " ".join(tb)

    emb_a = model.encode(text_a, convert_to_tensor=True, normalize_embeddings=True)
    emb_b = model.encode(text_b, convert_to_tensor=True, normalize_embeddings=True)

    return util.cos_sim(emb_a, emb_b).item()

def jaccard_set(ta: List[str], tb: List[str]) -> float:
    A, B = set(ta), set(tb)
    if not A and not B: return 0.0
    return len(A & B) / float(len(A | B))

def levenshtein(a: str, b: str) -> int:
    if a == b: return 0
    la, lb = len(a), len(b)
    if la == 0: return lb
    if lb == 0: return la
    prev = list(range(lb+1))
    cur = [0]*(lb+1)
    for i in range(1, la+1):
        cur[0]=i
        ca=a[i-1]
        for j in range(1, lb+1):
            cost = 0 if ca == b[j-1] else 1
            cur[j] = min(prev[j]+1, cur[j-1]+1, prev[j-1]+cost)
        prev, cur = cur, prev
    return prev[lb]

# Pre-tokenize all definitions for speed
DEF_TOKENS: Dict[str, List[str]] = {}  # row_index -> tokens

def ensure_def_tokens():
    if DEF_TOKENS:
        return
    for rid, sense in ITEMS.items():
        DEF_TOKENS[rid] = norm_tokens(sense.definition or "")

# ------------------------------
# Elo / θ Estimation (3PL MLE with fallback)
# ------------------------------

def p_correct(theta: float, a: float, b: float, c: float=C_GUESS) -> float:
    # 3PL: P = c + (1-c) * sigmoid(D*a*(theta-b)), D≈1.7 (use 1.7)
    D = 1.7
    z = D * a * (theta - b)
    s = 1.0/(1.0+math.exp(-z))
    return c + (1.0-c) * s

def mle_theta(events: List[Tuple[bool, float, float]]) -> float:
    """
    events: list of (correct, a_scaled, b_eff), blanks already removed.
    Returns theta that maximizes likelihood. Robust fallback to grid if Newton fails.
    """
    if not events:
        return 0.0  # neutral

    # Newton-Raphson with damping, fallback to grid
    theta = 0.0
    D = 1.7
    for _ in range(15):
        # gradient and hessian
        g = 0.0
        h = 0.0
        for y, a, b in events:
            z = D*a*(theta-b)
            s = 1.0/(1.0+math.exp(-z))
            p = C_GUESS + (1.0-C_GUESS)*s
            # derivative wrt theta
            dp = (1.0-C_GUESS) * s * (1.0-s) * D*a
            # log-likelihood derivatives
            # grad += (y - p)/ (p*(1-p)) * dp * ??? -> safer: g += (y - p) * dp / (p*(1-p)) * ???
            # Instead use: d/dθ log L = Σ ( (y - p) * dp / (p*(1-p)) ) * ???
            # Simpler: use direct derivative of Bernoulli log-likelihood:
            g += ( (y - p) / (p * (1.0 - p) + 1e-9) ) * dp
            # Hessian approx: - Σ [ (dp^2) * ( (1-2p) / (p^2*(1-p)^2) ) * ??? ] is messy
            # Use numeric second derivative approx per item:
            h -= (dp*dp) / (p*(1.0-p) + 1e-9)

        if abs(h) < 1e-6:
            break
        step = g / h
        # damping
        if abs(step) > 2.0:
            step = 2.0 if step > 0 else -2.0
        theta_new = theta - step
        if abs(theta_new - theta) < 1e-4:
            theta = theta_new
            break
        theta = theta_new

    # Sanity clamp
    theta = max(-6.0, min(6.0, theta))

    # Fallback grid refine if degenerate:
    # Evaluate a small grid around theta to ensure near-optimal.
    def nll(th):
        s = 0.0
        for y, a, b in events:
            p = p_correct(th, a, b)
            p = min(max(p, 1e-6), 1-1e-6)
            s -= y*math.log(p) + (1.0-y)*math.log(1.0-p)
        return s

    best_th = theta
    best = nll(theta)
    for delta in [-1.0,-0.5,-0.25,-0.1,0.1,0.25,0.5,1.0]:
        th = max(-6.0, min(6.0, theta+delta))
        val = nll(th)
        if val < best:
            best = val
            best_th = th
    return best_th

def compute_elo_for_user(up: UserProfile) -> Tuple[float, int, int]:
    """
    Returns (displayed_elo, N_total, N_nonblank)
    """
    # build last-200 window view
    events = up.history_list[-LAST200_LIMIT:]
    N_total = len(events)  # includes blanks
    # build MLE events excluding blanks
    mle_events = []
    for ev in events:
        if ev.qid in up.blanked_qids:
            continue
        sense = effective_sense(ev.row_index)
        a_scaled = SCALE_A * sense.a
        b_eff = sense.b
        mle_events.append( (1.0 if ev.correct else 0.0, a_scaled, b_eff) )
    N_nonblank = len(mle_events)
    # theta
    theta = mle_theta(mle_events) if mle_events else 0.0
    elo_unpen = 1600.0 + 500.0*theta
    elo_unpen = max(100.0, min(3100.0, elo_unpen))

    # penalty uses N_total (blanks count as 1)
    if N_total >= 200:
        penalty = 0.0
    else:
        q = 200 - N_total
        if elo_unpen > 2000.0:
            penalty = 1.5 * q
        elif elo_unpen > 1000.0:
            lam = (elo_unpen - 1000.0)/1000.0
            penalty = 1.5 * q * lam
        else:
            penalty = 0.0

    displayed = elo_unpen - penalty
    return displayed, N_total, N_nonblank

# ------------------------------
# Item parameter helpers
# ------------------------------

def effective_sense(row_index: str) -> ItemSense:
    # apply b_override if present
    base = ITEMS[row_index]
    if row_index in B_OVERRIDES:
        return ItemSense(row_index=base.row_index, key=base.key, definition=base.definition, a=base.a, b=B_OVERRIDES[row_index])
    return base

# ------------------------------
# Selection & Collision (sense-aware)
# ------------------------------

def is_globally_excluded_rowidx(row_index: str) -> bool:
    return row_index in GLOBAL_EXCLUDED_ROWIDX

def per_sense_meets_policy_b(row_index: str) -> bool:
    """
    Option B: sense-aware threshold only for Easy bucket B: b < -2.5
    For the skeleton, we expose a hook that decides active policy per-user based on window_12.
    TODO: Implement full routing per irt_key_selection_spec.txt if desired.
    """
    # If in Easy-B policy mode, enforce b < -2.5, else allow all.
    # We'll read the active mode from session policy passed by the quiz runner.
    return True  # policy is applied externally in candidate filters

def get_freq_rank(key: str) -> int:
    return FREQ_RANK_OF_KEY.get(key, 10**9)

def outward_rank_wave(start_rank: int) -> List[int]:
    # Generates r, r+1, r-1, r+2, r-2, ...
    # For performance, we will stream this lazily in the selection loop,
    # but we keep this helper for clarity.
    # Here we just return a placeholder; actual iteration in choose_next_item().
    return []

def choose_next_item(up: UserProfile, policy: Dict[str, Any]) -> Optional[str]:
    mode = route_policy_from_window12(up)

    rid = choose_key_and_sense_by_policy(up, mode)
    if rid:
        def ok(r):
            if is_globally_excluded_rowidx(r): return False
            if up.mode == "freedom": return r not in up.last_200_rowidx
            return ITEMS[r].key not in up.seen_keys_all_senses
        if ok(rid): 
            return rid
        seed_key = ITEMS[rid].key
    else:
        if mode == "uniform_bgt1":
            seed_key = draw_key_uniform(KEYS_HIGH_B) or draw_key_uniform(KEYS_ALL)
        elif mode == "uniform_all":
            seed_key = draw_key_uniform(KEYS_ALL)
        elif mode == "h15":
            seed_key = draw_key_by_cdf(CDF_15) or draw_key_uniform(KEYS_ALL)
        elif mode == "h25":
            seed_key = draw_key_by_cdf(CDF_25) or draw_key_uniform(KEYS_ALL)
        elif mode == "h40":
            seed_key = draw_key_by_cdf(CDF_40) or draw_key_uniform(KEYS_ALL)
        else:
            u = random.random()
            pool = KEYS_A if (u < 5/8 and KEYS_A) or not KEYS_B else (KEYS_B if KEYS_B else KEYS_A)
            seed_key = draw_key_uniform(pool) or draw_key_uniform(KEYS_ALL)
        if not seed_key:
            return None

    return next_item_via_collision(up, ITEMS[rid].key if rid else seed_key, mode)

# ------------------------------
# Distractor selection (THR1/THR2) with fast similarity
# ------------------------------

def select_distractors_type1(correct_rid: str) -> Tuple[List[str], List[str]]:
    """
    Given Indonesian word (key), choose 3 definition distractors (by subdefinition)
    Returns (distractor_rowidxs, rejection_reasons)
    """
    ensure_def_tokens()
    correct = effective_sense(correct_rid)
    key = correct.key
    tok_corr = DEF_TOKENS[correct_rid]
    # D = tokens of other senses of the same key
    D = [DEF_TOKENS[r] for r in SENSES_BY_KEY.get(key, []) if r != correct_rid]

    rejs = []
    chosen: List[str] = []
    tries = 0
    while len(chosen) < 3:
        tries += 1
        if tries > 10000:
            break
        # sample by subdefinition (uniform over all senses)
        rid, sense = random.choice(list(ITEMS.items()))
        if sense.key == key:
            continue
        tok = DEF_TOKENS[rid]
        # pre-screens
        if jaccard_set(tok, tok_corr) >= JACCARD_MAX:
            rejs.append("jac")
            continue
        # 3.1 sim(candidate, D) < THR1
        s1 = 0.0
        for t in D:
            s1 = max(s1, cosine_tokens(tok, t))
            if s1 >= THR1:
                break
        if s1 >= THR1:
            rejs.append("thr1")
            continue
        # 3.2 sim(candidate, chosen) < THR2
        ok = True
        for rid2 in chosen:
            t2 = DEF_TOKENS[rid2]
            s2 = cosine_tokens(tok, t2)
            if s2 >= THR2:
                ok = False
                break
        if not ok:
            rejs.append("thr2")
            continue
        chosen.append(rid)
    return chosen, rejs

def select_distractors_type2(correct_rid: str) -> Tuple[List[str], List[str]]:
    """
    Given English definition, choose 3 *words* (keys) as distractors.
    We compare full sense-sets of words.
    Returns (distractor_rowidxs_best_sense, rejection_reasons)
    """
    ensure_def_tokens()
    corr = effective_sense(correct_rid)
    key_correct = corr.key
    tok_corr = DEF_TOKENS[correct_rid]

    def sim_def_to_word(tok_def, key_word: str) -> float:
        smax = 0.0
        for rid in SENSES_BY_KEY.get(key_word, []):
            s = cosine_tokens(tok_def, DEF_TOKENS[rid])
            if s > smax:
                smax = s
                if smax >= 1.0:
                    break
        return smax

    chosen_keys: List[str] = []
    chosen_rids: List[str] = []
    rejs = []
    tries = 0
    while len(chosen_keys) < 3:
        tries += 1
        if tries > 10000:
            break
        # sample by key (uniform over keys)
        k = random.choice(list(SENSES_BY_KEY.keys()))
        if k == key_correct:
            continue
        # Levenshtein screen on words
        if levenshtein(k.lower(), key_correct.lower()) <= LEVEN_MAX:
            rejs.append("lev")
            continue
        # jaccard screen vs def (max over senses of k)
        jbad = False
        for rid in SENSES_BY_KEY.get(k, []):
            if jaccard_set(DEF_TOKENS[rid], tok_corr) >= JACCARD_MAX:
                jbad = True
                break
        if jbad:
            rejs.append("jac")
            continue
        # 3.1 sim(d_correct, K_k) < THR1
        s1 = sim_def_to_word(tok_corr, k)
        if s1 >= THR1:
            rejs.append("thr1")
            continue
        # 3.2 pairwise against other chosen keys' sense-sets
        ok = True
        for kprev in chosen_keys:
            # compute max sim between sense sets
            smax = 0.0
            for rid_a in SENSES_BY_KEY.get(kprev, []):
                ta = DEF_TOKENS[rid_a]
                for rid_b in SENSES_BY_KEY.get(k, []):
                    s = cosine_tokens(ta, DEF_TOKENS[rid_b])
                    if s > smax:
                        smax = s
                        if smax >= THR2:
                            break
                if smax >= THR2:
                    break
            if smax >= THR2:
                ok = False
                break
        if not ok:
            rejs.append("thr2")
            continue
        # accept; choose a representative rid for that word (best farthest sense)
        # choose the sense with minimal sim to the correct definition
        best_rid = None
        best_s = 1e9
        for rid in SENSES_BY_KEY.get(k, []):
            s = cosine_tokens(DEF_TOKENS[rid], tok_corr)
            if s < best_s:
                best_s = s
                best_rid = rid
        if not best_rid:
            continue
        chosen_keys.append(k)
        chosen_rids.append(best_rid)

    return chosen_rids, rejs

# ------------------------------
# Admin Helpers: Approvals / Ledger
# ------------------------------

@dataclass
class ApprovalEntry:
    approval_id: int
    kind: str                 # "report" or "complaint"
    user_id: int
    qid: int
    row_index: str
    action: str               # "blank" or "blank_if_wrong"
    distractors: List[str]
    admin_id: int
    date: str
    undone: bool = False

APPROVALS: Dict[int, ApprovalEntry] = {}  # approval_id -> entry

def next_qid() -> int:
    global NEXT_QID
    qid = NEXT_QID
    NEXT_QID += 1
    save_counter()
    return qid

def next_approval_id() -> int:
    global NEXT_APPROVAL_ID
    aid = NEXT_APPROVAL_ID
    NEXT_APPROVAL_ID += 1
    save_counter()
    return aid

def apply_report_approval(ae: ApprovalEntry):
    # Make the qid a blank for that user
    up = USERS.get(ae.user_id)
    if not up: return
    up.blanked_qids.add(ae.qid)
    # globally exclude this row_index for future correct items
    REPORTED_ROWIDX_APPROVED[ae.row_index] = REPORTED_ROWIDX_APPROVED.get(ae.row_index, 0) + 1
    GLOBAL_EXCLUDED_ROWIDX.add(ae.row_index)
    # remove from pending if present
    REPORTED_ROWIDX_PENDING.discard(ae.row_index)
    # CSV log
    ensure_csv_headers(APPROVED_REPORTS_CSV, ["approval_id","user_id","qid","row_index_correct","correct_key","d1","d2","d3","action","admin_id","date"])
    sense = ITEMS.get(ae.row_index) or ItemSense(ae.row_index,"","",1.0,0.0)
    d1,d2,d3 = (ae.distractors + ["","",""])[:3]
    append_csv(APPROVED_REPORTS_CSV, [ae.approval_id, ae.user_id, ae.qid, ae.row_index, sense.key, d1,d2,d3, ae.action, ae.admin_id, ae.date])

def undo_report_approval(ae: ApprovalEntry):
    up = USERS.get(ae.user_id)
    if up:
        up.blanked_qids.discard(ae.qid)
    # decrement refcount; if zero, remove from global exclusion
    cnt = REPORTED_ROWIDX_APPROVED.get(ae.row_index, 0)
    if cnt > 1:
        REPORTED_ROWIDX_APPROVED[ae.row_index] = cnt - 1
    else:
        REPORTED_ROWIDX_APPROVED.pop(ae.row_index, None)
        GLOBAL_EXCLUDED_ROWIDX.discard(ae.row_index)

def apply_complaint_approval(ae: ApprovalEntry):
    up = USERS.get(ae.user_id)
    if not up: return
    # blank only if that event was wrong
    was_wrong = False
    for ev in up.history_list:
        if ev.qid == ae.qid:
            was_wrong = (not ev.correct)
            break
    if was_wrong:
        up.blanked_qids.add(ae.qid)
    # CSV log
    ensure_csv_headers(APPROVED_COMPLAINTS_CSV, ["approval_id","user_id","qid","row_index_correct","correct_key","d1","d2","d3","action","admin_id","date"])
    sense = ITEMS.get(ae.row_index) or ItemSense(ae.row_index,"","",1.0,0.0)
    d1,d2,d3 = (ae.distractors + ["","",""])[:3]
    append_csv(APPROVED_COMPLAINTS_CSV, [ae.approval_id, ae.user_id, ae.qid, ae.row_index, sense.key, d1,d2,d3, ae.action, ae.admin_id, ae.date])

def undo_complaint_approval(ae: ApprovalEntry):
    up = USERS.get(ae.user_id)
    if up:
        up.blanked_qids.discard(ae.qid)

# ------------------------------
# Discord Helpers
# ------------------------------

def is_admin(user_id: int) -> bool:
    return user_id in ADMIN_ALLOWLIST

def today_date_str() -> str:
    # Date-only, no time
    return time.strftime("%Y-%m-%d", time.localtime())

async def send_long_text(ctx, filename: str, text: str):
    # Send as a file if too long for message
    data = text.encode("utf-8")
    if len(data) <= 1800:
        await ctx.send(f"```\n{text}\n```")
    else:
        await ctx.send(file=discord.File(io.BytesIO(data), filename=filename))

# ------------------------------
# Bot Events
# ------------------------------

@bot.event
async def on_ready():
    print(f"Logged in as {bot.user} (id={bot.user.id})")

# ------------------------------
# Core Commands
# ------------------------------


def _display_name_for(ctx, uid: int) -> str:
    # Prefer the current guild's display name; fallback to global user name; then ID.
    try:
        if ctx.guild:
            m = ctx.guild.get_member(uid)
            if m:
                return m.display_name
        u = ctx.bot.get_user(uid)
        return u.name if u else f"User {uid}"
    except Exception:
        return f"User {uid}"

@bot.command(name="help")
async def cmd_help(ctx: commands.Context):
    msg = dedent(f"""
    Commands:
    • {COMMAND_PREFIX}register <freedom|anki> — register or switch mode (default freedom)
    • {COMMAND_PREFIX}reviewmode <all|manual|mistakes> — set what gets printed at quiz end (default manual)
    • {COMMAND_PREFIX}mine — add the previous question's key to your mined list
    • {COMMAND_PREFIX}mineid <row_index> — add that key to mined list
    • {COMMAND_PREFIX}myseen — download a deduped list of all keys you've seen (first occurrence order)
    • {COMMAND_PREFIX}myhistory — download your entire history as 'row_index correct|wrong' (trim oldest if file >8MB)
    • {COMMAND_PREFIX}report — report the most recent question as unfair (bad definition). Won't accept duplicates for the same row_index if pending/approved.
    • {COMMAND_PREFIX}reportid <row_index> — report by row_index
    • {COMMAND_PREFIX}complain — complain that the options were too similar for the most recent question
    • {COMMAND_PREFIX}complainid <row_index> — complain by row_index
    • {COMMAND_PREFIX}nerf / {COMMAND_PREFIX}nerfid <row_index> — request difficulty decrease
    • {COMMAND_PREFIX}buff / {COMMAND_PREFIX}buffid <row_index> — request difficulty increase
    • {COMMAND_PREFIX}quiz [N] — start a quiz (default N={DEFAULT_QUIZ_LEN}), type 'exit' to stop
    • {COMMAND_PREFIX}hide — toggle whether to show freq-rank and b after each answer
    • {COMMAND_PREFIX}joinleaderboard — opt in to the public Elo leaderboard (off by default)
    • {COMMAND_PREFIX}leaderboard — show the leaderboard of opted-in users (Elo visible after MIN_NONBLANK_FOR_ELO)
    • {COMMAND_PREFIX}helpadmin — shows admin-only commands
    """).strip()
    await send_long_text(ctx, "help.txt", msg)

@bot.command(name="helpadmin")
async def cmd_helpadmin(ctx: commands.Context):
    if not is_admin(ctx.author.id):
        await ctx.send("Admins only.")
        return
    msg = dedent(f"""
    Admin-only:
    • {COMMAND_PREFIX}admin — open the approval console (one admin at a time)
    • {COMMAND_PREFIX}adminforceexit — force-release the admin console lock
    • {COMMAND_PREFIX}rebalance <row_index> <new_b> — set runtime difficulty (b) for an item
    • {COMMAND_PREFIX}undo <approval_id> — undo a prior approval (report/complaint)
    • {COMMAND_PREFIX}list_approvals [report|complaint] [active|undone] [limit] — list approvals
    • {COMMAND_PREFIX}recalcelo — recompute Elo for all users (requires typing 'recalc' to confirm)
    • {COMMAND_PREFIX}recalcelo @user — recompute Elo for a single user
    """).strip()
    await send_long_text(ctx, "helpadmin.txt", msg)

@bot.command(name="register")
async def cmd_register(ctx: commands.Context, mode: Optional[str]=None):
    mode = (mode or "freedom").lower().strip()
    if mode not in ("freedom","anki"):
        await ctx.send("Mode must be 'freedom' or 'anki'.")
        return
    up = USERS.get(ctx.author.id) or UserProfile(user_id=ctx.author.id)
    up.mode = mode
    USERS[ctx.author.id] = up
    save_user_store()
    await ctx.send(f"Registered in **{mode}** mode.")

@bot.command(name="reviewmode")
async def cmd_reviewmode(ctx: commands.Context, mode: Optional[str]=None):
    mode = (mode or "manual").lower().strip()
    if mode not in ("all","manual","mistakes"):
        await ctx.send("Reviewmode must be 'all', 'manual', or 'mistakes'.")
        return
    up = USERS.get(ctx.author.id) or UserProfile(user_id=ctx.author.id)
    up.reviewmode = mode
    USERS[ctx.author.id] = up
    save_user_store()
    await ctx.send(f"Review mode set to **{mode}**.")

@bot.command(name="mine")
async def cmd_mine(ctx: commands.Context):
    up = USERS.get(ctx.author.id)
    if not up or not up.history_list:
        await ctx.send("Nothing to mine yet.")
        return
    last = up.history_list[-1]
    key = ITEMS[last.row_index].key if last.row_index in ITEMS else last.key
    # enforce limits
    s = " OR ".join(f"key:{k}" for k in up.session_mine_list + [key])
    if len(up.session_mine_list) >= MINE_MAX_KEYS or len(s) >= MINE_MAX_CHARS:
        await ctx.send("Mine list is full for this session (limit reached).")
        return
    up.session_mine_list.append(key)
    save_user_store()
    await ctx.send(f"Mined: **{key}**")

@bot.command(name="mineid")
async def cmd_mineid(ctx: commands.Context, row_index: Optional[str]=None):
    if not row_index:
        await ctx.send("Usage: ?mineid <row_index>")
        return
    sense = ITEMS.get(row_index)
    if not sense:
        await ctx.send("Unknown row_index.")
        return
    up = USERS.get(ctx.author.id) or UserProfile(user_id=ctx.author.id)
    s = " OR ".join(f"key:{k}" for k in up.session_mine_list + [sense.key])
    if len(up.session_mine_list) >= MINE_MAX_KEYS or len(s) >= MINE_MAX_CHARS:
        await ctx.send("Mine list is full for this session (limit reached).")
        return
    up.session_mine_list.append(sense.key)
    USERS[ctx.author.id] = up
    save_user_store()
    await ctx.send(f"Mined: **{sense.key}**")

@bot.command(name="myseen")
async def cmd_myseen(ctx: commands.Context):
    up = USERS.get(ctx.author.id)
    if not up or not up.history_list:
        await ctx.send("No history yet.")
        return
    # stable dedupe by first occurrence order
    seen = []
    seen_set = set()
    for ev in up.history_list:
        key = ITEMS.get(ev.row_index, ItemSense(ev.row_index, ev.key, "", 1.0, 0.0)).key or ev.key
        if key not in seen_set:
            seen_set.add(key)
            seen.append(key)
    text = "\n".join(seen)
    await send_long_text(ctx, "myseen.txt", text)

@bot.command(name="myhistory")
async def cmd_myhistory(ctx: commands.Context):
    up = USERS.get(ctx.author.id)
    if not up or not up.history_list:
        await ctx.send("No history yet.")
        return
    lines = [f"{ev.row_index} {'correct' if ev.correct else 'wrong'}" for ev in up.history_list]
    text = "\n".join(lines)
    data = text.encode("utf-8")
    if len(data) > 8*1024*1024:
        # Trim oldest lines; prepend header
        while len(data) > 8*1024*1024 and lines:
            lines.pop(0)
            text = "(trimmed)\n" + "\n".join(lines)
            data = text.encode("utf-8")
    await ctx.send(file=discord.File(io.BytesIO(data), filename="myhistory.txt"))

@bot.command(name="hide")
async def cmd_hide(ctx: commands.Context):
    up = USERS.get(ctx.author.id) or UserProfile(user_id=ctx.author.id)
    up.hide_item_info = not up.hide_item_info
    USERS[ctx.author.id] = up
    save_user_store()
    await ctx.send("Per-question info: **hidden**" if up.hide_item_info else "Per-question info: **shown**")

# ------------------------------
# Report / Complaint requests
# ------------------------------

async def queue_report(ctx: commands.Context, row_index: Optional[str]):
    if not row_index:
        # default to last question
        up = USERS.get(ctx.author.id)
        if not up or not up.history_list:
            await ctx.send("No question to report.")
            return
        row_index = up.history_list[-1].row_index
    sense = ITEMS.get(row_index)
    if not sense:
        await ctx.send("Unknown row_index.")
        return
    # de-dup logic: disallow if pending or already approved
    if row_index in REPORTED_ROWIDX_PENDING or row_index in REPORTED_ROWIDX_APPROVED:
        await ctx.send("This row_index has already been reported or approved previously.")
        return
    # Enqueue a REPORT ticket (simple append to a global list)
    TICKETS.append({"kind":"report", "user_id":ctx.author.id, "row_index":row_index, "qid": find_last_qid(ctx.author.id, row_index), "distractors": find_last_distractors(ctx.author.id, row_index)})
    REPORTED_ROWIDX_PENDING.add(row_index)
    await ctx.send(f"Report queued for **{sense.key}** (row_index={row_index}).")

async def queue_complaint(ctx: commands.Context, row_index: Optional[str]):
    if not row_index:
        up = USERS.get(ctx.author.id)
        if not up or not up.history_list:
            await ctx.send("No question to complain about.")
            return
        row_index = up.history_list[-1].row_index
    sense = ITEMS.get(row_index)
    if not sense:
        await ctx.send("Unknown row_index.")
        return
    TICKETS.append({"kind":"complaint", "user_id":ctx.author.id, "row_index":row_index, "qid": find_last_qid(ctx.author.id, row_index), "distractors": find_last_distractors(ctx.author.id, row_index)})
    await ctx.send(f"Complaint queued for **{sense.key}** (row_index={row_index}).")

def find_last_qid(user_id: int, row_index: str) -> int:
    up = USERS.get(user_id)
    if not up: return 0
    # last occurrence of this row_index
    for ev in reversed(up.history_list):
        if ev.row_index == row_index:
            return ev.qid
    # fallback last event
    return up.history_list[-1].qid if up.history_list else 0

def find_last_distractors(user_id: int, row_index: str) -> List[str]:
    up = USERS.get(user_id)
    if not up: return []
    for ev in reversed(up.history_list):
        if ev.row_index == row_index:
            return ev.distractors
    return []

@bot.command(name="report")
async def cmd_report(ctx: commands.Context):
    await queue_report(ctx, None)

@bot.command(name="reportid")
async def cmd_reportid(ctx: commands.Context, row_index: Optional[str]=None):
    if not row_index:
        await ctx.send("Usage: ?reportid <row_index>")
        return
    await queue_report(ctx, row_index)

@bot.command(name="complain")
async def cmd_complain(ctx: commands.Context):
    await queue_complaint(ctx, None)

@bot.command(name="complainid")
async def cmd_complainid(ctx: commands.Context, row_index: Optional[str]=None):
    if not row_index:
        await ctx.send("Usage: ?complainid <row_index>")
        return
    await queue_complaint(ctx, row_index)

# ------------------------------
# Nerf / Buff requests (tickets)
# ------------------------------

@bot.command(name="nerf")
async def cmd_nerf(ctx: commands.Context):
    await queue_diff_request(ctx, None, "nerf")

@bot.command(name="nerfid")
async def cmd_nerfid(ctx: commands.Context, row_index: Optional[str]=None):
    if not row_index:
        await ctx.send("Usage: ?nerfid <row_index>")
        return
    await queue_diff_request(ctx, row_index, "nerf")

@bot.command(name="buff")
async def cmd_buff(ctx: commands.Context):
    await queue_diff_request(ctx, None, "buff")

@bot.command(name="buffid")
async def cmd_buffid(ctx: commands.Context, row_index: Optional[str]=None):
    if not row_index:
        await ctx.send("Usage: ?buffid <row_index>")
        return
    await queue_diff_request(ctx, row_index, "buff")

async def queue_diff_request(ctx: commands.Context, row_index: Optional[str], kind: str):
    if not row_index:
        up = USERS.get(ctx.author.id)
        if not up or not up.history_list:
            await ctx.send("No question to request difficulty change for.")
            return
        row_index = up.history_list[-1].row_index
    sense = ITEMS.get(row_index)
    if not sense:
        await ctx.send("Unknown row_index.")
        return
    TICKETS.append({"kind": kind, "user_id": ctx.author.id, "row_index": row_index, "qid": find_last_qid(ctx.author.id, row_index), "distractors": find_last_distractors(ctx.author.id, row_index)})
    await ctx.send(f"Request queued: **{kind}** for {sense.key} (row_index={row_index}).")

# ------------------------------
# Admin Console
# ------------------------------

TICKETS: List[Dict[str, Any]] = []

@bot.command(name="admin")
async def cmd_admin(ctx: commands.Context):
    global ADMIN_LOCK
    if not is_admin(ctx.author.id):
        await ctx.send("Admins only.")
        return
    if ADMIN_LOCK and ADMIN_LOCK != ctx.author.id:
        await ctx.send("Another admin is using the console. Use ?adminforceexit to take control.")
        return
    ADMIN_LOCK = ctx.author.id
    await ctx.send("Admin console acquired. Type 'exit' to leave.")

    try:
        while True:
            # display counts
            cnt_report = sum(1 for t in TICKETS if t["kind"]=="report")
            cnt_compl = sum(1 for t in TICKETS if t["kind"]=="complaint")
            cnt_diff   = sum(1 for t in TICKETS if t["kind"] in ("nerf","buff"))
            await ctx.send(f"Pending: reports={cnt_report}, complaints={cnt_compl}, diffs={cnt_diff}")
            if not TICKETS:
                await ctx.send("No tickets. Type 'exit' to leave, or wait for new tickets.")
                # wait for admin input
            else:
                ticket = TICKETS[0]
                kind = ticket["kind"]
                # render
                rid = ticket["row_index"]
                s = ITEMS.get(rid)
                if not s:
                    await ctx.send(f"[Ticket] kind={kind} row_index={rid} (not found) — type 'reject' or 'undo'")
                else:
                    # build prompt
                    qid = ticket.get("qid",0)
                    d1,d2,d3 = (ticket.get("distractors") or ["","",""])[:3]
                    if kind in ("nerf","buff"):
                        rank = get_freq_rank(s.key)
                        b_eff = effective_sense(rid).b
                        en = ENGLISH_NERF.get(rid) or ENGLISH_NERF.get(s.key) or {}
                        nerf_info = f"english_nerf={en.get('english_nerf','?')}" if isinstance(en, dict) else ""
                        await ctx.send(f"[{kind.upper()}] qid={qid} row_index={rid} key={s.key} rank={rank} b={b_eff} {nerf_info}\nType new b in [-3,3], or 'reject', or 'undo'.")
                    else:
                        await ctx.send(f"[{kind.upper()}] qid={qid} row_index={rid} key={s.key} d1={d1} d2={d2} d3={d3}\nType: definition/distractor/approve/approvewrong/report/complain/reject/undo/ctrlz/yes/no/deny/skillissue")
                # wait for response or 'exit'
            try:
                resp: discord.Message = await bot.wait_for(
                    "message",
                    check=lambda m: m.author.id==ctx.author.id and m.channel.id==ctx.channel.id,
                    timeout=300.0
                )
            except asyncio.TimeoutError:
                await ctx.send("Admin console timeout. Re-open with ?admin.")
                break

            txt = resp.content.strip().lower()
            if txt == "exit":
                await ctx.send("Admin console released.")
                break

            if not TICKETS:
                continue

            ticket = TICKETS[0]
            kind = ticket["kind"]
            rid = ticket["row_index"]
            s = ITEMS.get(rid)

            if kind in ("nerf","buff"):
                if txt in ("reject","deny","no","skillissue","undo","ctrlz"):
                    # reject diff request (no state change)
                    TICKETS.pop(0)
                    await ctx.send("Rejected.")
                    continue
                # try parse new b
                try:
                    new_b = float(txt)
                except:
                    await ctx.send("Please type a numeric b in [-3,3], or 'reject'.")
                    continue
                new_b = max(-3.0, min(3.0, new_b))
                # apply runtime override
                B_OVERRIDES[rid] = new_b
                ensure_csv_headers(DIFF_ADJUSTMENTS_CSV, ["row_index","key","googletranslate","a","b","old_b","admin_id","date"])
                old_b = s.b
                append_csv(DIFF_ADJUSTMENTS_CSV, [rid, s.key, s.definition, s.a, new_b, old_b, ctx.author.id, today_date_str()])
                # pop ticket
                TICKETS.pop(0)
                await ctx.send(f"Rebalanced row_index={rid} b={old_b} → {new_b}.")
                continue

            # reports / complaints with synonyms
            synonyms_def = {"definition","subdefinition","approve","yes","accept","report"}
            synonyms_dist = {"distractor","distractors","approvewrong","complain","complaint"}
            synonyms_rej = {"reject","deny","no","skillissue"}
            synonyms_undo = {"undo","ctrlz"}

            if txt in synonyms_rej:
                # If report was pending, drop from pending set
                if kind=="report":
                    REPORTED_ROWIDX_PENDING.discard(rid)
                TICKETS.pop(0)
                await ctx.send("Rejected.")
                continue
            if txt in synonyms_undo:
                # No-op here; use ?undo <id> outside console
                await ctx.send("Use ?undo <approval_id> to undo a past approval.")
                continue

            # determine approval kind (cross-approve support)
            if txt in synonyms_def:
                # approve as REPORT (blank + global exclude)
                # qid should exist; if not, it's okay (we still globally exclude)
                aid = next_approval_id()
                ae = ApprovalEntry(
                    approval_id=aid,
                    kind="report",
                    user_id=ticket["user_id"],
                    qid=ticket.get("qid", 0),
                    row_index=rid,
                    action="blank",
                    distractors=(ticket.get("distractors") or []),
                    admin_id=ctx.author.id,
                    date=today_date_str(),
                )
                APPROVALS[aid] = ae
                apply_report_approval(ae)
                TICKETS.pop(0)
                await ctx.send(f"Approved REPORT (id={aid}).")
                continue

            if txt in synonyms_dist:
                # approve as COMPLAINT (blank_if_wrong only, no global)
                aid = next_approval_id()
                ae = ApprovalEntry(
                    approval_id=aid,
                    kind="complaint",
                    user_id=ticket["user_id"],
                    qid=ticket.get("qid", 0),
                    row_index=rid,
                    action="blank_if_wrong",
                    distractors=(ticket.get("distractors") or []),
                    admin_id=ctx.author.id,
                    date=today_date_str(),
                )
                APPROVALS[aid] = ae
                apply_complaint_approval(ae)
                # if the report ticket was in pending set, remove
                REPORTED_ROWIDX_PENDING.discard(rid)
                TICKETS.pop(0)
                await ctx.send(f"Approved COMPLAINT (id={aid}).")
                continue

            await ctx.send("Unrecognized input. Try: definition / distractor / reject / exit.")

    finally:
        ADMIN_LOCK = None

@bot.command(name="adminforceexit")
async def cmd_adminforceexit(ctx: commands.Context):
    global ADMIN_LOCK
    if not is_admin(ctx.author.id):
        await ctx.send("Admins only.")
        return
    ADMIN_LOCK = None
    await ctx.send("Admin console lock cleared.")

@bot.command(name="undo")
async def cmd_undo(ctx: commands.Context, approval_id: Optional[str]=None):
    if not is_admin(ctx.author.id):
        await ctx.send("Admins only.")
        return
    if not approval_id or not approval_id.isdigit():
        await ctx.send("Usage: ?undo <approval_id>")
        return
    aid = int(approval_id)
    ae = APPROVALS.get(aid)
    if not ae or ae.undone:
        await ctx.send("Unknown or already undone approval id.")
        return
    # revert
    if ae.kind == "report":
        undo_report_approval(ae)
    else:
        undo_complaint_approval(ae)
    ae.undone = True
    await ctx.send(f"Undone approval id={aid}.")

@bot.command(name="list_approvals")
async def cmd_list_approvals(ctx: commands.Context, kind: Optional[str]=None, state: Optional[str]=None, limit: Optional[str]=None):
    if not is_admin(ctx.author.id):
        await ctx.send("Admins only.")
        return
    kind = (kind or "").strip().lower()
    state = (state or "").strip().lower()
    lim = int(limit) if (limit and limit.isdigit()) else 20
    items = list(APPROVALS.values())
    if kind in ("report","complaint"):
        items = [x for x in items if x.kind == kind]
    if state in ("active","undone"):
        want = (state == "undone")
        items = [x for x in items if x.undone == want]
    items.sort(key=lambda x: x.approval_id, reverse=True)
    lines = []
    for x in items[:lim]:
        lines.append(f"{x.approval_id}: {x.kind} user={x.user_id} qid={x.qid} rid={x.row_index} undone={x.undone}")
    await send_long_text(ctx, "approvals.txt", "\n".join(lines) or "(none)")

@bot.command(name="recalcelo")
async def cmd_recalcelo(ctx: commands.Context, target: Optional[str]=None):
    if not is_admin(ctx.author.id):
        await ctx.send("Admins only.")
        return
    if target and target.startswith("<@") and target.endswith(">"):
        # single user
        user_id = int(target.strip("<@!>"))
        up = USERS.get(user_id)
        if not up:
            await ctx.send("User not found.")
            return
        elo, n_total, n_nonblank = compute_elo_for_user(up)
        await ctx.send(f"Recalculated Elo for <@{user_id}> → {int(round(elo))} (window {n_nonblank}/{n_total} nonblank/total).")
        return
    # all users warning
    await ctx.send("This recalculates Elo for ALL users. Type `recalc` to proceed, or `?recalcelo @user` to limit scope.")
    try:
        msg: discord.Message = await bot.wait_for(
            "message",
            check=lambda m: m.author.id==ctx.author.id and m.channel.id==ctx.channel.id,
            timeout=30.0
        )
    except asyncio.TimeoutError:
        await ctx.send("Cancelled.")
        return
    if msg.content.strip().lower() != "recalc":
        await ctx.send("Cancelled.")
        return
    cnt = 0
    for uid, up in USERS.items():
        elo, n_total, n_nonblank = compute_elo_for_user(up)
        cnt += 1
    await ctx.send(f"Recalculated Elo for {cnt} users.")

@bot.command(name="rebalance")
async def cmd_rebalance(ctx: commands.Context, row_index: Optional[str]=None, new_b: Optional[str]=None):
    if not is_admin(ctx.author.id):
        await ctx.send("Admins only.")
        return
    if not row_index or new_b is None:
        await ctx.send("Usage: ?rebalance <row_index> <new_b>")
        return
    if row_index not in ITEMS:
        await ctx.send("Unknown row_index.")
        return
    try:
        b_val = float(new_b)
    except:
        await ctx.send("new_b must be numeric.")
        return
    b_val = max(-3.0, min(3.0, b_val))
    s = ITEMS[row_index]
    old_b = effective_sense(row_index).b
    B_OVERRIDES[row_index] = b_val
    ensure_csv_headers(DIFF_ADJUSTMENTS_CSV, ["row_index","key","googletranslate","a","b","old_b","admin_id","date"])
    append_csv(DIFF_ADJUSTMENTS_CSV, [row_index, s.key, s.definition, s.a, b_val, old_b, ctx.author.id, today_date_str()])
    await ctx.send(f"Rebalanced row_index={row_index} b={old_b} → {b_val}.")

# ------------------------------
# QUIZ
# ------------------------------
# === QUIZ helpers (drop-in) ===================================================

def is_hide_enabled(ctx: commands.Context) -> bool:
    """Return whether the user wants to hide per-question info."""
    up = USERS.get(ctx.author.id)
    return bool(up and up.hide_item_info)

async def apply_answer_effects(
    ctx: commands.Context,
    session: Dict[str, any],
    item: Dict[str, any],
    is_correct: bool,
    pick_idx: int
) -> None:
    """
    Log the answer to user history and update rolling windows so selection
    policies (window_12, last_200) work as expected.
    """
    user_id = ctx.author.id
    up = USERS.get(user_id) or UserProfile(user_id=user_id)
    USERS[user_id] = up

    # Allocate a new QID for the event
    qid = next_qid()

    # Record history (distractors as row_index list; qtype = 1 or 2)
    ev = HistoryEvent(
        qid=qid,
        row_index=item["row_index"],
        key=item["key"],
        correct=is_correct,
        date=today_date_str(),
        qtype=item["qtype"],
        distractors=list(item.get("distractor_rowidxs", [])),
    )
    up.history_list.append(ev)

    # Maintain last_200 row_index window (freedom mode uses this to avoid repeats)
    up.last_200_rowidx.append(item["row_index"])
    if len(up.last_200_rowidx) > LAST200_LIMIT:
        # trim oldest
        del up.last_200_rowidx[0 : len(up.last_200_rowidx) - LAST200_LIMIT]

    # In anki mode we avoid repeating the same key — mark the key as seen
    if up.mode == "anki":
        up.seen_keys_all_senses.add(item["key"])

    save_user_store()

async def get_next_quiz_item(ctx: commands.Context, session: Dict[str, any]) -> Optional[Dict[str, any]]:
    """
    Build a quiz item dict for the rewritten ?quiz command.

    Returns a dict with:
      {
        "qtype": 1|2,
        "key": str,
        "definition": str,
        "options": [str, str, str, str],
        "correct_idx": 0..3,
        "rank": int|float,
        "b": float,
        "row_index": str,
        "distractor_rowidxs": [str, str, str]
      }
    or None if we couldn't produce a valid item after several attempts.
    """
    user_id = ctx.author.id
    up = USERS.get(user_id) or UserProfile(user_id=user_id)
    USERS[user_id] = up

    # Try a handful of times to find a valid item + distractors
    for _ in range(50):
        rid = choose_next_item(up, policy={})
        if not rid:
            return None  # no eligible correct item
        sense = effective_sense(rid)

        # Randomize question type (1: choose definition; 2: choose word)
        qtype = 1 if random.random() < 0.5 else 2

        if qtype == 1:
            # Type 1: show (key), options are definitions
            dists, _rejs = select_distractors_type1(rid)
            if len(dists) < 3:
                continue  # try a different item
            correct_text = sense.definition
            option_texts = [correct_text] + [ITEMS[d].definition for d in dists]
        else:
            # Type 2: show (definition), options are words (keys)
            dists, _rejs = select_distractors_type2(rid)
            if len(dists) < 3:
                continue  # try a different item
            correct_text = sense.key
            option_texts = [correct_text] + [ITEMS[d].key for d in dists]

        # Shuffle options; track correct index
        idxs = list(range(4))
        random.shuffle(idxs)
        options = [option_texts[i] for i in idxs]
        correct_idx = idxs.index(0)  # 0 was the correct slot before shuffling

        return {
            "qtype": qtype,
            "key": sense.key,
            "definition": sense.definition,
            "options": options,
            "correct_idx": correct_idx,
            "rank": get_freq_rank(sense.key),
            "b": sense.b,
            "row_index": rid,
            "distractor_rowidxs": dists,
        }

    # Could not find a valid item with 3 distractors in reasonable time
    return None

@bot.command(name="quiz")
async def cmd_quiz(ctx: commands.Context, limit: Optional[str] = None):
    """
    ?quiz  — rewritten per spec:
      1) Prompt A (Type 1):   "({sense.key})\n**Choose the correct English definition.**\n"
         Prompt B (Type 2):   "{sense.definition}\n**Choose the correct Indonesian word:**\n"
      2) On invalid input (not a/b/c/d/exit): do NOT send anything and keep listening.
      3) On wrong answers: always show "❌ Wrong (Correct: X)" regardless of ?hide; then append tail
         "  [rank {rank}, b={b:.2f}]" only if ?hide is OFF.
      4) Combined-message flow:
           - If there’s another question remaining, DO NOT send feedback immediately.
             Prepend the feedback line to the NEXT question message (right before the prompt).
           - If it was the final question, send the feedback as a normal message.
    """

    # ---- config / guards -----------------------------------------------------
    IDLE_SECS = SESSION_IDLE_TIMEOUT_SEC  # idle timeout waiting for an answer
    global ACTIVE_QUIZ
    if ACTIVE_QUIZ.get(ctx.author.id):
        await ctx.send("You already have an active quiz. Type 'exit' to stop.")
        return

    # parse limit (fallback to your project’s default if parsing fails)
    try:
        total_q = int(limit) if limit is not None else DEFAULT_QUIZ_LEN  # use your existing default
    except Exception:
        total_q = DEFAULT_QUIZ_LEN

    session = {
        "remaining": total_q,
        "pending_feedback": None,     # str to prepend to next question, or None
        "stats": {"correct": 0, "wrong": 0},
    }
    ACTIVE_QUIZ[ctx.author.id] = session
    try:
        await ctx.send(f"Starting quiz ({total_q} questions). Type 'exit' to stop.")

        while session["remaining"] > 0:
            # ---- get next item ------------------------------------------------
            item = await get_next_quiz_item(ctx, session)
            if not item:
                await ctx.send("No eligible question found. Try again later.")
                break

            # item schema expected:
            # item = {
            #   "qtype": 1 or 2,
            #   "key": str,                   # for Type 1
            #   "definition": str,            # for Type 2
            #   "options": [str, str, str, str],
            #   "correct_idx": int,           # 0..3
            #   "rank": int | float,
            #   "b": float,
            # }

            # ---- format the question message ---------------------------------
            parts = []
            if session["pending_feedback"]:
                parts.append(session["pending_feedback"])
                session["pending_feedback"] = None  # consume it

            if item["qtype"] == 1:
                # Type 1 (definition choices for a key)
                prompt = f"({item['key']})\n**Choose the correct English definition.**"
            else:
                # Type 2 (word choices for a definition)
                prompt = f"{item['definition']}\n**Choose the correct Indonesian word:**"

            opts = [
                f"a) {item['options'][0]}",
                f"b) {item['options'][1]}",
                f"c) {item['options'][2]}",
                f"d) {item['options'][3]}",
                "(type a/b/c/d or 'exit')",
            ]
            parts.append(prompt)
            parts.append("\n".join(opts))
            await ctx.send("\n".join(parts))

            # ---- wait for a/b/c/d/exit (silent on invalid) -------------------
            def _check(m: discord.Message) -> bool:
                return (m.author.id == ctx.author.id) and (m.channel.id == ctx.channel.id)

            while True:
                try:
                    msg = await bot.wait_for("message", check=_check, timeout=IDLE_SECS)
                except asyncio.TimeoutError:
                    await ctx.send("Quiz expired due to inactivity.")
                    return  # will fall into finally and pop ACTIVE_QUIZ

                content = msg.content.strip().lower()
                if content in ("a", "b", "c", "d", "exit"):
                    break  # valid; proceed
                # Change #2: do nothing on invalid input; keep listening
                continue

            if content == "exit":
                await ctx.send("Quiz ended.")
                return

            # ---- grade --------------------------------------------------------
            pick_idx = "abcd".index(content)
            correct_idx = item["correct_idx"]
            is_correct = (pick_idx == correct_idx)
            correct_letter = "ABCD"[correct_idx]

            # update your scoring / persistence
            await apply_answer_effects(ctx, session, item, is_correct, pick_idx)

            if is_correct:
                feedback = "✅ Correct!"
                session["stats"]["correct"] += 1
            else:
                # Change #3: always show the correct letter even if ?hide is ON
                feedback = f"❌ Wrong (Correct: {correct_letter})"
                session["stats"]["wrong"] += 1

            # Append tail only if ?hide is OFF (unchanged logic)
            if not is_hide_enabled(ctx):
                # note: rank might be int or float in your code; leave as-is
                feedback += f"  [rank {item['rank']}, b={item['b']:.2f}]"

            # ---- combined-message flow ---------------------------------------
            session["remaining"] -= 1
            if session["remaining"] > 0:
                # Change #4: don't send feedback now; prepend to next question
                session["pending_feedback"] = feedback
                continue
            else:
                # final question -> send feedback as usual
                await ctx.send(feedback)

    finally:
        # end-of-quiz summary (restored)
        up = USERS.get(ctx.author.id)
        if up:
            # 1) Mined keys text
            if up.session_mine_list:
                s = " OR ".join(f"key:{k}" for k in up.session_mine_list)
                if len(s) > MINE_MAX_CHARS:
                    # trim greedily to MINE_MAX_CHARS
                    parts = []
                    total = 0
                    for k in up.session_mine_list:
                        frag = ("" if not parts else " OR ") + f"key:{k}"
                        if total + len(frag) > MINE_MAX_CHARS:
                            break
                        parts.append(f"key:{k}")
                        total += len(frag)
                    s = " OR ".join(parts)
                await send_long_text(ctx, "mined_keys.txt", s)
                up.session_mine_list.clear()

            # 2) Reviewmode text (all/mistakes)
            try:
                recent_events = up.history_list[-DEFAULT_QUIZ_LEN:]
            except Exception:
                recent_events = up.history_list[-25:]
            if up.reviewmode == "all":
                keys_encountered = [
                    (ITEMS[e.row_index].key if e.row_index in ITEMS else e.key)
                    for e in recent_events
                ]
                s = " OR ".join(f"key:{k}" for k in keys_encountered)
                if s:
                    await send_long_text(ctx, "review_all.txt", s)
            elif up.reviewmode == "mistakes":
                wrong_keys = [
                    (ITEMS[e.row_index].key if e.row_index in ITEMS else e.key)
                    for e in recent_events
                    if not e.correct
                ]
                s = " OR ".join(f"key:{k}" for k in wrong_keys)
                if s:
                    await send_long_text(ctx, "review_mistakes.txt", s)

            # 3) Total correct/total + Elo (gated by MIN_NONBLANK_FOR_ELO)
            elo, n_total, n_nonblank = compute_elo_for_user(up)
            correct_recent = sum(1 for e in up.history_list[-LAST200_LIMIT:]
                                  if (e.qid not in up.blanked_qids and e.correct))
            total_recent = sum(1 for e in up.history_list[-LAST200_LIMIT:]
                                if (e.qid not in up.blanked_qids))
            line1 = f"Total: {correct_recent}/{total_recent} questions"
            if n_nonblank >= MIN_NONBLANK_FOR_ELO:
                await ctx.send(f"{line1}\nElo: {int(round(elo))}")
            else:
                await ctx.send(f"{line1}\nElo: hidden (solve {MIN_NONBLANK_FOR_ELO} non-blank questions for Elo rating!)")

        ACTIVE_QUIZ.pop(ctx.author.id, None)
        try:
            await save_user_store()  # keep if you persist sessions; no-op if you don't
        except Exception:
            pass


# ------------------------------
# Startup
# ------------------------------

def bootstrap():
    # Load stores
    load_user_store()
    load_counter()
    load_items()
    load_freqs()
    load_english_nerf()
    ensure_def_tokens()
    build_key_pools()

    # CSV headers
    ensure_csv_headers(DIFF_ADJUSTMENTS_CSV, ["row_index","key","googletranslate","a","b","old_b","admin_id","date"])
    ensure_csv_headers(APPROVED_REPORTS_CSV, ["approval_id","user_id","qid","row_index_correct","correct_key","d1","d2","d3","action","admin_id","date"])
    ensure_csv_headers(APPROVED_COMPLAINTS_CSV, ["approval_id","user_id","qid","row_index_correct","correct_key","d1","d2","d3","action","admin_id","date"])

def read_token() -> str:
    tok = os.environ.get("DISCORD_TOKEN","").strip()
    if tok:
        return tok
    if os.path.isfile("token.txt"):
        with open("token.txt","r",encoding="utf-8") as f:
            return f.read().strip()
    raise RuntimeError("No Discord token found. Set DISCORD_TOKEN env var or create token.txt")

if __name__ == "__main__":
    bootstrap()
    TOKEN = read_token()
    bot.run(TOKEN)


@bot.command(name="joinleaderboard")
async def cmd_join_leaderboard(ctx: commands.Context):
    uid = ctx.author.id
    up = USERS.get(uid)
    if up is None:
        up = UserProfile(user_id=uid)
        USERS[uid] = up
    if up.leaderboard_opt_in:
        await ctx.send("You're already opted in to the leaderboard.")
        return
    up.leaderboard_opt_in = True
    save_user_store()
    await ctx.send("✅ You have opted in to the leaderboard. Use `!leaderboard` to see the rankings.")

@bot.command(name="leaderboard")
async def cmd_leaderboard(ctx: commands.Context):
    # Build rows for opted-in users with visible Elo
    rows = []
    for uid, up in USERS.items():
        if not getattr(up, "leaderboard_opt_in", False):
            continue
        elo, n_total, n_nonblank = compute_elo_for_user(up)
        if n_nonblank >= MIN_NONBLANK_FOR_ELO:
            rows.append((int(round(elo)), uid, n_nonblank, n_total))

    if not rows:
        await ctx.send(f"No eligible entries yet. Users must `!joinleaderboard` **and** reach ≥{MIN_NONBLANK_FOR_ELO} non-blank answers.")
        return

    rows.sort(key=lambda t: (-t[0], t[1]))  # Elo desc, then uid
    lines = [f"🏆 **Leaderboard** — opted-in users (Elo visible after ≥{MIN_NONBLANK_FOR_ELO} non-blank answers)"]
    for i, (eloi, uid, nnb, ntot) in enumerate(rows, 1):
        name = _display_name_for(ctx, uid)
        lines.append(f"{i:>2}. {name}: **{eloi}**  (non-blank {nnb}, total {ntot})")

    text_out = "\\n".join(lines)
    await send_long_text(ctx, "leaderboard.txt", text_out)

