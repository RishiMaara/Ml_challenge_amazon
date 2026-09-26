"""Record normalisation: names and addresses -> canonical strings and parsed components.

Two passes:
  1. learn   - learn transliteration maps (native-script token -> Latin token, native-script state -> state)
               from TRAIN matched pairs only.
  2. apply   - normalise every source file with those maps -> artifacts/normalized/*.parquet

Usage:
    python -m src.normalize --processed ../../artifacts/processed --out ../../artifacts/normalized
"""
import argparse
import json
import re
import unicodedata
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import polars as pl
from anyascii import anyascii

from .dictionaries import (
    ADDRESS_NOISE, ALIAS_MARKERS, DOMAIN_TLDS, LEGAL_CANON, LEGAL_PHRASES, LEGAL_TOKENS,
    NAME_STOPWORDS, NUMBER_PREFIXES, STATE_ABBREVS, STATE_CANON, STREET_CANON, STREET_CANON_DEFAULT,
)

# ------------------------------------------------------------------ character level
_ZERO_WIDTH = re.compile(r"[­​-‏‪-‮⁠-⁤﻿]")
_NON_ASCII = re.compile(r"[^\x00-\x7f]")
_INDIC = re.compile(r"[ऀ-෿]")
_DROP_CHARS = re.compile(r"[.'`’]")          # "p.c." -> "pc", "o'brien" -> "obrien"
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_SPACES = re.compile(r"\s+")


def to_ascii(s: str) -> str:
    """NFKC, strip zero-width chars, transliterate/fold to ASCII, lowercase."""
    s = _ZERO_WIDTH.sub("", unicodedata.normalize("NFKC", s))
    s = s.replace("°", " ").replace("º", " ")
    if _NON_ASCII.search(s):
        s = anyascii(s)
    return s.lower()


def simple_tokens(s: str) -> list[str]:
    s = _DROP_CHARS.sub("", s.replace("&", " and "))
    return _NON_ALNUM.sub(" ", s).split()


# ------------------------------------------------------------------ names
_URL_TAIL = re.compile(r"\s*\|\s*.*$")
_URL_ANY = re.compile(rf"\b(?:https?://)?(?:www\.)?[a-z0-9-]+\.{DOMAIN_TLDS}\b")
_DOMAIN = re.compile(rf"^\s*[#@]?(?:https?://)?(?:www\.)?([a-z0-9-]+)\.{DOMAIN_TLDS}\s*$")
_HASHTAG = re.compile(r"^\s*#([a-z0-9]{4,})\s*$")
_ALIAS = re.compile(ALIAS_MARKERS)
_LEGAL_PHRASES = [(re.compile(p), r) for p, r in LEGAL_PHRASES]
_OCR = str.maketrans({"0": "o", "1": "l", "3": "e", "5": "s", "8": "b"})


def _ocr_fix(tok: str) -> str:
    """Fix digit-for-letter OCR noise inside mostly-alphabetic tokens (fell0wship, 8uildcon)."""
    if tok.isalpha() or tok.isdigit():
        return tok
    n_alpha = sum(c.isalpha() for c in tok)
    if n_alpha >= 3 and all(c in "01358" for c in tok if c.isdigit()):
        return tok.translate(_OCR)
    return tok


_VOWELS = re.compile(r"[aeiouy]")


def skeleton(tok: str) -> str:
    """Consonant skeleton robust to vowel loss in transliteration and common typos: gujarat/gujrat -> gjrt."""
    if not tok or tok.isdigit():
        return tok
    t = tok.replace("ph", "f").replace("sh", "s").replace("ch", "c").replace("th", "t").replace("kh", "k")
    t = t.replace("bh", "b").replace("dh", "d").replace("gh", "g").replace("jh", "j")
    t = t.translate(str.maketrans({"z": "j", "w": "v", "q": "k", "c": "k", "x": "k"}))
    head, rest = t[0], _VOWELS.sub("", t[1:]).replace("h", "")
    out = [head]
    for c in rest:
        if c != out[-1]:
            out.append(c)
    return "".join(out)


def normalize_name(raw: str, token_map: dict | None = None) -> dict:
    had_indic = bool(_INDIC.search(raw))
    s = to_ascii(raw)
    is_domain = False
    m = _DOMAIN.match(s) or _HASHTAG.match(s)
    if m:
        is_domain, s = True, m.group(1).replace("-", " ")
    else:
        s = _URL_TAIL.sub("", s)
        s = _URL_ANY.sub(" ", s)
    toks = simple_tokens(s)
    s = " ".join(toks)
    for pat, rep in _LEGAL_PHRASES:
        s = pat.sub(rep, s)
    alt = ""
    parts = _ALIAS.split(s, maxsplit=1)
    if len(parts) == 2 and parts[1].strip():
        alt, s = parts[0].strip(), parts[1].strip()
    toks = [_ocr_fix(t) for t in s.split()]
    if had_indic and token_map:
        toks = [token_map.get(t, t) for t in toks]
    toks = [LEGAL_CANON.get(t, t) for t in toks]
    core = [t for t in toks if t not in LEGAL_TOKENS and t not in NAME_STOPWORDS] or toks
    legal = sorted({t for t in toks if t in LEGAL_TOKENS})
    return {
        "name_norm": " ".join(toks),
        "name_core": " ".join(core),
        "name_key": " ".join(sorted(set(core))),
        "name_skel": " ".join(skeleton(t) for t in core),
        "name_legal": " ".join(legal),
        "name_alt": alt,
        "name_is_domain": is_domain,
        "name_indic": had_indic,
    }


# ------------------------------------------------------------------ addresses
_ADDR_NOISE = [re.compile(p) for p in ADDRESS_NOISE]
_ORDINAL = re.compile(r"\b(\d+)(?:st|nd|rd|th)\b")  # generator corrupts suffixes: 54rd / 7nd -> 54th / 7th


def _ordinal(m: re.Match) -> str:
    n = int(m.group(1))
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{m.group(1)}{suffix}"
_DIGITS = re.compile(r"\d+")
_POSTCODE_LEN = {"US": (5,), "France": (5,), "India": (6,)}


def _state_of(part: str, country: str, had_indic: bool, state_map: dict | None):
    canon = STATE_CANON.get(country, {})
    if part in canon:
        return canon[part]
    if had_indic and state_map:
        return state_map.get(part)
    return None


def normalize_address(raw: str, country: str, state_map: dict | None = None) -> dict:
    street_canon = STREET_CANON.get(country, STREET_CANON_DEFAULT)
    pc_lens = _POSTCODE_LEN.get(country, (5, 6))
    parts = []  # (normalised part string, had_indic)
    for p in raw.split(","):
        had_indic = bool(_INDIC.search(p))
        a = to_ascii(p)
        for pat in _ADDR_NOISE:
            a = pat.sub(" ", a)
        a = " ".join(simple_tokens(a))
        if a:
            parts.append((a, had_indic))

    # --- state: prefer 2-letter abbreviations, then the last matching part
    state, state_idx = "", -1
    abbrevs = STATE_ABBREVS.get(country, frozenset())
    cands = [(i, _state_of(a, country, ind, state_map), a in abbrevs) for i, (a, ind) in enumerate(parts)]
    cands = [c for c in cands if c[1]]
    if cands:
        best = max(cands, key=lambda c: (c[2], c[0]))
        state_idx, state = best[0], best[1]

    # --- postcode: a standalone 5/6-digit token that ends its part
    postcode = ""
    for i, (a, _) in enumerate(parts):
        last = a.rsplit(" ", 1)[-1]
        if last.isdigit() and len(last) in pc_lens and (" " in a or i > 0):
            postcode = last

    streets, locs, all_toks, nums = [], [], [], []
    for i, (a, _) in enumerate(parts):
        if i == state_idx:
            continue
        toks = a.split()
        toks = [t for j, t in enumerate(toks)
                if not (t in NUMBER_PREFIXES and j + 1 < len(toks) and toks[j + 1][:1].isdigit())]
        toks = [_ORDINAL.sub(_ordinal, street_canon.get(t, t)) for t in toks]
        if not toks:
            continue
        canon_part = " ".join(toks)
        all_toks.extend(toks)
        if any(c.isdigit() for c in canon_part):
            streets.append(canon_part)
            nums.extend(d for d in _DIGITS.findall(canon_part) if d != postcode)
        else:
            locs.append(canon_part)
    seen, uniq_nums = set(), []
    for d in nums:
        d = d.lstrip("0") or "0"
        if d not in seen:
            seen.add(d)
            uniq_nums.append(d)
    return {
        "addr_norm": " ".join(all_toks),
        "addr_key": " ".join(sorted(set(all_toks))),
        "addr_street": " | ".join(streets),
        "addr_localities": " | ".join(locs),
        "addr_state": state,
        "addr_postcode": postcode,
        "addr_nums": " ".join(uniq_nums),
        "addr_house_no": uniq_nums[0] if uniq_nums else "",
        "addr_empty": not parts,
    }


# ------------------------------------------------------------------ learning translit maps (train only)
def learn_maps(processed: Path, min_count: int = 3, min_share: float = 0.5) -> dict:
    """Learn native-script -> Latin maps from train matched pairs (India only has native scripts).

    token_map: transliterated name token -> Latin S1 token, via positional alignment of equal-length names.
    state_map: transliterated address part -> canonical state, via the matched S1's parsed state.
    """
    indic = r"[ऀ-෿]"
    s1 = (pl.scan_parquet(processed / "train_source1.parquet").filter(pl.col("country") == "India")
            .select(pl.col("entity_id").alias("s1_id"), pl.col("business_name").alias("n1"), pl.col("business_address").alias("a1")))
    s23 = (pl.scan_parquet([processed / f"train_source{i}.parquet" for i in (2, 3)])
             .filter((pl.col("country") == "India") &
                     (pl.col("business_name").str.contains(indic) | pl.col("business_address").str.contains(indic)))
             .select(pl.col("entity_id").alias("match_id"), pl.col("business_name").alias("n2"), pl.col("business_address").alias("a2")))
    gt = pl.scan_parquet(processed / "train_gt_pairs.parquet").drop_nulls("match_id").select("s1_id", "match_id")
    pairs = gt.join(s23, on="match_id").join(s1, on="s1_id").select("n1", "n2", "a1", "a2").collect()

    tok_src, tok_dst, st_src, st_dst = [], [], [], []
    for n1, n2, a1, a2 in pairs.iter_rows():
        if _INDIC.search(n2):
            t2, t1 = simple_tokens(to_ascii(n2)), simple_tokens(to_ascii(n1))
            if len(t1) == len(t2):
                for x, y in zip(t2, t1):
                    if x != y:
                        tok_src.append(x)
                        tok_dst.append(LEGAL_CANON.get(y, y))
        s1_state = normalize_address(a1, "India")["addr_state"]
        if s1_state:
            for p in a2.split(","):
                if _INDIC.search(p):
                    key = " ".join(simple_tokens(to_ascii(p)))
                    if key:
                        st_src.append(key)
                        st_dst.append(s1_state)

    def pick(src, dst):
        votes = pl.DataFrame({"k": src, "v": dst}).group_by("k", "v").len()
        return dict(
            votes.with_columns(pl.col("len").sum().over("k").alias("total"))
                 .sort("len", descending=True).unique("k", keep="first")
                 .filter((pl.col("len") >= min_count) & (pl.col("len") / pl.col("total") >= min_share))
                 .select("k", "v").iter_rows())

    return {"token_map": pick(tok_src, tok_dst), "state_map": pick(st_src, st_dst), "n_pairs": pairs.height}


# ------------------------------------------------------------------ apply to files
_MAPS: dict = {}


def _init(maps):
    global _MAPS
    _MAPS = maps


def _normalize_chunk(df: pl.DataFrame) -> pl.DataFrame:
    """Normalise one chunk; builds columns (not per-row dicts) to keep memory flat."""
    tm, sm = _MAPS.get("token_map"), _MAPS.get("state_map")
    cols = defaultdict(list)
    for eid, name, addr, country in df.select("entity_id", "business_name", "business_address", "country").iter_rows():
        cols["entity_id"].append(eid)
        cols["country"].append(country)
        for k, v in normalize_name(name, tm).items():
            cols[k].append(v)
        for k, v in normalize_address(addr, country, sm).items():
            cols[k].append(v)
    return pl.DataFrame(cols)


def normalize_frame(df: pl.DataFrame, maps: dict, workers: int = 0, chunk: int = 50_000) -> pl.DataFrame:
    chunks = (df.slice(i, chunk) for i in range(0, df.height, chunk))
    if workers == 1:
        _init(maps)
        return pl.concat([_normalize_chunk(c) for c in chunks])
    out, pending = [], []
    with ProcessPoolExecutor(max_workers=workers or None, initializer=_init, initargs=(maps,)) as ex:
        n_workers = ex._max_workers
        for c in chunks:  # bounded in-flight work so the parent never holds the whole file as Python objects
            pending.append(ex.submit(_normalize_chunk, c))
            if len(pending) >= 2 * n_workers:
                out.append(pending.pop(0).result())
        out.extend(p.result() for p in pending)
    return pl.concat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--only", nargs="*", help="subset of files, e.g. train_source1")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    maps_path = args.out / "translit_maps.json"
    if maps_path.exists():
        maps = json.loads(maps_path.read_text(encoding="utf-8"))
    else:
        maps = learn_maps(args.processed)
        maps_path.write_text(json.dumps(maps, ensure_ascii=False, indent=0), encoding="utf-8")
    print(f"maps: {len(maps['token_map'])} tokens, {len(maps['state_map'])} states (from {maps['n_pairs']:,} pairs)")

    for split in ("train", "test"):
        for src in ("source1", "source2", "source3"):
            name = f"{split}_{src}"
            if args.only and name not in args.only:
                continue
            df = pl.read_parquet(args.processed / f"{name}.parquet")
            norm = normalize_frame(df, maps, workers=args.workers)
            norm.write_parquet(args.out / f"{name}.parquet", compression="zstd")
            print(f"{name}: {norm.height:,} rows normalised")


if __name__ == "__main__":
    main()
