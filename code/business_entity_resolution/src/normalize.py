"""Country-agnostic text normalization for business names and addresses.

Nothing here keys on a specific country label: the same rules run for US, India and
the unseen test country (France). Abbreviation tables cover English, Indian and French
conventions, so all variants of a word map to one canonical token
(e.g. 'avenue'/'av'/'ave' -> 'ave', 'private'/'pvt' -> 'pvt').
"""

import re
import unicodedata

import pandas as pd

# ---- legal / corporate suffixes (canonical form on the right) ----
LEGAL = {
    "private": "pvt", "pvt": "pvt", "pte": "pvt", "prv": "pvt",
    "limited": "ltd", "ltd": "ltd", "ltda": "ltd",
    "corporation": "corp", "corp": "corp", "incorporated": "inc", "inc": "inc",
    "company": "co", "co": "co", "cie": "co", "compagnie": "co",
    "llc": "llc", "llp": "llp", "lp": "lp", "plc": "plc", "pllc": "pllc",
    "sarl": "sarl", "sas": "sas", "sasu": "sas", "sa": "sa", "eurl": "eurl", "sci": "sci",
    "snc": "snc", "gmbh": "gmbh", "ag": "ag", "bv": "bv", "nv": "nv",
    "opc": "opc", "huf": "huf", "group": "group", "groupe": "group",
    "enterprises": "ent", "enterprise": "ent", "ent": "ent",
    "industries": "ind", "industry": "ind", "ind": "ind",
    "brothers": "bros", "bros": "bros", "associates": "assoc", "assoc": "assoc",
    "international": "intl", "intl": "intl", "services": "svc", "service": "svc",
    "and": "and", "et": "and", "&": "and",
    "the": "the", "le": "the", "la": "the", "les": "the",
    "of": "of", "de": "of", "du": "of", "des": "of", "d": "of",
}
# tokens dropped to build name_core (legal form + filler); descriptive words are kept
NAME_STOP = {
    "pvt", "ltd", "corp", "inc", "co", "llc", "llp", "lp", "plc", "pllc", "sarl", "sas",
    "sa", "eurl", "sci", "snc", "gmbh", "ag", "bv", "nv", "opc", "huf",
    "and", "the", "of", "m", "s", "ms", "dba", "aka", "t", "a",
}

ADDR = {
    # street types
    "street": "st", "st": "st", "str": "st", "road": "rd", "rd": "rd",
    "avenue": "ave", "ave": "ave", "av": "ave", "avn": "ave",
    "boulevard": "blvd", "blvd": "blvd", "bd": "blvd", "bld": "blvd",
    "drive": "dr", "dr": "dr", "lane": "ln", "ln": "ln", "court": "ct", "ct": "ct",
    "place": "pl", "pl": "pl", "square": "sq", "sq": "sq", "highway": "hwy", "hwy": "hwy",
    "parkway": "pkwy", "pkwy": "pkwy", "circle": "cir", "cir": "cir", "terrace": "ter",
    "expressway": "expy", "freeway": "fwy", "turnpike": "tpke", "way": "way",
    "marg": "rd", "path": "rd", "salai": "rd", "chemin": "ch", "ch": "ch",
    "impasse": "imp", "imp": "imp", "allee": "all", "quai": "qu", "qu": "qu",
    "route": "rte", "rte": "rte", "rue": "rue", "cours": "crs", "faubourg": "fbg", "fbg": "fbg",
    # units / building
    "suite": "ste", "ste": "ste", "unit": "unit", "apartment": "apt", "apt": "apt",
    "floor": "fl", "fl": "fl", "flr": "fl", "etage": "fl", "building": "bldg", "bldg": "bldg",
    "batiment": "bldg", "bat": "bldg", "room": "rm", "rm": "rm", "number": "no", "no": "no",
    "num": "no", "plot": "plot", "shop": "shop", "door": "no", "flat": "apt",
    "bis": "bis", "ter": "ter",
    # landmark words
    "near": "nr", "nr": "nr", "opposite": "opp", "opp": "opp", "behind": "bhd",
    "beside": "bsd", "besides": "bsd", "next": "nxt", "adjacent": "adj", "adj": "adj",
    "pres": "nr", "face": "opp", "cote": "bsd",
    # directions
    "north": "n", "south": "s", "east": "e", "west": "w",
    "northeast": "ne", "northwest": "nw", "southeast": "se", "southwest": "sw",
    # India-specific admin words
    "nagar": "ngr", "ngr": "ngr", "colony": "col", "sector": "sec", "sec": "sec",
    "phase": "ph", "ph": "ph", "cross": "crs", "main": "main", "layout": "lyt",
    "extension": "extn", "extn": "extn", "district": "dist", "dist": "dist",
    "post": "po", "po": "po", "taluk": "tk", "tehsil": "tk", "village": "vill", "vill": "vill",
    "saint": "st", "sainte": "ste",
    "mount": "mt", "mt": "mt", "fort": "ft", "ft": "ft",
}
LANDMARK_WORDS = {"nr", "opp", "bhd", "bsd", "nxt", "adj"}
UNIT_WORDS = {"ste", "unit", "apt", "fl", "bldg", "rm", "no", "plot", "shop"}

_punct = re.compile(r"[^\w\s]", flags=re.UNICODE)
_space = re.compile(r"\s+")
_num = re.compile(r"\d+")


def ascii_fold(s):
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))


def basic_clean(s):
    if not s:
        return ""
    s = ascii_fold(s).lower()
    s = s.replace("&", " and ").replace("@", " at ").replace("'", "")
    s = s.replace("p.o.", "po ").replace("s/o", " ").replace("c/o", " ")
    s = _punct.sub(" ", s)
    s = _space.sub(" ", s).strip()
    return s


def _collapse_initials(tokens):
    """'p v t l t d' style or 'a b c' -> join runs of single letters into one token."""
    out, run = [], []
    for t in tokens:
        if len(t) == 1 and t.isalpha():
            run.append(t)
        else:
            if len(run) >= 2:
                out.append("".join(run))
            else:
                out.extend(run)
            run = []
            out.append(t)
    if len(run) >= 2:
        out.append("".join(run))
    else:
        out.extend(run)
    return out


def norm_name(s):
    """Return (name_norm, name_core)."""
    toks = _collapse_initials(basic_clean(s).split())
    toks = [LEGAL.get(t, t) for t in toks]
    norm = " ".join(toks)
    core = [t for t in toks if t not in NAME_STOP]
    if not core:  # name was only legal words; keep something
        core = toks
    return norm, " ".join(core)


_pin_split = re.compile(r"\b(\d{3}) (\d{3})\b")
_zip4 = re.compile(r"\b(\d{5}) \d{4}\b")


def norm_addr(s):
    s = basic_clean(s)
    s = _pin_split.sub(r"\1\2", s)   # '560 001' -> '560001'
    s = _zip4.sub(r"\1", s)           # '78701 1234' -> '78701'
    toks = [ADDR.get(t, t) for t in s.split()]
    return " ".join(toks)


_postcode = re.compile(r"(?<![\d-])(\d{3} ?\d{3}|\d{5}(?:[- ]?\d{4})?)(?![\d-])")


def extract_postcode(raw):
    """Last 5-6 digit group in the back part of the address (PIN 6, ZIP 5(+4), French CP 5).

    A match in the first 35% of the string is treated as a house number, not a postcode.
    """
    if not raw:
        return ""
    raw = ascii_fold(raw)
    cands = [m for m in _postcode.finditer(raw) if m.start() >= 0.35 * len(raw)]
    if not cands:
        return ""
    pc = cands[-1].group(1).replace(" ", "").replace("-", "")
    if len(pc) == 9:  # ZIP+4 -> ZIP5
        pc = pc[:5]
    return pc


def addr_parts(addr_norm, postcode):
    toks = addr_norm.split()
    nums = [t for t in toks if any(ch.isdigit() for ch in t) and t.replace(" ", "") != postcode]
    house = nums[0] if nums else ""
    landmark = []
    for i, t in enumerate(toks):
        if t in LANDMARK_WORDS:
            landmark.extend(t2 for t2 in toks[i + 1:i + 3] if not any(ch.isdigit() for ch in t2))
    alpha = [t for t in toks if not any(ch.isdigit() for ch in t)]
    tail = alpha[-3:]  # usually city / state / country words
    return house, " ".join(nums), " ".join(landmark), " ".join(tail)


def acronym(core):
    toks = core.split()
    return "".join(t[0] for t in toks) if len(toks) >= 2 else ""


def normalize_frame(df):
    """Add all normalized columns. Pure function of each row -> identical for train and test."""
    out = df.copy()
    nn = [norm_name(x) for x in out["business_name"].tolist()]
    out["name_norm"] = [a for a, _ in nn]
    out["name_core"] = [b for _, b in nn]
    out["addr_norm"] = [norm_addr(x) for x in out["business_address"].tolist()]
    out["postcode"] = [extract_postcode(x) for x in out["business_address"].tolist()]
    parts = [addr_parts(a, p) for a, p in zip(out["addr_norm"], out["postcode"])]
    out["house_no"] = [p[0] for p in parts]
    out["addr_nums"] = [p[1] for p in parts]
    out["landmark"] = [p[2] for p in parts]
    out["addr_tail"] = [p[3] for p in parts]
    out["acronym"] = [acronym(c) for c in out["name_core"]]
    out["country_key"] = [basic_clean(c) or "unknown" for c in out["country"]]
    return out


def token_idf(series_list):
    """Document-frequency based IDF over tokens of the given Series (fit per split, no labels)."""
    import math
    from collections import Counter
    df = Counter()
    n = 0
    for s in series_list:
        for text in s:
            n += 1
            df.update(set(text.split()))
    return {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}, n


def as_str(s):
    return s if isinstance(s, str) else ("" if pd.isna(s) else str(s))
