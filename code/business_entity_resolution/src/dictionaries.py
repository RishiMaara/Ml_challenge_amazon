"""Hand-written normalisation dictionaries (general language knowledge, not entity data).

Keys are already in normalised form: ASCII, lowercase, punctuation replaced by spaces.
"""

# ---------------------------------------------------------------- names
# Legal-form / honorific tokens -> canonical token. Applied to name tokens for every country.
LEGAL_CANON = {
    "private": "pvt", "pvt": "pvt", "pte": "pvt", "limited": "ltd", "ltd": "ltd", "ltda": "ltd",
    "incorporated": "inc", "inc": "inc", "corporation": "corp", "corp": "corp",
    "company": "co", "co": "co", "llc": "llc", "llp": "llp", "lp": "lp", "pllc": "pllc",
    "pc": "pc", "plc": "plc", "pa": "pa", "opc": "opc",
    "md": "md", "dds": "dds", "cpa": "cpa", "esq": "esq",
    # France
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "eurl": "eurl", "sa": "sa", "sci": "sci",
    "snc": "snc", "scop": "scop", "selarl": "selarl", "scm": "scm", "sca": "sca",
    "cie": "cie", "compagnie": "cie", "ets": "ets", "etablissements": "ets", "etablissement": "ets",
}
LEGAL_TOKENS = frozenset(LEGAL_CANON.values())

# Multi-token legal forms collapsed before tokenising (after dots are removed).
LEGAL_PHRASES = [
    (r"\bl l c\b", "llc"), (r"\bl l p\b", "llp"), (r"\bp l l c\b", "pllc"), (r"\bp c\b", "pc"),
    (r"\bp a\b", "pa"), (r"\bm d\b", "md"), (r"\bs a s u\b", "sasu"), (r"\bs a s\b", "sas"),
    (r"\bs a r l\b", "sarl"), (r"\bs a\b", "sa"), (r"\bs c i\b", "sci"), (r"\bs n c\b", "snc"),
    (r"\bpvt\s*ltd\b", "pvt ltd"),
]

NAME_STOPWORDS = frozenset({"the", "dr", "and", "of", "de", "du", "des", "la", "le", "les", "l", "d", "et", "a"})

# Alias markers: the S1-style name is the part AFTER the marker.
ALIAS_MARKERS = r"\b(?:d\s*b\s*a|doing business as|t\s+a|trading as|formerly known as|f\s*k\s*a|a\s+k\s+a|aka|also known as)\b"

DOMAIN_TLDS = r"(?:com|net|org|in|co\.in|co|fr|biz|info|us|io)"

# ---------------------------------------------------------------- addresses
_US_STREET = {
    "street": "st", "str": "st", "st": "st", "avenue": "ave", "av": "ave", "ave": "ave",
    "road": "rd", "rd": "rd", "boulevard": "blvd", "blvd": "blvd", "lane": "ln", "ln": "ln",
    "drive": "dr", "dr": "dr", "court": "ct", "ct": "ct", "place": "pl", "pl": "pl",
    "circle": "cir", "cir": "cir", "highway": "hwy", "hwy": "hwy", "parkway": "pkwy", "pkwy": "pkwy",
    "trail": "trl", "trl": "trl", "terrace": "ter", "ter": "ter", "square": "sq", "sq": "sq",
    "mount": "mt", "mt": "mt", "fort": "ft", "ft": "ft", "point": "pt", "pt": "pt",
    "north": "n", "south": "s", "east": "e", "west": "w",
    "northeast": "ne", "northwest": "nw", "southeast": "se", "southwest": "sw",
    "apartment": "apt", "apt": "apt", "suite": "ste", "ste": "ste", "floor": "fl", "fl": "fl",
    "building": "bldg", "bldg": "bldg", "route": "rte", "rte": "rte", "saint": "st",
    "county": "cty", "cty": "cty", "expressway": "expy", "freeway": "fwy", "crossing": "xing",
    "first": "1st", "second": "2nd", "third": "3rd", "fourth": "4th", "fifth": "5th",
    "sixth": "6th", "seventh": "7th", "eighth": "8th", "ninth": "9th", "tenth": "10th", "eleventh": "11th",
    "twelfth": "12th", "thirteenth": "13th", "fourteenth": "14th", "fifteenth": "15th", "sixteenth": "16th",
    "seventeenth": "17th", "eighteenth": "18th", "nineteenth": "19th", "twentieth": "20th",
}
_IN_STREET = dict(_US_STREET, **{
    "opposite": "opp", "opp": "opp", "near": "near", "nr": "near", "behind": "behind",
    "sector": "sec", "sec": "sec", "phase": "ph", "ph": "ph", "colony": "colony",
    "nagar": "nagar", "marg": "marg", "chowk": "chowk", "district": "dist", "dist": "dist",
    "taluka": "tal", "tal": "tal", "village": "vill", "vill": "vill", "post": "po",
})
_FR_STREET = {
    "rue": "rue", "r": "rue", "avenue": "avenue", "av": "avenue", "ave": "avenue",
    "boulevard": "boulevard", "bd": "boulevard", "bld": "boulevard", "blvd": "boulevard",
    "allee": "allee", "all": "allee", "chemin": "chemin", "ch": "chemin", "chem": "chemin",
    "place": "place", "pl": "place", "impasse": "impasse", "imp": "impasse",
    "route": "route", "rte": "route", "square": "square", "sq": "square",
    "saint": "saint", "st": "saint", "sainte": "sainte", "ste": "sainte",
    "faubourg": "faubourg", "fg": "faubourg", "fbg": "faubourg", "quai": "quai", "qu": "quai",
    "cours": "cours", "crs": "cours", "residence": "residence", "res": "residence",
    "lotissement": "lotissement", "lot": "lotissement", "cite": "cite", "hameau": "hameau", "ham": "hameau",
    "passage": "passage", "pass": "passage", "promenade": "promenade", "prom": "promenade",
}
STREET_CANON = {"US": _US_STREET, "India": _IN_STREET, "France": _FR_STREET}
STREET_CANON_DEFAULT = _US_STREET  # open-set countries

# Tokens that only introduce a number ("H.No 12", "Door No 7/72", "#831"): dropped when followed by a digit.
NUMBER_PREFIXES = frozenset({"no", "hno", "h", "door", "house", "plot", "flat", "shop", "sno", "s", "khasra", "kh", "survey", "unit", "apt", "ste", "suite", "n", "numero", "num"})

ADDRESS_NOISE = [r"\bnull\b", r"\bnone\b", r"\bn a\b", r"\bpmb\s*\d+\b", r"\bp\s*o\s*box\s*\d+\b", r"\bpo\s*box\s*\d+\b", r"\bbox\s*\d+\b"]

# ---------------------------------------------------------------- states / regions
US_STATES = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas", "ca": "california",
    "co": "colorado", "ct": "connecticut", "de": "delaware", "fl": "florida", "ga": "georgia",
    "hi": "hawaii", "id": "idaho", "il": "illinois", "in": "indiana", "ia": "iowa", "ks": "kansas",
    "ky": "kentucky", "la": "louisiana", "me": "maine", "md": "maryland", "ma": "massachusetts",
    "mi": "michigan", "mn": "minnesota", "ms": "mississippi", "mo": "missouri", "mt": "montana",
    "ne": "nebraska", "nv": "nevada", "nh": "new hampshire", "nj": "new jersey", "nm": "new mexico",
    "ny": "new york", "nc": "north carolina", "nd": "north dakota", "oh": "ohio", "ok": "oklahoma",
    "or": "oregon", "pa": "pennsylvania", "ri": "rhode island", "sc": "south carolina",
    "sd": "south dakota", "tn": "tennessee", "tx": "texas", "ut": "utah", "vt": "vermont",
    "va": "virginia", "wa": "washington", "wv": "west virginia", "wi": "wisconsin", "wy": "wyoming",
    "dc": "district of columbia", "pr": "puerto rico",
}

IN_STATES = {
    "ap": "andhra pradesh", "ar": "arunachal pradesh", "as": "assam", "br": "bihar",
    "cg": "chhattisgarh", "ct": "chhattisgarh", "ga": "goa", "gj": "gujarat", "hr": "haryana",
    "hp": "himachal pradesh", "jh": "jharkhand", "ka": "karnataka", "kl": "kerala",
    "mp": "madhya pradesh", "mh": "maharashtra", "mn": "manipur", "ml": "meghalaya", "mz": "mizoram",
    "nl": "nagaland", "od": "odisha", "or": "odisha", "pb": "punjab", "rj": "rajasthan", "sk": "sikkim",
    "tn": "tamil nadu", "ts": "telangana", "tg": "telangana", "tr": "tripura", "up": "uttar pradesh",
    "uk": "uttarakhand", "ua": "uttarakhand", "ut": "uttarakhand", "wb": "west bengal",
    "an": "andaman and nicobar islands", "ch": "chandigarh", "dn": "dadra and nagar haveli and daman and diu",
    "dd": "dadra and nagar haveli and daman and diu", "dl": "delhi", "jk": "jammu and kashmir",
    "la": "ladakh", "ld": "lakshadweep", "py": "puducherry",
}
IN_STATE_ALIASES = {
    "orissa": "odisha", "pondicherry": "puducherry", "uttaranchal": "uttarakhand",
    "nct of delhi": "delhi", "new delhi": None,  # New Delhi is a city, not a state
    "chattisgarh": "chhattisgarh", "tamilnadu": "tamil nadu", "dadra and nagar haveli": "dadra and nagar haveli and daman and diu",
    "daman and diu": "dadra and nagar haveli and daman and diu", "andaman and nicobar": "andaman and nicobar islands",
    "jammu kashmir": "jammu and kashmir", "keralam": "kerala",
}

FR_REGIONS_DEPTS = {
    "auvergne rhone alpes": ["ain", "allier", "ardeche", "cantal", "drome", "isere", "loire", "haute loire", "puy de dome", "rhone", "savoie", "haute savoie"],
    "bourgogne franche comte": ["cote d or", "doubs", "jura", "nievre", "haute saone", "saone et loire", "yonne", "territoire de belfort"],
    "bretagne": ["cotes d armor", "finistere", "ille et vilaine", "morbihan"],
    "centre val de loire": ["cher", "eure et loir", "indre", "indre et loire", "loir et cher", "loiret"],
    "corse": ["corse du sud", "haute corse"],
    "grand est": ["ardennes", "aube", "marne", "haute marne", "meurthe et moselle", "meuse", "moselle", "bas rhin", "haut rhin", "vosges"],
    "hauts de france": ["aisne", "nord", "oise", "pas de calais", "somme"],
    "ile de france": ["seine et marne", "yvelines", "essonne", "hauts de seine", "seine saint denis", "val de marne", "val d oise"],  # 'paris' omitted: it is also a city
    "normandie": ["calvados", "eure", "manche", "orne", "seine maritime"],
    "nouvelle aquitaine": ["charente", "charente maritime", "correze", "creuse", "dordogne", "gironde", "landes", "lot et garonne", "pyrenees atlantiques", "deux sevres", "vienne", "haute vienne"],
    "occitanie": ["ariege", "aude", "aveyron", "gard", "haute garonne", "gers", "herault", "lot", "lozere", "hautes pyrenees", "pyrenees orientales", "tarn", "tarn et garonne"],
    "pays de la loire": ["loire atlantique", "maine et loire", "mayenne", "sarthe", "vendee"],
    "provence alpes cote d azur": ["alpes de haute provence", "hautes alpes", "alpes maritimes", "bouches du rhone", "var", "vaucluse"],
}


def _build_state_maps():
    us = {**US_STATES, **{v: v for v in US_STATES.values()}}
    ind = {**IN_STATES, **{v: v for v in IN_STATES.values()}}
    ind.update({k: v for k, v in IN_STATE_ALIASES.items() if v})
    fr = {}
    for region, depts in FR_REGIONS_DEPTS.items():
        fr[region] = region
        for d in depts:
            fr[d] = region
    fr["paca"] = "provence alpes cote d azur"
    fr["idf"] = "ile de france"
    return {"US": us, "India": ind, "France": fr}


STATE_CANON = _build_state_maps()
# 2-letter abbreviations are preferred when several parts look like a state (a city can share a state's full name).
STATE_ABBREVS = {"US": frozenset(US_STATES), "India": frozenset(IN_STATES), "France": frozenset({"paca", "idf"})}
