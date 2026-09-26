"""Unit tests for src.normalize, built from real examples seen in EDA. Run: python -m pytest tests -q"""
from src.normalize import normalize_address, normalize_name, skeleton


def n(raw, **kw):
    return normalize_name(raw, **kw)


# ------------------------------------------------------------------ names
def test_legal_suffix_variants_share_core():
    forms = ["Lloyd Trading Private Limited", "Lloyd Trading Pvt. Ltd.", "Lloyd Trading Limited Private",
             "LLOYD TRADING PVT LTD", "The Lloyd Trading Private  Limited"]
    assert {n(f)["name_core"] for f in forms} == {"lloyd trading"}
    assert n("Lloyd Trading Pvt. Ltd.")["name_legal"] == "ltd pvt"


def test_us_legal_punctuation():
    assert n("Metro Motors (L.L.C.)")["name_core"] == "metro motors"
    assert n("Metro Motors (L.L.C.)")["name_legal"] == "llc"
    assert n("Saint Cloud Youth Association P.C.")["name_legal"] == "pc"
    assert n("Leslie K. Flagg, ([M.D.])")["name_core"] == "leslie k flagg"


def test_word_order_key():
    assert n("Association Tri-State LLC")["name_key"] == n("Tri-State Association LLC")["name_key"]


def test_alias_markers_keep_part_after_marker():
    r = n("Arckelo D.B.A. Metro Motors LLC")
    assert r["name_core"] == "metro motors" and r["name_alt"] == "arckelo"
    assert n("Ectoarc t/a Northern Mercantile Clinic")["name_core"] == "northern mercantile clinic"
    assert n("Cirawex formerly known as Thane Agro Private Limited")["name_core"] == "thane agro"
    assert n("Ectolumdrex dba X+ Madison Inc")["name_core"] == "x madison"


def test_domains_and_urls():
    r = n("saintcloudyouth.com")
    assert r["name_is_domain"] and r["name_core"] == "saintcloudyouth"
    assert n("#centraleducation")["name_is_domain"]
    assert n("SHIVSHAKTI VIDYALAYA OVERSEAS CORPORATION | www.shivshakti.com")["name_core"] == "shivshakti vidyalaya overseas"


def test_junk_accents_ocr_zero_width():
    assert n("-- Holloway Peak Inc Seafood")["name_core"] == "holloway peak seafood"
    assert n("<< Team Ecole")["name_core"] == "team ecole"
    assert n("Tónet")["name_core"] == "tonet"
    assert n("Midwest  Fell0wship​ LLC")["name_core"] == "midwest fellowship"
    assert n("Energy Dgshit 8uildcon Private Limited")["name_core"] == "energy dgshit buildcon"
    assert n("Mahima & Sons Private Limited")["name_norm"] == "mahima and sons pvt ltd"


def test_french_legal_forms():
    assert n("Thermal & Fils SASU")["name_legal"] == "sasu"
    assert n("Fractales Amis Groupe S.A.S")["name_legal"] == "sas"
    assert n("Établissements Team")["name_core"] == "team"


def test_native_script_with_learned_map():
    raw = "ரெட் குளோபல் பிரைவேட் லிமிடெட்"
    r = n(raw, token_map={"ret": "red", "kulopl": "global", "piraivet": "pvt", "limitet": "ltd"})
    assert r["name_indic"] and r["name_core"] == "red global" and r["name_legal"] == "ltd pvt"


def test_skeleton():
    assert skeleton("gujarat") == skeleton("gujrat") == "gjrt"
    assert skeleton("12") == "12"


# ------------------------------------------------------------------ addresses
def test_us_address_order_and_abbreviations():
    a = normalize_address("1795 Westchester Drive, High Point, NC", "US")
    b = normalize_address("NC, HIGH POINT, 1795 WESTCHESTER DR", "US")
    c = normalize_address("1795 Westchester Dr, High Point, North Carolina", "US")
    assert a["addr_key"] == b["addr_key"] == c["addr_key"]
    assert a["addr_state"] == c["addr_state"] == "north carolina"
    assert a["addr_house_no"] == "1795" and a["addr_localities"] == "high pt"


def test_us_noise_tokens_and_house_numbers():
    r = normalize_address("1545 R Mount Read Blvd, NULL, PMB 4092, Rochester, New York", "US")
    assert "null" not in r["addr_norm"] and "4092" not in r["addr_nums"]
    assert r["addr_state"] == "new york" and r["addr_house_no"] == "1545"
    assert normalize_address("4328-D ARCADIA LANE, PHOENIX, AZ", "US")["addr_house_no"] == "4328"
    assert normalize_address("Unit 2007, 1111 Church Street, Nashville, TN", "US")["addr_nums"] == "2007 1111"


def test_postcode():
    assert normalize_address("12 Main St, Raleigh, NC 27601", "US")["addr_postcode"] == "27601"
    assert normalize_address("17560 Ellis Road, Tahlequah, OK", "US")["addr_postcode"] == ""


def test_india_state_variants():
    a = normalize_address("67/93, Tatabad Iii Street, Coimbatore, Tamil Nadu", "India")
    b = normalize_address("67/93, Tatabad Iii Street, Coimbatore, TN", "India")
    c = normalize_address("67/93, TATABAD III STREET, COIMBATORE, தமிழ்நாடு", "India", state_map={"tmilnatu": "tamil nadu"})
    assert a["addr_state"] == b["addr_state"] == c["addr_state"] == "tamil nadu"
    assert a["addr_key"] == b["addr_key"] == c["addr_key"]
    assert normalize_address("H.NO 7/72 E2, 7, ST-6, THAKKALIVILAI", "India")["addr_nums"] == "7 72 2 6"


def test_india_delhi_city_vs_state():
    r = normalize_address("304, 3Rd Floor 56 Eros Appartments Nehru Place, Delhi, South Delhi, Delhi", "India")
    assert r["addr_state"] == "delhi" and "delhi" in r["addr_localities"]


def test_france_department_to_region_and_street_abbrev():
    a = normalize_address("20 Rue Parmentier, Dunkerque, Hauts-de-France", "France")
    b = normalize_address("20 R. PARMENTIER, DUNKERQUE, Nord", "France")
    assert a["addr_state"] == b["addr_state"] == "hauts de france"
    assert a["addr_key"] == b["addr_key"]
    c = normalize_address("3 ALL BOIRON, PESSAC", "France")
    assert c["addr_street"] == "3 allee boiron" and c["addr_state"] == ""


def test_unknown_country_does_not_crash():
    r = normalize_address("Calle Mayor 5, Madrid", "Spain")
    assert r["addr_house_no"] == "5" and r["addr_state"] == ""


def test_empty_address():
    assert normalize_address("", "US")["addr_empty"]


def test_ordinal_suffix_noise():
    a = normalize_address("825 B 54th St, Seattle, WA", "US")
    b = normalize_address("825 B 54rd St, Seattle, WA", "US")
    c = normalize_address("8215 Fourteenth Pl, Broken Arrow, OK", "US")
    assert a["addr_key"] == b["addr_key"] and "54th" in a["addr_norm"]
    assert c["addr_street"] == "8215 14th pl"
    assert normalize_address("1 7nd Ave, X, NY", "US")["addr_street"] == "1 7th ave"
