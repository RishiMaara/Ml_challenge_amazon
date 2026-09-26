# EDA findings (Step 2)

Scripts: `notebooks/eda_01_structure.py` plus the ad-hoc analyses whose outputs are in `artifacts/eda/*.txt`. All numbers are from the full data unless marked as a sample.

## Structure
- **Sizes:** train S1 2,206,821 / S2 5,034,616 / S3 5,285,603. Test S1 1,732,544 / S2 4,887,273 / S3 5,082,316.
- **IDs:** no duplicate `entity_id`s. S2/S3 contain ~0.4–0.5% exact duplicate (name, address, country) rows; S1 has none.
- **Encoding:** files are clean UTF-8, with no U+FFFD characters. French accents are real characters.
- **One S1 per record (confirmed):** every matched S2/S3 id belongs to exactly **one** S1. This justifies the one-S1-per-record assignment constraint.
- **Country always agrees:** every train pair has the same `country` value on both sides, so blocking can be partitioned by country.
- **Matches per S1:** singleton rate is 5.6%, identical for US and India; the mean is 3.46 matches. 13.0% of S1 have no S2 match and 12.1% have no S3 match.
- **Distractors:** 73–75% of S2/S3 records are matched. The ~2.68M unmatched records look like ordinary businesses that simply have no S1 record.

## Noise catalogue
**Names**
- Word permutation ("LLC Tri-State Association") and OCR/typo edits (0↔o, 8↔B, transposed letters, dropped letters).
- Accents injected ("Tónet"), double spaces, casing changes (S2 is often UPPERCASE).
- Legal suffixes swapped or dropped (Private Limited / Pvt Ltd / Limited Private / LLC / (L.L.C.)).
- Descriptors appended ("Ventures", "Center", "[Co]", "Corporation"), and "The" or "Dr" prepended.
- Domain-style names (`saintcloudyouth.com`, ~3.5–4.5%).
- Junk prefixes (`--`, `<<`, `#`, ~2–3%).
- **Alias constructions:** "X D.B.A. Name", "X t/a Name", "X formerly known as Name"; the S1 name appears after the marker.
- **Native-script names:** whole names transliterated into Devanagari, Gujarati, Tamil, Telugu, Kannada, Bengali, Malayalam, Oriya or Gurmukhi ("ગુજરાત ઇન્ડસ્ટ્રીઝ" = Gujarat Industries). 41% of India S2 rows and 33% of India S3 rows contain an Indic script (in the name or the address).
- Zero-width characters injected (~1.3% of India S2, 0.7% of India S3).

**Addresses**
- Components reordered (state first, city first) and house numbers dropped or altered ("4328-D", "4328 1/2", "6536.").
- City substituted (Greece → Rochester, Phoenix → Arcadia).
- Tokens injected: `NULL`, `PMB 4092`, `PO Box`, "Floor 1", "Divreportingcircle".
- State written as an abbreviation (S1, S2), the full name (S3), or in native script ("महाराष्ट्र", "தமிழ்நாடு").
- ~2.7–3.4% of S2/S3 addresses are empty.
- **Postcodes are rare:** a 5-digit ZIP appears in ~10% of US addresses, and a 6-digit PIN appears in ~0% of Indian ones. Postcodes are *not* a primary blocking key.
- US S2 addresses are ~90% UPPERCASE with USPS abbreviations (AVE, BLVD, RD). S3 spells out state names.

## Difficulty
- **Hard negatives exist.** 49% of S1 rows share a normalised name key with another S1 in the same country (generic names: "Meridian LLC", "Summit LLC", "Cedar Inc"). 22% of unmatched distractors have an exact name-key match in S1. The address and house number must disambiguate.
- **String similarity** (60k sampled true pairs vs random pairs in the same country):
  - true matches: name token-set ratio median 100 (US) / 95 (India); non-empty address token-set median 93 / 97
  - random pairs: name median ~33–38, address ~36–38
  - Separation from random pairs is easy; the difficulty is concentrated in the hard negatives.

## France (test only, no labels)
- **Size:** 259k S1, 703k S2, 732k S3, heavily concentrated in 3 regions (Hauts-de-France, Nouvelle-Aquitaine, Pays de la Loire) and ~10 cities (Bordeaux, Nantes, Lille, Tourcoing, Dunkerque, Roubaix, Calais, Saint-Nazaire, Pessac, La Teste-de-Buch). City blocks will be huge.
- **Legal forms:** SARL, SAS, SASU, EURL, SA, SCI, SNC, "& Fils", "& Cie", "Ets/Établissements".
- **Generic words:** Club, Amicale, Ecole, Comité, Union, Maison, Centre, Pharmacie.
- **Address abbreviations:** R / R. → Rue, AV / Av. → Avenue, ALL → Allée, BD → Boulevard, plus Bis/Ter and N°/nº.
- **S1 uses the region** (Hauts-de-France) while S2/S3 often use the **department** (Nord, Gironde, Loire-Atlantique). We need a department ↔ region map.
- The noise generator looks the same as for US/India (typos, accents, reordering, domain names, empty addresses), so country-agnostic features should transfer.

## Implications for the pipeline
1. **Normalisation must:**
   - strip zero-width and junk characters
   - transliterate Indic scripts (anyascii) and fold accents
   - expand abbreviations for US, India and France
   - map native-script, abbreviated and department state names to one canonical form
   - split alias constructions (D.B.A., t/a, formerly known as) and use the best-matching part
2. **Blocking** can partition by `country`. It cannot rely on postcodes, and must handle word permutations and scripts:
   - TF-IDF over character n-grams of transliterated text
   - embeddings
   - rare-token inverted index
3. **Features** must carry the address and house number strongly, because 49% of S1 names are not unique. Add name-rarity (IDF) features.
4. **Decision layer:** use the one-S1-per-record assignment (confirmed). The 5.6% singleton rate is low but carries a full 1.0 each.
5. **France:** add dictionaries (legal forms, street types, department ↔ region). Validate via leave-one-country-out.
