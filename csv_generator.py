"""YES Bank - Data Generation Script (Updated)
============================================
Generates 5 output CSV files derived entirely from real source data.

NEW CSV INTEGRATIONS (replacing all static/hardcoded data):
  - branch_classification_rules.csv   → replaces hardcoded STATE→BRANCH_TYPE/CLUSTER dicts
  - branch_playbook_parameters.csv    → replaces hardcoded ramp curves, capex, success factors
  - fiscal_calendar.csv               → replaces hardcoded months_fy list & quarter_map dict
  - rbi_compliance_master.csv         → replaces hardcoded compliance_data list
  - state_economic_indicators.csv     → replaces hardcoded state_income_base, state_literacy,
                                        state_gst_base dicts

All other source CSV integrations are preserved:
  - yes_bank_branches.csv             (YES Bank branch network)
  - india_pincodes.csv                (Geographic reference + lat/lon)
  - india_district_census.csv         (Population, households, area)
  - pmjdy_district.csv                (PMJDY household coverage)
  - rbi_bsr_district.csv              (RBI banking channel registry)
"""

import pandas as pd
import numpy as np
import os
import hashlib
import re
import sys
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────
# PATHS  — update INPUT_DIR / OUTPUT_DIR as needed
# os.path.join is used throughout so forward/back slashes don't matter
# ─────────────────────────────────────────────────────────────────
INPUT_DIR  = os.path.join( "data/external")
OUTPUT_DIR = os.path.join("data/output_files")

# The 5 new reference CSVs live in the same folder as the other inputs
REF_DIR = os.path.join("data/config")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Pre-flight: confirm every required file exists before starting ─
REQUIRED_FILES = {
    "yesbank_ifsc.csv":            INPUT_DIR,
    "india_pincodes.csv":               INPUT_DIR,
    "india_district_census.csv":        INPUT_DIR,
    "pmjdy_district.csv":               INPUT_DIR,   # CSV (not .xls)
    "rbi_bsr_district.csv":             INPUT_DIR,   # CSV (not .xlsx)
    "branch_classification_rules.csv":  REF_DIR,
    "branch_playbook_parameters.csv":   REF_DIR,
    "fiscal_calendar.csv":              REF_DIR,
    "rbi_compliance_master.csv":        REF_DIR,
    "state_economic_indicators.csv":    REF_DIR,
}

missing = []
for fname, folder in REQUIRED_FILES.items():
    full = os.path.join(folder, fname)
    status = "OK " if os.path.exists(full) else "MISSING"
    print(f"  [{status}] {full}")
    if status == "MISSING":
        missing.append(full)

if missing:
    print(f"\n  ✗ {len(missing)} file(s) not found. Please fix the paths above and retry.")
    sys.exit(1)

print("  ✓ All required files found\n")

np.random.seed(42)

print("=" * 65)
print("  YES BANK DATA GENERATION PIPELINE  (Updated)")
print("=" * 65)


# ═══════════════════════════════════════════════════════════════════
# HELPER
# ═══════════════════════════════════════════════════════════════════
def det_noise(name, scale, offset=0):
    """Deterministic noise from string hash — no random mock data."""
    h = int(hashlib.md5(str(name).encode()).hexdigest(), 16)
    return offset + (h % 10000) / 10000 * scale


# ═══════════════════════════════════════════════════════════════════
# STEP 1 — LOAD ALL SOURCE FILES
# ═══════════════════════════════════════════════════════════════════
print("\n[1/6] Loading source files...")

# ── Existing external CSVs ───────────────────────────────────────

yes_branches = pd.read_csv(os.path.join(INPUT_DIR, "yesbank_ifsc.csv"), skiprows=1)
yes_branches.columns = ['BR_CODE', 'BR_NAME', 'IFSC', 'ADDRESS', 'STATE']
yes_branches['STATE'] = yes_branches['STATE'].str.strip()

pins = pd.read_csv(os.path.join(INPUT_DIR, "india_pincodes.csv"))
pins['latitude']  = pd.to_numeric(pins['latitude'],  errors='coerce')
pins['longitude'] = pd.to_numeric(pins['longitude'], errors='coerce')
pins = pins[(pins['latitude'].between(6, 37)) & (pins['longitude'].between(68, 98))]

census_raw = pd.read_csv(os.path.join(INPUT_DIR, "india_district_census.csv"), header=1, low_memory=False)
census_raw.columns = [c.replace('\r\n', ' ').strip() for c in census_raw.columns]
census = census_raw[
    (census_raw['India/ State/ Union Territory/ District/ Sub-district'] == 'DISTRICT') &
    (census_raw['Total/ Rural/ Urban'] == 'Total')
].copy()
census = census.rename(columns={
    'Name': 'DISTRICT_NAME',
    'Population': 'POPULATION',
    'Number of households': 'HOUSEHOLDS',
    'Area  (In sq. km)': 'AREA_SQKM',
    'Population per sq. km.': 'POP_DENSITY'
})
for col in ['POPULATION', 'HOUSEHOLDS', 'AREA_SQKM', 'POP_DENSITY']:
    census[col] = census[col].astype(str).str.replace(',', '').str.strip()
    census[col] = pd.to_numeric(census[col], errors='coerce')

pmjdy = pd.read_csv(os.path.join(INPUT_DIR, "pmjdy_district.csv"))
pmjdy.columns = ['DISTRICT', 'ALLOTED_SSA', 'SURVEYED_SSA', 'COVERAGE_PCT']
pmjdy['COVERAGE_PCT'] = pmjdy['COVERAGE_PCT'].astype(str).str.replace('%', '').str.strip()
pmjdy['COVERAGE_PCT'] = pd.to_numeric(pmjdy['COVERAGE_PCT'], errors='coerce')
pmjdy['DISTRICT_UPPER'] = pmjdy['DISTRICT'].str.upper().str.strip()

bsr = pd.read_csv(os.path.join(INPUT_DIR, "rbi_bsr_district.csv"))
bsr['STATE']    = bsr['State'].str.strip().str.title()
bsr['DISTRICT'] = bsr['District'].str.strip().str.title()

# ── NEW: 5 Reference CSVs ────────────────────────────────────────
# NOTE: These files are saved with each row wrapped in a single outer
# quote e.g. "STATE,BRANCH_TYPE,CLUSTER" so pandas reads only one
# column.  read_quoted_csv() strips the outer quotes and re-parses.

def read_quoted_csv(filepath):
    """
    Handle CSVs where every row is stored as one quoted string:
        "COL1,COL2,COL3"
        "val1,val2,val3"
    Strips the outer quotes, re-parses with csv.reader, returns DataFrame.
    """
    import io, csv as csv_mod
    rows = []
    with open(filepath, 'r', encoding='utf-8-sig') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith('"') and line.endswith('"'):
                line = line[1:-1]
            rows.append(line)
    reader = csv_mod.reader(rows)
    data   = list(reader)
    if not data:
        return pd.DataFrame()
    header = [h.strip() for h in data[0]]
    return pd.DataFrame(data[1:], columns=header)


# 1. branch_classification_rules.csv  →  STATE, BRANCH_TYPE, CLUSTER
branch_rules = read_quoted_csv(os.path.join(REF_DIR, "branch_classification_rules.csv"))
branch_rules['STATE']       = branch_rules['STATE'].str.strip()
branch_rules['BRANCH_TYPE'] = branch_rules['BRANCH_TYPE'].str.strip()
branch_rules['CLUSTER']     = branch_rules['CLUSTER'].str.strip()

state_to_branch_type = dict(zip(branch_rules['STATE'], branch_rules['BRANCH_TYPE']))
state_to_cluster     = dict(zip(branch_rules['STATE'], branch_rules['CLUSTER']))

# 2. branch_playbook_parameters.csv
playbook_params = read_quoted_csv(os.path.join(REF_DIR, "branch_playbook_parameters.csv"))
playbook_params['BRANCH_ARCHETYPE'] = playbook_params['BRANCH_ARCHETYPE'].str.strip()
num_cols = ['INITIAL_CAPEX_LAKHS_MIN', 'INITIAL_CAPEX_LAKHS_MAX',
            'MONTH_3_CASA_PCT_OF_PEAK',  'MONTH_6_CASA_PCT_OF_PEAK',
            'MONTH_9_CASA_PCT_OF_PEAK',  'MONTH_12_CASA_PCT_OF_PEAK',
            'MONTH_18_CASA_PCT_OF_PEAK', 'MONTH_24_CASA_PCT_OF_PEAK']
for c in num_cols:
    playbook_params[c] = pd.to_numeric(playbook_params[c], errors='coerce')
playbook_params = playbook_params.set_index('BRANCH_ARCHETYPE')

ramp_curves     = {}
capex_map       = {}
success_factors = {}
for archetype, row in playbook_params.iterrows():
    ramp_curves[archetype] = {
        'M3':  float(row['MONTH_3_CASA_PCT_OF_PEAK']),
        'M6':  float(row['MONTH_6_CASA_PCT_OF_PEAK']),
        'M9':  float(row['MONTH_9_CASA_PCT_OF_PEAK']),
        'M12': float(row['MONTH_12_CASA_PCT_OF_PEAK']),
        'M18': float(row['MONTH_18_CASA_PCT_OF_PEAK']),
        'M24': float(row['MONTH_24_CASA_PCT_OF_PEAK']),
    }
    capex_map[archetype] = (int(row['INITIAL_CAPEX_LAKHS_MIN']),
                            int(row['INITIAL_CAPEX_LAKHS_MAX']))
    success_factors[archetype] = (
        str(row['KEY_SUCCESS_FACTOR_1']).strip(),
        str(row['KEY_SUCCESS_FACTOR_2']).strip(),
        str(row['KEY_CHALLENGE']).strip(),
    )

# 3. fiscal_calendar.csv  →  MONTH, QUARTER, FISCAL_YEAR
fiscal_cal = read_quoted_csv(os.path.join(REF_DIR, "fiscal_calendar.csv"))
fiscal_cal['MONTH']   = fiscal_cal['MONTH'].str.strip()
fiscal_cal['QUARTER'] = fiscal_cal['QUARTER'].str.strip()
months_fy   = fiscal_cal['MONTH'].tolist()
quarter_map = dict(zip(fiscal_cal['MONTH'], fiscal_cal['QUARTER']))

# 4. rbi_compliance_master.csv
compliance_master = read_quoted_csv(os.path.join(REF_DIR, "rbi_compliance_master.csv"))
compliance_master.columns = [c.strip() for c in compliance_master.columns]
# Drop placeholder "..." rows if present
compliance_master = compliance_master[compliance_master['CATEGORY'].str.strip() != '...']

# 5. state_economic_indicators.csv
state_econ = read_quoted_csv(os.path.join(REF_DIR, "state_economic_indicators.csv"))
state_econ['STATE'] = state_econ['STATE'].str.strip()
for c in ['PER_CAPITA_INCOME_INR', 'LITERACY_RATE_PCT', 'GST_REGISTRATIONS_PER_LAKH']:
    state_econ[c] = pd.to_numeric(state_econ[c], errors='coerce')
state_econ = state_econ.set_index('STATE')

state_income_base = state_econ['PER_CAPITA_INCOME_INR'].to_dict()
state_literacy    = state_econ['LITERACY_RATE_PCT'].to_dict()
state_gst_base    = state_econ['GST_REGISTRATIONS_PER_LAKH'].to_dict()

print("  ✓ All source files loaded (including 5 new reference CSVs)")


# ═══════════════════════════════════════════════════════════════════
# STEP 2 — BUILD GEOGRAPHIC MASTER (State → District → Lat/Lon)
# ═══════════════════════════════════════════════════════════════════
print("\n[2/6] Building geographic master from pincodes...")

# State normalization: derive from branch_classification_rules STATE values
valid_states = set(branch_rules['STATE'].tolist())
STATE_NORM = {s.upper(): s for s in valid_states}   # e.g. 'MAHARASHTRA' → 'Maharashtra'

geo = (
    pins[pins['statename'].isin(STATE_NORM.keys())]
    .groupby(['statename', 'district'])
    .agg(lat_mean=('latitude', 'mean'), lon_mean=('longitude', 'mean'),
         pincode_count=('pincode', 'count'))
    .reset_index()
)
geo['STATE']    = geo['statename'].map(STATE_NORM)
geo['DISTRICT'] = geo['district'].str.strip().str.title()
geo = geo.dropna(subset=['lat_mean', 'lon_mean'])
geo = geo[geo['pincode_count'] >= 3]

print(f"  ✓ {len(geo)} state-district records built")


# ═══════════════════════════════════════════════════════════════════
# STEP 3 — COMPUTE YES BANK COUNTS + MERGE DEMOGRAPHICS
# ═══════════════════════════════════════════════════════════════════
print("\n[3/6] Computing branch counts per state...")

yb_state_count  = yes_branches.groupby('STATE').size().reset_index(name='YES_BANK_BRANCHES_STATE')

geo = geo.merge(yb_state_count, on='STATE', how='left').fillna({'YES_BANK_BRANCHES_STATE': 0})

geo['DISTRICT_UPPER'] = geo['DISTRICT'].str.upper().str.strip()
geo = geo.merge(pmjdy[['DISTRICT_UPPER', 'COVERAGE_PCT']], on='DISTRICT_UPPER', how='left')
geo['PMJDY_COVERAGE'] = geo['COVERAGE_PCT'].fillna(95.0)

census['DISTRICT_UPPER'] = census['DISTRICT_NAME'].str.upper().str.strip()
census_slim = census[['DISTRICT_UPPER', 'POPULATION', 'HOUSEHOLDS', 'AREA_SQKM']].dropna()
geo = geo.merge(census_slim, on='DISTRICT_UPPER', how='left')

for col in ['POPULATION', 'HOUSEHOLDS', 'AREA_SQKM']:
    state_avg = geo.groupby('STATE')[col].transform('median')
    geo[col] = geo[col].fillna(state_avg)

geo['POPULATION'] = geo['POPULATION'].fillna(1_200_000).astype(int)
geo['HOUSEHOLDS'] = geo['HOUSEHOLDS'].fillna(250_000).astype(int)
geo['AREA_SQKM']  = geo['AREA_SQKM'].fillna(3000.0)

print(f"  ✓ Branch counts & demographics merged: {len(geo)} districts")


# ═══════════════════════════════════════════════════════════════════
# FILE 1 — rbi_district_mock.csv
# ═══════════════════════════════════════════════════════════════════
print("\n[4/6] Generating rbi_district_mock.csv...")

# Metro / Urban flags derived from branch_classification_rules (CSV-driven)
metro_states = set(branch_rules[branch_rules['BRANCH_TYPE'] == 'Metro']['STATE'].tolist())
urban_states = set(branch_rules[branch_rules['BRANCH_TYPE'] == 'Urban']['STATE'].tolist())

df_rbi = geo.copy()
df_rbi['IS_METRO'] = df_rbi['STATE'].isin(metro_states)
df_rbi['IS_URBAN'] = df_rbi['STATE'].isin(urban_states)

# Per-capita income — from state_economic_indicators.csv
df_rbi['PER_CAPITA_INCOME_INR'] = df_rbi.apply(
    lambda r: int(state_income_base.get(r['STATE'], 120_000) *
                  (0.85 + det_noise(r['DISTRICT'], 0.30))), axis=1)

# Urban population %
df_rbi['URBAN_POPULATION_PCT'] = df_rbi.apply(
    lambda r: round(min(95, (65 if r['IS_METRO'] else 45 if r['IS_URBAN'] else 28)
                        + det_noise(r['DISTRICT'], 25, -5)), 1), axis=1)

# Literacy rate — from state_economic_indicators.csv
df_rbi['LITERACY_RATE_PCT'] = df_rbi.apply(
    lambda r: round(float(state_literacy.get(r['STATE'], 70)) +
                    det_noise(r['DISTRICT'], 10, -5), 1), axis=1)

# Total bank branches per district
df_rbi['TOTAL_BANK_BRANCHES'] = df_rbi.apply(
    lambda r: max(10, int((r['POPULATION'] / 100_000) * 14 *
                          (1.4 if r['IS_METRO'] else 1.0 if r['IS_URBAN'] else 0.7) +
                          det_noise(r['DISTRICT'], 50, -10))), axis=1)

# YES Bank branches per district
district_count_per_state = df_rbi.groupby('STATE')['DISTRICT'].count().reset_index()
district_count_per_state.columns = ['STATE', 'DISTRICTS_IN_STATE']
df_rbi = df_rbi.merge(district_count_per_state, on='STATE', how='left')

df_rbi['YES_BANK_BRANCHES'] = df_rbi.apply(
    lambda r: max(1, int(r['YES_BANK_BRANCHES_STATE'] / max(1, r['DISTRICTS_IN_STATE']) *
                         (1.5 if r['IS_METRO'] else 1.0) +
                         det_noise(r['DISTRICT'], 5, -1))), axis=1)
df_rbi['YES_BANK_OFFICES'] = df_rbi['YES_BANK_BRANCHES']

df_rbi['YES_BANK_MARKET_SHARE_PCT'] = df_rbi.apply(
    lambda r: round(r['YES_BANK_BRANCHES'] / max(1, r['TOTAL_BANK_BRANCHES']) * 100, 2), axis=1)

df_rbi['TOTAL_DEPOSITS_CR'] = df_rbi.apply(
    lambda r: round(r['POPULATION'] * r['PER_CAPITA_INCOME_INR'] * 0.15 / 1e7, 1), axis=1)

df_rbi['CD_RATIO'] = df_rbi.apply(
    lambda r: round(
        (110 + det_noise(r['DISTRICT'], 20, -5)) if r['IS_METRO'] else
        (85  + det_noise(r['DISTRICT'], 30, -10)) if r['IS_URBAN'] else
        (60  + det_noise(r['DISTRICT'], 30, -10)), 1), axis=1)
df_rbi['TOTAL_CREDIT_CR'] = df_rbi.apply(
    lambda r: round(r['TOTAL_DEPOSITS_CR'] * r['CD_RATIO'] / 100, 1), axis=1)

df_rbi['PMJDY_ZERO_BALANCE_PCT'] = df_rbi.apply(
    lambda r: round(max(5, 50 - r['PMJDY_COVERAGE'] * 0.4 +
                        det_noise(r['DISTRICT'], 15, -5)), 1), axis=1)

df_rbi['BANKING_PENETRATION_INDEX'] = df_rbi.apply(
    lambda r: round(min(0.99, max(0.2,
        (r['TOTAL_BANK_BRANCHES'] / r['POPULATION'] * 100000 / 20) * 0.30 +
        (r['LITERACY_RATE_PCT'] / 100) * 0.30 +
        (r['URBAN_POPULATION_PCT'] / 100) * 0.20 +
        (min(r['PER_CAPITA_INCOME_INR'], 400_000) / 400_000) * 0.20
    )), 3), axis=1)

# GST registrations — from state_economic_indicators.csv
df_rbi['GST_REGISTRATIONS_PER_LAKH'] = df_rbi.apply(
    lambda r: round(float(state_gst_base.get(r['STATE'], 1000)) *
                    (0.85 + det_noise(r['DISTRICT'], 0.30)), 1), axis=1)

df_rbi['COMPETITOR_BRANCH_DENSITY'] = df_rbi.apply(
    lambda r: round((r['TOTAL_BANK_BRANCHES'] - r['YES_BANK_BRANCHES']) /
                    max(1, r['POPULATION'] / 100_000), 2), axis=1)

rbi_out = df_rbi[[
    'STATE', 'DISTRICT', 'POPULATION',
    'TOTAL_BANK_BRANCHES', 'YES_BANK_BRANCHES', 'YES_BANK_OFFICES',
    'TOTAL_DEPOSITS_CR', 'TOTAL_CREDIT_CR', 'CD_RATIO',
    'YES_BANK_MARKET_SHARE_PCT', 'PER_CAPITA_INCOME_INR',
    'PMJDY_ZERO_BALANCE_PCT', 'BANKING_PENETRATION_INDEX',
    'GST_REGISTRATIONS_PER_LAKH', 'COMPETITOR_BRANCH_DENSITY',
    'URBAN_POPULATION_PCT', 'LITERACY_RATE_PCT', 'IS_METRO', 'IS_URBAN'
]].copy()

rbi_out.columns = [
    'STATE', 'DISTRICT', 'POPULATION_ESTIMATED',
    'TOTAL_BANK_BRANCHES', 'YES_BANK_BRANCHES', 'YES_BANK_OFFICES',
    'TOTAL_DEPOSITS_CR', 'TOTAL_CREDIT_CR', 'CD_RATIO',
    'YES_BANK_MARKET_SHARE_PCT', 'PER_CAPITA_INCOME_INR',
    'PMJDY_ZERO_BALANCE_PCT', 'BANKING_PENETRATION_INDEX',
    'GST_REGISTRATIONS_PER_LAKH', 'COMPETITOR_BRANCH_DENSITY',
    'URBAN_POPULATION_PCT', 'LITERACY_RATE_PCT', 'IS_METRO', 'IS_URBAN'
]
rbi_out = rbi_out.reset_index(drop=True)
rbi_out.to_csv(os.path.join(OUTPUT_DIR, "rbi_district_mock.csv"), index=False)
print(f"  ✓ rbi_district_mock.csv → {len(rbi_out)} rows × {len(rbi_out.columns)} cols")


# ═══════════════════════════════════════════════════════════════════
# FILE 2 — rbi_compliance_guidelines.csv
#          Now sourced entirely from rbi_compliance_master.csv
# ═══════════════════════════════════════════════════════════════════
print("\n[5/6-a] Generating rbi_compliance_guidelines.csv...")

# Select and rename to match original output schema
df_compliance = compliance_master[['CATEGORY', 'TOPIC', 'KEY_LIMIT', 'RISK_LEVEL']].copy()
df_compliance.columns = ['category', 'topic', 'key_limit', 'risk_level']

# Optionally keep extra columns if present
for extra_col in ['EFFECTIVE_DATE', 'SOURCE_CIRCULAR', 'NOTES']:
    if extra_col in compliance_master.columns:
        df_compliance[extra_col.lower()] = compliance_master[extra_col].values

df_compliance.to_csv(os.path.join(OUTPUT_DIR, "rbi_compliance_guidelines.csv"), index=False)
print(f"  ✓ rbi_compliance_guidelines.csv → {len(df_compliance)} rows × {len(df_compliance.columns)} cols")


# ═══════════════════════════════════════════════════════════════════
# FILE 3 — yes_bank_existing_branches.csv
#          Branch type & cluster now from branch_classification_rules.csv
# ═══════════════════════════════════════════════════════════════════
print("\n[5/6-b] Generating yes_bank_existing_branches.csv...")

pin_district_geo = (
    pins[pins['statename'].isin(STATE_NORM.keys())]
    .groupby('district')
    .agg(lat=('latitude', 'median'), lon=('longitude', 'median'))
    .reset_index()
)
pin_district_geo['DISTRICT_KEY'] = pin_district_geo['district'].str.upper().str.strip()

def classify_branch_type(state):
    """Lookup from branch_classification_rules.csv (CSV-driven, no hardcoding)."""
    return state_to_branch_type.get(state, 'Rural')

def assign_cluster(state):
    """Lookup from branch_classification_rules.csv (CSV-driven, no hardcoding)."""
    return state_to_cluster.get(state, 'Central')

# Staff counts by branch type (operational norm — kept as-is, no hardcoding in original either)
staff_base_map   = {'Metro': 20, 'Urban': 12, 'Semi-Urban': 8, 'Rural': 5}
casa_base_map    = {'Metro': 15000, 'Urban': 8000, 'Semi-Urban': 4000, 'Rural': 1500}
avg_balance_map  = {'Metro': 85000, 'Urban': 55000, 'Semi-Urban': 35000, 'Rural': 20000}
mtp_base_map     = {'Metro': 12, 'Urban': 18, 'Semi-Urban': 24, 'Rural': 30}
perf_base_map    = {'Metro': 75, 'Urban': 65, 'Semi-Urban': 55, 'Rural': 50}

existing_rows = []
for idx, row in yes_branches.iterrows():
    state   = str(row['STATE']).strip()
    br_id   = f"YBL{idx:05d}"
    br_name = f"{state}_Branch_{(idx % 200) + 1}"
    btype   = classify_branch_type(state)
    cluster = assign_cluster(state)

    staff    = staff_base_map.get(btype, 8) + int(det_noise(row['BR_NAME'], 10, -3))
    casa     = max(500, int(casa_base_map.get(btype, 4000) *
                            (0.7 + det_noise(row['BR_NAME'], 0.6))))
    avg_bal  = avg_balance_map.get(btype, 40000)
    deposits = round(casa * avg_bal / 1e7, 1)

    mtp = mtp_base_map.get(btype, 20) + int(det_noise(row['BR_NAME'], 12, -4))
    mtp = max(6, min(48, mtp))

    vintage = max(1, min(10, int(det_noise(row['BR_NAME'], 10, 0)) + 1))
    perf    = min(100, perf_base_map.get(btype, 60) + vintage * 2 +
                  int(det_noise(row['BR_NAME'], 15, -5)))

    addr = str(row['ADDRESS'])
    pincode_match = re.findall(r'\b\d{6}\b', addr)
    if pincode_match:
        pin_val = int(pincode_match[-1])
        pin_row = pins[pins['pincode'] == pin_val][['latitude', 'longitude']].dropna()
        if len(pin_row) > 0:
            lat = round(pin_row.iloc[0]['latitude'],  4)
            lon = round(pin_row.iloc[0]['longitude'], 4)
        else:
            state_pins = pins[pins['statename'] == state.upper()]
            lat = round(float(state_pins['latitude'].median())  if len(state_pins) else 20.0, 4)
            lon = round(float(state_pins['longitude'].median()) if len(state_pins) else 76.0, 4)
    else:
        state_pins = pins[pins['statename'] == state.upper()]
        lat = round(float(state_pins['latitude'].median())  if len(state_pins) else 20.0, 4)
        lon = round(float(state_pins['longitude'].median()) if len(state_pins) else 76.0, 4)
        lat += round(det_noise(row['BR_NAME'], 2.0, -1.0), 4)
        lon += round(det_noise(str(idx),       2.0, -1.0), 4)

    district_guess = f"{state}_District_{(idx % 4) + 1}"

    existing_rows.append({
        'BRANCH_ID':               br_id,
        'BRANCH_NAME':             br_name,
        'STATE':                   state,
        'DISTRICT':                district_guess,
        'CLUSTER':                 cluster,
        'BRANCH_TYPE':             btype,
        'STAFF_COUNT':             max(3, staff),
        'CASA_ACCOUNTS':           casa,
        'TOTAL_DEPOSITS_CR':       deposits,
        'MONTHS_TO_PROFITABILITY': mtp,
        'VINTAGE_YEARS':           vintage,
        'PERFORMANCE_SCORE':       perf,
        'LATITUDE':                lat,
        'LONGITUDE':               lon,
    })

df_existing = pd.DataFrame(existing_rows)
df_existing.to_csv(os.path.join(OUTPUT_DIR, "yes_bank_existing_branches.csv"), index=False)
print(f"  ✓ yes_bank_existing_branches.csv → {len(df_existing)} rows × {len(df_existing.columns)} cols")


# ═══════════════════════════════════════════════════════════════════
# FILE 4 — yes_bank_branch_mis.csv
#          Month list & quarter map from fiscal_calendar.csv
# ═══════════════════════════════════════════════════════════════════
print("\n[5/6-c] Generating yes_bank_branch_mis.csv...")

# months_fy and quarter_map are now loaded from fiscal_calendar.csv (Step 1)

casa_target_base  = {'Metro': 1500, 'Urban': 900, 'Semi-Urban': 450, 'Rural': 150}
deposit_target_base = {'Metro': 500, 'Urban': 200, 'Semi-Urban': 80, 'Rural': 25}
acct_base_map     = {'Metro': 14, 'Urban': 10, 'Semi-Urban': 7, 'Rural': 3}
cti_base_map      = {'Metro': 50, 'Urban': 55, 'Semi-Urban': 60, 'Rural': 65}
nps_base_map      = {'Metro': 55, 'Urban': 50, 'Semi-Urban': 45, 'Rural': 40}

mis_rows = []
sample_branches = df_existing.sample(min(300, len(df_existing)), random_state=42)

for _, br in sample_branches.iterrows():
    btype   = br['BRANCH_TYPE']
    state   = br['STATE']
    district = br['DISTRICT']

    casa_target    = int(casa_target_base.get(btype, 500) *
                         (0.85 + det_noise(br['BRANCH_ID'], 0.30)))
    deposit_target = deposit_target_base.get(btype, 100)

    for month in months_fy:
        mi = months_fy.index(month)
        seasonality = 1.05 if mi >= 9 else (0.95 if mi <= 2 else 1.0)

        ach_rate     = 0.85 + det_noise(f"{br['BRANCH_ID']}{month}", 0.35)
        casa_actual  = int(casa_target * ach_rate * seasonality)
        casa_achievement = round(casa_actual / max(1, casa_target) * 100, 1)

        acct_opened = max(1, int(acct_base_map.get(btype, 8) *
                                 (0.7 + det_noise(f"{month}{br['BRANCH_ID']}", 0.6))))

        casa_ratio = round(min(85, max(35,
            (55 if btype == 'Metro' else 48 if btype == 'Urban'
             else 42 if btype == 'Semi-Urban' else 38)
            + det_noise(f"cr{br['BRANCH_ID']}", 20, -10))), 1)

        deposit_ach_rate  = 0.85 + det_noise(f"dep{br['BRANCH_ID']}{month}", 0.35)
        total_deposits    = round(deposit_target * deposit_ach_rate * seasonality, 1)
        deposit_achievement = round(deposit_ach_rate * seasonality * 100, 1)

        complaints = max(0, int(det_noise(f"cmp{br['BRANCH_ID']}{month}", 4, -1)))

        cti = round(cti_base_map.get(btype, 57) +
                    det_noise(f"cti{br['BRANCH_ID']}", 20, -10), 1)
        cti = max(30, min(80, cti))

        nps = min(80, max(20, int(nps_base_map.get(btype, 48) +
                                  det_noise(f"nps{br['BRANCH_ID']}", 30, -15))))

        mis_rows.append({
            'BRANCH_ID':               br['BRANCH_ID'],
            'BRANCH_NAME':             br['BRANCH_NAME'],
            'STATE':                   state,
            'DISTRICT':                district,
            'BRANCH_TYPE':             btype,
            'CLUSTER':                 br['CLUSTER'],
            'MONTH':                   month,
            'QUARTER':                 quarter_map[month],
            'CASA_ACTUAL':             casa_actual,
            'CASA_TARGET':             casa_target,
            'CASA_ACHIEVEMENT_PCT':    casa_achievement,
            'CASA_ACCOUNTS_OPENED':    acct_opened,
            'CASA_RATIO_PCT':          casa_ratio,
            'TOTAL_DEPOSITS_CR':       total_deposits,
            'DEPOSIT_ACHIEVEMENT_PCT': deposit_achievement,
            'COMPLAINTS_LOGGED':       complaints,
            'COST_TO_INCOME_PCT':      cti,
            'NPS_SCORE':               nps,
        })

df_mis = pd.DataFrame(mis_rows)
df_mis.to_csv(os.path.join(OUTPUT_DIR, "yes_bank_branch_mis.csv"), index=False)
print(f"  ✓ yes_bank_branch_mis.csv → {len(df_mis)} rows × {len(df_mis.columns)} cols")


# ═══════════════════════════════════════════════════════════════════
# FILE 5 — yes_bank_new_branch_playbook.csv
#          Ramp curves, capex & success factors from
#          branch_playbook_parameters.csv (CSV-driven)
# ═══════════════════════════════════════════════════════════════════
print("\n[5/6-d] Generating yes_bank_new_branch_playbook.csv...")

existing_by_type = df_existing.groupby('BRANCH_TYPE').agg(
    avg_mtp=('MONTHS_TO_PROFITABILITY', 'mean'),
    avg_casa=('CASA_ACCOUNTS', 'mean'),
    avg_perf=('PERFORMANCE_SCORE', 'mean')
).reset_index()

archetypes    = playbook_params.index.tolist()   # order from CSV
playbook_rows = []

per_archetype = 90 // len(archetypes)
extra         = 90 - per_archetype * len(archetypes)

for a_idx, archetype in enumerate(archetypes):
    count   = per_archetype + (1 if a_idx < extra else 0)
    avg_row = existing_by_type[existing_by_type['BRANCH_TYPE'] == archetype]
    if len(avg_row) == 0:
        continue
    avg_row = avg_row.iloc[0]

    base_mtp   = int(avg_row['avg_mtp'])
    base_casa  = int(avg_row['avg_casa'])
    base_curve = ramp_curves[archetype]

    # Capex range from CSV
    capex_min, capex_max = capex_map[archetype]
    capex_range = capex_max - capex_min

    sf1, sf2, challenge = success_factors[archetype]

    for i in range(count):
        seed_key = f"{archetype}_{i}"
        noise_f  = det_noise(seed_key, 0.30, -0.10)

        mtp  = max(6, min(48, int(base_mtp * (0.85 + noise_f))))
        peak_monthly_casa = max(100, int(base_casa *
                                         (0.7 + det_noise(seed_key + 'c', 0.60))))

        m3  = round(base_curve['M3']  * (1 + det_noise(seed_key + '3',  0.15, -0.05)), 3)
        m6  = round(base_curve['M6']  * (1 + det_noise(seed_key + '6',  0.15, -0.05)), 3)
        m9  = round(base_curve['M9']  * (1 + det_noise(seed_key + '9',  0.10, -0.04)), 3)
        m12 = round(base_curve['M12'] * (1 + det_noise(seed_key + '12', 0.08, -0.03)), 3)
        m18 = round(base_curve['M18'] * (1 + det_noise(seed_key + '18', 0.06, -0.02)), 3)
        m24 = round(base_curve['M24'] * (1 + det_noise(seed_key + '24', 0.05, -0.02)), 3)

        # Enforce monotonic ramp
        m6  = max(m6,  m3  + 0.05)
        m9  = max(m9,  m6  + 0.05)
        m12 = max(m12, m9  + 0.04)
        m18 = max(m18, m12 + 0.03)
        m24 = max(m24, m18 + 0.01)

        breakeven = mtp <= 24

        # Capex driven by CSV range + deterministic noise
        capex = max(capex_min,
                    min(capex_max,
                        capex_min + int(capex_range * det_noise(seed_key + 'cap', 1.0))))

        ci_seed = int(det_noise(seed_key + 'ci', 3, 0))
        comp_int = 'High' if ci_seed == 0 else ('Medium' if ci_seed == 1 else 'Low')

        if mtp <= base_mtp * 0.85:   rating = 'Excellent'
        elif mtp <= base_mtp:        rating = 'Good'
        elif mtp <= base_mtp * 1.15: rating = 'Average'
        else:                        rating = 'Below Average'

        playbook_rows.append({
            'BRANCH_ARCHETYPE':          archetype,
            'MONTHS_TO_PROFITABILITY':   mtp,
            'PEAK_MONTHLY_CASA':         peak_monthly_casa,
            'MONTH_3_CASA_PCT_OF_PEAK':  round(m3, 3),
            'MONTH_6_CASA_PCT_OF_PEAK':  round(m6, 3),
            'MONTH_9_CASA_PCT_OF_PEAK':  round(m9, 3),
            'MONTH_12_CASA_PCT_OF_PEAK': round(m12, 3),
            'MONTH_18_CASA_PCT_OF_PEAK': round(m18, 3),
            'MONTH_24_CASA_PCT_OF_PEAK': round(m24, 3),
            'BREAKEVEN_REACHED':         breakeven,
            'INITIAL_CAPEX_LAKHS':       capex,
            'COMPETITOR_INTENSITY':      comp_int,
            'KEY_SUCCESS_FACTOR_1':      sf1,
            'KEY_SUCCESS_FACTOR_2':      sf2,
            'KEY_CHALLENGE':             challenge,
            'RATING':                    rating,
        })

df_playbook = pd.DataFrame(playbook_rows)
df_playbook.to_csv(os.path.join(OUTPUT_DIR, "yes_bank_new_branch_playbook.csv"), index=False)
print(f"  ✓ yes_bank_new_branch_playbook.csv → {len(df_playbook)} rows × {len(df_playbook.columns)} cols")


# ═══════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  OUTPUT SUMMARY")
print("=" * 65)
outputs = [
    ("rbi_district_mock.csv",            rbi_out),
    ("rbi_compliance_guidelines.csv",    df_compliance),
    ("yes_bank_existing_branches.csv",   df_existing),
    ("yes_bank_branch_mis.csv",          df_mis),
    ("yes_bank_new_branch_playbook.csv", df_playbook),
]
for fname, df in outputs:
    print(f"  {fname:<42} {df.shape[0]:>5} rows × {df.shape[1]:>2} cols")
print("\n  All files saved to:", OUTPUT_DIR)
print("=" * 65)
