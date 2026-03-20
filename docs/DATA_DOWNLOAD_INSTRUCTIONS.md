# External Data Download Instructions
## YES Bank AI PoC — Solutions 1, 2 & 4

All data below is FREE, publicly available, and requires no login unless noted.
Download each file and place it in the `data/external/` folder.

---

## DATASET 1: YES Bank Branch Master List
**Used by:** Solution 1 (Location Engine), Solution 4 (Ramp-Up Accelerator)
**File to save as:** `data/external/yes_bank_branches.csv`

### Option A — RBI Official IFSC Master (Most Reliable)
1. Go to: https://www.rbi.org.in/Scripts/IFSCMICRDetails.aspx
2. Select Bank: YES BANK
3. Click Download → Save as CSV
4. OR directly: https://www.paisabazaar.com/ifsc-code/yes-bank/ → scroll to "Download all IFSC Codes"

### Option B — RazorPay IFSC API (Programmatic)
```
https://ifsc.razorpay.com/YESB0000001
```
Full database: https://github.com/razorpay/ifsc/releases → Download `IFSC.csv`
Filter rows where BANK = "YESB"

### Option C — YES Bank Official PDF
URL: https://www.yes.bank.in → About Us → Branch Locator → Download Branch List PDF

**Expected columns:** BRANCH_CODE, BRANCH_NAME, ADDRESS, CITY, DISTRICT, STATE, PINCODE, LATITUDE, LONGITUDE

---

## DATASET 2: RBI District-Level Banking Penetration (BSR Data)
**Used by:** Solution 1 (Location Engine) — core scoring dataset
**File to save as:** `data/external/rbi_bsr_district.xlsx`

### Steps:
1. Go to: https://data.rbi.org.in
2. Navigate: Publications → Statistical Tables Relating to Banks in India → 2023-24
3. Download Table 1.1: "Number of Offices of Scheduled Commercial Banks - State/District Wise"
4. Also download Table 1.2: "Deposits and Credit of Scheduled Commercial Banks - District Wise"

**Alternative direct path:**
- URL: https://dbie.rbi.org.in/DBIE/dbie.rbi?site=statistics
- Section: Banking Statistics → Branch Banking Statistics
- Select: All Banks, All States, Latest Year → Export to Excel

**Expected columns:** STATE, DISTRICT, TOTAL_BANK_OFFICES, TOTAL_DEPOSITS_CR, TOTAL_CREDIT_CR, POPULATION_PER_BRANCH

---

## DATASET 3: India District Population & Demographics (Census)
**Used by:** Solution 1 (Location Engine)
**File to save as:** `data/external/india_district_census.csv`

### Steps:
1. Go to: https://censusindia.gov.in/census.website/data/data-visualizations
2. Download: Primary Census Abstract - District Level
3. OR use the pre-cleaned version from:
   - https://github.com/datameet/india-district-data → districts.csv

**Expected columns:** STATE, DISTRICT, POPULATION_2011, URBAN_POPULATION_PCT, LITERACY_RATE, HOUSEHOLDS, WORKERS_MAIN

---

## DATASET 4: PMJDY Financial Inclusion Data (District-Level)
**Used by:** Solution 1 (Location Engine) — measures banking saturation
**File to save as:** `data/external/pmjdy_district.xlsx`

### Steps:
1. Go to: https://pmjdy.gov.in/statewise-statistics
2. Click: Download → District-wise data
3. Select latest quarter

**Expected columns:** STATE, DISTRICT, TOTAL_ACCOUNTS, TOTAL_DEPOSITS, AADHAAR_SEEDED_PCT, RUPAY_CARDS_ISSUED

---

## DATASET 5: India Pincode Database (For Geo-mapping)
**Used by:** Solution 1 (Location Engine), Solution 2 (Performance Copilot maps)
**File to save as:** `data/external/india_pincodes.csv`

### Steps:
1. Free download from: https://data.gov.in/resource/all-india-pincode-directory
2. OR: https://github.com/vinitshahdeo/pincodes → all-pincodes.csv

**Expected columns:** PINCODE, DISTRICT, STATE, LATITUDE, LONGITUDE, OFFICE_TYPE

---

## DATASET 6: RBI Trend & Progress of Banking Report 2023-24
**Used by:** Solution 2 (Performance Copilot context), Solution 4 (Ramp-Up knowledge base)
**File to save as:** `data/external/rbi_trend_progress_2024.pdf`

### Steps:
1. Go to: https://rbi.org.in/Scripts/AnnualPublications.aspx?head=Trend+and+Progress+of+Banking+in+India
2. Click: 2024 → Download PDF (approx 300 pages)
3. Key sections to use: Chapter II (Banking Sector), Chapter IV (Financial Inclusion)

---

## DATASET 7: RBI FREE-AI Framework Report (For Solution 4 Knowledge Base)
**Used by:** Solution 4 (Ramp-Up knowledge base — compliance content)
**File to save as:** `data/external/rbi_free_ai_framework.pdf`

### Steps:
1. Go to: https://www.rbi.org.in/Scripts/PublicationReportDetails.aspx?ID=1415
2. Download the August 2025 FREE-AI Framework PDF

---

## DATASET 8: Competitor Bank Branch Data (For Location Engine)
**Used by:** Solution 1 (competitive gap analysis)
**File to save as:** `data/external/competitor_branches.csv`

### Steps:
1. Go to: https://github.com/razorpay/ifsc/releases → IFSC.csv
2. Filter for: HDFC, ICICI, AXIS, KOTAK, SBI, INDUSIND
3. Keep columns: BANK, BRANCH, ADDRESS, DISTRICT, STATE, PINCODE

This single file gives you all major competitor branch locations for gap analysis.

---

## DATASET 9: YES Bank Annual Report 2023-24 (For Mocked Data Baseline)
**Used by:** Solution 2 (calibrating mock performance data)
**File to save as:** `data/external/yes_bank_annual_report_2024.pdf`

### Steps:
1. Go to: https://www.yesbank.in/about-us/investor-relations/annual-reports
2. Download FY 2023-24 Annual Report

---

## QUICK PRIORITY ORDER
If time is limited, download in this order:

| Priority | Dataset | Time to Download | Impact |
|----------|---------|-----------------|--------|
| 1 | Razorpay IFSC.csv (YES Bank + Competitors) | 5 min | S1 Core |
| 2 | RBI BSR District Data | 10 min | S1 Core |
| 3 | India Pincode DB | 5 min | S1 + S2 |
| 4 | PMJDY District Data | 10 min | S1 Enhancement |
| 5 | Census District Data | 5 min | S1 Enhancement |
| 6 | YES Bank Annual Report PDF | 2 min | S2 + S4 |
| 7 | RBI Trend & Progress PDF | 2 min | S4 Knowledge Base |
| 8 | RBI FREE-AI PDF | 2 min | S4 Knowledge Base |

---
*Once downloaded, place files in `data/external/` and run `python setup_and_validate.py` to validate all datasets.*
