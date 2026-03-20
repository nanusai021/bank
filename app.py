"""
YES Bank Branch Intelligence Suite — Flask Backend
Reads all CSV files and serves dynamic Jinja2 template.
Includes /api/chat endpoint with GROQ or OLLAMA support.

─── LLM CONFIGURATION ────────────────────────────────────────
 Set LLM_PROVIDER to "groq" or "ollama" below.

 GROQ  → pip install groq
         Get free key: https://console.groq.com
         Set env:  export GROQ_API_KEY="gsk_..."

 OLLAMA → Install from https://ollama.com
          Run: ollama pull llama3
          No API key needed — runs locally on port 11434
──────────────────────────────────────────────────────────────
"""

import json
import os
import requests
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ─────────────────────────────────────────────────────────
# ★ LLM CONFIGURATION — change this to switch providers ★
# ─────────────────────────────────────────────────────────
LLM_PROVIDER = "groq"          # "groq" | "ollama"

# Groq settings
GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")   # set env: export GROQ_API_KEY="gsk_..."
GROQ_MODEL    = "llama3-8b-8192"     # fast & free; alternatives: mixtral-8x7b-32768, llama3-70b-8192

# Ollama settings (local)
OLLAMA_HOST   = "http://localhost:11434"
OLLAMA_MODEL  = "llama3"             # run: ollama pull llama3  (or mistral, gemma, phi3, etc.)

# ─────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")

CSV = {
    "mis":         os.path.join(DATA, "output_files/yes_bank_branch_mis.csv"),
    "existing":    os.path.join(DATA, "output_files/yes_bank_existing_branches.csv"),
    "playbook":    os.path.join(DATA, "output_files/yes_bank_new_branch_playbook.csv"),
    "compliance":  os.path.join(DATA, "output_files/rbi_compliance_guidelines.csv"),
    "districts":   os.path.join(DATA, "output_files/rbi_district_mock.csv"),
    # ── NEW: 3 Huddle Engine CSV files ────────────────────────────────────────
    "huddle_log":   os.path.join(DATA, "output_files/huddle_daily_log.csv"),
    "huddle_staff": os.path.join(DATA, "output_files/huddle_staff_performance.csv"),
    "huddle_block": os.path.join(DATA, "output_files/huddle_blockers.csv"),
}


# ─────────────────────────────────────────────────────────
# DATA LOADERS
# ─────────────────────────────────────────────────────────
def load_data():
    mis        = pd.read_csv(CSV["mis"])
    existing   = pd.read_csv(CSV["existing"])
    playbook   = pd.read_csv(CSV["playbook"])
    compliance = pd.read_csv(CSV["compliance"])
    districts  = pd.read_csv(CSV["districts"])
    # ── NEW: 3 real huddle CSV files ──────────────────────────────────────────
    dl = pd.read_csv(CSV["huddle_log"],   parse_dates=["DATE"])
    sp = pd.read_csv(CSV["huddle_staff"], parse_dates=["DATE"])
    bl = pd.read_csv(CSV["huddle_block"], parse_dates=["DATE"])
    return mis, existing, playbook, compliance, districts, dl, sp, bl


# ─────────────────────────────────────────────────────────
# VIABILITY SCORE
# ─────────────────────────────────────────────────────────
def compute_viability_score(dist_df):
    d = dist_df.copy()
    def minmax(s):
        mn, mx = s.min(), s.max()
        return (s - mn) / (mx - mn + 1e-9)
    d["f_market"]    = minmax(d["POPULATION_ESTIMATED"] * d["PER_CAPITA_INCOME_INR"])
    d["f_gap"]       = minmax(100 - d["YES_BANK_MARKET_SHARE_PCT"])
    d["f_business"]  = minmax(d["GST_REGISTRATIONS_PER_LAKH"] + d["CD_RATIO"])
    d["f_inclusion"] = minmax(d["PMJDY_ZERO_BALANCE_PCT"])
    d["f_catchment"] = minmax(d["URBAN_POPULATION_PCT"] * d["LITERACY_RATE_PCT"])
    d["f_comp"]      = minmax(1 / (d["COMPETITOR_BRANCH_DENSITY"] + 1))
    d["VIABILITY_SCORE"] = (
        d["f_market"] * 25 + d["f_gap"] * 20 + d["f_business"] * 20 +
        d["f_inclusion"] * 15 + d["f_catchment"] * 10 + d["f_comp"] * 10
    ).round(1)
    p95 = d["VIABILITY_SCORE"].quantile(0.95)
    p60 = d["VIABILITY_SCORE"].quantile(0.60)
    p30 = d["VIABILITY_SCORE"].quantile(0.30)
    def tier(s):
        if s >= p95: return "Top Priority"
        elif s >= p60: return "Recommended"
        elif s >= p30: return "Watch List"
        return "Low Priority"
    d["TIER"] = d["VIABILITY_SCORE"].apply(tier)
    d["M2P"] = (35 - d["VIABILITY_SCORE"] * 0.25).clip(10, 35).round(0).astype(int)
    d["CASA_12M"] = (d["POPULATION_ESTIMATED"] * 0.003 * (d["VIABILITY_SCORE"] / 100) * 1.5).clip(500, 50000).round(0).astype(int)
    d["GAP_PCT"] = (100 - d["YES_BANK_MARKET_SHARE_PCT"] * 10).clip(10, 90).round(0).astype(int)
    d = d.sort_values("VIABILITY_SCORE", ascending=False).reset_index(drop=True)
    d["RANK"] = d.index + 1
    return d


def build_opportunities(dist_df, existing_df):
    scored = compute_viability_score(dist_df)
    presence = existing_df.groupby("DISTRICT").size().reset_index(name="YB_PRESENCE")
    scored = scored.merge(presence, on="DISTRICT", how="left")
    scored["YB_PRESENCE"] = scored["YB_PRESENCE"].fillna(0).astype(int)
    result = []
    for _, row in scored.head(9).iterrows():
        result.append({"rank":int(row["RANK"]),"tier":row["TIER"],"district":row["DISTRICT"],"state":row["STATE"],"score":float(row["VIABILITY_SCORE"]),"m2p":int(row["M2P"]),"casa12m":int(row["CASA_12M"]),"gap":int(row["GAP_PCT"]),"ybPresence":int(row["YB_PRESENCE"])})
    return result


def build_cluster_data(mis_df):
    q4 = mis_df[mis_df["QUARTER"] == "Q4"].copy()
    agg = q4.groupby("CLUSTER").agg(branches=("BRANCH_ID","nunique"),casa_achv=("CASA_ACHIEVEMENT_PCT","mean"),total_casa=("CASA_ACCOUNTS_OPENED","sum"),deposits=("TOTAL_DEPOSITS_CR","mean"),casa_ratio=("CASA_RATIO_PCT","mean"),nps=("NPS_SCORE","mean")).reset_index()
    agg = agg.sort_values("casa_achv", ascending=False).reset_index(drop=True)
    return [{"cluster":row["CLUSTER"],"branches":int(row["branches"]),"casa":round(float(row["casa_achv"]),1),"totalCasa":int(row["total_casa"]),"deposits":round(float(row["deposits"]),1),"casaRatio":round(float(row["casa_ratio"]),1),"nps":round(float(row["nps"]),1)} for _,row in agg.iterrows()]



LAUNCH_SEQUENCE = {
    "Metro_FinanceDistrict": [
        {"period": "Week 1–2",  "bg": "#f0f4ff", "color": "#003087", "items": ["Corporate salary account drives", "Premium KYC + wealth onboarding"]},
        {"period": "Week 3–4",  "bg": "#f0f4ff", "color": "#003087", "items": ["Fixed deposit campaigns", "Credit card cross-sell to HNI"]},
        {"period": "Month 2",   "bg": "#e8f5f0", "color": "#1a7f5a", "items": ["Insurance + Demat opening", "Priority banking activation"]},
        {"period": "Month 3+",  "bg": "#e8f5f0", "color": "#1a7f5a", "items": ["MSME & trade finance", "Locker + NRI services"]},
    ],
    "Metro_Residential": [
        {"period": "Week 1–2",  "bg": "#f0f4ff", "color": "#003087", "items": ["CASA camps + KYC drives", "Salary account partnerships"]},
        {"period": "Week 3–4",  "bg": "#f0f4ff", "color": "#003087", "items": ["Home loan pre-approvals", "FD + RD campaigns"]},
        {"period": "Month 2",   "bg": "#e8f5f0", "color": "#1a7f5a", "items": ["Insurance + YES PAY UPI", "Demat account activation"]},
        {"period": "Month 3+",  "bg": "#e8f5f0", "color": "#1a7f5a", "items": ["MSME business accounts", "Locker availability"]},
    ],
    "Urban_Commercial": [
        {"period": "Week 1–2",  "bg": "#f0f4ff", "color": "#003087", "items": ["Digital onboarding + UPI activation", "CASA + GST-linked accounts"]},
        {"period": "Week 3–4",  "bg": "#f0f4ff", "color": "#003087", "items": ["FD campaign + trade credit", "Business current accounts"]},
        {"period": "Month 2",   "bg": "#e8f5f0", "color": "#1a7f5a", "items": ["Insurance + loan cross-sell", "Digital banking suite rollout"]},
        {"period": "Month 3+",  "bg": "#e8f5f0", "color": "#1a7f5a", "items": ["MSME credit + OD facility", "Supply-chain finance"]},
    ],
    "Urban_Residential": [
        {"period": "Week 1–2",  "bg": "#f0f4ff", "color": "#003087", "items": ["CASA + KYC camps", "Salary account drives"]},
        {"period": "Week 3–4",  "bg": "#f0f4ff", "color": "#003087", "items": ["FD campaign", "Credit card cross-sell"]},
        {"period": "Month 2",   "bg": "#e8f5f0", "color": "#1a7f5a", "items": ["Insurance + Demat", "YES PAY UPI activation"]},
        {"period": "Month 3+",  "bg": "#e8f5f0", "color": "#1a7f5a", "items": ["MSME business accounts", "Locker availability"]},
    ],
    "SemiUrban_Market": [
        {"period": "Week 1–2",  "bg": "#f0f4ff", "color": "#003087", "items": ["PMJDY + Jan Dhan CASA drives", "Farmer + agri account camps"]},
        {"period": "Week 3–4",  "bg": "#f0f4ff", "color": "#003087", "items": ["Fixed deposit for local traders", "Self-help group linkage"]},
        {"period": "Month 2",   "bg": "#e8f5f0", "color": "#1a7f5a", "items": ["Insurance + micro-loan launch", "YES PAY & mBanking activation"]},
        {"period": "Month 3+",  "bg": "#e8f5f0", "color": "#1a7f5a", "items": ["Kisan credit card", "MSME + agri working capital"]},
    ],
    "Rural_Agri": [
        {"period": "Week 1–2",  "bg": "#f0f4ff", "color": "#003087", "items": ["PMJDY zero-balance accounts", "Basic savings + KYC camps"]},
        {"period": "Week 3–4",  "bg": "#f0f4ff", "color": "#003087", "items": ["Agri deposit drives", "BC (Business Correspondent) setup"]},
        {"period": "Month 2",   "bg": "#e8f5f0", "color": "#1a7f5a", "items": ["Crop insurance + Fasal Bima", "Agri credit + KCC launch"]},
        {"period": "Month 3+",  "bg": "#e8f5f0", "color": "#1a7f5a", "items": ["SHG linkage + micro-finance", "Rural DBT disbursement"]},
    ],
}
DEFAULT_LAUNCH = LAUNCH_SEQUENCE["Urban_Residential"]
def build_briefs(playbook_df, dist_df, existing_df):
    scored = compute_viability_score(dist_df)
    top3 = scored.head(3).copy()
    pb_avg = playbook_df.groupby("BRANCH_ARCHETYPE").agg(avg_m2p=("MONTHS_TO_PROFITABILITY","mean"),avg_peak=("PEAK_MONTHLY_CASA","mean"),avg_m3pct=("MONTH_3_CASA_PCT_OF_PEAK","mean"),avg_m6pct=("MONTH_6_CASA_PCT_OF_PEAK","mean"),avg_m12pct=("MONTH_12_CASA_PCT_OF_PEAK","mean"),avg_capex=("INITIAL_CAPEX_LAKHS","mean"),hist_count=("BRANCH_ARCHETYPE","count"),factors=("KEY_SUCCESS_FACTOR_1",lambda x:list(x.unique()[:3])),factors2=("KEY_SUCCESS_FACTOR_2",lambda x:list(x.unique()[:2])),challenges=("KEY_CHALLENGE",lambda x:list(x.unique()[:2])),success_rt=("BREAKEVEN_REACHED",lambda x:round(x.map({"True":1,True:1,"False":0,False:0}).mean()*100))).reset_index()
    def pick_arch(row):
        if row["IS_METRO"] in (True,"True"): return "Metro_FinanceDistrict" if row["CD_RATIO"]>100 else "Metro_Residential"
        elif row["IS_URBAN"] in (True,"True"): return "Urban_Commercial" if row["GST_REGISTRATIONS_PER_LAKH"]>1000 else "Urban_Residential"
        elif row["URBAN_POPULATION_PCT"]>30: return "SemiUrban_Market"
        return "Rural_Agri"
    top3["ARCHETYPE"] = top3.apply(pick_arch, axis=1)
    pres = existing_df.groupby("DISTRICT").size().reset_index(name="YB_PRESENCE")
    top3 = top3.merge(pres, on="DISTRICT", how="left")
    top3["YB_PRESENCE"] = top3["YB_PRESENCE"].fillna(0).astype(int)
    briefs = []
    for _, row in top3.iterrows():
        pb_row = pb_avg[pb_avg["BRANCH_ARCHETYPE"] == row["ARCHETYPE"]]
        if pb_row.empty: pb_row = pb_avg.iloc[0:1]
        pb = pb_row.iloc[0]
        peak = int(pb["avg_peak"]); m3pct=int(pb["avg_m3pct"]*100); m6pct=int(pb["avg_m6pct"]*100); m12pct=int(pb["avg_m12pct"]*100)
        briefs.append({"location":f"{row['DISTRICT']}, {row['STATE']}","archetype":row["ARCHETYPE"].replace("_"," "),"score":float(row["VIABILITY_SCORE"]),"tier":row["TIER"],"m2p":int(round(pb["avg_m2p"])),"casa12m":int(row["CASA_12M"]),"ybPresence":int(row["YB_PRESENCE"]),"gap":int(row["GAP_PCT"]),"capex":int(round(pb["avg_capex"])),"peakCasa":peak,"m3":int(peak*m3pct/100),"m6":int(peak*m6pct/100),"m12":int(peak*m12pct/100),"m3pct":m3pct,"m6pct":m6pct,"m12pct":m12pct,"historicalBranches":int(pb["hist_count"]),"successRate":int(pb["success_rt"]),"successFactors":list(pb["factors"])[:2]+list(pb["factors2"])[:1],"challenges":list(pb["challenges"])[:2],"pcIncome":int(row["PER_CAPITA_INCOME_INR"]),"launchSequence":LAUNCH_SEQUENCE.get(row["ARCHETYPE"], DEFAULT_LAUNCH)})
    return briefs


def build_summary(mis_df, existing_df, playbook_df, dist_df):
    scored = compute_viability_score(dist_df)
    q4 = mis_df[mis_df["QUARTER"] == "Q4"]
    type_counts = existing_df["BRANCH_TYPE"].value_counts(); total = len(existing_df)
    colors = {"Metro":"#003087","Urban":"#00a0b0","Semi-Urban":"#c9a84c","Rural":"#1a7f5a"}
    return {"network_branches":total,"states_covered":int(existing_df["STATE"].nunique()),"avg_casa_branch":int(existing_df["CASA_ACCOUNTS"].mean()),"cost_to_income":round(float(mis_df["COST_TO_INCOME_PCT"].mean()),1),"districts_scored":int(len(dist_df)),"mis_records":int(len(mis_df)),"playbook_entries":int(len(playbook_df)),"top_priority":int((scored["TIER"]=="Top Priority").sum()),"recommended":int((scored["TIER"]=="Recommended").sum()),"at_risk":int((q4["CASA_ACHIEVEMENT_PCT"]<80).sum()),"avg_capex":round(float(playbook_df["INITIAL_CAPEX_LAKHS"].mean()),0),"use_real_data":True,"type_dist":[{"type":bt,"count":int(c),"pct":round(c/total*100),"color":colors.get(bt,"#888")} for bt,c in type_counts.items()]}


def build_s1_kpis(dist_df):
    scored = compute_viability_score(dist_df)
    top=scored[scored["TIER"]=="Top Priority"]; rec=scored[scored["TIER"]=="Recommended"]; watch=scored[scored["TIER"]=="Watch List"]
    def td(t,b,df,a): return {"tier":t,"badge":b,"count":int(len(df)),"avg_v":round(float(df["VIABILITY_SCORE"].mean()),1) if len(df) else 0,"avg_m2p":f"{int(round(df['M2P'].mean()))} months" if len(df) else "—","action":a}
    return {
        "districts_scored": int(len(scored)),
        "total_districts":  int(len(scored)),
        "num_states":        int(dist_df["STATE"].nunique()),
        "top_priority":      int(len(top)),
        "recommended":       int(len(rec)),
        "watch":             int(len(watch)),
        "avg_m2p":           int(round(scored.head(20)["M2P"].mean())),
        "avg_m2p_top20":     int(round(scored.head(20)["M2P"].mean())),
        "tier_breakdown":    [td("Top Priority","badge-red",top,"Open all in FY26 Q1 →"),
                              td("Recommended","badge-amber",rec,"Filter by region →"),
                              td("Watch List","badge-blue",watch,"Monitor quarterly →")],
    }


def build_s2_kpis(mis_df):
    q4=mis_df[mis_df["QUARTER"]=="Q4"]
    at_risk   = q4[q4["CASA_ACHIEVEMENT_PCT"]<80].drop_duplicates("BRANCH_ID")
    critical  = q4[q4["CASA_ACHIEVEMENT_PCT"]<60].drop_duplicates("BRANCH_ID")
    cluster_perf = (q4.groupby("CLUSTER")
                      .agg(branches=("BRANCH_ID","nunique"), casa_achv=("CASA_ACHIEVEMENT_PCT","mean"))
                      .reset_index()
                      .sort_values("casa_achv", ascending=False))
    best  = cluster_perf.iloc[0]
    worst = cluster_perf.iloc[-1]
    return {
        "at_risk":          int(len(at_risk)),
        "critical":         int(len(critical)),
        "avg_casa":         round(float(q4["CASA_ACHIEVEMENT_PCT"].mean()),1),
        "avg_nps":          round(float(q4["NPS_SCORE"].mean()),1),
        "branches_tracked": int(mis_df["BRANCH_ID"].nunique()),
        "states":           int(mis_df["STATE"].nunique()),
        "best_casa":        round(float(best["casa_achv"]),1),
        "best_cluster":     str(best["CLUSTER"]),
        "best_n":           int(best["branches"]),
        "worst_casa":       round(float(worst["casa_achv"]),1),
        "worst_cluster":    str(worst["CLUSTER"]),
        "top_branches":     q4.groupby("BRANCH_NAME")["CASA_ACHIEVEMENT_PCT"].mean().nlargest(5).reset_index().to_dict(orient="records"),
    }


def build_s4_kpis(playbook_df):
    arch = (playbook_df.groupby("BRANCH_ARCHETYPE")
                       .agg(avg_m2p=("MONTHS_TO_PROFITABILITY","mean"),
                            peak=("PEAK_MONTHLY_CASA","mean"),
                            m12=("MONTH_12_CASA_PCT_OF_PEAK","mean"),
                            capex=("INITIAL_CAPEX_LAKHS","mean"),
                            success_rt=("BREAKEVEN_REACHED", lambda x: round(x.map({True:1,"True":1,False:0,"False":0}).mean()*100)))
                       .reset_index()
                       .sort_values("avg_m2p"))
    def badge(sr):
        if sr>=90: return "badge-red"
        elif sr>=70: return "badge-amber"
        return "badge-blue"
    arch_table = [{"name":r["BRANCH_ARCHETYPE"].replace("_"," "),"avg_m2p":int(round(r["avg_m2p"])),"peak":int(r["peak"]),"m12pct":int(round(r["m12"]*100)),"capex":int(round(r["capex"])),"success_rt":int(r["success_rt"]),"badge":badge(int(r["success_rt"]))} for _,r in arch.iterrows()]
    best  = arch.iloc[0]; worst = arch.iloc[-1]
    return {
        "archetypes":      int(playbook_df["BRANCH_ARCHETYPE"].nunique()),
        "playbook_entries":int(len(playbook_df)),
        "entries":         int(len(playbook_df)),
        "avg_m2p":         round(float(playbook_df["MONTHS_TO_PROFITABILITY"].mean()),1),
        "avg_capex":       round(float(playbook_df["INITIAL_CAPEX_LAKHS"].mean()),1),
        "best_m2p":        int(round(best["avg_m2p"])),
        "best_archetype":  str(best["BRANCH_ARCHETYPE"]).replace("_"," "),
        "worst_m2p":       int(round(worst["avg_m2p"])),
        "worst_archetype": str(worst["BRANCH_ARCHETYPE"]).replace("_"," "),
        "arch_table":      arch_table,
    }


def build_copilot_queries(mis_df):
    q4 = mis_df[mis_df["QUARTER"] == "Q4"].copy()
    mah = q4[q4["STATE"] == "Maharashtra"]
    at_risk_mah = mah[mah["CASA_ACHIEVEMENT_PCT"] < 85].drop_duplicates("BRANCH_NAME").sort_values("CASA_ACHIEVEMENT_PCT")
    risk_rows = "".join(f'<tr><td>{r["BRANCH_NAME"]}</td><td>{r["CLUSTER"]}</td><td><span style="color:{"#d93025" if round(r["CASA_ACHIEVEMENT_PCT"],1)<70 else "#e67e22"};font-weight:700">{round(r["CASA_ACHIEVEMENT_PCT"],1)}%</span></td><td><b>{"Critical" if round(r["CASA_ACHIEVEMENT_PCT"],1)<70 else "At Risk"}</b></td></tr>' for _,r in at_risk_mah.head(6).iterrows())
    q1_resp=f'<div class="response-query">❓ Which Maharashtra branches are at CASA risk?</div><table class="data-table"><thead><tr><th>Branch</th><th>Cluster</th><th>CASA Achievement</th><th>Status</th></tr></thead><tbody>{risk_rows}</tbody></table><div class="response-insight">💡 {len(at_risk_mah)} branches at risk in Maharashtra. Immediate RM intervention recommended.</div>'

    ca = q4.groupby("CLUSTER").agg(branches=("BRANCH_ID","nunique"),casa_achv=("CASA_ACHIEVEMENT_PCT","mean"),total_casa=("CASA_ACCOUNTS_OPENED","sum"),deposits=("TOTAL_DEPOSITS_CR","mean")).reset_index().sort_values("casa_achv",ascending=False).reset_index(drop=True)
    cr = "".join(f'<tr><td>#{i+1}</td><td><b>{r["CLUSTER"]}</b></td><td>{int(r["branches"])}</td><td><span style="color:{"#1a7f5a" if round(r["casa_achv"],1)>=92 else "#e67e22"};font-weight:700">{round(r["casa_achv"],1)}%</span></td><td>{int(r["total_casa"]):,}</td><td>{round(r["deposits"],1)}</td></tr>' for i,(_,r) in enumerate(ca.iterrows()))
    q2_resp=f'<div class="response-query">❓ Show cluster-wise performance ranking for Q4</div><table class="data-table"><thead><tr><th>Rank</th><th>Cluster</th><th>Branches</th><th>CASA Achv.</th><th>Total CASA</th><th>Deposits ₹Cr</th></tr></thead><tbody>{cr}</tbody></table><div class="response-insight">💡 {ca.iloc[0]["CLUSTER"]} leads at {round(ca.iloc[0]["casa_achv"],1)}%. Gap vs bottom: {round(ca.iloc[0]["casa_achv"]-ca.iloc[-1]["casa_achv"],1)}pp.</div>'

    bq = q4.drop_duplicates("BRANCH_NAME").sort_values("CASA_ACHIEVEMENT_PCT",ascending=False)
    def br(df,color): return "".join(f'<tr><td>{r["BRANCH_NAME"]}</td><td>{r["STATE"]}</td><td>{r["BRANCH_TYPE"]}</td><td><span style="color:{color};font-weight:700">{round(r["CASA_ACHIEVEMENT_PCT"],1)}%</span></td><td>{int(r["CASA_ACCOUNTS_OPENED"]):,}</td></tr>' for _,r in df.iterrows())
    q3_resp=f'<div class="response-query">❓ Top 5 and bottom 5 branches by CASA achievement</div><strong style="color:#1a7f5a">🏆 TOP 5</strong><table class="data-table" style="margin:8px 0 14px"><thead><tr><th>Branch</th><th>State</th><th>Type</th><th>CASA Achv.</th><th>CASA Opened</th></tr></thead><tbody>{br(bq.head(5),"#1a7f5a")}</tbody></table><strong style="color:#d93025">⚠️ BOTTOM 5</strong><table class="data-table" style="margin-top:8px"><thead><tr><th>Branch</th><th>State</th><th>Type</th><th>CASA Achv.</th><th>CASA Opened</th></tr></thead><tbody>{br(bq.tail(5).sort_values("CASA_ACHIEVEMENT_PCT"),"#d93025")}</tbody></table>'

    ln = q4[q4["NPS_SCORE"]<30].drop_duplicates("BRANCH_NAME").sort_values("NPS_SCORE")[["BRANCH_NAME","STATE","NPS_SCORE","CASA_ACHIEVEMENT_PCT","COMPLAINTS_LOGGED"]].head(8)
    nr = "".join(f'<tr><td>{r["BRANCH_NAME"]}</td><td>{r["STATE"]}</td><td><span style="color:{"#d93025" if int(r["NPS_SCORE"])<25 else "#e67e22"};font-weight:700">{int(r["NPS_SCORE"])}</span></td><td>{round(r["CASA_ACHIEVEMENT_PCT"],1)}%</td><td>{"High" if r["COMPLAINTS_LOGGED"]>8 else "Med"}</td><td>{round(r["COMPLAINTS_LOGGED"] * 2.5, 1)}</td></tr>' for _,r in ln.iterrows())
    q4_resp=f'<div class="response-query">❓ Which branches have NPS below 30?</div><table class="data-table"><thead><tr><th>Branch</th><th>State</th><th>NPS</th><th>CASA Achv.</th><th>Complaints</th><th>Est. Wait(min)</th></tr></thead><tbody>{nr}</tbody></table><div class="response-insight">💡 {len(ln)} branches with NPS &lt;30. Elevated wait times likely root cause — queue management review needed.</div>'

    urban=q4[q4["BRANCH_TYPE"]=="Urban"]; semi=q4[q4["BRANCH_TYPE"]=="Semi-Urban"]
    def sm(df,c): return round(float(df[c].mean()),1) if len(df)>0 else 0
    def dlt(a,b): d=round(a-b,1); return f'<span style="color:#e67e22">{("+" if d>0 else "")}{d}</span>'
    q5_resp=f'<div class="response-query">❓ Compare Urban vs Semi-Urban branch productivity</div><table class="data-table"><thead><tr><th>Metric</th><th>Urban</th><th>Semi-Urban</th><th>Delta</th></tr></thead><tbody><tr><td>CASA Achievement</td><td>{sm(urban,"CASA_ACHIEVEMENT_PCT")}%</td><td>{sm(semi,"CASA_ACHIEVEMENT_PCT")}%</td><td>{dlt(sm(urban,"CASA_ACHIEVEMENT_PCT"),sm(semi,"CASA_ACHIEVEMENT_PCT"))}</td></tr><tr><td>Avg Deposits ₹Cr</td><td>{sm(urban,"TOTAL_DEPOSITS_CR")}</td><td>{sm(semi,"TOTAL_DEPOSITS_CR")}</td><td>{dlt(sm(urban,"TOTAL_DEPOSITS_CR"),sm(semi,"TOTAL_DEPOSITS_CR"))}</td></tr><tr><td>NPS Score</td><td>{sm(urban,"NPS_SCORE")}</td><td>{sm(semi,"NPS_SCORE")}</td><td>{dlt(sm(urban,"NPS_SCORE"),sm(semi,"NPS_SCORE"))}</td></tr><tr><td>Cost-to-Income %</td><td>{sm(urban,"COST_TO_INCOME_PCT")}%</td><td>{sm(semi,"COST_TO_INCOME_PCT")}%</td><td>{dlt(sm(urban,"COST_TO_INCOME_PCT"),sm(semi,"COST_TO_INCOME_PCT"))}</td></tr></tbody></table><div class="response-insight">💡 Urban leads on volume. Semi-Urban has untapped cross-sell opportunity — RM capacity not fully deployed.</div>'

    return [
        {"q":"Which Maharashtra branches are at CASA risk this quarter?","response":q1_resp},
        {"q":"Show cluster-wise performance ranking for Q4","response":q2_resp},
        {"q":"Top 5 and bottom 5 branches by CASA achievement","response":q3_resp},
        {"q":"Which branches have NPS below 30?","response":q4_resp},
        {"q":"Compare Urban vs Semi-Urban branch productivity","response":q5_resp},
    ]


# ─────────────────────────────────────────────────────────
# CSV CONTEXT FOR CHATBOT
# ─────────────────────────────────────────────────────────
def build_chat_context():
    try:
        mis, existing, playbook, compliance, districts, dl, sp, bl = load_data()
        scored = compute_viability_score(districts)
        q4 = mis[mis["QUARTER"] == "Q4"].copy()
        at_risk  = q4[q4["CASA_ACHIEVEMENT_PCT"] < 80].drop_duplicates("BRANCH_ID")
        critical = q4[q4["CASA_ACHIEVEMENT_PCT"] < 60].drop_duplicates("BRANCH_ID")
        top5 = q4.groupby("BRANCH_NAME")["CASA_ACHIEVEMENT_PCT"].mean().nlargest(5).reset_index()
        bot5 = q4.groupby("BRANCH_NAME")["CASA_ACHIEVEMENT_PCT"].mean().nsmallest(5).reset_index()
        cluster_perf = q4.groupby("CLUSTER").agg(branches=("BRANCH_ID","nunique"),casa_achv=("CASA_ACHIEVEMENT_PCT","mean"),nps=("NPS_SCORE","mean"),deposits=("TOTAL_DEPOSITS_CR","mean")).reset_index().sort_values("casa_achv",ascending=False)
        state_perf = q4.groupby("STATE").agg(branches=("BRANCH_ID","nunique"),casa_achv=("CASA_ACHIEVEMENT_PCT","mean")).reset_index().sort_values("casa_achv",ascending=False)
        type_perf = q4.groupby("BRANCH_TYPE").agg(count=("BRANCH_ID","nunique"),casa_achv=("CASA_ACHIEVEMENT_PCT","mean"),deposits=("TOTAL_DEPOSITS_CR","mean"),nps=("NPS_SCORE","mean")).reset_index()
        top10 = scored.head(10)[["DISTRICT","STATE","VIABILITY_SCORE","TIER","M2P","CASA_12M","GAP_PCT"]]

        return f"""You are the YES Bank Branch Intelligence Copilot — an expert AI assistant embedded in the Branch Intelligence Suite.
You have access to YES Bank's branch performance data, district viability scores, branch playbook, and RBI compliance guidelines.
Be concise, data-backed, and action-oriented. Use bullet points for lists. Keep answers under 200 words unless detail is needed.

=== MIS OVERVIEW ===
Total Branches: {mis["BRANCH_ID"].nunique()} | Quarters: {", ".join(mis["QUARTER"].unique())}
Overall Avg CASA Achievement: {round(float(mis["CASA_ACHIEVEMENT_PCT"].mean()),1)}% | Avg NPS: {round(float(mis["NPS_SCORE"].mean()),1)}
Q4 At-Risk (CASA<80%): {len(at_risk)} branches | Critical (CASA<60%): {len(critical)} branches

=== TOP 5 BRANCHES — Q4 CASA ===
{top5.to_string(index=False)}

=== BOTTOM 5 BRANCHES — Q4 CASA ===
{bot5.to_string(index=False)}

=== CLUSTER PERFORMANCE (Q4) ===
{cluster_perf.to_string(index=False)}

=== STATE PERFORMANCE TOP 10 (Q4) ===
{state_perf.head(10).to_string(index=False)}

=== BRANCH TYPE BREAKDOWN (Q4) ===
{type_perf.to_string(index=False)}

=== TOP 10 EXPANSION DISTRICTS ===
{top10.to_string(index=False)}

=== DISTRICT TIER COUNTS ===
{json.dumps(scored["TIER"].value_counts().to_dict())}

=== EXISTING NETWORK ===
Total: {len(existing)} branches | States: {existing["STATE"].nunique()} | Types: {existing["BRANCH_TYPE"].value_counts().to_dict()}

=== NEW BRANCH PLAYBOOK ===
Archetypes: {", ".join(playbook["BRANCH_ARCHETYPE"].unique())}
Avg Months to Profitability: {round(float(playbook["MONTHS_TO_PROFITABILITY"].mean()),1)} | Avg Capex: ₹{round(float(playbook["INITIAL_CAPEX_LAKHS"].mean()),1)} Lakhs

=== RBI COMPLIANCE (sample) ===
{json.dumps(compliance.to_dict(orient="records")[:6], indent=2)}

Answer questions about this data only. Be specific with numbers. Redirect off-topic questions politely."""

    except Exception as e:
        return f"YES Bank AI Copilot. Data context error: {e}. Please check your CSV file paths."


# ─────────────────────────────────────────────────────────
# LLM PROVIDERS
# ─────────────────────────────────────────────────────────
def call_groq(system_prompt, messages):
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role":"system","content":system_prompt}] + messages,
        max_tokens=1024,
        temperature=0.4,
    )
    return response.choices[0].message.content


def call_ollama(system_prompt, messages):
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role":"system","content":system_prompt}] + messages,
        "stream": False,
        "options": {"temperature": 0.4, "num_predict": 1024},
    }
    resp = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def call_llm(system_prompt, messages):
    if LLM_PROVIDER == "groq":
        return call_groq(system_prompt, messages)
    elif LLM_PROVIDER == "ollama":
        return call_ollama(system_prompt, messages)
    raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER!r}")


# ─────────────────────────────────────────────────────────
# /api/chat ENDPOINT
# ─────────────────────────────────────────────────────────
@app.route("/api/chat", methods=["POST"])
def chat():
    data    = request.get_json()
    message = data.get("message", "").strip()
    history = data.get("history", [])
    if not message:
        return jsonify({"error": "Empty message"}), 400
    try:
        system_prompt = build_chat_context()
        messages = [{"role":m["role"],"content":m["content"]} for m in history[-10:]]
        messages.append({"role":"user","content":message})
        reply = call_llm(system_prompt, messages)
        return jsonify({"reply": reply, "provider": LLM_PROVIDER})
    except Exception as e:
        return jsonify({"error":str(e),"reply":f"⚠️ Error ({LLM_PROVIDER}): {str(e)}"}), 500



# ─────────────────────────────────────────────────────────
# HUDDLE ENGINE BUILDER  ★ NOW READS FROM 3 REAL CSV FILES ★
# ─────────────────────────────────────────────────────────
def build_huddle_engine(dl, sp, bl):
    """
    Build all Huddle Engine data from 3 real CSV files.

    Parameters
    ----------
    dl : huddle_daily_log.csv        — 1 row per branch per day  (430 rows)
    sp : huddle_staff_performance.csv — 1 row per staff per day (2150 rows)
    bl : huddle_blockers.csv          — 1 row per blocker raised  (216 rows)

    Returns dict consumed by:
      • Jinja2 template tags  {{ huddle.xxx }}
      • JavaScript payload    const huddleData = {{ huddle_json | safe }}
    """

    # ── 1. NETWORK-LEVEL KPIs (from huddle_daily_log) ─────────────────────────
    total_branches  = int(dl["BRANCH_ID"].nunique())
    working_days    = int(dl["DATE"].nunique())
    total_records   = total_branches * working_days

    occurrence_rate = round(float(dl["HUDDLE_HELD"].mean()) * 100, 1)
    avg_compliance  = round(float(dl["COMPLIANCE_SCORE"].mean()), 1)
    shame_flag_rate = round(float(dl["SHAME_FLAG"].mean()) * 100, 1)

    huddle_casa = float(dl[dl["HUDDLE_HELD"] == 1]["CASA_ACTUAL"].mean())
    miss_mask   = dl["HUDDLE_HELD"] == 0
    miss_casa   = float(dl[miss_mask]["CASA_ACTUAL"].mean()) if miss_mask.any() else huddle_casa
    uplift_pct  = round((huddle_casa - miss_casa) / max(miss_casa, 1) * 100, 1)
    extra_casa  = int(total_branches * working_days * (huddle_casa - miss_casa))
    escalations = int(dl[dl["ESCALATION_RAISED"] == 1]["BRANCH_ID"].nunique())

    # ── 2. BRANCH AGGREGATION (from huddle_daily_log) ─────────────────────────
    branch_agg = dl.groupby(
        ["BRANCH_ID", "BRANCH_NAME", "STATE", "BRANCH_TYPE", "CLUSTER"]
    ).agg(
        avg_compliance = ("COMPLIANCE_SCORE", "mean"),
        avg_sentiment  = ("SENTIMENT_SCORE",  "mean"),
        total_casa     = ("CASA_ACTUAL",       "sum"),
        total_target   = ("CASA_TARGET",       "sum"),
        last_casa      = ("CASA_ACTUAL",       "last"),
        last_target    = ("CASA_TARGET",       "last"),
        occurrence     = ("HUDDLE_HELD",       "mean"),
        shame_days     = ("SHAME_FLAG",        "sum"),
    ).reset_index()

    branch_agg["mtd_pct"] = (
        branch_agg["total_casa"] / branch_agg["total_target"].clip(lower=1) * 100
    ).round(1)

    def rag_status(pct):
        if pct < 70:  return ("Critical",              "#7f1d1d", "SPRINT MODE", "#d93025", "🚨")
        if pct < 88:  return ("Intervention Required", "#ef4444", "SPRINT MODE", "#d93025", "🔴")
        if pct < 100: return ("Watch",                 "#f59e0b", "FOCUS MODE",  "#e67e22", "🟡")
        return               ("On Track",              "#22c55e", "GROWTH MODE", "#1a7f5a", "🟢")

    branch_agg[["rag","rag_color","mode","mode_color","rag_icon"]] = pd.DataFrame(
        branch_agg["mtd_pct"].apply(rag_status).tolist(), index=branch_agg.index)

    # ── 3. STAR PERFORMERS (from huddle_staff_performance IS_STAR_DAY=1) ──────
    stars = (
        sp[sp["IS_STAR_DAY"] == 1]
        .sort_values("DATE")
        .groupby("BRANCH_ID").last()
        .reset_index()
        [["BRANCH_ID", "STAFF_NAME", "STAFF_ROLE", "CASA_OPENED"]]
    )

    # ── 4. OPEN BLOCKERS (from huddle_blockers RESOLVED_SAME_DAY=0) ───────────
    open_bl = (
        bl[bl["RESOLVED_SAME_DAY"] == 0]
        .groupby("BRANCH_ID")
        .agg(
            open_count  = ("BLOCKER_TYPE", "count"),
            latest_desc = ("BLOCKER_DESC", "last"),
            latest_type = ("BLOCKER_TYPE", "last"),
        )
        .reset_index()
    )

    # ── 5. BUILD 3 BRIEFS: worst / mid / best branch ──────────────────────────
    SEGMENT_MAP = {
        "Metro":      "Salaried professionals, corporates, HNI wallet-share",
        "Urban":      "Self-employed traders, salaried mid-income, digital-first",
        "Semi-Urban": "Self-employed traders, govt employees, school/college staff",
        "Rural":      "Farmers, daily-wage workers, SHG members, local merchants",
    }
    COMP_PULSES = [
        {"tag":"Compliance | CASA",    "title":"Interest Rate on Savings Accounts",
         "body":"Banks free to set SA rates. YES Bank offers tiered rates. Min 3.5% on balances under Rs.1L.",
         "limit":"Deregulated by RBI in 2011"},
        {"tag":"Compliance | KYC",     "title":"Aadhaar eKYC for Account Opening",
         "body":"Aadhaar OTP-based eKYC permitted for SA. Biometric eKYC required above Rs.50,000.",
         "limit":"Balance cap Rs.50,000 for OTP-only eKYC"},
        {"tag":"Compliance | Digital", "title":"Video KYC (V-CIP)",
         "body":"V-CIP allowed for new accounts. Customer must be in India. Live facial match required.",
         "limit":"Customer must be physically in India"},
        {"tag":"Compliance | Data",    "title":"Customer Data Localization",
         "body":"All payment system data must be stored in India. Cannot be shared with foreign entities.",
         "limit":"India-only storage mandatory"},
        {"tag":"Compliance | PSL",     "title":"Priority Sector Lending Targets",
         "body":"40% of Adjusted Net Bank Credit must go to priority sectors. Agriculture sub-target 18%.",
         "limit":"40% ANBC mandatory"},
    ]

    sorted_br = branch_agg.sort_values("mtd_pct").reset_index(drop=True)
    n = len(sorted_br)
    briefs_huddle = []

    for i, idx in enumerate([0, n // 2, n - 1]):
        row = sorted_br.iloc[idx]
        pct = float(row["mtd_pct"])
        gap = max(0, int(row["total_target"] - row["total_casa"]))
        rag, rag_color, mode, mode_color, rag_icon = rag_status(pct)

        # Star from huddle_staff_performance
        star_row  = stars[stars["BRANCH_ID"] == row["BRANCH_ID"]]
        star_name = str(star_row.iloc[0]["STAFF_NAME"])   if not star_row.empty else "Top Performer"
        star_role = str(star_row.iloc[0]["STAFF_ROLE"])   if not star_row.empty else "Sales Officer"
        star_casa = int(star_row.iloc[0]["CASA_OPENED"])  if not star_row.empty else int(row["last_casa"])

        # Open blocker from huddle_blockers
        bl_row        = open_bl[open_bl["BRANCH_ID"] == row["BRANCH_ID"]]
        blocker_desc  = str(bl_row["latest_desc"].values[0]) if not bl_row.empty else ""
        blocker_type  = str(bl_row["latest_type"].values[0]) if not bl_row.empty else ""
        blocker_count = int(bl_row["open_count"].values[0])  if not bl_row.empty else 0

        if pct >= 100:
            tasks = [
                "Cross-sell day: every new SA gets a credit card offer at the same sitting",
                "FD conversion: call SA customers with Rs.50K+ idle balance",
                "NPS push: one staff calls 3 recent account holders for feedback"
            ]
            mission_desc = "On Track — shift focus to cross-sell and wallet depth"
        elif pct < 88:
            tasks = [
                f"CASA SPRINT: need {gap} accounts today — every conversation starts with account opening",
                "RM Call Blitz: each RM calls 5 existing customers for FD top-up or referral",
                "Door-to-door: visit top 3 nearby employers or societies before 11am"
            ]
            mission_desc = f"CASA Sprint — {gap} accounts gap to close"
        else:
            tasks = [
                f"CASA Focus: {gap}-account gap — assign each RM a daily mini-target",
                "Mid-day check-in: BM reviews progress at 1pm and re-deploys if needed",
                "Cross-sell to walk-ins: offer FD or insurance to non-CASA visitors"
            ]
            mission_desc = f"Focus Mode — close {gap}-account gap before month-end"

        briefs_huddle.append({
            "branch_id":    str(row["BRANCH_ID"]),
            "branch_name":  str(row["BRANCH_NAME"]),
            "state":        str(row["STATE"]),
            "branch_type":  str(row["BRANCH_TYPE"]),
            "cluster":      str(row["CLUSTER"]),
            "rag":          rag,      "rag_color":  rag_color,  "rag_icon": rag_icon,
            "mode":         mode,     "mode_color": mode_color,
            "casa_pct":     round(pct, 1),
            "gap":          gap,
            "yest_opened":  int(row["last_casa"]),
            "yest_target":  int(row["last_target"]),
            "yest_pct":     round(float(row["last_casa"]) / max(float(row["last_target"]), 1) * 100, 1),
            "trend":        "Up Growing" if pct >= 100 else ("Stable" if pct >= 90 else "Down Declining"),
            "deposit_pct":  100.0,
            "nps":          50,
            "mission_desc": mission_desc,
            "segment":      SEGMENT_MAP.get(str(row["BRANCH_TYPE"]), "General banking customers"),
            "tasks":        tasks,
            "star_name":    star_name,
            "star_role":    star_role,
            "star_casa":    star_casa,
            "blocker_desc":  blocker_desc,
            "blocker_type":  blocker_type,
            "blocker_count": blocker_count,
            "pulse": COMP_PULSES[i % len(COMP_PULSES)],
        })

    # ── 6. COMPLIANCE TIER DISTRIBUTION (COMPLIANCE_SCORE from daily log) ─────
    def tier_fn(score):
        if score >= 85: return "Excellent"
        if score >= 70: return "Good"
        if score >= 55: return "Needs Improvement"
        if score >= 40: return "Poor"
        return "Non-Compliant"

    branch_agg["tier"] = branch_agg["avg_compliance"].apply(tier_fn)
    tier_colors = {
        "Excellent": "#1a7f5a", "Good": "#4ade80",
        "Needs Improvement": "#f59e0b", "Poor": "#ef4444", "Non-Compliant": "#7f1d1d"
    }
    tc = branch_agg["tier"].value_counts()
    compliance_tiers = [
        {"tier": t, "count": int(tc.get(t, 0)),
         "pct": round(tc.get(t, 0) / max(total_branches, 1) * 100, 1),
         "color": tier_colors[t]}
        for t in ["Excellent", "Good", "Needs Improvement", "Poor", "Non-Compliant"]
    ]

    # ── 7. CLUSTER LEAGUE TABLE (CLUSTER groupby from daily log) ──────────────
    cl = dl.groupby("CLUSTER").agg(
        branches    = ("BRANCH_ID",         "nunique"),
        avg_score   = ("COMPLIANCE_SCORE",  "mean"),
        occurrence  = ("HUDDLE_HELD",       "mean"),
        shame_flags = ("SHAME_FLAG",        "sum"),
        escalations = ("ESCALATION_RAISED", "sum"),
    ).reset_index().sort_values("avg_score", ascending=False).reset_index(drop=True)

    cl["avg_score"]   = cl["avg_score"].round(1)
    cl["occurrence"]  = (cl["occurrence"] * 100).round(1)
    cl["score_color"] = cl["avg_score"].apply(
        lambda s: "#1a7f5a" if s >= 70 else ("#e67e22" if s >= 55 else "#d93025")
    )

    # ── 8. TOP 5 BRANCHES (by avg COMPLIANCE_SCORE from daily log) ────────────
    top5 = branch_agg.nlargest(5, "avg_compliance")[
        ["BRANCH_NAME", "CLUSTER", "avg_compliance", "occurrence"]
    ].reset_index(drop=True)

    top5_list = [
        {"branch":     str(r["BRANCH_NAME"]),
         "cluster":    str(r["CLUSTER"]),
         "score":      round(float(r["avg_compliance"]), 1),
         "occurrence": round(float(r["occurrence"]) * 100, 1)}
        for _, r in top5.iterrows()
    ]

    # ── 9. RETURN — maps 1-to-1 with {{ huddle.xxx }} in dashboard.html ───────
    return {
        "occurrence_rate":  occurrence_rate,
        "avg_compliance":   avg_compliance,
        "uplift_pct":       uplift_pct,
        "extra_casa":       f"{extra_casa:,}",
        "total_branches":   total_branches,
        "total_records":    f"{total_records:,}",
        "working_days":     working_days,
        "huddle_casa_day":  round(huddle_casa, 2),
        "miss_casa_day":    round(miss_casa,   2),
        "shame_flag_rate":  shame_flag_rate,
        "escalations":      escalations,
        "briefs":           briefs_huddle,
        "compliance_tiers": compliance_tiers,
        "cluster_league":   cl.to_dict(orient="records"),
        "top5_branches":    top5_list,
    }


def build_feature_weights():
    """Returns the viability score feature weights — matches compute_viability_score() formula."""
    return [
        {"label": "Market Size (Population × Per Capita Income)", "pct": 25, "color": "#003087"},
        {"label": "YES Bank Whitespace Gap",                       "pct": 20, "color": "#003087"},
        {"label": "Business Activity (GST Density + CD Ratio)",    "pct": 20, "color": "#003087"},
        {"label": "Financial Inclusion (PMJDY Zero-Balance %)",    "pct": 15, "color": "#00a0b0"},
        {"label": "CASA Catchment (Urban Pop × Literacy)",         "pct": 10, "color": "#c9a84c"},
        {"label": "Competition Score (1 / Competitor Density)",    "pct": 10, "color": "#1a7f5a"},
    ]

# ─────────────────────────────────────────────────────────
# MAIN ROUTE
# ─────────────────────────────────────────────────────────

@app.route("/")
def dashboard():
    # load_data() now returns 8 DataFrames (5 original + 3 huddle CSVs)
    mis, existing, playbook, compliance, districts, dl, sp, bl = load_data()
    summary         = build_summary(mis, existing, playbook, districts)
    opportunities   = build_opportunities(districts, existing)
    cluster_data    = build_cluster_data(mis)
    briefs          = build_briefs(playbook, districts, existing)
    s1_kpis         = build_s1_kpis(districts)
    s2_kpis         = build_s2_kpis(mis)
    s4_kpis         = build_s4_kpis(playbook)
    queries         = build_copilot_queries(mis)
    compliance_list = compliance.to_dict(orient="records")
    feature_weights = build_feature_weights()
    huddle          = build_huddle_engine(dl, sp, bl)   # ★ real CSVs

    # Load IFSC file for total branch count — fall back gracefully if missing
    ifsc_path = os.path.join(BASE, "data", "yesbank_ifsc.csv")
    try:
        branches_count_df = pd.read_csv(ifsc_path)
        branches_counts = int(branches_count_df['Branch Code'].count()) - 1
    except FileNotFoundError:
        branches_counts = len(existing)   # use existing network count as fallback

    return render_template(
        "dashboard.html",
        summary=summary, opportunities=opportunities, cluster_data=cluster_data,
        briefs=briefs, s1_kpis=s1_kpis, s2_kpis=s2_kpis, s4_kpis=s4_kpis,
        queries=queries, compliance=compliance_list, branches_counts=branches_counts,
        feature_weights=feature_weights,
        huddle=huddle,
        huddle_json=json.dumps(huddle),
        opp_json=json.dumps(opportunities), cluster_json=json.dumps(cluster_data),
        briefs_json=json.dumps(briefs), summary_json=json.dumps(summary),
        queries_json=json.dumps(queries), compliance_json=json.dumps(compliance_list),
        llm_provider=LLM_PROVIDER,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)