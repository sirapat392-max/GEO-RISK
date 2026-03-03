import math
import re
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple

import feedparser
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# ===== Data Sources (multi-source, global + SEA/TH relevance) =====
RSS_SOURCES = [
    "https://www.reuters.com/world/rss",
    "https://feeds.bbci.co.uk/news/world/rss.xml",
    "https://feeds.bbci.co.uk/news/world/asia/rss.xml",
    "https://www.aljazeera.com/xml/rss/all.xml",
    "https://apnews.com/hub/apf-topnews?output=amp",
    "https://www.usgs.gov/feeds/news",
]

# Source reliability baseline (simple prior)
SOURCE_WEIGHT_HINTS = {
    "reuters": 0.92,
    "bbc": 0.90,
    "associated press": 0.90,
    "ap": 0.88,
    "al jazeera": 0.82,
    "usgs": 0.95,
}

TH_KEYWORDS = [
    "thailand", "thai", "bangkok", "myanmar", "cambodia", "laos", "malaysia",
    "south china sea", "andaman", "mekong", "gulf of thailand"
]

CATEGORY_KEYWORDS = {
    "conflict": ["conflict", "clash", "strike", "missile", "military", "troops", "protest", "riot"],
    "sanctions": ["sanction", "embargo", "restriction", "tariff", "export control"],
    "shipping_disruption": ["shipping", "port", "strait", "freight", "container", "route", "blockade"],
    "cyber_incidents": ["cyber", "ransomware", "breach", "ddos", "malware", "hack"],
    "commodity_shock": ["oil", "gas", "lng", "commodity", "food price", "wheat", "rice", "inflation"],
    "flood_storm": ["flood", "storm", "typhoon", "cyclone", "heavy rain", "landslide"],
    "earthquake_geology": ["earthquake", "quake", "tsunami", "volcano"],
    "air_quality_wildfire": ["pm2.5", "haze", "wildfire", "air quality", "smoke"],
    "public_health": ["outbreak", "epidemic", "pandemic", "disease", "hospital surge"],
}

app = FastAPI(title="GeoRisk Thailand API", version="0.3.0")
FRONTEND_DIR = "/app/frontend"
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def clamp(v: float, lo=0, hi=100) -> int:
    return max(lo, min(hi, int(round(v))))


def norm_text(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def source_weight(source_name: str) -> float:
    n = norm_text(source_name)
    for k, w in SOURCE_WEIGHT_HINTS.items():
        if k in n:
            return w
    return 0.75


def classify_categories(text: str) -> Dict[str, int]:
    t = norm_text(text)
    out = {k: 0 for k in CATEGORY_KEYWORDS.keys()}
    for cat, kws in CATEGORY_KEYWORDS.items():
        for kw in kws:
            if kw in t:
                out[cat] += 1
    return out


def thailand_relevance(text: str) -> float:
    t = norm_text(text)
    hits = sum(1 for k in TH_KEYWORDS if k in t)
    return min(1.0, hits / 3)


def collect_news(limit: int = 120) -> List[Dict[str, Any]]:
    raw: List[Dict[str, Any]] = []
    for url in RSS_SOURCES:
        try:
            feed = feedparser.parse(url)
            feed_title = feed.feed.get("title", url)
            for e in feed.entries[:60]:
                title = e.get("title", "")
                summary = e.get("summary", "")
                if not title:
                    continue
                text = f"{title} {summary}"
                raw.append({
                    "title": title,
                    "summary": summary,
                    "link": e.get("link", ""),
                    "published": e.get("published", ""),
                    "source": feed_title,
                    "source_weight": source_weight(feed_title),
                    "categories": classify_categories(text),
                    "thailand_relevance": thailand_relevance(text),
                })
        except Exception:
            continue

    # Dedupe + fuse by normalized title key
    fused: Dict[str, Dict[str, Any]] = {}
    for it in raw:
        key = norm_text(it["title"])[:180]
        if key not in fused:
            fused[key] = {
                "title": it["title"],
                "summary": it["summary"],
                "link": it["link"],
                "published": it["published"],
                "sources": [it["source"]],
                "source_weights": [it["source_weight"]],
                "categories": it["categories"].copy(),
                "thailand_relevance": it["thailand_relevance"],
            }
        else:
            f = fused[key]
            if it["source"] not in f["sources"]:
                f["sources"].append(it["source"])
                f["source_weights"].append(it["source_weight"])
            for c, v in it["categories"].items():
                f["categories"][c] += v
            f["thailand_relevance"] = max(f["thailand_relevance"], it["thailand_relevance"])

    events = list(fused.values())
    for ev in events:
        src_count = len(ev["sources"])
        avg_w = sum(ev["source_weights"]) / max(1, len(ev["source_weights"]))
        ev["confidence"] = round(min(1.0, avg_w * (1 + math.log1p(src_count) / 2)), 3)

    # prioritize TH relevance + confidence
    events.sort(key=lambda x: (x["thailand_relevance"], x["confidence"], len(x["sources"])), reverse=True)
    return events[:limit]


def scores_from_events(events: List[Dict[str, Any]]) -> Dict[str, int]:
    cats = {k: 0.0 for k in CATEGORY_KEYWORDS.keys()}
    for ev in events:
        w = ev["confidence"] * (0.5 + ev["thailand_relevance"])
        for c, v in ev["categories"].items():
            cats[c] += w * v

    # normalize to 0..100 with soft cap
    mx = max(cats.values()) if cats else 1.0
    if mx <= 0:
        return {k: 10 for k in cats.keys()}
    return {k: clamp((v / mx) * 70 + 15) for k, v in cats.items()}


def normalize_three(best: int, base: int, worst: int) -> Tuple[int, int, int]:
    total = max(best + base + worst, 1)
    b = round(best * 100 / total)
    c = round(base * 100 / total)
    w = 100 - b - c
    return b, c, w


def scenario_from_scores(scores: Dict[str, int]) -> Dict[str, Any]:
    core = (scores["conflict"] + scores["shipping_disruption"] + scores["cyber_incidents"] + scores["commodity_shock"]) / 4
    worst = clamp(10 + core * 0.35, 10, 55)
    best = clamp(25 - core * 0.10, 8, 30)
    base = max(100 - best - worst, 20)
    best, base, worst = normalize_three(best, base, worst)
    dyn = scenario_text_from_scores(scores)

    return {
        "best": {
            "probability": best,
            "what_could_happen": ["แรงกดดันหลักชะลอลง", "ระบบขนส่งเสถียรขึ้น", "ความผันผวนตลาดลดลง"],
            "impact_thailand": "บวกเล็กน้อย"
        },
        "base": {
            "probability": base,
            "what_could_happen": dyn,
            "impact_thailand": "กลาง"
        },
        "worst": {
            "probability": worst,
            "what_could_happen": ["สัญญาณหลักลุกลามพร้อมกัน", "ต้นทุนพลังงาน/ขนส่งพุ่ง", "ความเสี่ยงไซเบอร์-ความมั่นคงเพิ่ม"],
            "impact_thailand": "ลบสูง"
        }
    }


def impact_from_scores(scores: Dict[str, int]) -> Dict[str, int]:
    return {
        "economy": clamp((scores["commodity_shock"] * 0.5 + scores["sanctions"] * 0.3 + scores["public_health"] * 0.2)),
        "logistics": clamp((scores["shipping_disruption"] * 0.6 + scores["flood_storm"] * 0.4)),
        "security": clamp((scores["conflict"] * 0.6 + scores["cyber_incidents"] * 0.4)),
        "energy_commodity": clamp((scores["commodity_shock"] * 0.7 + scores["shipping_disruption"] * 0.3)),
        "tourism_investment": clamp((scores["conflict"] * 0.35 + scores["air_quality_wildfire"] * 0.25 + scores["public_health"] * 0.4)),
    }


def action_plan_percent() -> Dict[str, Any]:
    return {
        "high_risk_50": {
            "personal": [
                {"step": "ติดตามสัญญาณเสี่ยงวันละ 2 รอบ", "percent": 20},
                {"step": "เตรียมน้ำ-อาหารฉุกเฉินและเอกสาร", "percent": 15},
                {"step": "หลีกเลี่ยงโซนเสี่ยงตามแผนที่", "percent": 15},
            ],
            "business": [
                {"step": "กระจาย supplier และเส้นทางขนส่ง", "percent": 20},
                {"step": "เพิ่ม cyber hardening + SOC watch", "percent": 15},
                {"step": "ทำ hedge ต้นทุนพลังงาน/วัตถุดิบ", "percent": 15},
            ],
        },
        "low_risk_10": {
            "personal": [
                {"step": "ตรวจข่าวเชิงความเสี่ยงรายวัน", "percent": 4},
                {"step": "ทบทวนแผนฉุกเฉินรายเดือน", "percent": 3},
                {"step": "สำรองยา/เวชภัณฑ์พื้นฐาน", "percent": 3},
            ],
            "business": [
                {"step": "monitor KPI ความเสี่ยงหลัก", "percent": 4},
                {"step": "ทดสอบ BCP/DR ตามรอบ", "percent": 3},
                {"step": "อัปเดต vendor risk score", "percent": 3},
            ],
        },
    }


def preparedness_quantities() -> Dict[str, Any]:
    return {
        "household_4_people": {
            "water_liters_per_day": 12,
            "water_liters_7d": 84,
            "water_liters_14d": 168,
            "water_liters_30d": 360,
            "rice_kg_14d": "14-18",
            "canned_protein_units_14d": "56-84",
            "medicine_days": 30,
            "power_banks_min": 2
        },
        "business_mid_size": {
            "safety_stock_days_base": 14,
            "safety_stock_days_stress": 21,
            "backup_suppliers_per_critical_item": 2,
            "incident_response_sla_hours": 4,
            "backup_connectivity_links": 2
        }
    }




def detailed_metrics(events: List[Dict[str, Any]], scores: Dict[str, int]) -> Dict[str, Any]:
    # event severity proxy
    severe_words = ["attack","explosion","fatal","dead","missile","earthquake","flood","riot","sanction"]
    severe_count = 0
    by_cat = {k: {"event_count": 0, "avg_confidence": 0.0, "top_events": []} for k in CATEGORY_KEYWORDS.keys()}

    for ev in events:
        txt = (ev.get("title","")+" "+ev.get("summary","")).lower()
        if any(w in txt for w in severe_words):
            severe_count += 1
        conf = float(ev.get("confidence", 0.0))
        for c, v in ev.get("categories", {}).items():
            if v > 0:
                by_cat[c]["event_count"] += 1
                by_cat[c]["avg_confidence"] += conf
                if len(by_cat[c]["top_events"]) < 3:
                    by_cat[c]["top_events"].append({
                        "title": ev.get("title"),
                        "confidence": conf,
                        "sources": ev.get("sources", [])
                    })

    for c in by_cat:
        ec = by_cat[c]["event_count"]
        if ec > 0:
            by_cat[c]["avg_confidence"] = round(by_cat[c]["avg_confidence"] / ec, 3)

    overall = {
        "event_volume": len(events),
        "severe_signal_count": severe_count,
        "thailand_relevance_avg": round(sum(e.get("thailand_relevance",0) for e in events)/max(1,len(events)),3),
        "risk_level": "high" if max(scores.values()) >= 70 else ("medium" if max(scores.values()) >= 45 else "low")
    }
    return {"overall": overall, "by_category": by_cat}
def quality_metrics(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    source_counter: Dict[str, int] = {}
    for e in events:
        for s in e.get("sources", []):
            source_counter[s] = source_counter.get(s, 0) + 1
    total_mentions = max(1, sum(source_counter.values()))
    top_share = max(source_counter.values()) / total_mentions if source_counter else 1.0
    warnings = []
    if top_share > 0.5:
        warnings.append("source_concentration_high")
    if len(source_counter) < 3:
        warnings.append("low_source_diversity")
    return {
        "sources_used": len(source_counter),
        "source_distribution": source_counter,
        "top_source_share": round(top_share, 3),
        "warnings": warnings,
    }




def top_categories(scores: Dict[str, int], n: int = 3):
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:n]


def scenario_text_from_scores(scores: Dict[str, int]):
    top = [k for k,_ in top_categories(scores,3)]
    def phrase(cat):
        m={
          'conflict':'แรงกดดันความขัดแย้งและการประท้วง',
          'sanctions':'แรงกดดันมาตรการคว่ำบาตร/การค้า',
          'shipping_disruption':'ความเสี่ยงสะดุดในเส้นทางขนส่ง',
          'cyber_incidents':'ความเสี่ยงเหตุการณ์ไซเบอร์',
          'commodity_shock':'ความผันผวนพลังงาน/สินค้าโภคภัณฑ์',
          'flood_storm':'ความเสี่ยงน้ำท่วม/พายุ',
          'earthquake_geology':'ความเสี่ยงแผ่นดินไหว/ธรณี',
          'air_quality_wildfire':'ความเสี่ยงคุณภาพอากาศ/ไฟป่า',
          'public_health':'ความเสี่ยงด้านสาธารณสุข',
        }
        return m.get(cat,cat)
    return [f"สัญญาณหลัก: {phrase(c)}" for c in top]


def formula_panel(scores: Dict[str,int], events: List[Dict[str,Any]]):
    return {
      "scores_formula": "score_cat = clamp((weighted_cat_signal/max_signal)*70 + 15)",
      "event_weight_formula": "event_weight = confidence * (0.5 + thailand_relevance)",
      "confidence_formula": "confidence = min(1, avg_source_weight * (1 + ln(1+source_count)/2))",
      "impact_formula": {
        "economy": "0.5*commodity_shock + 0.3*sanctions + 0.2*public_health",
        "logistics": "0.6*shipping_disruption + 0.4*flood_storm",
        "security": "0.6*conflict + 0.4*cyber_incidents",
      },
      "scenario_formula": "worst=clamp(10+core*0.35), best=clamp(25-core*0.10), base=100-best-worst",
      "input_event_count": len(events)
    }


def data_lineage(events: List[Dict[str, Any]], scores: Dict[str,int]):
    top = [k for k,_ in top_categories(scores,3)]
    lineage={}
    for cat in top:
        related=[]
        for ev in events:
            if ev.get('categories',{}).get(cat,0)>0:
                related.append({
                    'title': ev.get('title'),
                    'sources': ev.get('sources',[]),
                    'confidence': ev.get('confidence',0),
                    'thailand_relevance': ev.get('thailand_relevance',0)
                })
        lineage[cat]={'evidence_count':len(related),'evidence':related[:8]}
    return lineage


def weight_breakdown(events: List[Dict[str, Any]]):
    src={}
    for ev in events:
        for s in ev.get('sources',[]):
            src[s]=src.get(s,0)+1
    by_source=[{'source':k,'mentions':v,'share':round(v/max(1,sum(src.values())),3)} for k,v in sorted(src.items(), key=lambda kv: kv[1], reverse=True)]
    return {
      'source_stage': by_source,
      'event_stage_note': 'dedupe by normalized title; merge multi-source confirmations',
      'category_stage_note': 'keyword category signals aggregated with event_weight',
      'risk_stage_note': 'normalized weighted category signal -> risk score 0..100'
    }

def build_analysis(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    scores = scores_from_events(events)
    scenarios = scenario_from_scores(scores)
    impact = impact_from_scores(scores)
    return {
        "scores": scores,
        "future_scenarios": scenarios,
        "impact_to_thailand": impact,
        "action_plan_percent": action_plan_percent(),
        "preparedness_quantities": preparedness_quantities(),
        "data_quality": quality_metrics(events),
        "detailed_metrics": detailed_metrics(events, scores),
        "formula_panel": formula_panel(scores, events),
        "data_lineage": data_lineage(events, scores),
        "weight_breakdown": weight_breakdown(events),
        "no_hardcoded_mode": True,
        "summary": "Thailand multi-risk: weighted fusion from multi-source global + regional feeds (defensive posture).",
    }


def parse_map_link(url: str) -> Dict[str, Any]:
    out = {"lat": None, "lng": None}
    try:
        if "q=" in url:
            q = url.split("q=", 1)[1].split("&", 1)[0]
            if "," in q:
                a, b = q.split(",", 1)
                out["lat"] = float(a)
                out["lng"] = float(b)
    except Exception:
        pass
    return out


@app.get("/health")
def health():
    return {"ok": True, "ts": now_iso()}


@app.get("/")
def root_ui():
    return FileResponse(f"{FRONTEND_DIR}/index.html")


@app.get("/api/news")
def api_news(limit: int = Query(60, ge=10, le=300)):
    events = collect_news(limit)
    return {"ts": now_iso(), "count": len(events), "items": events}


@app.get("/api/risk")
def api_risk():
    events = collect_news(120)
    analysis = build_analysis(events)
    return {
        "ts": now_iso(),
        "focus": "Thailand",
        "event_count": len(events),
        "analysis": analysis,
    }


@app.get("/api/geo-assess")
def api_geo_assess(map_url: str = Query(...), radius_km: float = Query(2.0, ge=0.5, le=20)):
    coords = parse_map_link(map_url)
    events = collect_news(80)
    analysis = build_analysis(events)
    sec = analysis["impact_to_thailand"]["security"]
    conf = analysis["scores"]["conflict"]
    hotspot = {
        "area_risk_percent": clamp((sec + conf) / 2),
        "window": "24-72h",
        "recommendation": [
            f"หลีกเลี่ยงรัศมี {radius_km} กม. รอบพื้นที่ที่มีสัญญาณความเสี่ยงสูง",
            "ติดตามแจ้งเตือนทุก 2 ชั่วโมง",
            "เตรียมเส้นทางสำรองอย่างน้อย 2 เส้นทาง"
        ]
    }
    return {
        "ts": now_iso(),
        "input": {"map_url": map_url, "radius_km": radius_km, **coords},
        "hotspot_assessment": hotspot,
        "analysis_hint": "defensive early-warning only",
        "evidence_count": len(events)
    }
