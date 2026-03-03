import os
import json
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple

import httpx
import feedparser
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/auto")

RSS_SOURCES = [
    "https://www.reuters.com/world/rss",
    "https://feeds.bbci.co.uk/news/world/asia/rss.xml",
    "https://www.aljazeera.com/xml/rss/all.xml",
    "https://www.usgs.gov/feeds/news",
]

TH_KEYWORDS = [
    "thailand", "thai", "bangkok", "myanmar", "cambodia", "laos", "malaysia",
    "south china sea", "flood", "storm", "earthquake", "wildfire", "pm2.5"
]

app = FastAPI(title="GeoRisk Thailand API", version="0.2.0")
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


def clamp(v: int, lo=0, hi=100) -> int:
    return max(lo, min(hi, int(v)))


def collect_news(limit: int = 80) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for url in RSS_SOURCES:
        feed = feedparser.parse(url)
        for e in feed.entries[:60]:
            title = e.get("title", "")
            summary = e.get("summary", "")
            blob = f"{title} {summary}".lower()
            if any(k in blob for k in TH_KEYWORDS):
                items.append({
                    "title": title,
                    "summary": summary,
                    "link": e.get("link", ""),
                    "published": e.get("published", ""),
                    "source": feed.feed.get("title", url),
                })

    seen = set()
    dedup = []
    for it in items:
        key = (it["title"][:150]).strip().lower()
        if key in seen:
            continue
        seen.add(key)
        dedup.append(it)
    return dedup[:limit]


def normalize_three(best: int, base: int, worst: int) -> Tuple[int, int, int]:
    total = max(best + base + worst, 1)
    b = round(best * 100 / total)
    c = round(base * 100 / total)
    w = 100 - b - c
    return b, c, w


def fallback_analysis(news_count: int) -> Dict[str, Any]:
    intensity = min(20, news_count // 4)
    scores = {
        "conflict": clamp(38 + intensity),
        "sanctions": clamp(16 + intensity // 2),
        "shipping_disruption": clamp(34 + intensity),
        "cyber_incidents": clamp(45 + intensity),
        "commodity_shock": clamp(40 + intensity),
        "flood_storm": clamp(36 + intensity // 2),
        "earthquake_geology": clamp(22 + intensity // 3),
        "air_quality_wildfire": clamp(30 + intensity // 2),
        "public_health": clamp(20 + intensity // 3),
    }

    best, base, worst = normalize_three(20, 55, 25)

    return {
        "scores": scores,
        "future_scenarios": {
            "best": {
                "probability": best,
                "what_could_happen": [
                    "แรงกดดันภูมิรัฐศาสตร์ชะลอลง",
                    "เส้นทางขนส่งกลับสู่ภาวะปกติ",
                    "ราคาพลังงานแกว่งตัวแคบ"
                ],
                "impact_thailand": "บวกเล็กน้อย"
            },
            "base": {
                "probability": base,
                "what_could_happen": [
                    "เหตุเสี่ยงเกิดเป็นช่วง ๆ",
                    "โลจิสติกส์ติดขัดบางช่วงเวลา",
                    "ต้นทุนพลังงาน/สินค้าโภคภัณฑ์ผันผวนปานกลาง"
                ],
                "impact_thailand": "กลาง"
            },
            "worst": {
                "probability": worst,
                "what_could_happen": [
                    "ความขัดแย้งขยายวงในภูมิภาค",
                    "เกิด disruption กับขนส่ง/พลังงาน",
                    "เหตุไซเบอร์กระทบภาคธุรกิจสำคัญ"
                ],
                "impact_thailand": "ลบสูง"
            }
        },
        "impact_to_thailand": {
            "economy": clamp(55 + intensity // 2),
            "logistics": clamp(52 + intensity // 2),
            "security": clamp(46 + intensity // 2),
            "energy_commodity": clamp(60 + intensity // 2),
            "tourism_investment": clamp(44 + intensity // 2),
        },
        "action_plan_percent": {
            "high_risk_50": {
                "personal": [
                    {"step": "ติดตามสัญญาณเสี่ยงวันละ 2 รอบ", "percent": 20},
                    {"step": "เตรียมน้ำ-อาหารฉุกเฉินและเอกสาร", "percent": 15},
                    {"step": "หลีกเลี่ยงโซนเสี่ยงตามแผนที่", "percent": 15}
                ],
                "business": [
                    {"step": "กระจาย supplier และเส้นทางขนส่ง", "percent": 20},
                    {"step": "เพิ่ม cyber hardening + SOC watch", "percent": 15},
                    {"step": "ทำ hedge ต้นทุนพลังงาน/วัตถุดิบ", "percent": 15}
                ]
            },
            "low_risk_10": {
                "personal": [
                    {"step": "ตรวจข่าวเชิงความเสี่ยงรายวัน", "percent": 4},
                    {"step": "ทบทวนแผนฉุกเฉินรายเดือน", "percent": 3},
                    {"step": "สำรองยา/เวชภัณฑ์พื้นฐาน", "percent": 3}
                ],
                "business": [
                    {"step": "monitor KPI ความเสี่ยงหลัก", "percent": 4},
                    {"step": "ทดสอบ BCP/DR ตามรอบ", "percent": 3},
                    {"step": "อัปเดต vendor risk score", "percent": 3}
                ]
            }
        },
        "preparedness_quantities": {
            "household_4_people": {
                "water_liters_per_day": 12,
                "water_liters_7d": 84,
                "water_liters_14d": 168,
                "rice_kg_14d": "14-18",
                "canned_protein_units_14d": "56-84",
                "medicine_days": 30
            },
            "business_mid_size": {
                "safety_stock_days_base": 14,
                "safety_stock_days_stress": 21,
                "backup_suppliers_per_critical_item": 2,
                "incident_response_sla_hours": 4
            }
        },
        "summary": "Thailand multi-risk level: medium with elevated energy/commodity + cyber + logistics coupling"
    }


async def llm_risk_scenario(news: List[Dict[str, Any]]) -> Dict[str, Any]:
    # keep deterministic baseline for stability; can switch to live LLM later
    return fallback_analysis(len(news))


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
def api_news(limit: int = Query(40, ge=10, le=200)):
    news = collect_news(limit)
    return {"ts": now_iso(), "count": len(news), "items": news}


@app.get("/api/risk")
async def api_risk():
    news = collect_news(80)
    analysis = await llm_risk_scenario(news)
    return {"ts": now_iso(), "focus": "Thailand", "news_count": len(news), "analysis": analysis}


@app.get("/api/geo-assess")
async def api_geo_assess(map_url: str = Query(...), radius_km: float = Query(2.0, ge=0.5, le=20)):
    coords = parse_map_link(map_url)
    news = collect_news(60)
    base = fallback_analysis(len(news))
    hotspot = {
        "area_risk_percent": clamp((base["impact_to_thailand"]["security"] + base["scores"]["conflict"]) // 2),
        "window": "24-72h",
        "recommendation": [
            f"หลีกเลี่ยงรัศมี {radius_km} กม. รอบจุดเสี่ยงช่วงเวลาชุมนุม",
            "ติดตามแจ้งเตือนทุก 2 ชั่วโมง",
            "เตรียมเส้นทางสำรองอย่างน้อย 2 เส้นทาง"
        ]
    }
    return {
        "ts": now_iso(),
        "input": {"map_url": map_url, "radius_km": radius_km, **coords},
        "hotspot_assessment": hotspot,
        "analysis_hint": "defensive early-warning only"
    }
