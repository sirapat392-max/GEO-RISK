# GEO-RISK (Thailand-first)

## Run (local server)
```bash
cd /root/GEO-RISK/deploy
docker compose up -d
```

## API
- `GET /health`
- `GET /api/news`
- `GET /api/risk`
- `GET /api/geo-assess?map_url=<google_maps_url>&radius_km=2`

## UI
- `GET /`

## Notes
- Forecast scenarios: Best/Base/Worst (%)
- Impact to Thailand (%): economy/logistics/security/energy/tourism
- Action plan %: high_risk_50 + low_risk_10
- Preparedness quantities included
