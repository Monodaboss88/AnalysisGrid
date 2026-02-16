"""
Run Sustainability Endpoints
==============================
FastAPI router for the Run Sustainability Analyzer.

Endpoints:
  GET  /api/sustainability/analyze?symbol=NVDA     — Single stock analysis
  POST /api/sustainability/scan                    — Multi-stock scan
  GET  /api/sustainability/quick?symbols=MU,LRCX   — Quick scorecard for multiple
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional
import traceback

from run_sustainability_analyzer import RunSustainabilityAnalyzer

sustainability_router = APIRouter(prefix="/api/sustainability", tags=["Run Sustainability"])

analyzer = RunSustainabilityAnalyzer()


class ScanRequest(BaseModel):
    symbols: List[str]


@sustainability_router.get("/analyze")
async def analyze_sustainability(symbol: str = Query(..., description="Stock ticker symbol")):
    """Full sustainability analysis for a single stock"""
    try:
        result = analyzer.analyze(symbol.upper().strip())
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@sustainability_router.post("/scan")
async def scan_sustainability(request: ScanRequest):
    """Analyze multiple stocks and rank by sustainability score"""
    try:
        symbols = [s.upper().strip() for s in request.symbols if s.strip()]
        if not symbols:
            raise HTTPException(status_code=400, detail="No symbols provided")
        if len(symbols) > 20:
            raise HTTPException(status_code=400, detail="Max 20 symbols per scan")
        
        results = analyzer.scan_multiple(symbols)
        return {
            "count": len(results),
            "results": results
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@sustainability_router.get("/quick")
async def quick_sustainability(symbols: str = Query(..., description="Comma-separated symbols")):
    """Quick scorecard: returns symbol, score, grade, verdict for multiple stocks"""
    try:
        sym_list = [s.upper().strip() for s in symbols.split(",") if s.strip()]
        if not sym_list:
            raise HTTPException(status_code=400, detail="No symbols provided")
        if len(sym_list) > 20:
            raise HTTPException(status_code=400, detail="Max 20 symbols")
        
        results = []
        for sym in sym_list:
            r = analyzer.analyze(sym)
            if "error" not in r:
                results.append({
                    "symbol": r["symbol"],
                    "company_name": r["company_name"],
                    "current_price": r["current_price"],
                    "market_cap_tier": r["market_cap_tier"],
                    "annual_return_pct": r["annual_return_pct"],
                    "overall_score": r["overall_score"],
                    "overall_grade": r["overall_grade"],
                    "sustainability_verdict": r["sustainability_verdict"],
                    "cycle_phase": r["cycle_position"]["estimated_cycle_phase"],
                    "multiple_signal": r["multiple_expansion"]["signal"],
                    "revenue_signal": r["revenue_health"]["signal"],
                    "recommended_action": r["recommended_action"],
                })
        
        results.sort(key=lambda x: x["overall_score"], reverse=True)
        return {"count": len(results), "scorecards": results}
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
