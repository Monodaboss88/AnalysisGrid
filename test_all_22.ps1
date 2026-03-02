$base = "https://analysisgrid-production.up.railway.app"
Write-Host "=== FULL 22-ENDPOINT TEST ==="
Write-Host "Server: $base"
Write-Host ""

function Test-EP($num, $name, $url, $method, $body) {
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    try {
        $params = @{ Uri=$url; TimeoutSec=90; UseBasicParsing=$true }
        if ($method -eq "POST") {
            $params.Method = "POST"
            $params.ContentType = "application/json"
            $params.Body = $body
        }
        $r = Invoke-WebRequest @params
        $t = $sw.Elapsed.TotalSeconds.ToString('F1')
        $snippet = $r.Content.Substring(0, [Math]::Min(120, $r.Content.Length))
        Write-Host "$num. $name : $($r.StatusCode) (${t}s) $snippet"
    } catch {
        $t = $sw.Elapsed.TotalSeconds.ToString('F1')
        $msg = $_.Exception.Message
        if ($msg -match '(\d{3})') { $code = $Matches[1] } else { $code = "ERR" }
        # Try to get response body for error details
        $detail = ""
        try { $detail = $_.ErrorDetails.Message } catch {}
        Write-Host "$num. $name : $code (${t}s) $detail"
    }
}

# 1. AI Advisor (POST - needs full body, 422 expected without scanner data)
Test-EP  1 "AI Advisor        " "$base/api/ai/analyze" "POST" '{"symbol":"AAPL","timeframe":"swing","signal":"bullish","confidence":75,"bull_score":70,"bear_score":30,"price":200,"vah":205,"poc":200,"val":195,"vwap":200,"position":"above_poc","vwap_zone":"above","rsi":55,"rsi_zone":"neutral","spy_price":500,"spy_sma_20":495,"spy_sma_50":490,"spy_sma_200":480,"vix":15}'

# 2. MTF Auction
Test-EP  2 "MTF Auction       " "$base/api/analyze/live/mtf/AAPL" "GET"

# 3. MTF + AI
Test-EP  3 "MTF + AI          " "$base/api/analyze/live/mtf/AAPL/ai" "POST" '{}'

# 4. Analyze Live
Test-EP  4 "Analyze Live      " "$base/api/analyze/live/AAPL" "GET"

# 5. Regime Scanner
Test-EP  5 "Regime Scanner    " "$base/api/regime-scan?tickers=AAPL,MSFT" "GET"

# 6. Regime Levels
Test-EP  6 "Regime Levels     " "$base/api/regime-levels/AAPL" "GET"

# 7. Buffett Scanner
Test-EP  7 "Buffett Scanner   " "$base/api/buffett-scan?tickers=AAPL" "GET"

# 8. War Room
Test-EP  8 "War Room          " "$base/api/war-room?tickers=AAPL" "GET"

# 9. Signal Probability
Test-EP  9 "Signal Probability" "$base/api/signal/AAPL" "GET"

# 10. Signal Quick
Test-EP 10 "Signal Quick      " "$base/api/signal/AAPL/quick" "GET"

# 11. Alpha Scanner
Test-EP 11 "Alpha Scanner     " "$base/api/alpha/scan?tickers=AAPL,MSFT" "GET"

# 12. Entry Scanner
Test-EP 12 "Entry Scanner     " "$base/api/entry-scan/scan/AAPL" "GET"

# 13. Combo Scanner
Test-EP 13 "Combo Scanner     " "$base/api/combo-scan?tickers=AAPL,MSFT" "GET"

# 14. Options Flow
Test-EP 14 "Options Flow      " "$base/api/options-flow?tickers=AAPL" "GET"

# 15. Research Builder
Test-EP 15 "Research Builder  " "$base/api/research/build" "POST" '{"config":{"title":"Test","layer2":[{"ticker":"AAPL"},{"ticker":"MSFT"}],"layer3":[]},"mode":"full"}'

# 16. Catalyst Scanner (scan/live)
Test-EP 16 "Catalyst Scanner  " "$base/api/scan/live?watchlist=default&limit=2" "GET"

# 17. Sustainability Quick
Test-EP 17 "Sustainability    " "$base/api/sustainability/quick?symbols=AAPL" "GET"

# 18. Sustainability Analyze
Test-EP 18 "Sustain Analyze   " "$base/api/sustainability/analyze?symbol=AAPL" "GET"

# 19. Sustainability Scan
Test-EP 19 "Sustain Scan      " "$base/api/sustainability/scan" "POST" '{"symbols":["AAPL"]}'

# 20. Trading Cards
Test-EP 20 "Trading Cards     " "$base/api/card/AAPL/data" "GET"

# 21. Quote
Test-EP 21 "Quote             " "$base/api/quote/AAPL" "GET"

# 22. Structure Reversal
Test-EP 22 "Structure Reversal" "$base/api/structure/reversals/AAPL" "GET"

Write-Host ""
Write-Host "=== POST-TEST HEALTH ==="
$r = Invoke-WebRequest -Uri "$base/api/health" -TimeoutSec 15 -UseBasicParsing
Write-Host $r.Content
