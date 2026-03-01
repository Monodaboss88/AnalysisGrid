$base = "https://analysisgrid-production.up.railway.app"
Write-Host "=== FINAL TEST: ALL ENDPOINTS ==="
Write-Host ""

function Test-Endpoint($name, $url, $method, $body) {
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
        Write-Host "$name : $($r.StatusCode) (${t}s)"
    } catch {
        $t = $sw.Elapsed.TotalSeconds.ToString('F1')
        $msg = $_.Exception.Message
        if ($msg -match '(\d{3})') { $code = $Matches[1] } else { $code = "ERR" }
        Write-Host "$name : $code (${t}s)"
    }
}

Test-Endpoint "Health         " "$base/api/health" "GET"
Test-Endpoint "AI Analyze     " "$base/api/ai/analyze" "POST" '{"symbol":"AAPL","timeframe":"swing"}'
Test-Endpoint "MTF Auction    " "$base/api/analyze/live/mtf/AAPL" "GET"
Test-Endpoint "Regime Levels  " "$base/api/regime-levels/AAPL" "GET"
Test-Endpoint "Buffett        " "$base/api/buffett-scan?tickers=AAPL" "GET"
Test-Endpoint "War Room       " "$base/api/war-room?tickers=AAPL" "GET"
Test-Endpoint "Signal         " "$base/api/signal/AAPL" "GET"
Test-Endpoint "Alpha Scanner  " "$base/api/alpha/scan?tickers=AAPL" "GET"
Test-Endpoint "Entry Scanner  " "$base/api/entry-scan/scan/AAPL" "GET"
Test-Endpoint "Combo Scanner  " "$base/api/combo-scan?tickers=AAPL,MSFT" "GET"
Test-Endpoint "Options Flow   " "$base/api/options-flow?tickers=AAPL,MSFT" "GET"
Test-Endpoint "Research Build " "$base/api/research/build" "POST" '{"config":{"title":"Final","layer2":[{"ticker":"AAPL"},{"ticker":"MSFT"}],"layer3":[]},"mode":"full"}'
Test-Endpoint "Sustainability " "$base/api/sustainability/quick?symbols=AAPL,MSFT" "GET"
Test-Endpoint "Card Data      " "$base/api/card/AAPL/data" "GET"
Test-Endpoint "Card Execution " "$base/api/card/AAPL/execution" "GET"
Test-Endpoint "Card Thesis    " "$base/api/card/AAPL/thesis" "GET"
Test-Endpoint "Card Both      " "$base/api/card/AAPL/both" "GET"

Write-Host ""
Write-Host "=== POST-TEST HEALTH ==="
$r = Invoke-WebRequest -Uri "$base/api/health" -TimeoutSec 10 -UseBasicParsing
Write-Host $r.Content
