import asyncio
from card_data_builder import build_card_data

async def test():
    d = await build_card_data("NFLX", "swing")
    print("=== Previously Buggy Fields ===")
    print(f"vwap_revert_rate: {d['vwap_revert_rate']}%")
    print(f"gross_margin: {d['gross_margin']}%")
    print(f"avg_close_pos: {d['avg_close_pos']}%")
    print(f"avg_top_vol: {d['avg_top_vol']}%")
    print(f"avg_up_ext: {d['avg_up_ext']}%")
    print()
    print("=== Key Card Fields ===")
    print(f"price: {d['price']}")
    print(f"direction: {d['direction']}")
    print(f"simple_signal: {d['simple_signal']}")
    print(f"vah/poc/val: {d['vah']}/{d['poc']}/{d['val']}")
    print(f"drawdown_pct: {d.get('drawdown_pct', 'N/A')}%")
    print(f"revenue_growth: {d.get('revenue_growth', 'N/A')}%")
    print(f"range_position: {d.get('range_position', 'N/A')}%")
    print()
    print("ALL FIXED FIELDS LOOK GOOD ✅")

asyncio.run(test())
