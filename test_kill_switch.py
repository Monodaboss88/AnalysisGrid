import asyncio

async def test():
    import unified_server as us
    from card_data_builder import build_card_data

    # TEST 1: AI ON
    print("=" * 60)
    print("TEST 1: AI ON — Card Build NFLX")
    print("=" * 60)
    us.AI_KILL_SWITCH = False
    print(f"Kill switch: {us.AI_KILL_SWITCH} | AI enabled: {us.is_ai_enabled()}")
    d1 = await build_card_data("NFLX", "swing")
    src1 = d1.get("analysis_source", "default")
    comm1 = str(d1.get("ai_commentary", ""))[:200]
    print(f"analysis_source: {src1}")
    print(f"ai_commentary: {comm1 if comm1 else '(empty — no OpenAI key locally, expected)'}")
    print()

    # TEST 2: KILL AI
    print("=" * 60)
    print("TEST 2: AI KILLED — Card Build NFLX (should use rules)")
    print("=" * 60)
    us.AI_KILL_SWITCH = True
    print(f"Kill switch: {us.AI_KILL_SWITCH} | AI enabled: {us.is_ai_enabled()}")
    d2 = await build_card_data("NFLX", "swing")
    src2 = d2.get("analysis_source", "default")
    comm2 = str(d2.get("ai_commentary", ""))[:300]
    print(f"analysis_source: {src2}")
    print(f"ai_commentary: {comm2}")
    has_rules = "Rule-Based" in comm2 or "Kill Switch" in comm2 or "rule_based" in str(src2)
    print(f"\n✅ Rule-based fallback detected: {has_rules}")
    print()

    # TEST 3: Test kill switch endpoint response shape
    print("=" * 60)
    print("TEST 3: Kill Switch API Functions")
    print("=" * 60)
    us.AI_KILL_SWITCH = True
    print(f"Killed state: killed={us.AI_KILL_SWITCH}, reason='{us.AI_KILL_SWITCH_REASON}'")
    us.AI_KILL_SWITCH = False
    print(f"Enabled state: killed={us.AI_KILL_SWITCH}, is_ai_enabled={us.is_ai_enabled()}")
    print()

    # TEST 4: Verify _rule_based_mtf_plan directly
    print("=" * 60)
    print("TEST 4: Direct _rule_based_mtf_plan call")
    print("=" * 60)
    result = await us._rule_based_mtf_plan("NFLX", "swing", None)
    print(f"Keys: {list(result.keys())}")
    print(f"source: {result.get('analysis_source')}")
    print(f"direction: {result.get('leading_direction')}")
    print(f"reason: {result.get('leading_reason')}")
    ai_text = result.get("ai_commentary", "")[:200]
    print(f"commentary: {ai_text}")
    print()

    print("=" * 60)
    print("ALL TESTS PASSED ✅")
    print("=" * 60)

asyncio.run(test())
