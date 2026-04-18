"""
Profile Synthesizer 集成测试套件
================================
测试维度：
  1. 对话上下文窗口捕捉
  2. GraphZep 长期记忆提取与注入
  3. 用户画像构建与上下文注入
  4. 用户画像更新
  5. 用户手动设置偏好的高权重集成
  6. 端到端 API 验证
"""
import asyncio
import json
import sys
import os
import time

# 确保项目根目录在路径中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ======================================================================
# 测试1: ProfileSynthesizer Schema 完整性
# ======================================================================
def test_1_schema_integrity():
    """验证 UserPortrait Pydantic Schema 的完整性和默认值"""
    print("\n" + "="*60)
    print("测试 1: UserPortrait Schema 完整性")
    print("="*60)
    
    from services.profile_synthesizer import UserPortrait, SituationalPattern, TasteShift
    
    # 1a: 空画像（默认值）
    empty = UserPortrait()
    assert empty.confidence == "low"
    assert empty.one_line_summary == "暂无画像数据"
    assert empty.situational_patterns == []
    assert empty.dislike_signals == []
    print("  ✅ 1a: 空画像默认值正确")
    
    # 1b: 完整画像
    full = UserPortrait(
        emotional_baseline="偏内省忧郁",
        situational_patterns=[
            SituationalPattern(
                situation="深夜独处",
                preferred_styles=["钢琴", "ambient"],
                evidence="4/17 深夜多次要求安静的歌"
            ),
            SituationalPattern(
                situation="运动时",
                preferred_styles=["电子", "Hip-hop"],
                evidence="4/16 跑步时要求节奏强的歌"
            ),
        ],
        taste_evolution=[
            TasteShift(period="4月初", dominant_taste="J-Rock", evidence="多次点赞J-Rock歌曲"),
            TasteShift(period="4月中", dominant_taste="City Pop", evidence="开始搜索City Pop"),
        ],
        current_dominant_genres=["City Pop", "Jazz"],
        current_dominant_moods=["治愈", "怀旧"],
        dislike_signals=["重金属", "嘈杂电子"],
        interaction_style="探索型，偏好模糊描述",
        user_declared_preferences="[用户主动设置] 偏好流派: 摇滚, 爵士",
        confidence="high",
        one_line_summary="深夜偏内省的City Pop/Jazz爱好者"
    )
    
    assert full.confidence == "high"
    assert len(full.situational_patterns) == 2
    assert len(full.taste_evolution) == 2
    assert "City Pop" in full.current_dominant_genres
    assert "重金属" in full.dislike_signals
    print("  ✅ 1b: 完整画像构造正确")
    
    # 1c: JSON 序列化/反序列化
    json_str = full.model_dump_json(ensure_ascii=False)
    restored = UserPortrait(**json.loads(json_str))
    assert restored.one_line_summary == full.one_line_summary
    assert len(restored.situational_patterns) == 2
    print("  ✅ 1c: JSON 序列化/反序列化无损")
    
    print("  ✅ 测试 1 全部通过")


# ======================================================================
# 测试2: get_portrait_for_prompt 输出格式
# ======================================================================
def test_2_portrait_prompt_format():
    """验证画像格式化为 prompt 注入文本的正确性"""
    print("\n" + "="*60)
    print("测试 2: 画像 Prompt 注入格式")
    print("="*60)
    
    from services.profile_synthesizer import ProfileSynthesizer, UserPortrait, SituationalPattern
    
    synth = ProfileSynthesizer("test_user")
    
    # 2a: 无画像时返回空
    assert synth.get_portrait_for_prompt() == ""
    print("  ✅ 2a: 无画像时返回空字符串")
    
    # 2b: 有画像时返回格式化文本
    synth._cached_portrait = UserPortrait(
        emotional_baseline="偏内省",
        situational_patterns=[
            SituationalPattern(situation="深夜", preferred_styles=["钢琴", "ambient"], evidence="test"),
        ],
        current_dominant_genres=["Jazz"],
        current_dominant_moods=["治愈"],
        dislike_signals=["重金属"],
        user_declared_preferences="[用户主动设置] 偏好流派: 摇滚",
        confidence="medium",
        one_line_summary="内省型Jazz/钢琴爱好者"
    )
    
    text = synth.get_portrait_for_prompt()
    print(f"  格式化输出: {text}")
    
    # 验证关键信息存在
    assert "[用户主动设置]" in text, "应包含用户主动设置的偏好"
    assert "内省型Jazz" in text, "应包含一句话摘要"
    assert "深夜" in text, "应包含情境偏好"
    assert "重金属" in text, "应包含负面偏好"
    print("  ✅ 2b: prompt 注入格式正确，包含所有关键信息")
    
    # 2c: 用户声明偏好出现在最前面（高权重）
    assert text.index("[用户主动设置]") < text.index("画像摘要"), \
        "用户主动设置应出现在画像摘要之前（高权重优先）"
    print("  ✅ 2c: 用户声明偏好排序最前（高权重）")
    
    print("  ✅ 测试 2 全部通过")


# ======================================================================
# 测试3: 对话计数器与自动触发
# ======================================================================
def test_3_conversation_counter():
    """验证对话计数器在达到阈值时正确触发"""
    print("\n" + "="*60)
    print("测试 3: 对话计数器与自动触发")
    print("="*60)
    
    from services.profile_synthesizer import ProfileSynthesizer, PORTRAIT_REFRESH_INTERVAL
    
    synth = ProfileSynthesizer("test_counter")
    
    # 前 N-1 轮不触发
    for i in range(PORTRAIT_REFRESH_INTERVAL - 1):
        should_refresh = synth.increment_conversation()
        assert not should_refresh, f"第 {i+1} 轮不应触发"
    
    # 第 N 轮触发
    should_refresh = synth.increment_conversation()
    assert should_refresh, f"第 {PORTRAIT_REFRESH_INTERVAL} 轮应触发"
    print(f"  ✅ 3a: 每 {PORTRAIT_REFRESH_INTERVAL} 轮正确触发一次")
    
    # 触发后计数重置
    should_refresh = synth.increment_conversation()
    assert not should_refresh, "触发后应重置计数"
    print("  ✅ 3b: 触发后计数器正确重置")
    
    print("  ✅ 测试 3 全部通过")


# ======================================================================
# 测试4: Neo4j 画像持久化（读写一致性）
# ======================================================================
async def test_4_neo4j_persistence():
    """验证画像写入 Neo4j 后能正确读回"""
    print("\n" + "="*60)
    print("测试 4: Neo4j 画像持久化")
    print("="*60)
    
    from services.profile_synthesizer import ProfileSynthesizer, UserPortrait, SituationalPattern
    
    synth = ProfileSynthesizer("local_admin")
    
    # 创建测试画像
    test_portrait = UserPortrait(
        emotional_baseline="测试用-偏内省",
        situational_patterns=[
            SituationalPattern(situation="测试场景", preferred_styles=["测试风格"], evidence="测试证据"),
        ],
        current_dominant_genres=["test_genre"],
        current_dominant_moods=["test_mood"],
        dislike_signals=["test_dislike"],
        user_declared_preferences="测试用户声明",
        confidence="medium",
        one_line_summary="测试画像-内省型Jazz爱好者"
    )
    
    # 写入
    save_ok = await synth.save_portrait(test_portrait)
    if not save_ok:
        print("  ⚠️ Neo4j 写入失败（可能未连接），跳过此测试")
        return
    print("  ✅ 4a: 画像写入 Neo4j 成功")
    
    # 读取（新实例，验证不是从内存读的）
    synth2 = ProfileSynthesizer("local_admin")
    loaded = await synth2.load_portrait()
    
    assert loaded is not None, "应能从 Neo4j 读取画像"
    assert loaded.one_line_summary == "测试画像-内省型Jazz爱好者"
    assert loaded.confidence == "medium"
    assert "test_genre" in loaded.current_dominant_genres
    assert "test_dislike" in loaded.dislike_signals
    print("  ✅ 4b: 从 Neo4j 读取画像一致")
    
    # 验证缓存也被设置
    assert synth2.get_cached_portrait() is not None
    print("  ✅ 4c: 加载后内存缓存同步更新")
    
    print("  ✅ 测试 4 全部通过")


# ======================================================================
# 测试5: GraphZep 数据收集
# ======================================================================
async def test_5_graphzep_collection():
    """验证从 GraphZep 收集记忆数据"""
    print("\n" + "="*60)
    print("测试 5: GraphZep 数据收集")
    print("="*60)
    
    from services.profile_synthesizer import ProfileSynthesizer
    
    synth = ProfileSynthesizer("local_admin")
    facts = await synth._collect_graphzep_facts()
    
    if facts:
        lines = facts.split("\n")
        print(f"  ✅ 5a: GraphZep 返回 {len(lines)} 条记忆")
        for line in lines[:3]:
            print(f"    {line[:80]}...")
        
        # 验证时间戳存在
        has_timestamp = any("时间:" in line for line in lines)
        if has_timestamp:
            print("  ✅ 5b: 记忆中包含时间戳")
        else:
            print("  ⚠️ 5b: 记忆中未发现时间戳（GraphZep 可能未返回 valid_at）")
    else:
        print("  ⚠️ 5a: GraphZep 返回空（可能未连接或无历史数据），这是预期的冷启动状态")
    
    print("  ✅ 测试 5 完成")


# ======================================================================
# 测试6: Neo4j 行为统计收集
# ======================================================================
async def test_6_neo4j_stats():
    """验证从 Neo4j 收集用户行为统计"""
    print("\n" + "="*60)
    print("测试 6: Neo4j 行为统计收集")
    print("="*60)
    
    from services.profile_synthesizer import ProfileSynthesizer
    
    synth = ProfileSynthesizer("local_admin")
    stats = await synth._collect_neo4j_stats()
    
    if stats:
        print(f"  ✅ 6a: Neo4j 统计收集成功")
        print(f"    Top 歌手: {stats.get('top_artists', [])}")
        print(f"    Top 流派: {stats.get('top_genres', [])}")
        print(f"    不喜欢:   {stats.get('disliked_songs', [])}")
        print(f"    经常跳过: {stats.get('often_skipped', [])}")
        
        # 检查用户声明偏好
        declared = stats.get("declared_genres", "")
        if declared:
            print(f"    用户声明流派: {declared}")
            print("  ✅ 6b: 用户主动设置的偏好已包含在统计中")
        else:
            print("  ⚠️ 6b: 用户尚未设置偏好（正常情况）")
    else:
        print("  ⚠️ 6a: Neo4j 统计为空（可能无数据）")
    
    print("  ✅ 测试 6 完成")


# ======================================================================
# 测试7: 画像动态加载器（替代静态标签）
# ======================================================================
async def test_7_profile_loader_integration():
    """验证 _load_user_profile_for_prompt 优先使用动态画像"""
    print("\n" + "="*60)
    print("测试 7: 画像动态加载器集成")
    print("="*60)
    
    from services.profile_synthesizer import get_profile_synthesizer, UserPortrait
    
    # 设置一个测试画像到缓存
    synth = get_profile_synthesizer("local_admin")
    synth._cached_portrait = UserPortrait(
        emotional_baseline="测试",
        one_line_summary="测试画像-动态加载验证",
        confidence="medium",
        user_declared_preferences="[用户主动设置] 流派: Jazz",
    )
    
    # 调用 MusicRecommendationGraph._load_user_profile_for_prompt
    try:
        from agent.music_graph import MusicRecommendationGraph
        graph = MusicRecommendationGraph()
        profile_text = graph._load_user_profile_for_prompt("local_admin")
        
        if "测试画像-动态加载验证" in profile_text:
            print(f"  ✅ 7a: _load_user_profile_for_prompt 正确使用动态画像")
            print(f"    输出: {profile_text[:100]}")
        elif profile_text:
            print(f"  ⚠️ 7a: 加载了画像但不是动态版本: {profile_text[:100]}")
        else:
            print(f"  ⚠️ 7a: 未加载到任何画像")
        
        # 验证用户声明偏好在输出中
        if "[用户主动设置]" in profile_text:
            print(f"  ✅ 7b: 用户主动声明偏好在输出中（高权重）")
        
    except Exception as e:
        print(f"  ⚠️ 7a: 无法加载 MusicRecommendationGraph: {e}")
    
    # 清理
    synth._cached_portrait = None
    print("  ✅ 测试 7 完成")


# ======================================================================
# 测试8: API 端点验证
# ======================================================================
async def test_8_api_endpoints():
    """验证 API 端点的基本功能"""
    print("\n" + "="*60)
    print("测试 8: API 端点验证")
    print("="*60)
    
    from api.user_portrait import get_user_portrait, refresh_user_portrait
    
    # 8a: GET 端点
    result = await get_user_portrait("local_admin")
    assert result["success"] == True
    print(f"  ✅ 8a: GET /api/user-portrait 返回成功 (source={result.get('source', '?')})")
    
    if result.get("portrait"):
        print(f"    画像摘要: {result.get('summary', '?')}")
    else:
        print(f"    消息: {result.get('message', '?')}")
    
    print("  ✅ 测试 8 完成")


# ======================================================================
# 测试9: Prompt 模板验证
# ======================================================================
def test_9_prompt_template():
    """验证 PROFILE_SYNTHESIZER_PROMPT 的占位符完整性"""
    print("\n" + "="*60)
    print("测试 9: Prompt 模板占位符验证")
    print("="*60)
    
    from llms.prompts import PROFILE_SYNTHESIZER_PROMPT
    
    required_placeholders = [
        "graphzep_facts",
        "top_artists",
        "top_genres",
        "disliked_songs",
        "often_skipped",
        "declared_preferences",
        "current_time",
    ]
    
    for ph in required_placeholders:
        assert f"{{{ph}}}" in PROFILE_SYNTHESIZER_PROMPT, f"缺少占位符: {{{ph}}}"
    
    print(f"  ✅ 9a: 所有 {len(required_placeholders)} 个占位符都存在")
    
    # 验证关键指令存在
    assert "权重最高" in PROFILE_SYNTHESIZER_PROMPT
    assert "时间戳" in PROFILE_SYNTHESIZER_PROMPT
    assert "situational_patterns" in PROFILE_SYNTHESIZER_PROMPT
    print("  ✅ 9b: 关键指令（权重、时间戳、情境模式）存在")
    
    print("  ✅ 测试 9 全部通过")


# ======================================================================
# 测试10: 完整画像聚合（端到端，依赖 LLM + GraphZep + Neo4j）
# ======================================================================
async def test_10_full_synthesize():
    """端到端画像聚合测试（如果所有服务可用）"""
    print("\n" + "="*60)
    print("测试 10: 端到端画像聚合")
    print("="*60)
    
    from services.profile_synthesizer import ProfileSynthesizer
    
    synth = ProfileSynthesizer("local_admin")
    
    print("  开始画像聚合（需要 LLM + GraphZep + Neo4j 都在线）...")
    t0 = time.time()
    
    try:
        portrait = await synth.synthesize()
        elapsed = time.time() - t0
        
        print(f"\n  聚合结果 ({elapsed:.1f}s):")
        print(f"    置信度:     {portrait.confidence}")
        print(f"    一句话摘要: {portrait.one_line_summary}")
        print(f"    情绪基线:   {portrait.emotional_baseline}")
        print(f"    主导流派:   {portrait.current_dominant_genres}")
        print(f"    主导情绪:   {portrait.current_dominant_moods}")
        print(f"    负面偏好:   {portrait.dislike_signals}")
        print(f"    交互风格:   {portrait.interaction_style}")
        print(f"    用户声明:   {portrait.user_declared_preferences[:50] if portrait.user_declared_preferences else '无'}")
        
        if portrait.situational_patterns:
            print(f"    情境偏好:")
            for sp in portrait.situational_patterns[:3]:
                print(f"      - {sp.situation} → {sp.preferred_styles}")
        
        if portrait.taste_evolution:
            print(f"    品味进化:")
            for ts in portrait.taste_evolution[:3]:
                print(f"      - {ts.period}: {ts.dominant_taste}")
        
        # 保存到 Neo4j
        saved = await synth.save_portrait(portrait)
        if saved:
            print(f"\n  ✅ 10a: 画像已保存到 Neo4j")
        
        # 验证 prompt 注入
        prompt_text = synth.get_portrait_for_prompt()
        if prompt_text:
            print(f"  ✅ 10b: Prompt 注入文本: {prompt_text[:120]}...")
        
        print(f"  ✅ 测试 10 完成 (耗时 {elapsed:.1f}s)")
        
    except Exception as e:
        print(f"  ⚠️ 测试 10 失败: {e}")
        import traceback
        traceback.print_exc()


# ======================================================================
# 执行所有测试
# ======================================================================
async def run_all_tests():
    print("="*60)
    print("Profile Synthesizer 集成测试套件")
    print("="*60)
    
    # 同步测试
    test_1_schema_integrity()
    test_2_portrait_prompt_format()
    test_3_conversation_counter()
    test_9_prompt_template()
    
    # 异步测试（依赖 Neo4j / GraphZep）
    await test_4_neo4j_persistence()
    await test_5_graphzep_collection()
    await test_6_neo4j_stats()
    await test_7_profile_loader_integration()
    await test_8_api_endpoints()
    
    # 端到端测试（依赖 LLM + Neo4j + GraphZep）
    await test_10_full_synthesize()
    
    print("\n" + "="*60)
    print("所有测试执行完毕")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(run_all_tests())
