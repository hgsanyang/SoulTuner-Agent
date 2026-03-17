/**
 * Next.js API路由：调用 WAN 大模型生成音乐推荐卡片文案
 *
 * 说明：
 * - 这里作为前端到 WAN 的轻量代理，避免在浏览器暴露密钥。
 * - 实际的 WAN API 地址 / 模型名称请根据你接入的服务商文档替换。
 */

import { NextRequest } from 'next/server';

// 优先使用专门的 WAN 环境变量，其次复用你在 setting.json 里已经配置的 SiliconFlow 网关
// （即：WAN 其实走的是 SiliconFlow 的大模型网关）
const WAN_API_URL =
  process.env.WAN_API_URL ||
  process.env.SILICONFLOW_BASE_URL ||
  ''; // 例如：https://api.siliconflow.cn/v1
const WAN_API_KEY =
  process.env.WAN_API_KEY ||
  process.env.SILICONFLOW_API_KEY ||
  '';

export async function POST(request: NextRequest) {
  if (!WAN_API_URL || !WAN_API_KEY) {
    return new Response(
      JSON.stringify({
        error: 'WAN_API_URL 或 WAN_API_KEY 未配置，请在环境变量中设置。',
      }),
      {
        status: 500,
        headers: { 'Content-Type': 'application/json' },
      }
    );
  }

  try {
    const body = await request.json();
    const { title, artist, mood, segmentLabel, story } = body || {};

    const prompt = [
      '你是一个为音乐旅程生成分享卡片文案的中文文案助手。',
      '请根据歌曲和情绪，输出一份适合社交媒体分享的卡片文案。',
      '',
      `歌曲：${title || '未知歌曲'}`,
      `演唱者：${artist || '未知艺术家'}`,
      mood ? `情绪：${mood}` : '',
      segmentLabel ? `所在旅程章节：${segmentLabel}` : '',
      story ? `用户的整体故事：${story}` : '',
      '',
      '请严格输出 JSON，字段包括：',
      '{',
      '  "headline": "卡片上的主标题，简短有画面感",',
      '  "subline": "一行补充说明，描述这段音乐适合的场景或心情",',
      '  "hashtags": ["#标签1", "#标签2"]',
      '}',
    ]
      .filter(Boolean)
      .join('\n');

    const wanResponse = await fetch(WAN_API_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${WAN_API_KEY}`,
      },
      body: JSON.stringify({
        // TODO: 根据实际 WAN 接口规范调整 model / messages 字段
        model: 'wan-music-card',
        messages: [
          {
            role: 'user',
            content: prompt,
          },
        ],
      }),
    });

    if (!wanResponse.ok) {
      const text = await wanResponse.text();
      console.error('WAN API error:', wanResponse.status, text);
      return new Response(
        JSON.stringify({
          error: '调用 WAN 大模型失败',
          detail: text,
        }),
        {
          status: 500,
          headers: { 'Content-Type': 'application/json' },
        }
      );
    }

    const data = await wanResponse.json();

    // 根据你实际的 WAN 返回结构解析出文本内容，这里假设为 data.choices[0].message.content
    const content: string =
      data?.choices?.[0]?.message?.content ||
      data?.output ||
      data?.result ||
      '';

    let parsed;
    try {
      parsed = JSON.parse(content);
    } catch {
      // 如果模型没有严格返回 JSON，可以做一次简易兜底
      parsed = { headline: title || 'AI 音乐旅程', subline: content.slice(0, 80), hashtags: [] };
    }

    return new Response(JSON.stringify(parsed), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    });
  } catch (error) {
    console.error('WAN music card API error:', error);
    return new Response(JSON.stringify({ error: '生成音乐卡片失败' }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' },
    });
  }
}


