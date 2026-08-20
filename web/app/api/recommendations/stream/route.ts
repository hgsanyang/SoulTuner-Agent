/**
 * Next.js API路由：音乐推荐流式接口（SSE代理）
 */

import { NextRequest } from 'next/server';

// Server-side proxy traffic must use the private backend address.  The public
// browser URL may intentionally be empty so Studio visitors stay same-origin.
const API_BASE_URL = process.env.BACKEND_INTERNAL_URL || 'http://localhost:8501';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();

    // 转发请求到后端FastAPI服务器
    const response = await fetch(`${API_BASE_URL}/api/recommendations/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-SoulTuner-Profile': request.headers.get('X-SoulTuner-Profile') || '',
        'X-SoulTuner-Mode': request.headers.get('X-SoulTuner-Mode') || '',
        'X-SoulTuner-Session': request.headers.get('X-SoulTuner-Session') || '',
      },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      throw new Error(`Backend responded with status: ${response.status}`);
    }

    // 返回流式响应
    return new Response(response.body, {
      headers: {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'X-Accel-Buffering': 'no',
      },
    });
  } catch (error) {
    console.error('SSE proxy error:', error);
    return new Response(
      JSON.stringify({ error: 'Failed to connect to backend' }),
      {
        status: 500,
        headers: { 'Content-Type': 'application/json' },
      }
    );
  }
}

