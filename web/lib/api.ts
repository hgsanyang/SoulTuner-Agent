export interface JourneySegment {
    segment_id: number;
    mood: string;
    description?: string;
    duration?: number;
    start_time?: number;
    total_songs?: number;
    songs?: any[];
}

export interface JourneyRequest {
    story?: string;
    mood_transitions?: { time: number; mood: string; intensity: number }[];
    duration?: number;
    user_preferences?: Record<string, any>;
    context?: Record<string, any>;
}

export type MoodTransitionInput = { time: number; mood: string; intensity: number };

export interface MusicCardResponse {
    headline?: string;
    subline?: string;
    hashtags?: string[];
}

export interface SSEEvent {
    type: 'start' | 'thinking' | 'response' | 'recommendations_start' | 'song'
        | 'recommendations_complete' | 'complete' | 'error'
        | 'journey_start' | 'journey_info' | 'journey_complete'
        | 'segment_start' | 'segment_complete' | 'transition_point';
    message?: string;
    text?: string;
    is_complete?: boolean;
    song?: { title: string; artist: string; [key: string]: any };
    error?: string;
    // Journey-specific fields
    segment?: JourneySegment;
    segment_id?: number;
    to_segment?: number;
    total_segments?: number;
    total_duration?: number;
    total_songs?: number;
    result?: {
        segments?: JourneySegment[];
        total_duration?: number;
        total_songs?: number;
    };
}

export interface StreamParams {
    query: string;
    chatHistory?: { role: string; content: string }[];
    llmProvider?: string;       // 模型供应商
    webSearchEnabled?: boolean; // 联网搜索开关
}

export function streamRecommendations(
    params: StreamParams,
    onEvent: (event: SSEEvent) => void
): () => void {
    const controller = new AbortController();

    const startStream = async () => {
        try {
            const response = await fetch(`http://localhost:8501/api/recommendations/stream`, {
                method: 'POST',
                headers: {
                    'Accept': 'text/event-stream',
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    query: params.query,
                    chat_history: params.chatHistory || [],
                    llm_provider: params.llmProvider || 'siliconflow',
                    web_search_enabled: params.webSearchEnabled !== false,  // 默认 true
                }),
                signal: controller.signal,
            });

            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }

            if (!response.body) {
                throw new Error('No body in response');
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();

                if (done) {
                    break;
                }

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');

                // Keep the last incomplete line in the buffer
                buffer = lines.pop() || '';

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const dataStr = line.slice(6);
                        if (dataStr === '[DONE]') {
                            onEvent({ type: 'complete' });
                            continue;
                        }

                        try {
                            const event: SSEEvent = JSON.parse(dataStr);
                            onEvent(event);
                        } catch (err) {
                            console.error('Failed to parse SSE JSON:', dataStr, err);
                        }
                    }
                }
            }
        } catch (err: any) {
            if (err.name === 'AbortError') {
                console.log('Stream aborted');
            } else {
                console.error('Stream error:', err);
                onEvent({ type: 'error', error: err.message || 'Unknown error' });
            }
        }
    };

    startStream();

    return () => {
        controller.abort();
    };
}

// ---- 用户行为事件上报 ----
export async function sendUserEvent(
    eventType: 'like' | 'unlike' | 'save' | 'skip' | 'dislike' | 'full_play' | 'repeat',
    songTitle: string,
    artist: string,
): Promise<void> {
    try {
        await fetch('http://localhost:8501/api/user-event', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                event_type: eventType,
                song_title: songTitle,
                artist: artist,
            }),
        });
    } catch (err) {
        console.warn('[UserEvent] 上报失败:', err);
    }
}

// ---- 加入本地（数据飞轮按需触发）----
export async function acquireSong(song: {
    title: string;
    artist: string;
    song_id?: string;
    platform?: string;
}): Promise<{ success: boolean; message: string; song?: any }> {
    const resp = await fetch('http://localhost:8501/api/acquire-song', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            title: song.title,
            artist: song.artist,
            song_id: song.song_id || '',
            platform: song.platform || 'netease',
        }),
    });
    if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: '加入本地失败' }));
        throw new Error(err.detail || `加入本地失败: ${resp.status}`);
    }
    return resp.json();
}

// ---- Journey 流式接口 ----
export function streamJourney(
    params: JourneyRequest,
    onEvent: (event: SSEEvent) => void,
): () => void {
    const controller = new AbortController();
    const run = async () => {
        try {
            const resp = await fetch('http://localhost:8501/api/journey/stream', {
                method: 'POST',
                headers: { 'Accept': 'text/event-stream', 'Content-Type': 'application/json' },
                body: JSON.stringify(params),
                signal: controller.signal,
            });
            if (!resp.ok) throw new Error(`Server error: ${resp.status}`);
            if (!resp.body) throw new Error('No body');
            const reader = resp.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop() || '';
                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const dataStr = line.slice(6);
                        if (dataStr === '[DONE]') { onEvent({ type: 'complete' }); continue; }
                        try { onEvent(JSON.parse(dataStr)); } catch { /* skip */ }
                    }
                }
            }
        } catch (err: any) {
            if (err.name !== 'AbortError') onEvent({ type: 'error', error: err.message });
        }
    };
    run();
    return () => controller.abort();
}

// ---- 搜索歌曲 ----
export async function searchMusic(query: string, genre?: string): Promise<any> {
    const resp = await fetch('http://localhost:8501/api/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, genre, limit: 20 }),
    });
    if (!resp.ok) throw new Error(`搜索失败: ${resp.status}`);
    return resp.json();
}

// ---- 生成音乐分享卡片 ----
export async function generateMusicCard(params: {
    title: string;
    artist: string;
    mood?: string;
    segmentLabel?: string;
}): Promise<MusicCardResponse> {
    // 简单的客户端生成（无额外后端端点）
    return {
        headline: `${params.mood || '旋律'} · ${params.title}`,
        subline: `${params.artist} — ${params.segmentLabel || '推荐'}`,
        hashtags: ['#音乐旅程', `#${params.mood || '推荐'}`, `#${params.artist}`],
    };
}
