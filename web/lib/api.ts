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
    llm_provider?: string;  // 已弃用：模型提供商统一由后端 settings 管理
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

// ---- 查询用户喜欢/不喜欢的歌曲（从 Neo4j 同步）----

export interface BackendSong {
    title: string;
    artist: string;
    audio_url?: string;
    cover_url?: string;
    album?: string;
    genre?: string;
    moods?: string[];
    themes?: string[];
}

export interface LikedSongBackend {
    song: BackendSong;
    reason: string;
    source: string;
    score: number;
}

export interface DislikedSongBackend {
    title: string;
    artist: string;
    audio_url?: string;
    cover_url?: string;
    album?: string;
    disliked_at?: number;
}

export async function fetchLikedSongs(limit: number = 50): Promise<LikedSongBackend[]> {
    try {
        const resp = await fetch(`http://localhost:8501/api/liked-songs?limit=${limit}`);
        if (!resp.ok) return [];
        const data = await resp.json();
        return data.success ? data.songs : [];
    } catch (err) {
        console.warn('[API] fetchLikedSongs 失败:', err);
        return [];
    }
}

export async function fetchDislikedSongs(limit: number = 50): Promise<DislikedSongBackend[]> {
    try {
        const resp = await fetch(`http://localhost:8501/api/disliked-songs?limit=${limit}`);
        if (!resp.ok) return [];
        const data = await resp.json();
        return data.success ? data.songs : [];
    } catch (err) {
        console.warn('[API] fetchDislikedSongs 失败:', err);
        return [];
    }
}

export async function removeDislike(songTitle: string, artist: string): Promise<boolean> {
    try {
        const resp = await fetch(
            `http://localhost:8501/api/disliked-songs?song_title=${encodeURIComponent(songTitle)}&artist=${encodeURIComponent(artist)}`,
            { method: 'DELETE' }
        );
        if (!resp.ok) return false;
        const data = await resp.json();
        return data.success;
    } catch (err) {
        console.warn('[API] removeDislike 失败:', err);
        return false;
    }
}

// ---- 从本地曲库彻底删除一首歌（图谱 + 音频 + 封面 + 歌词 + 元数据）----
export async function deleteSongFromLibrary(
    songTitle: string,
    artist: string,
): Promise<{ success: boolean; message: string; deleted_files?: string[] }> {
    try {
        const resp = await fetch(
            `http://localhost:8501/api/songs?song_title=${encodeURIComponent(songTitle)}&artist=${encodeURIComponent(artist)}`,
            { method: 'DELETE' },
        );
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({ detail: '删除失败' }));
            throw new Error(err.detail || `删除失败: ${resp.status}`);
        }
        return resp.json();
    } catch (err: any) {
        console.error('[API] deleteSongFromLibrary 失败:', err);
        return { success: false, message: err.message || '删除失败' };
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
            // 读取和推荐页同步的持久化模型选择
            const provider = (typeof window !== 'undefined'
                ? localStorage.getItem('music_selected_provider')
                : null) || 'siliconflow';
            const resp = await fetch('http://localhost:8501/api/journey/stream', {
                method: 'POST',
                headers: { 'Accept': 'text/event-stream', 'Content-Type': 'application/json' },
                body: JSON.stringify({ ...params, llm_provider: params.llm_provider || provider }),
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

// ==================================================================
// 待入库 (Pending) 管理 API
// ==================================================================

export interface PendingSong {
    music_id: string;
    title: string;
    artist: string;
    album: string;
    duration: number;
    format: string;
    file_basename: string;
    audio_url: string;
    cover_url: string;
    lrc_url: string;
    acquired_at: string;
}

export async function fetchPendingSongs(): Promise<PendingSong[]> {
    try {
        const resp = await fetch('http://localhost:8501/api/pending-songs');
        if (!resp.ok) return [];
        const data = await resp.json();
        return data.success ? data.songs : [];
    } catch (err) {
        console.warn('[API] fetchPendingSongs 失败:', err);
        return [];
    }
}

export async function ingestPendingSongs(songs: {
    file_basename: string;
    ext: string;
    music_id: string;
    title: string;
    artist: string;
    album: string;
    duration: number;
}[]): Promise<{ success: boolean; ingested: number; message: string }> {
    try {
        const resp = await fetch('http://localhost:8501/api/pending-songs/ingest', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ songs }),
        });
        if (!resp.ok) throw new Error(`入库失败: ${resp.status}`);
        return resp.json();
    } catch (err: any) {
        console.error('[API] ingestPendingSongs 失败:', err);
        return { success: false, ingested: 0, message: err.message || '入库失败' };
    }
}

export async function deletePendingSong(
    fileBasename: string, ext: string = 'mp3'
): Promise<{ success: boolean }> {
    try {
        const resp = await fetch(
            `http://localhost:8501/api/pending-songs?file_basename=${encodeURIComponent(fileBasename)}&ext=${encodeURIComponent(ext)}`,
            { method: 'DELETE' },
        );
        if (!resp.ok) return { success: false };
        return resp.json();
    } catch (err) {
        console.warn('[API] deletePendingSong 失败:', err);
        return { success: false };
    }
}

// ==================================================================
// 我的曲库 (Library) API — 查询 Neo4j 图谱中全部歌曲
// ==================================================================

export interface LibrarySong {
    title: string;
    artist: string;
    album: string;
    audio_url: string;
    cover_url: string;
    lrc_url: string;
    source: string;
    music_id: string;
    duration: number;
    format: string;
    vibe: string;
    moods: string[];
    themes: string[];
}

export async function fetchLibrarySongs(
    offset: number = 0, limit: number = 200
): Promise<{ songs: LibrarySong[]; total: number }> {
    try {
        const resp = await fetch(
            `http://localhost:8501/api/library-songs?offset=${offset}&limit=${limit}`
        );
        if (!resp.ok) return { songs: [], total: 0 };
        const data = await resp.json();
        return data.success ? { songs: data.songs, total: data.total } : { songs: [], total: 0 };
    } catch (err) {
        console.warn('[API] fetchLibrarySongs 失败:', err);
        return { songs: [], total: 0 };
    }
}

