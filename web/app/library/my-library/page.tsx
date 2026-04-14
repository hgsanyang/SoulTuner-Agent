'use client';

/**
 * 🎵 我的曲库页面 (My Library)
 * 显示 Neo4j 知识图谱中的所有 Song 节点。
 * 支持搜索筛选、播放、查看标签、删除管理。
 */

import { useState, useEffect, useCallback } from 'react';
import { theme } from '@/styles/theme';
import { usePlayer } from '@/context/PlayerContext';
import { useLibrary } from '@/context/LibraryContext';
import { useRouter } from 'next/navigation';
import { fetchLibrarySongs, deleteSongFromLibrary, LibrarySong } from '@/lib/api';

export default function MyLibraryPage() {
    const [songs, setSongs] = useState<LibrarySong[]>([]);
    const [total, setTotal] = useState(0);
    const [loading, setLoading] = useState(true);
    const [searchQuery, setSearchQuery] = useState('');
    const [deleting, setDeleting] = useState<string | null>(null);
    const { playSong } = usePlayer();
    const { showToast } = useLibrary();
    const router = useRouter();

    const loadSongs = useCallback(async () => {
        setLoading(true);
        const data = await fetchLibrarySongs(0, 500);
        setSongs(data.songs);
        setTotal(data.total);
        setLoading(false);
    }, []);

    useEffect(() => { loadSongs(); }, [loadSongs]);

    const handleDelete = async (song: LibrarySong) => {
        const key = `${song.title}_${song.artist}`;
        setDeleting(key);
        const result = await deleteSongFromLibrary(song.title, song.artist);
        setDeleting(null);
        if (result.success) {
            showToast(`🗑️ 已从曲库中移除「${song.title}」`);
            setSongs(prev => prev.filter(s => !(s.title === song.title && s.artist === song.artist)));
            setTotal(prev => prev - 1);
        } else {
            showToast(`❌ 删除失败: ${result.message}`);
        }
    };

    // Filter by search (null-safe: title/artist/album may be null from Neo4j)
    const filtered = searchQuery.trim()
        ? songs.filter(s => {
            const q = searchQuery.toLowerCase();
            return (s.title || '').toLowerCase().includes(q) ||
                (s.artist || '').toLowerCase().includes(q) ||
                (s.album || '').toLowerCase().includes(q) ||
                (s.moods || []).some(m => (m || '').toLowerCase().includes(q)) ||
                (s.vibe || '').toLowerCase().includes(q);
        })
        : songs;

    const sourceLabel = (src: string) => {
        switch (src) {
            case 'online': return { text: '联网', color: '#3b82f6' };
            case 'mtg': return { text: 'MTG', color: '#8b5cf6' };
            default: return { text: '本地', color: theme.colors.primary.accent };
        }
    };

    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', padding: '1rem', color: theme.colors.text.primary, minHeight: '100%' }}>
            {/* 返回按钮 */}
            <button onClick={() => router.back()} style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', background: 'none', border: 'none', color: theme.colors.text.secondary, cursor: 'pointer', fontSize: '0.9rem', fontWeight: 500, padding: '0.25rem 0', width: 'fit-content', transition: 'color 0.2s' }}
                onMouseEnter={e => (e.currentTarget.style.color = '#fff')}
                onMouseLeave={e => (e.currentTarget.style.color = theme.colors.text.secondary)}
            >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><polyline points="15 18 9 12 15 6" /></svg>
                返回
            </button>

            {/* Header */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '1.5rem', marginBottom: '0.5rem' }}>
                <div style={{ width: '100px', height: '100px', borderRadius: theme.borderRadius.md, background: 'linear-gradient(135deg, #8b5cf6 0%, #6d28d9 100%)', display: 'flex', alignItems: 'center', justifyContent: 'center', boxShadow: theme.shadows.md }}>
                    <svg width="42" height="42" viewBox="0 0 24 24" fill="white" stroke="none">
                        <path d="M12 3v10.55c-.59-.34-1.27-.55-2-.55-2.21 0-4 1.79-4 4s1.79 4 4 4 4-1.79 4-4V7h4V3h-6z" />
                    </svg>
                </div>
                <div>
                    <p style={{ margin: 0, fontSize: '0.8rem', fontWeight: 600, letterSpacing: '0.05em', color: theme.colors.text.muted }}>知识图谱</p>
                    <h1 style={{ margin: '0.2rem 0', fontSize: '2.5rem', fontWeight: 800, letterSpacing: '-0.02em' }}>我的曲库</h1>
                    <p style={{ margin: 0, fontSize: '0.9rem', color: theme.colors.text.secondary }}>
                        {loading ? '加载中...' : `图谱中共有 ${total} 首歌曲`}
                    </p>
                </div>
            </div>

            {/* Search bar */}
            <div style={{ position: 'relative', maxWidth: '400px' }}>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke={theme.colors.text.muted} strokeWidth="2" style={{ position: 'absolute', left: '0.85rem', top: '50%', transform: 'translateY(-50%)' }}>
                    <circle cx="11" cy="11" r="8" /><line x1="21" y1="21" x2="16.65" y2="16.65" />
                </svg>
                <input
                    type="text"
                    placeholder="搜索歌名、歌手、专辑..."
                    value={searchQuery}
                    onChange={e => setSearchQuery(e.target.value)}
                    style={{
                        width: '100%', padding: '0.65rem 0.85rem 0.65rem 2.5rem',
                        background: 'rgba(255,255,255,0.05)', border: `1px solid ${theme.colors.border.default}`,
                        borderRadius: theme.borderRadius.sm, color: theme.colors.text.primary,
                        fontSize: '0.88rem', outline: 'none', transition: 'border-color 0.2s',
                    }}
                    onFocus={e => (e.currentTarget.style.borderColor = theme.colors.primary.accent)}
                    onBlur={e => (e.currentTarget.style.borderColor = theme.colors.border.default)}
                />
            </div>

            {/* Song List */}
            {!loading && filtered.length === 0 ? (
                <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '1rem', padding: '4rem', borderRadius: theme.borderRadius.lg, backgroundColor: 'rgba(255,255,255,0.02)', border: `1px dashed ${theme.colors.border.default}`, textAlign: 'center' }}>
                    <div style={{ width: '64px', height: '64px', borderRadius: '50%', backgroundColor: 'rgba(255,255,255,0.05)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke={theme.colors.text.muted} strokeWidth="2">
                            <path d="M9 18V5l12-2v13" /><circle cx="6" cy="18" r="3" /><circle cx="18" cy="16" r="3" />
                        </svg>
                    </div>
                    <h3 style={{ margin: 0, fontSize: '1.2rem', fontWeight: 600 }}>
                        {searchQuery ? '没有匹配的歌曲' : '曲库为空'}
                    </h3>
                    <p style={{ margin: 0, fontSize: '0.9rem', color: theme.colors.text.muted, maxWidth: '24rem' }}>
                        {searchQuery ? '试试其他关键词' : '通过 AI 对话获取新歌后，在待入库页面确认入库即可添加到这里。'}
                    </p>
                </div>
            ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
                    {filtered.map((song) => {
                        const src = sourceLabel(song.source);
                        const key = `${song.title}_${song.artist}`;
                        const isDeleting = deleting === key;
                        return (
                            <div key={key}
                                style={{
                                    display: 'flex', alignItems: 'center', gap: '0.75rem',
                                    padding: '0.7rem 1rem', borderRadius: theme.borderRadius.md,
                                    backgroundColor: 'rgba(255,255,255,0.02)',
                                    transition: 'background-color 0.2s', cursor: 'pointer',
                                    opacity: isDeleting ? 0.4 : 1,
                                }}
                                onMouseEnter={e => (e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.05)')}
                                onMouseLeave={e => (e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.02)')}
                                onClick={() => {
                                    if (song.audio_url) {
                                        const baseUrl = song.audio_url.startsWith('http') ? '' : 'http://localhost:8501';
                                        playSong({
                                            title: song.title, artist: song.artist,
                                            preview_url: `${baseUrl}${song.audio_url}`,
                                            coverUrl: song.cover_url ? `${baseUrl}${song.cover_url}` : undefined,
                                            lrc_url: song.lrc_url ? `${baseUrl}${song.lrc_url}` : undefined,
                                        });
                                    }
                                }}
                            >
                                {/* Cover */}
                                <div style={{
                                    width: '46px', height: '46px', borderRadius: '6px', flexShrink: 0,
                                    background: song.cover_url
                                        ? `url(${song.cover_url.startsWith('http') ? song.cover_url : 'http://localhost:8501' + song.cover_url}) center/cover, linear-gradient(135deg, #333, #222)`
                                        : 'linear-gradient(135deg, #333, #222)',
                                    backgroundSize: 'cover',
                                    backgroundPosition: 'center',
                                }} />

                                {/* Info */}
                                <div style={{ flex: 1, minWidth: 0 }}>
                                    <div style={{ fontWeight: 600, fontSize: '0.95rem', color: theme.colors.text.primary, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                        {song.title}
                                    </div>
                                    <div style={{ fontSize: '0.82rem', color: theme.colors.text.secondary, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                        {song.artist}{song.album ? ` · ${song.album}` : ''}
                                    </div>
                                </div>

                                {/* Tags */}
                                <div style={{ display: 'flex', gap: '0.3rem', flexShrink: 0, flexWrap: 'wrap', maxWidth: '180px' }}>
                                    {song.moods?.slice(0, 2).map(m => (
                                        <span key={m} style={{ fontSize: '0.7rem', padding: '0.15rem 0.45rem', borderRadius: '9999px', background: 'rgba(29,185,84,0.12)', color: theme.colors.primary.accent, whiteSpace: 'nowrap' }}>{m}</span>
                                    ))}
                                    {song.vibe && (
                                        <span style={{ fontSize: '0.7rem', padding: '0.15rem 0.45rem', borderRadius: '9999px', background: 'rgba(139,92,246,0.12)', color: '#a78bfa', whiteSpace: 'nowrap' }}>{song.vibe}</span>
                                    )}
                                </div>

                                {/* Source badge */}
                                <span style={{ fontSize: '0.7rem', padding: '0.15rem 0.5rem', borderRadius: '9999px', border: `1px solid ${src.color}33`, color: src.color, whiteSpace: 'nowrap', flexShrink: 0 }}>
                                    {src.text}
                                </span>

                                {/* Play */}
                                <button title={song.audio_url ? '播放' : '暂无音源'}
                                    onClick={e => {
                                        e.stopPropagation();
                                        if (song.audio_url) {
                                            const baseUrl = song.audio_url.startsWith('http') ? '' : 'http://localhost:8501';
                                            playSong({
                                                title: song.title, artist: song.artist,
                                                preview_url: `${baseUrl}${song.audio_url}`,
                                                coverUrl: song.cover_url ? (song.cover_url.startsWith('http') ? song.cover_url : `http://localhost:8501${song.cover_url}`) : undefined,
                                                lrc_url: song.lrc_url ? (song.lrc_url.startsWith('http') ? song.lrc_url : `http://localhost:8501${song.lrc_url}`) : undefined,
                                            });
                                        }
                                    }}
                                    disabled={!song.audio_url}
                                    style={{ background: 'none', border: 'none', color: song.audio_url ? theme.colors.primary.accent : theme.colors.text.muted, cursor: song.audio_url ? 'pointer' : 'not-allowed', padding: '0.4rem', display: 'flex', opacity: song.audio_url ? 1 : 0.35 }}>
                                    <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3" /></svg>
                                </button>

                                {/* Delete */}
                                <button title="从曲库移除" onClick={e => { e.stopPropagation(); handleDelete(song); }}
                                    disabled={isDeleting}
                                    style={{ background: 'none', border: 'none', color: theme.colors.text.muted, cursor: isDeleting ? 'wait' : 'pointer', padding: '0.4rem', display: 'flex', transition: 'color 0.2s' }}
                                    onMouseEnter={e => (e.currentTarget.style.color = '#ef4444')}
                                    onMouseLeave={e => (e.currentTarget.style.color = theme.colors.text.muted)}
                                >
                                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="3 6 5 6 21 6" /><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" /></svg>
                                </button>
                            </div>
                        );
                    })}
                </div>
            )}
        </div>
    );
}
