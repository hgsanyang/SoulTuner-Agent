'use client';

/**
 * 🎵 待入库页面 (Pending Import)
 * 显示已下载但未入库到 Neo4j 图谱的歌曲。
 * 支持试听、勾选批量入库、单曲/批量删除。
 */

import { useState, useEffect, useCallback } from 'react';
import { theme } from '@/styles/theme';
import { usePlayer } from '@/context/PlayerContext';
import { useLibrary } from '@/context/LibraryContext';
import { useRouter } from 'next/navigation';
import { fetchPendingSongs, ingestPendingSongs, deletePendingSong, PendingSong } from '@/lib/api';

export default function PendingPage() {
    const [songs, setSongs] = useState<PendingSong[]>([]);
    const [selected, setSelected] = useState<Set<string>>(new Set());
    const [loading, setLoading] = useState(true);
    const [ingesting, setIngesting] = useState(false);
    const { playSong } = usePlayer();
    const { showToast } = useLibrary();
    const router = useRouter();

    const loadSongs = useCallback(async () => {
        setLoading(true);
        const data = await fetchPendingSongs();
        setSongs(data);
        setLoading(false);
    }, []);

    useEffect(() => { loadSongs(); }, [loadSongs]);

    const toggleSelect = (basename: string) => {
        setSelected(prev => {
            const next = new Set(prev);
            if (next.has(basename)) next.delete(basename);
            else next.add(basename);
            return next;
        });
    };

    const toggleSelectAll = () => {
        if (selected.size === songs.length) {
            setSelected(new Set());
        } else {
            setSelected(new Set(songs.map(s => s.file_basename)));
        }
    };

    const handleIngest = async () => {
        const toIngest = songs.filter(s => selected.has(s.file_basename));
        if (toIngest.length === 0) return;
        setIngesting(true);
        const result = await ingestPendingSongs(toIngest.map(s => ({
            file_basename: s.file_basename,
            ext: s.format,
            music_id: s.music_id,
            title: s.title,
            artist: s.artist,
            album: s.album,
            duration: s.duration,
        })));
        setIngesting(false);
        if (result.success) {
            showToast(`✅ 已成功入库 ${result.ingested} 首歌曲`);
            setSelected(new Set());
            loadSongs();
        } else {
            showToast('❌ 入库失败，请重试');
        }
    };

    const handleDelete = async (song: PendingSong) => {
        const result = await deletePendingSong(song.file_basename, song.format);
        if (result.success) {
            showToast(`🗑️ 已删除「${song.title}」`);
            setSongs(prev => prev.filter(s => s.file_basename !== song.file_basename));
            setSelected(prev => {
                const next = new Set(prev);
                next.delete(song.file_basename);
                return next;
            });
        }
    };

    const handleDeleteSelected = async () => {
        const toDelete = songs.filter(s => selected.has(s.file_basename));
        for (const song of toDelete) {
            await deletePendingSong(song.file_basename, song.format);
        }
        showToast(`🗑️ 已删除 ${toDelete.length} 首歌曲`);
        setSelected(new Set());
        loadSongs();
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
                <div style={{ width: '100px', height: '100px', borderRadius: theme.borderRadius.md, background: 'linear-gradient(135deg, #f59e0b 0%, #d97706 100%)', display: 'flex', alignItems: 'center', justifyContent: 'center', boxShadow: theme.shadows.md }}>
                    <svg width="42" height="42" viewBox="0 0 24 24" fill="white" stroke="none">
                        <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 14H9V8h2v8zm4 0h-2V8h2v8z" />
                    </svg>
                </div>
                <div>
                    <p style={{ margin: 0, fontSize: '0.8rem', fontWeight: 600, letterSpacing: '0.05em', color: theme.colors.text.muted }}>暂存区</p>
                    <h1 style={{ margin: '0.2rem 0', fontSize: '2.5rem', fontWeight: 800, letterSpacing: '-0.02em' }}>待入库</h1>
                    <p style={{ margin: 0, fontSize: '0.9rem', color: theme.colors.text.secondary }}>
                        {loading ? '加载中...' : `共 ${songs.length} 首歌曲等待确认入库`}
                    </p>
                </div>
            </div>

            {/* Song List */}
            {!loading && songs.length === 0 ? (
                <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '1rem', padding: '4rem', borderRadius: theme.borderRadius.lg, backgroundColor: 'rgba(255,255,255,0.02)', border: `1px dashed ${theme.colors.border.default}`, textAlign: 'center' }}>
                    <div style={{ width: '64px', height: '64px', borderRadius: '50%', backgroundColor: 'rgba(255,255,255,0.05)', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '0.5rem' }}>
                        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke={theme.colors.text.muted} strokeWidth="2">
                            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><polyline points="7 10 12 15 17 10" /><line x1="12" y1="15" x2="12" y2="3" />
                        </svg>
                    </div>
                    <h3 style={{ margin: 0, fontSize: '1.2rem', fontWeight: 600 }}>暂无待入库歌曲</h3>
                    <p style={{ margin: 0, fontSize: '0.9rem', color: theme.colors.text.muted, maxWidth: '24rem' }}>
                        通过 AI 对话获取新歌后，歌曲会先下载到这里等待你确认入库。
                    </p>
                </div>
            ) : (
                <>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
                        {songs.map((song) => {
                            const isSelected = selected.has(song.file_basename);
                            return (
                                <div key={song.file_basename}
                                    style={{
                                        display: 'flex', alignItems: 'center', gap: '0.75rem',
                                        padding: '0.75rem 1rem', borderRadius: theme.borderRadius.md,
                                        backgroundColor: isSelected ? 'rgba(245,158,11,0.08)' : 'rgba(255,255,255,0.02)',
                                        border: isSelected ? '1px solid rgba(245,158,11,0.3)' : '1px solid transparent',
                                        transition: 'all 0.2s', cursor: 'pointer',
                                    }}
                                    onMouseEnter={e => { if (!isSelected) e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.05)'; }}
                                    onMouseLeave={e => { if (!isSelected) e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.02)'; }}
                                    onClick={() => toggleSelect(song.file_basename)}
                                >
                                    {/* Checkbox */}
                                    <div style={{
                                        width: '20px', height: '20px', borderRadius: '4px', flexShrink: 0,
                                        border: isSelected ? '2px solid #f59e0b' : `2px solid ${theme.colors.border.focus}`,
                                        backgroundColor: isSelected ? '#f59e0b' : 'transparent',
                                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                                        transition: 'all 0.2s',
                                    }}>
                                        {isSelected && (
                                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#000" strokeWidth="3"><polyline points="20 6 9 17 4 12" /></svg>
                                        )}
                                    </div>

                                    {/* Cover */}
                                    <div style={{
                                        width: '46px', height: '46px', borderRadius: '6px', flexShrink: 0,
                                        background: `url(http://localhost:8501${song.cover_url}) center/cover, linear-gradient(135deg, #333, #222)`,
                                    }} />

                                    {/* Info */}
                                    <div style={{ flex: 1, minWidth: 0 }}>
                                        <div style={{ fontWeight: 600, fontSize: '0.95rem', color: theme.colors.text.primary, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{song.title}</div>
                                        <div style={{ fontSize: '0.82rem', color: theme.colors.text.secondary, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{song.artist} · {song.album}</div>
                                    </div>

                                    {/* Time */}
                                    <div style={{ fontSize: '0.75rem', color: theme.colors.text.muted, whiteSpace: 'nowrap', paddingRight: '0.5rem' }}>
                                        {song.acquired_at ? new Date(song.acquired_at).toLocaleDateString('zh-CN') : ''}
                                    </div>

                                    {/* Play */}
                                    <button title="试听" onClick={e => { e.stopPropagation(); playSong({ title: song.title, artist: song.artist, preview_url: `http://localhost:8501${song.audio_url}`, coverUrl: `http://localhost:8501${song.cover_url}`, lrc_url: `http://localhost:8501${song.lrc_url}` }); }}
                                        style={{ background: 'none', border: 'none', color: theme.colors.primary.accent, cursor: 'pointer', padding: '0.4rem', borderRadius: '50%', display: 'flex', transition: 'transform 0.2s' }}
                                        onMouseEnter={e => (e.currentTarget.style.transform = 'scale(1.15)')}
                                        onMouseLeave={e => (e.currentTarget.style.transform = 'scale(1)')}
                                    >
                                        <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3" /></svg>
                                    </button>

                                    {/* Delete */}
                                    <button title="删除" onClick={e => { e.stopPropagation(); handleDelete(song); }}
                                        style={{ background: 'none', border: 'none', color: theme.colors.text.muted, cursor: 'pointer', padding: '0.4rem', borderRadius: '50%', display: 'flex', transition: 'color 0.2s' }}
                                        onMouseEnter={e => (e.currentTarget.style.color = '#ef4444')}
                                        onMouseLeave={e => (e.currentTarget.style.color = theme.colors.text.muted)}
                                    >
                                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="3 6 5 6 21 6" /><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" /></svg>
                                    </button>
                                </div>
                            );
                        })}
                    </div>

                    {/* Bottom Action Bar */}
                    {songs.length > 0 && (
                        <div style={{
                            position: 'sticky', bottom: '5rem', display: 'flex', alignItems: 'center', gap: '1rem',
                            padding: '0.85rem 1.25rem', borderRadius: theme.borderRadius.lg,
                            background: 'rgba(18,18,18,0.95)', backdropFilter: 'blur(12px)',
                            border: `1px solid ${theme.colors.border.default}`, boxShadow: theme.shadows.lg,
                        }}>
                            {/* Select All */}
                            <button onClick={toggleSelectAll}
                                style={{ background: 'none', border: `1px solid ${theme.colors.border.focus}`, color: theme.colors.text.secondary, cursor: 'pointer', padding: '0.5rem 1rem', borderRadius: theme.borderRadius.sm, fontSize: '0.82rem', transition: 'all 0.2s' }}
                                onMouseEnter={e => { e.currentTarget.style.borderColor = theme.colors.primary.accent; e.currentTarget.style.color = theme.colors.text.primary; }}
                                onMouseLeave={e => { e.currentTarget.style.borderColor = theme.colors.border.focus; e.currentTarget.style.color = theme.colors.text.secondary; }}
                            >
                                {selected.size === songs.length ? '取消全选' : '全选'}
                            </button>

                            <span style={{ fontSize: '0.82rem', color: theme.colors.text.muted }}>
                                已选 {selected.size} / {songs.length}
                            </span>

                            <div style={{ flex: 1 }} />

                            {/* Delete Selected */}
                            {selected.size > 0 && (
                                <button onClick={handleDeleteSelected}
                                    style={{ background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.3)', color: '#ef4444', cursor: 'pointer', padding: '0.5rem 1.2rem', borderRadius: theme.borderRadius.sm, fontSize: '0.85rem', fontWeight: 600, transition: 'all 0.2s' }}
                                    onMouseEnter={e => (e.currentTarget.style.background = 'rgba(239,68,68,0.2)')}
                                    onMouseLeave={e => (e.currentTarget.style.background = 'rgba(239,68,68,0.1)')}
                                >
                                    🗑️ 删除选中
                                </button>
                            )}

                            {/* Ingest Selected */}
                            <button onClick={handleIngest} disabled={selected.size === 0 || ingesting}
                                style={{
                                    background: selected.size > 0 ? theme.colors.primary.accent : theme.colors.primary[400],
                                    border: 'none', color: selected.size > 0 ? '#000' : theme.colors.text.muted,
                                    cursor: selected.size > 0 ? 'pointer' : 'not-allowed',
                                    padding: '0.5rem 1.5rem', borderRadius: theme.borderRadius.sm,
                                    fontSize: '0.85rem', fontWeight: 700, transition: 'all 0.2s',
                                    opacity: ingesting ? 0.6 : 1,
                                }}
                            >
                                {ingesting ? '入库中...' : `✅ 入库选中 (${selected.size})`}
                            </button>
                        </div>
                    )}
                </>
            )}
        </div>
    );
}
