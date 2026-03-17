'use client';

/**
 * 🎧 全局抽屉与底部控制栏组件 (GlobalPlayer)
 */
import React, { useState, useEffect, useRef } from 'react';
import { usePlayer } from '@/context/PlayerContext';
import { useLibrary } from '@/context/LibraryContext';
import { theme } from '@/styles/theme';
import { motion, AnimatePresence } from 'framer-motion';
import StarryBackground from './StarryBackground';

// ────────────────────────────────────────────────────────
// LRC 解析
// ────────────────────────────────────────────────────────
interface LrcLine { time: number; text: string; }

function parseLrc(raw: string): LrcLine[] {
    const lines: LrcLine[] = [];
    const regex = /\[(\d{2}):(\d{2})\.(\d{2,3})\](.*)/;
    raw.split('\n').forEach(l => {
        const m = l.match(regex);
        if (m) {
            const min = parseInt(m[1]), sec = parseInt(m[2]);
            const ms = parseInt(m[3].padEnd(3, '0'));
            lines.push({ time: min * 60 + sec + ms / 1000, text: m[4].trim() });
        }
    });
    return lines.sort((a, b) => a.time - b.time);
}

export default function GlobalPlayer() {
    const {
        currentSong, isPlaying, volume, duration, currentTime,
        playMode, queue, isExpanded,
        togglePlay, playNext, playPrev, playSong, setVolume, seek,
        setPlayMode, setExpanded, removeFromQueue,
    } = usePlayer();

    const { toggleLike, isLiked: checkIsLiked, collections, addToCollection } = useLibrary();

    // ── All hooks must be before any conditional return ──
    const [showFolderPicker, setShowFolderPicker] = useState(false);
    const [showQueue, setShowQueue] = useState(false);
    const [activeTab, setActiveTab] = useState<'lyrics' | 'album' | 'artist'>('lyrics');
    const [lrcLines, setLrcLines] = useState<LrcLine[]>([]);
    const [lrcLoading, setLrcLoading] = useState(false);
    const [hoveredQueueIdx, setHoveredQueueIdx] = useState<number | null>(null);
    const lyricsRef = useRef<HTMLDivElement>(null);

    // Fetch LRC when song changes
    useEffect(() => {
        if (!currentSong?.lrc_url) { setLrcLines([]); return; }
        setLrcLoading(true);
        fetch(currentSong.lrc_url)
            .then(r => r.text())
            .then(text => { setLrcLines(parseLrc(text)); setLrcLoading(false); })
            .catch(() => { setLrcLines([]); setLrcLoading(false); });
    }, [currentSong?.lrc_url]);

    // Auto-scroll lyrics
    useEffect(() => {
        if (!lyricsRef.current || lrcLines.length === 0) return;
        let idx = 0;
        for (let i = 0; i < lrcLines.length; i++) {
            if (lrcLines[i].time <= currentTime) idx = i; else break;
        }
        const el = lyricsRef.current.children[idx] as HTMLElement | undefined;
        el?.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }, [currentTime, lrcLines]);

    // ── Early return AFTER all hooks ──
    if (!currentSong) return null;

    const isLiked = checkIsLiked(currentSong.title, currentSong.artist);

    // Current lyric index
    let currentLrcIdx = 0;
    for (let i = 0; i < lrcLines.length; i++) {
        if (lrcLines[i].time <= currentTime) currentLrcIdx = i; else break;
    }

    const formatTime = (s: number) => {
        if (isNaN(s)) return '0:00';
        return `${Math.floor(s / 60)}:${Math.floor(s % 60).toString().padStart(2, '0')}`;
    };

    const PlayModeIcon = () => {
        if (playMode === 'random') return (
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="16 3 21 3 21 8"/><line x1="4" y1="20" x2="21" y2="3"/>
                <polyline points="21 16 21 21 16 21"/><line x1="15" y1="15" x2="21" y2="21"/><line x1="4" y1="4" x2="9" y2="9"/>
            </svg>
        );
        if (playMode === 'loop') return (
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="17 1 21 5 17 9"/><path d="M3 11V9a4 4 0 0 1 4-4h14"/>
                <polyline points="7 23 3 19 7 15"/><path d="M21 13v2a4 4 0 0 1-4 4H3"/>
            </svg>
        );
        return (
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <line x1="21" y1="12" x2="3" y2="12"/><line x1="21" y1="6" x2="3" y2="6"/><line x1="21" y1="18" x2="3" y2="18"/>
            </svg>
        );
    };

    return (
        <>
            {/* ════════════════════════════════
                全屏播放器
            ════════════════════════════════ */}
            <AnimatePresence>
                {isExpanded && (
                    <motion.div
                        initial={{ y: '100%' }} animate={{ y: 0 }} exit={{ y: '100%' }}
                        transition={{ type: 'spring', bounce: 0, duration: 0.4 }}
                        style={{ position: 'fixed', top: 0, left: 0, right: 0, bottom: '90px', zIndex: 100, backgroundColor: '#050914', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}
                    >
                        {/* 绚烂星空背景 */}
                        <StarryBackground />

                        <div style={{ position: 'relative', zIndex: 1, display: 'flex', flexDirection: 'column', height: '100%' }}>
                            {/* Header */}
                            <div style={{ padding: '1.5rem 2rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexShrink: 0 }}>
                                <button onClick={() => setExpanded(false)} style={{ background: 'none', border: 'none', color: '#fff', cursor: 'pointer', padding: '0.25rem' }}>
                                    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                        <polyline points="6 15 12 9 18 15"/>
                                    </svg>
                                </button>

                                {/* 右上角标签 */}
                                <div style={{ display: 'flex', gap: '0.2rem', backgroundColor: 'rgba(255,255,255,0.09)', borderRadius: '2rem', padding: '0.22rem' }}>
                                    {([['lyrics', '歌词'], ['album', '专辑'], ['artist', '歌手']] as const).map(([tab, label]) => (
                                        <button
                                            key={tab}
                                            onClick={() => setActiveTab(tab)}
                                            style={{
                                                background: activeTab === tab ? 'rgba(255,255,255,0.2)' : 'none',
                                                border: 'none', color: activeTab === tab ? '#fff' : 'rgba(255,255,255,0.45)',
                                                cursor: 'pointer', padding: '0.32rem 1.05rem', borderRadius: '2rem',
                                                fontSize: '0.85rem', fontWeight: activeTab === tab ? 600 : 400, transition: 'all 0.2s',
                                            }}
                                        >{label}</button>
                                    ))}
                                </div>
                            </div>

                            {/* Main Content */}
                            <div style={{ flex: 1, display: 'flex', padding: '0 4rem', overflow: 'hidden', gap: '4rem' }}>
                                {/* Left: 黑胶旋转封面 */}
                                <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'flex-end', paddingRight: '2rem' }}>
                                    <div style={{
                                        width: 'clamp(300px, 45vh, 500px)', height: 'clamp(300px, 45vh, 500px)', borderRadius: '50%',
                                        backgroundColor: '#1a1a1a',
                                        boxShadow: '0 0 60px rgba(0,0,0,0.7)',
                                        display: 'flex', alignItems: 'center', justifyContent: 'center',
                                        overflow: 'hidden',
                                        animation: isPlaying ? 'spin 22s linear infinite' : 'none',
                                        flexShrink: 0,
                                    }}>
                                        {currentSong.coverUrl ? (
                                            <img src={currentSong.coverUrl} style={{ width: '78%', height: '78%', borderRadius: '50%', objectFit: 'cover' }} alt="Album Cover" />
                                        ) : (
                                            <svg width="100" height="100" viewBox="0 0 24 24" fill={theme.colors.text.muted}>
                                                <path d="M9 18V5l12-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="16" r="3"/>
                                            </svg>
                                        )}
                                    </div>
                                </div>

                                {/* Right: 信息 + 面板 */}
                                <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden', paddingTop: '2rem', paddingLeft: '10%', paddingRight: '4%' }}>
                                    {/* Song info */}
                                    <div style={{ marginBottom: '1.2rem', flexShrink: 0 }}>
                                        <h1 style={{ fontSize: '2rem', fontWeight: 700, color: '#fff', margin: '0 0 0.35rem 0', lineHeight: 1.2 }}>{currentSong.title}</h1>
                                        <p style={{ fontSize: '1rem', color: 'rgba(255,255,255,0.55)', margin: 0 }}>{currentSong.artist}</p>
                                        {currentSong.genre && (
                                            <span style={{ display: 'inline-block', marginTop: '0.4rem', padding: '0.18rem 0.65rem', fontSize: '0.74rem', backgroundColor: 'rgba(255,255,255,0.09)', color: 'rgba(255,255,255,0.5)', borderRadius: '2rem' }}>
                                                {currentSong.genre}
                                            </span>
                                        )}
                                    </div>

                                    {/* 歌词面板 */}
                                    {activeTab === 'lyrics' && (
                                        <div ref={lyricsRef} style={{ flex: 1, overflowY: 'auto', paddingRight: '0.5rem', scrollbarWidth: 'none' }}>
                                            {lrcLoading && <div style={{ color: 'rgba(255,255,255,0.35)', fontSize: '1rem', paddingTop: '2rem' }}>加载歌词中...</div>}
                                            {!lrcLoading && lrcLines.length === 0 && (
                                                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: 'rgba(255,255,255,0.28)', fontSize: '1.1rem', letterSpacing: '0.05em' }}>
                                                    暂无歌词
                                                </div>
                                            )}
                                            {lrcLines.map((line, i) => (
                                                <div
                                                    key={i}
                                                    onClick={() => seek(line.time)}
                                                    style={{
                                                        padding: '0.45rem 0',
                                                        fontSize: i === currentLrcIdx ? '1.45rem' : '1.1rem',
                                                        fontWeight: i === currentLrcIdx ? 700 : 400,
                                                        color: i === currentLrcIdx ? '#fff' : 'rgba(255,255,255,0.32)',
                                                        lineHeight: 1.55, transition: 'all 0.3s',
                                                        cursor: 'pointer',
                                                    }}
                                                >{line.text || '\u00a0'}</div>
                                            ))}
                                        </div>
                                    )}

                                    {/* 专辑面板 */}
                                    {activeTab === 'album' && (
                                        <div style={{ flex: 1, paddingTop: '0.5rem' }}>
                                            <div style={{ display: 'flex', alignItems: 'center', gap: '1.5rem', marginBottom: '1.5rem' }}>
                                                {currentSong.coverUrl && (
                                                    <img src={currentSong.coverUrl} style={{ width: '110px', height: '110px', borderRadius: '10px', objectFit: 'cover' }} alt="Album" />
                                                )}
                                                <div>
                                                    <div style={{ fontSize: '1.3rem', fontWeight: 600, color: '#fff', marginBottom: '0.3rem' }}>{currentSong.title}</div>
                                                    <div style={{ fontSize: '1rem', color: 'rgba(255,255,255,0.5)' }}>{currentSong.artist}</div>
                                                    {currentSong.genre && <div style={{ fontSize: '0.85rem', color: 'rgba(255,255,255,0.3)', marginTop: '0.2rem' }}>{currentSong.genre}</div>}
                                                </div>
                                            </div>
                                        </div>
                                    )}

                                    {/* 歌手面板 */}
                                    {activeTab === 'artist' && (
                                        <div style={{ flex: 1, paddingTop: '0.5rem' }}>
                                            <div style={{ fontSize: '1.5rem', fontWeight: 700, color: '#fff', marginBottom: '0.6rem' }}>{currentSong.artist}</div>
                                            <div style={{ fontSize: '0.95rem', color: 'rgba(255,255,255,0.4)' }}>正在播放：{currentSong.title}</div>
                                        </div>
                                    )}
                                </div>
                            </div>

                        </div>

                        <style jsx global>{`@keyframes spin { 100% { transform: rotate(360deg); } }`}</style>
                    </motion.div>
                )}
            </AnimatePresence>

            {/* ════════════════════════════════
                播放队列浮窗
            ════════════════════════════════ */}
            <AnimatePresence>
                {showQueue && (
                    <motion.div
                        initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 12 }}
                        transition={{ duration: 0.22 }}
                        style={{
                            position: 'fixed', bottom: '100px', right: '1.5rem',
                            width: '360px', maxHeight: '480px',
                            backgroundColor: 'rgba(16,16,16,0.97)',
                            border: '1px solid rgba(255,255,255,0.12)',
                            borderRadius: '16px', boxShadow: '0 20px 60px rgba(0,0,0,0.85)',
                            backdropFilter: 'blur(20px)', zIndex: 60,
                            display: 'flex', flexDirection: 'column', overflow: 'hidden',
                        }}
                    >
                        {/* Queue header */}
                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0.85rem 1rem', borderBottom: '1px solid rgba(255,255,255,0.07)', flexShrink: 0 }}>
                            <span style={{ fontSize: '0.9rem', fontWeight: 600, color: '#fff' }}>
                                播放列表 <span style={{ fontSize: '0.8rem', color: 'rgba(255,255,255,0.38)', fontWeight: 400 }}>({queue.length})</span>
                            </span>
                            <button onClick={() => setShowQueue(false)} style={{ background: 'none', border: 'none', color: 'rgba(255,255,255,0.45)', cursor: 'pointer', padding: '0.2rem' }}>
                                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
                                    <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
                                </svg>
                            </button>
                        </div>

                        {/* Queue list */}
                        <div style={{ flex: 1, overflowY: 'auto', scrollbarWidth: 'none' }}>
                            {queue.length === 0 ? (
                                <div style={{ textAlign: 'center', color: 'rgba(255,255,255,0.28)', padding: '2.5rem 1rem', fontSize: '0.88rem' }}>播放列表为空</div>
                            ) : (
                                queue.map((song, idx) => {
                                    const isCurrent = currentSong.title === song.title && currentSong.artist === song.artist;
                                    const isHov = hoveredQueueIdx === idx;
                                    const songLiked = checkIsLiked(song.title, song.artist);
                                    return (
                                        <div
                                            key={`${song.title}_${song.artist}_${idx}`}
                                            onMouseEnter={() => setHoveredQueueIdx(idx)}
                                            onMouseLeave={() => setHoveredQueueIdx(null)}
                                            onClick={() => playSong(song)}
                                            style={{
                                                display: 'flex', alignItems: 'center', gap: '0.72rem',
                                                padding: '0.62rem 1rem', cursor: 'pointer',
                                                backgroundColor: isCurrent ? 'rgba(29,185,84,0.08)' : isHov ? 'rgba(255,255,255,0.05)' : 'transparent',
                                                transition: 'background-color 0.14s',
                                            }}
                                        >
                                            {/* thumb */}
                                            <div style={{ width: '40px', height: '40px', borderRadius: '6px', backgroundColor: '#252525', flexShrink: 0, overflow: 'hidden', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                                                {song.coverUrl
                                                    ? <img src={song.coverUrl} alt="" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                                                    : <svg width="16" height="16" viewBox="0 0 24 24" fill="rgba(255,255,255,0.28)"><path d="M9 18V5l12-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="16" r="3"/></svg>
                                                }
                                            </div>
                                            {/* info */}
                                            <div style={{ flex: 1, minWidth: 0 }}>
                                                <div style={{ fontSize: '0.87rem', fontWeight: isCurrent ? 600 : 400, color: isCurrent ? '#1DB954' : '#fff', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{song.title}</div>
                                                <div style={{ fontSize: '0.75rem', color: 'rgba(255,255,255,0.42)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{song.artist}</div>
                                            </div>
                                            {/* hover actions */}
                                            {isHov && (
                                                <div style={{ display: 'flex', alignItems: 'center', gap: '0.25rem', flexShrink: 0 }} onClick={e => e.stopPropagation()}>
                                                    <button onClick={e => { e.stopPropagation(); toggleLike(song); }} title={songLiked ? '取消喜欢' : '喜欢'}
                                                        style={{ background: 'none', border: 'none', cursor: 'pointer', padding: '0.28rem', display: 'flex', color: songLiked ? '#e91e63' : 'rgba(255,255,255,0.45)' }}>
                                                        <svg width="15" height="15" viewBox="0 0 24 24" fill={songLiked ? '#e91e63' : 'none'} stroke="currentColor" strokeWidth="2">
                                                            <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/>
                                                        </svg>
                                                    </button>
                                                    <button onClick={e => { e.stopPropagation(); removeFromQueue(song.title, song.artist); }} title="从列表移除"
                                                        style={{ background: 'none', border: 'none', cursor: 'pointer', padding: '0.28rem', display: 'flex', color: 'rgba(255,100,100,0.65)' }}>
                                                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
                                                            <polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14H6L5 6"/>
                                                            <path d="M10 11v6"/><path d="M14 11v6"/><path d="M9 6V4h6v2"/>
                                                        </svg>
                                                    </button>
                                                </div>
                                            )}
                                        </div>
                                    );
                                })
                            )}
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>

            {/* ════════════════════════════════
                常驻底部控制栏
            ════════════════════════════════ */}
            <div style={{
                position: 'fixed', bottom: 0, left: 0, right: 0, height: '90px',
                backgroundColor: '#181818', borderTop: `1px solid ${theme.colors.border.default}`,
                display: 'flex', alignItems: 'center', padding: '0 1rem',
                zIndex: 101, backdropFilter: 'blur(10px)',
            }}>
                {/* Left: Cover + Title + Actions */}
                <div style={{ flex: '1', display: 'flex', alignItems: 'center', gap: '1rem', position: 'relative' }}>
                    {/* 可点击封面 */}
                    <div
                        onClick={() => setExpanded(true)}
                        title="打开播放器"
                        style={{
                            width: '56px', height: '56px', borderRadius: theme.borderRadius.sm,
                            backgroundColor: '#333', flexShrink: 0, overflow: 'hidden',
                            display: 'flex', alignItems: 'center', justifyContent: 'center',
                            cursor: 'pointer', transition: 'transform 0.15s, box-shadow 0.15s',
                        }}
                        onMouseEnter={e => { (e.currentTarget as HTMLDivElement).style.transform = 'scale(1.07)'; (e.currentTarget as HTMLDivElement).style.boxShadow = '0 0 0 2px rgba(255,255,255,0.28)'; }}
                        onMouseLeave={e => { (e.currentTarget as HTMLDivElement).style.transform = 'scale(1)'; (e.currentTarget as HTMLDivElement).style.boxShadow = 'none'; }}
                    >
                        {currentSong.coverUrl
                            ? <img src={currentSong.coverUrl} alt="cover" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                            : <svg width="24" height="24" viewBox="0 0 24 24" fill={theme.colors.text.muted}><path d="M9 18V5l12-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="16" r="3"/></svg>
                        }
                    </div>

                    <div style={{ minWidth: 0 }}>
                        <div style={{ color: theme.colors.text.primary, fontWeight: 500, fontSize: '0.9rem', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: '160px' }}>{currentSong.title}</div>
                        <div style={{ color: theme.colors.text.secondary, fontSize: '0.8rem' }}>{currentSong.artist}</div>
                    </div>

                    {/* ❤ Like - 红色 */}
                    <button onClick={() => toggleLike(currentSong)} style={{ background: 'none', border: 'none', color: isLiked ? '#e91e63' : theme.colors.text.muted, cursor: 'pointer', marginLeft: '0.4rem', padding: '0.25rem', transition: 'color 0.2s' }}>
                        <svg width="20" height="20" viewBox="0 0 24 24" fill={isLiked ? '#e91e63' : 'none'} stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/>
                        </svg>
                    </button>

                    {/* 收藏到歌单 */}
                    <button onClick={() => setShowFolderPicker(p => !p)} style={{ background: 'none', border: 'none', color: showFolderPicker ? '#fff' : theme.colors.text.muted, cursor: 'pointer', padding: '0.25rem', transition: 'color 0.2s' }}>
                        <svg width="20" height="20" viewBox="0 0 24 24" fill={showFolderPicker ? 'currentColor' : 'none'} stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z"/>
                        </svg>
                    </button>

                    {showFolderPicker && (
                        <div style={{ position: 'absolute', bottom: '72px', left: '155px', backgroundColor: 'rgba(22,22,22,0.97)', border: '1px solid rgba(255,255,255,0.14)', borderRadius: '12px', boxShadow: '0 8px 32px rgba(0,0,0,0.7)', minWidth: '210px', zIndex: 200, overflow: 'hidden', backdropFilter: 'blur(16px)' }}>
                            <div style={{ padding: '0.6rem 1rem', fontSize: '0.76rem', color: 'rgba(255,255,255,0.4)', borderBottom: '1px solid rgba(255,255,255,0.07)', fontWeight: 600 }}>收藏到歌单</div>
                            {collections.length === 0
                                ? <div style={{ padding: '1rem', fontSize: '0.88rem', color: 'rgba(255,255,255,0.38)', textAlign: 'center' }}>暂无歌单</div>
                                : collections.map(col => (
                                    <button key={col.id}
                                        onClick={() => { addToCollection(col.id, { title: currentSong.title, artist: currentSong.artist, genre: currentSong.genre, preview_url: currentSong.preview_url, coverUrl: currentSong.coverUrl }); setShowFolderPicker(false); }}
                                        style={{ display: 'flex', alignItems: 'center', gap: '0.7rem', width: '100%', padding: '0.7rem 1rem', background: 'none', border: 'none', cursor: 'pointer', color: '#fff', fontSize: '0.9rem', textAlign: 'left', transition: 'background-color 0.14s' }}
                                        onMouseEnter={e => (e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.07)')}
                                        onMouseLeave={e => (e.currentTarget.style.backgroundColor = 'transparent')}
                                    >
                                        <div style={{ width: '26px', height: '26px', borderRadius: '4px', backgroundColor: col.coverColor || '#5B8DEF', flexShrink: 0 }} />
                                        <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{col.name}</span>
                                    </button>
                                ))
                            }
                        </div>
                    )}
                </div>

                {/* Center Controls */}
                <div style={{ flex: '1', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.5rem' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '1.5rem' }}>
                        <button onClick={() => setPlayMode(playMode === 'sequence' ? 'random' : playMode === 'random' ? 'loop' : 'sequence')} style={{ background: 'none', border: 'none', color: theme.colors.text.muted, cursor: 'pointer' }} title="切换播放模式">
                            <PlayModeIcon />
                        </button>
                        <button onClick={playPrev} style={{ background: 'none', border: 'none', color: theme.colors.text.primary, cursor: 'pointer' }}>
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor" stroke="none"><polygon points="19 20 9 12 19 4 19 20"/><line x1="5" y1="19" x2="5" y2="5" stroke="currentColor" strokeWidth="2"/></svg>
                        </button>
                        <button onClick={togglePlay} style={{ width: '36px', height: '36px', borderRadius: '50%', backgroundColor: '#fff', display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer', border: 'none' }}>
                            {isPlaying
                                ? <svg width="16" height="16" viewBox="0 0 24 24" fill="#000"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>
                                : <svg width="18" height="18" viewBox="0 0 24 24" fill="#000" style={{ marginLeft: '2px' }}><polygon points="5 3 19 12 5 21 5 3"/></svg>
                            }
                        </button>
                        <button onClick={playNext} style={{ background: 'none', border: 'none', color: theme.colors.text.primary, cursor: 'pointer' }}>
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor" stroke="none"><polygon points="5 4 15 12 5 20 5 4"/><line x1="19" y1="5" x2="19" y2="19" stroke="currentColor" strokeWidth="2"/></svg>
                        </button>
                        {/* 播放列表按钮（替换原下载按钮） */}
                        <button onClick={() => setShowQueue(p => !p)} title="播放列表" style={{ background: 'none', border: 'none', color: showQueue ? '#fff' : theme.colors.text.muted, cursor: 'pointer', transition: 'color 0.2s' }}>
                            <svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="21" y2="18"/>
                                <line x1="3" y1="6" x2="3.01" y2="6"/><line x1="3" y1="12" x2="3.01" y2="12"/><line x1="3" y1="18" x2="3.01" y2="18"/>
                            </svg>
                        </button>
                    </div>
                    {/* Progress */}
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', width: '100%', maxWidth: '400px', fontSize: '0.75rem', color: theme.colors.text.muted }}>
                        <span>{formatTime(currentTime)}</span>
                        <input type="range" min="0" max={duration || 100} value={currentTime} onChange={e => seek(Number(e.target.value))} style={{ flex: 1, accentColor: theme.colors.text.primary, height: '4px', cursor: 'pointer' }} />
                        <span>{formatTime(duration)}</span>
                    </div>
                </div>

                {/* Right Controls */}
                <div style={{ flex: '1', display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: '1rem' }}>
                    <button style={{ background: 'none', border: 'none', color: theme.colors.text.muted, cursor: 'pointer' }}>
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/>
                            <path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07"/>
                        </svg>
                    </button>
                    <input type="range" min="0" max="1" step="0.01" value={volume} onChange={e => setVolume(Number(e.target.value))} style={{ width: '80px', accentColor: theme.colors.text.primary, height: '4px', cursor: 'pointer' }} />
                    <button onClick={() => setExpanded(true)} title="展开播放器" style={{ background: 'none', border: 'none', color: theme.colors.text.muted, cursor: 'pointer', marginLeft: '0.5rem' }}>
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <polyline points="15 3 21 3 21 9"/><polyline points="9 21 3 21 3 15"/>
                            <line x1="21" y1="3" x2="14" y2="10"/><line x1="3" y1="21" x2="10" y2="14"/>
                        </svg>
                    </button>
                </div>
            </div>
        </>
    );
}
