'use client';

/**
 * 🎵 app/library/likes/page.tsx (我的喜欢页面 UI组件)
 * 作用：渲染并展示用户所收藏(点赞)的所有音乐列表。
 * 功能特性：
 * 1. 订阅 LibraryContext 获取当前设备已喜欢的歌曲数据并进行可视化渲染。
 * 2. 提供点击单曲卡片即可快捷调用 PlayerContext 播放对应歌曲的入口。
 * 3. 在列表项右侧支持直接点击爱心 SVG 取消该曲目的喜欢状态，数据将自动同步清理。
 */

import { theme } from '@/styles/theme';
import { useLibrary } from '@/context/LibraryContext';
import { usePlayer } from '@/context/PlayerContext';
import { useRouter } from 'next/navigation';

export default function LikesPage() {
    const { likedSongs, toggleLike } = useLibrary();
    const { playSong } = usePlayer();
    const router = useRouter();
    return (
        <div
            style={{
                display: 'flex',
                flexDirection: 'column',
                gap: '2rem',
                padding: '1rem',
                color: theme.colors.text.primary,
                minHeight: '100%',
            }}
        >
            {/* 返回按钮 */}
            <button
                onClick={() => router.back()}
                style={{
                    display: 'flex', alignItems: 'center', gap: '0.4rem',
                    background: 'none', border: 'none', color: theme.colors.text.secondary,
                    cursor: 'pointer', fontSize: '0.9rem', fontWeight: 500,
                    padding: '0.25rem 0', width: 'fit-content',
                    transition: 'color 0.2s',
                }}
                onMouseEnter={e => (e.currentTarget.style.color = '#fff')}
                onMouseLeave={e => (e.currentTarget.style.color = theme.colors.text.secondary)}
            >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                    <polyline points="15 18 9 12 15 6" />
                </svg>
                返回
            </button>
            <div style={{ display: 'flex', alignItems: 'center', gap: '1.5rem', marginBottom: '1rem' }}>
                <div
                    style={{
                        width: '120px',
                        height: '120px',
                        borderRadius: theme.borderRadius.md,
                        background: 'linear-gradient(135deg, #1db954 0%, #179342 100%)',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        boxShadow: theme.shadows.md,
                    }}
                >
                    <svg width="48" height="48" viewBox="0 0 24 24" fill="white" stroke="none">
                        <path d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z" />
                    </svg>
                </div>
                <div>
                    <p style={{ margin: 0, fontSize: '0.85rem', fontWeight: 600, letterSpacing: '0.05em', color: theme.colors.text.muted }}>
                        歌曲夹
                    </p>
                    <h1 style={{ margin: '0.2rem 0', fontSize: '3rem', fontWeight: 800, letterSpacing: '-0.02em' }}>
                        我的喜欢
                    </h1>
                    <p style={{ margin: 0, fontSize: '0.9rem', color: theme.colors.text.secondary }}>
                        当前共有 {likedSongs.length} 首已赞歌曲
                    </p>
                </div>
            </div>

            {likedSongs.length === 0 ? (
                <div
                    style={{
                        flex: 1,
                        display: 'flex',
                        flexDirection: 'column',
                        alignItems: 'center',
                        justifyContent: 'center',
                        gap: '1rem',
                        padding: '4rem',
                        borderRadius: theme.borderRadius.lg,
                        backgroundColor: 'rgba(255,255,255,0.02)',
                        border: `1px dashed ${theme.colors.border.default}`,
                        textAlign: 'center',
                    }}
                >
                    <div
                        style={{
                            width: '64px',
                            height: '64px',
                            borderRadius: '50%',
                            backgroundColor: 'rgba(255,255,255,0.05)',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            marginBottom: '0.5rem',
                        }}
                    >
                        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke={theme.colors.text.muted} strokeWidth="2">
                            <path d="M9 18V5l12-2v13" />
                            <circle cx="6" cy="18" r="3" />
                            <circle cx="18" cy="16" r="3" />
                        </svg>
                    </div>
                    <h3 style={{ margin: 0, fontSize: '1.2rem', fontWeight: 600 }}>暂无喜欢的歌曲</h3>
                    <p style={{ margin: 0, fontSize: '0.95rem', color: theme.colors.text.muted, maxWidth: '24rem' }}>
                        在音乐推荐与发现流程中，点击爱心即可将喜欢的歌曲收藏到这里。
                    </p>
                </div>
            ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                    {likedSongs.map((song, index) => (
                        <div
                            key={song.id}
                            style={{
                                display: 'flex',
                                alignItems: 'center',
                                padding: '1rem',
                                borderRadius: theme.borderRadius.md,
                                backgroundColor: 'rgba(255,255,255,0.02)',
                                transition: 'background-color 0.2s',
                                cursor: 'pointer',
                            }}
                            onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.05)')}
                            onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.02)')}
                            onClick={() => playSong({ title: song.title, artist: song.artist, genre: song.genre, preview_url: song.preview_url })}
                        >
                            <div style={{ width: '3rem', color: theme.colors.text.muted, fontSize: '1rem', textAlign: 'center' }}>
                                {index + 1}
                            </div>
                            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '0.2rem' }}>
                                <div style={{ fontWeight: 600, color: theme.colors.text.primary, fontSize: '1.05rem' }}>{song.title}</div>
                                <div style={{ fontSize: '0.85rem', color: theme.colors.text.secondary }}>{song.artist}</div>
                            </div>
                            {song.genre && (
                                <div style={{ padding: '0 1.5rem', color: theme.colors.text.muted, fontSize: '0.85rem' }}>
                                    {song.genre}
                                </div>
                            )}
                            <div
                                style={{
                                    padding: '0.5rem',
                                    color: theme.colors.primary.accent,
                                    cursor: 'pointer',
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'center',
                                }}
                                title="移除喜欢"
                                onClick={(e) => {
                                    e.stopPropagation();
                                    toggleLike(song);
                                }}
                            >
                                <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                    <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"></path>
                                </svg>
                            </div>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}
