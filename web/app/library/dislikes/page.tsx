'use client';

/**
 * 🚫 app/library/dislikes/page.tsx (不喜欢管理页面)
 * 功能：展示用户标记为「不喜欢」的歌曲列表，支持撤销操作。
 * 被标记为不喜欢的歌曲会在推荐中自动被过滤。
 */

import { theme } from '@/styles/theme';
import { useLibrary } from '@/context/LibraryContext';
import { useRouter } from 'next/navigation';

export default function DislikesPage() {
    const { dislikedSongs, undoDislike, syncFromBackend, isSyncing } = useLibrary();
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

            {/* 标题区 */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '1.5rem', marginBottom: '1rem' }}>
                <div
                    style={{
                        width: '120px',
                        height: '120px',
                        borderRadius: theme.borderRadius.md,
                        background: 'linear-gradient(135deg, #e74c3c 0%, #c0392b 100%)',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        boxShadow: theme.shadows.md,
                    }}
                >
                    <svg width="48" height="48" viewBox="0 0 24 24" fill="white" stroke="none">
                        <circle cx="12" cy="12" r="10" fill="none" stroke="white" strokeWidth="2" />
                        <line x1="4.93" y1="4.93" x2="19.07" y2="19.07" stroke="white" strokeWidth="2" />
                    </svg>
                </div>
                <div>
                    <p style={{ margin: 0, fontSize: '0.85rem', fontWeight: 600, letterSpacing: '0.05em', color: theme.colors.text.muted }}>
                        屏蔽列表
                    </p>
                    <h1 style={{ margin: '0.2rem 0', fontSize: '3rem', fontWeight: 800, letterSpacing: '-0.02em' }}>
                        不喜欢管理
                    </h1>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                        <p style={{ margin: 0, fontSize: '0.9rem', color: theme.colors.text.secondary }}>
                            共 {dislikedSongs.length} 首被屏蔽歌曲
                        </p>
                        <button
                            onClick={syncFromBackend}
                            disabled={isSyncing}
                            style={{
                                background: 'none',
                                border: `1px solid ${theme.colors.border.default}`,
                                color: theme.colors.text.secondary,
                                padding: '0.3rem 0.8rem',
                                borderRadius: '1rem',
                                fontSize: '0.75rem',
                                cursor: isSyncing ? 'wait' : 'pointer',
                                transition: 'all 0.2s',
                                opacity: isSyncing ? 0.5 : 1,
                            }}
                        >
                            {isSyncing ? '同步中...' : '🔄 从后端同步'}
                        </button>
                    </div>
                    <p style={{ margin: '0.5rem 0 0', fontSize: '0.8rem', color: theme.colors.text.muted }}>
                        这些歌曲将不会出现在推荐结果中。点击「撤销」可恢复。
                    </p>
                </div>
            </div>

            {dislikedSongs.length === 0 ? (
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
                            <path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3" />
                        </svg>
                    </div>
                    <h3 style={{ margin: 0, fontSize: '1.2rem', fontWeight: 600 }}>没有屏蔽的歌曲</h3>
                    <p style={{ margin: 0, fontSize: '0.95rem', color: theme.colors.text.muted, maxWidth: '24rem' }}>
                        在推荐结果中点击"不喜欢"按钮，歌曲将被添加到此列表并从后续推荐中排除。
                    </p>
                </div>
            ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                    {dislikedSongs.map((song, index) => (
                        <div
                            key={song.id}
                            style={{
                                display: 'flex',
                                alignItems: 'center',
                                padding: '1rem',
                                borderRadius: theme.borderRadius.md,
                                backgroundColor: 'rgba(255,255,255,0.02)',
                                transition: 'background-color 0.2s',
                            }}
                            onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.05)')}
                            onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.02)')}
                        >
                            <div style={{ width: '3rem', color: theme.colors.text.muted, fontSize: '1rem', textAlign: 'center' }}>
                                {index + 1}
                            </div>
                            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '0.2rem' }}>
                                <div style={{ fontWeight: 600, color: theme.colors.text.primary, fontSize: '1.05rem' }}>
                                    {song.title}
                                </div>
                                <div style={{ fontSize: '0.85rem', color: theme.colors.text.secondary }}>
                                    {song.artist}
                                </div>
                            </div>
                            <button
                                style={{
                                    padding: '0.4rem 1rem',
                                    borderRadius: '2rem',
                                    border: `1px solid ${theme.colors.border.default}`,
                                    background: 'transparent',
                                    color: theme.colors.text.secondary,
                                    cursor: 'pointer',
                                    fontSize: '0.8rem',
                                    fontWeight: 500,
                                    transition: 'all 0.2s',
                                }}
                                title="撤销不喜欢"
                                onClick={() => undoDislike(song.title, song.artist)}
                                onMouseEnter={(e) => {
                                    e.currentTarget.style.borderColor = '#1db954';
                                    e.currentTarget.style.color = '#1db954';
                                }}
                                onMouseLeave={(e) => {
                                    e.currentTarget.style.borderColor = theme.colors.border.default;
                                    e.currentTarget.style.color = theme.colors.text.secondary;
                                }}
                            >
                                撤销
                            </button>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}
