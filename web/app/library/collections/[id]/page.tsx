'use client';

import { theme } from '@/styles/theme';
import { useRouter, useParams } from 'next/navigation';
import { useLibrary } from '@/context/LibraryContext';
import { usePlayer } from '@/context/PlayerContext';

export default function CollectionDetailPage() {
    const router = useRouter();
    const params = useParams();
    const collectionId = Number(params?.id);
    const { collections, removeFromCollection } = useLibrary();
    const { playSong } = usePlayer();

    const collection = collections.find(c => c.id === collectionId);

    if (!collection) {
        return (
            <div style={{ padding: '2rem', color: theme.colors.text.primary, textAlign: 'center' }}>
                <p>歌单未找到</p>
                <button onClick={() => router.back()} style={{ color: theme.colors.primary.accent, background: 'none', border: 'none', cursor: 'pointer', fontSize: '1rem' }}>← 返回</button>
            </div>
        );
    }

    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem', padding: '1rem', color: theme.colors.text.primary, minHeight: '100%' }}>
            {/* 返回按钮 */}
            <button
                onClick={() => router.back()}
                style={{
                    display: 'flex', alignItems: 'center', gap: '0.4rem',
                    background: 'none', border: 'none', color: theme.colors.text.secondary,
                    cursor: 'pointer', fontSize: '0.9rem', fontWeight: 500,
                    padding: '0.25rem 0', width: 'fit-content', transition: 'color 0.2s',
                }}
                onMouseEnter={e => (e.currentTarget.style.color = '#fff')}
                onMouseLeave={e => (e.currentTarget.style.color = theme.colors.text.secondary)}
            >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                    <polyline points="15 18 9 12 15 6" />
                </svg>
                返回收藏夹
            </button>

            {/* 歌单头部 */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '1.5rem' }}>
                <div style={{
                    width: '120px', height: '120px', borderRadius: theme.borderRadius.md,
                    backgroundColor: collection.coverColor || '#5B8DEF',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    boxShadow: theme.shadows.md, flexShrink: 0,
                }}>
                    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="rgba(255,255,255,0.4)" strokeWidth="1.2">
                        <path d="M9 18V5l12-2v13" />
                        <circle cx="6" cy="18" r="3" />
                        <circle cx="18" cy="16" r="3" />
                    </svg>
                </div>
                <div>
                    <p style={{ margin: 0, fontSize: '0.85rem', fontWeight: 600, letterSpacing: '0.05em', color: theme.colors.text.muted }}>歌单</p>
                    <h1 style={{ margin: '0.2rem 0', fontSize: '2.4rem', fontWeight: 800, letterSpacing: '-0.02em' }}>{collection.name}</h1>
                    <p style={{ margin: 0, fontSize: '0.9rem', color: theme.colors.text.secondary }}>共 {collection.songs.length} 首歌曲</p>
                </div>
            </div>

            {/* 歌曲列表 */}
            {collection.songs.length === 0 ? (
                <div style={{
                    flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
                    gap: '1rem', padding: '4rem', borderRadius: theme.borderRadius.lg,
                    backgroundColor: 'rgba(255,255,255,0.02)', border: `1px dashed ${theme.colors.border.default}`,
                    textAlign: 'center',
                }}>
                    <h3 style={{ margin: 0, fontSize: '1.2rem', fontWeight: 600 }}>歌单还是空的</h3>
                    <p style={{ margin: 0, fontSize: '0.95rem', color: theme.colors.text.muted, maxWidth: '24rem' }}>
                        在音乐推荐页面点击歌曲卡片上的收藏图标，选择这个歌单即可添加歌曲。
                    </p>
                </div>
            ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                    {collection.songs.map((song, index) => (
                        <div
                            key={song.id}
                            style={{
                                display: 'flex', alignItems: 'center', padding: '1rem',
                                borderRadius: theme.borderRadius.md, backgroundColor: 'rgba(255,255,255,0.02)',
                                transition: 'background-color 0.2s', cursor: 'pointer',
                            }}
                            onMouseEnter={e => (e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.05)')}
                            onMouseLeave={e => (e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.02)')}
                            onClick={() => playSong({ title: song.title, artist: song.artist, genre: song.genre, preview_url: song.preview_url })}
                        >
                            <div style={{ width: '3rem', color: theme.colors.text.muted, fontSize: '1rem', textAlign: 'center' }}>{index + 1}</div>
                            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '0.2rem' }}>
                                <div style={{ fontWeight: 600, color: theme.colors.text.primary, fontSize: '1.05rem' }}>{song.title}</div>
                                <div style={{ fontSize: '0.85rem', color: theme.colors.text.secondary }}>{song.artist}</div>
                            </div>
                            {song.genre && (
                                <div style={{ padding: '0 1.5rem', color: theme.colors.text.muted, fontSize: '0.85rem' }}>{song.genre}</div>
                            )}
                            <div
                                style={{ padding: '0.5rem', color: theme.colors.text.muted, cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center' }}
                                title="从歌单中移除"
                                onClick={e => { e.stopPropagation(); removeFromCollection(collectionId, song.id); }}
                            >
                                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                    <polyline points="3 6 5 6 21 6" />
                                    <path d="M19 6l-1 14H6L5 6" />
                                    <path d="M10 11v6" />
                                    <path d="M14 11v6" />
                                    <path d="M9 6V4h6v2" />
                                </svg>
                            </div>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}
