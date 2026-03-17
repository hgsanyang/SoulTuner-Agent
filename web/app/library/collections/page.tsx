'use client';

import { theme } from '@/styles/theme';
import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { useLibrary } from '@/context/LibraryContext';

const COVER_COLORS = ['#5B8DEF', '#EF5B9C', '#EFA65B', '#5BEFA6', '#A65BEF', '#EF5B5B'];

export default function CollectionsPage() {
    const { collections, addCollection } = useLibrary();
    const [isCreating, setIsCreating] = useState(false);
    const [newTitle, setNewTitle] = useState('');
    const router = useRouter();

    const handleCreate = () => {
        if (newTitle.trim()) {
            addCollection(newTitle.trim());
            setNewTitle('');
            setIsCreating(false);
        }
    };

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

            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <div>
                    <h1 style={{ margin: '0', fontSize: '2.4rem', fontWeight: 800, letterSpacing: '-0.02em' }}>
                        我的收藏夹
                    </h1>
                    <p style={{ margin: '0.5rem 0 0', fontSize: '0.95rem', color: theme.colors.text.secondary }}>
                        管理并沉淀你的专属听歌体验
                    </p>
                </div>
                <button
                    type="button"
                    onClick={() => setIsCreating(true)}
                    style={{
                        display: 'flex', alignItems: 'center', gap: '0.5rem',
                        padding: '0.6rem 1.2rem', borderRadius: theme.borderRadius.full,
                        backgroundColor: 'rgba(255,255,255,0.1)', color: '#fff',
                        border: `1px solid rgba(255,255,255,0.2)`, fontWeight: 600,
                        cursor: 'pointer', transition: 'background-color 0.2s',
                    }}
                    onMouseOver={e => (e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.18)')}
                    onMouseOut={e => (e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.1)')}
                >
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                        <line x1="12" y1="5" x2="12" y2="19" />
                        <line x1="5" y1="12" x2="19" y2="12" />
                    </svg>
                    新建歌单
                </button>
            </div>

            {isCreating && (
                <div style={{
                    padding: '1.5rem', backgroundColor: theme.colors.background.elevated,
                    borderRadius: theme.borderRadius.md, border: `1px solid ${theme.colors.border.focus}`,
                    display: 'flex', gap: '1rem', alignItems: 'center'
                }}>
                    <input
                        type="text" autoFocus
                        placeholder="为你的新歌单起个名字..."
                        value={newTitle} onChange={e => setNewTitle(e.target.value)}
                        onKeyDown={e => e.key === 'Enter' && handleCreate()}
                        style={{
                            flex: 1, padding: '0.8rem 1rem', borderRadius: theme.borderRadius.md,
                            backgroundColor: 'rgba(0,0,0,0.5)', border: `1px solid ${theme.colors.border.default}`,
                            color: '#fff', fontSize: '1rem', outline: 'none',
                        }}
                    />
                    <button onClick={handleCreate}
                        style={{ padding: '0.8rem 1.5rem', borderRadius: theme.borderRadius.md, backgroundColor: theme.colors.primary.accent, color: '#000', border: 'none', fontWeight: 600, cursor: 'pointer' }}>
                        保存
                    </button>
                    <button onClick={() => { setIsCreating(false); setNewTitle(''); }}
                        style={{ padding: '0.8rem 1.5rem', borderRadius: theme.borderRadius.md, backgroundColor: 'transparent', color: theme.colors.text.secondary, border: 'none', fontWeight: 600, cursor: 'pointer' }}>
                        取消
                    </button>
                </div>
            )}

            <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))',
                gap: '1.5rem',
            }}>
                {collections.map((p, idx) => (
                    <div
                        key={p.id}
                        onClick={() => router.push(`/library/collections/${p.id}`)}
                        style={{
                            backgroundColor: theme.colors.background.card,
                            padding: '1rem', borderRadius: theme.borderRadius.md,
                            cursor: 'pointer', transition: 'all 0.2s',
                        }}
                        onMouseEnter={e => {
                            e.currentTarget.style.backgroundColor = theme.colors.background.hover;
                            e.currentTarget.style.transform = 'translateY(-3px)';
                        }}
                        onMouseLeave={e => {
                            e.currentTarget.style.backgroundColor = theme.colors.background.card;
                            e.currentTarget.style.transform = 'translateY(0)';
                        }}
                    >
                        <div style={{
                            width: '100%', aspectRatio: '1',
                            backgroundColor: p.coverColor || COVER_COLORS[idx % COVER_COLORS.length],
                            borderRadius: theme.borderRadius.sm, marginBottom: '1rem',
                            boxShadow: theme.shadows.sm,
                            display: 'flex', alignItems: 'center', justifyContent: 'center'
                        }}>
                            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="rgba(255,255,255,0.3)" strokeWidth="1">
                                <path d="M9 18V5l12-2v13" />
                                <circle cx="6" cy="18" r="3" />
                                <circle cx="18" cy="16" r="3" />
                            </svg>
                        </div>
                        <p style={{ margin: 0, fontWeight: 600, fontSize: '1rem', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{p.name}</p>
                        <span style={{ fontSize: '0.85rem', color: theme.colors.text.muted }}>{p.songs.length} 首歌曲</span>
                    </div>
                ))}
            </div>
        </div>
    );
}
