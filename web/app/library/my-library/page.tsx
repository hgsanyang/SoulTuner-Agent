'use client';

/**
 * 🎵 我的曲库页面 (My Library)
 * 显示 Neo4j 知识图谱中的所有 Song 节点。
 * 支持搜索筛选、播放、查看标签、删除管理。
 */

import { useState, useEffect, useCallback } from 'react';
import { useLang } from '@/context/LanguageContext';
import { theme } from '@/styles/theme';
import { usePlayer } from '@/context/PlayerContext';
import { useLibrary } from '@/context/LibraryContext';
import { useRouter } from 'next/navigation';
import { fetchLibrarySongs, deleteSongFromLibrary, retainOnlineAudio, updateLibrarySongTags, purgeCatalogCandidates, LibrarySong, CatalogTier } from '@/lib/api';
import { resolveOptionalMediaUrl } from '@/lib/runtime-url';

export default function MyLibraryPage() {
  const { t } = useLang();
    const [songs, setSongs] = useState<LibrarySong[]>([]);
    const [total, setTotal] = useState(0);
    // 曲库 vs 临时候选。推荐时联网抓来的缓存不算"我的曲库"，默认不显示。
    const [tier, setTier] = useState<CatalogTier>('library');
    const [tierCounts, setTierCounts] = useState({ library: 0, candidate: 0 });
    const [purging, setPurging] = useState(false);
    const [loading, setLoading] = useState(true);
    const [searchQuery, setSearchQuery] = useState('');
    const [sourceFilter, setSourceFilter] = useState('all');
    const [languageFilter, setLanguageFilter] = useState('all');
    const [moodFilter, setMoodFilter] = useState('all');
    const [qualityFilter, setQualityFilter] = useState('all');
    const [selectedSong, setSelectedSong] = useState<LibrarySong | null>(null);
    const [selectedKeys, setSelectedKeys] = useState<Set<string>>(new Set());
    const [tagDraft, setTagDraft] = useState({
        genres: '',
        moods: '',
        themes: '',
        scenarios: '',
        language: '',
    });
    const [savingTags, setSavingTags] = useState(false);
    const [retainingAudio, setRetainingAudio] = useState(false);
    const [deleting, setDeleting] = useState<string | null>(null);
    const { playSong } = usePlayer();
    const { showToast } = useLibrary();
    const router = useRouter();

    const songKey = (song: LibrarySong) => song.music_id || `${song.title}_${song.artist}_${song.audio_url || ''}`;

    const loadSongs = useCallback(async () => {
        setLoading(true);
        const data = await fetchLibrarySongs(0, 500, tier);
        setSongs(data.songs);
        setTotal(data.total);
        setTierCounts(data.counts);
        setLoading(false);
    }, [tier]);

    useEffect(() => { loadSongs(); }, [loadSongs]);

    const runPurge = async () => {
        // 两步：先干跑拿到数量给用户看，确认后才真删。删节点不可撤销。
        setPurging(true);
        const preview = await purgeCatalogCandidates(true);
        if (!preview.success) {
            setPurging(false);
            showToast(`❌ ${preview.error || t('清理失败')}`);
            return;
        }
        const eligible = preview.eligible || 0;
        if (eligible === 0) {
            setPurging(false);
            showToast(t('没有可清理的临时候选（还在缓存期内或你操作过的都会保留）'));
            return;
        }
        if (!window.confirm(t('将删除 {v0} 条已过期且从未被你操作过的临时候选，此操作不可撤销。继续？', { v0: eligible }))) {
            setPurging(false);
            return;
        }
        const result = await purgeCatalogCandidates(false);
        setPurging(false);
        if (result.success) {
            showToast(t('✅ 已清理 {v0} 条临时候选', { v0: result.deleted || 0 }));
            loadSongs();
        } else {
            showToast(`❌ ${result.error || t('清理失败')}`);
        }
    };

    useEffect(() => {
        if (!selectedSong) return;
        setTagDraft({
            genres: (selectedSong.genres || []).join(', '),
            moods: (selectedSong.moods || []).join(', '),
            themes: (selectedSong.themes || []).join(', '),
            scenarios: (selectedSong.scenarios || []).join(', '),
            language: selectedSong.language || '',
        });
    }, [selectedSong]);

    const parseTags = (value: string) => value
        .split(/[,，/]/)
        .map(v => v.trim())
        .filter(Boolean)
        .slice(0, 5);

    const saveSelectedTags = async () => {
        if (!selectedSong) return;
        setSavingTags(true);
        const next = {
            music_id: selectedSong.music_id,
            title: selectedSong.title,
            artist: selectedSong.artist,
            genres: parseTags(tagDraft.genres),
            moods: parseTags(tagDraft.moods),
            themes: parseTags(tagDraft.themes),
            scenarios: parseTags(tagDraft.scenarios),
            language: tagDraft.language.trim(),
        };
        const result = await updateLibrarySongTags(next);
        setSavingTags(false);
        if (result.success) {
            const updatedSong = {
                ...selectedSong,
                genres: next.genres,
                moods: next.moods,
                themes: next.themes,
                scenarios: next.scenarios,
                language: next.language,
            };
            setSelectedSong(updatedSong);
            setSongs(prev => prev.map(song => (
                (selectedSong.music_id && song.music_id === selectedSong.music_id)
                || (!selectedSong.music_id && song.title === selectedSong.title && song.artist === selectedSong.artist)
            ) ? updatedSong : song));
            showToast(t('✅ 标签已更新'));
        } else {
            showToast(t('❌ 标签更新失败: {v0}', { v0: result.error || t('未知错误') }));
        }
    };

    const retainSelectedAudio = async () => {
        if (!selectedSong) return;
        setRetainingAudio(true);
        const result = await retainOnlineAudio({
            music_id: selectedSong.music_id,
            song_id: selectedSong.source_id,
            title: selectedSong.title,
            artist: selectedSong.artist,
        });
        setRetainingAudio(false);
        if (result.success) {
            const updatedSong = { ...selectedSong, audio_retention: 'saved', audio_status: 'cached' };
            setSelectedSong(updatedSong);
            setSongs(prev => prev.map(song => songKey(song) === songKey(selectedSong) ? updatedSong : song));
            showToast(t('✅ 「{v0}」音源已长期保存', { v0: selectedSong.title }));
        } else {
            showToast(t('❌ 保存音源失败: {v0}', { v0: result.error || result.message || t('未知错误') }));
        }
    };

    const handleDelete = async (song: LibrarySong) => {
        const key = songKey(song);
        setDeleting(key);
        const result = await deleteSongFromLibrary(song.title, song.artist);
        setDeleting(null);
        if (result.success) {
            showToast(t('🗑️ 已从曲库中移除「{v0}」', { v0: song.title }));
            setSongs(prev => prev.filter(s => songKey(s) !== key));
            setSelectedKeys(prev => {
                const next = new Set(prev);
                next.delete(key);
                return next;
            });
            if (selectedSong && songKey(selectedSong) === key) setSelectedSong(null);
            setTotal(prev => prev - 1);
        } else {
            showToast(t('❌ 删除失败: {v0}', { v0: result.message }));
        }
    };

    const toggleSelected = (song: LibrarySong) => {
        const key = songKey(song);
        setSelectedKeys(prev => {
            const next = new Set(prev);
            if (next.has(key)) next.delete(key);
            else next.add(key);
            return next;
        });
    };

    const clearSelected = () => setSelectedKeys(new Set());

    const selectFiltered = () => {
        setSelectedKeys(new Set(filtered.map(songKey)));
    };

    const handleBulkDelete = async () => {
        const targets = songs.filter(song => selectedKeys.has(songKey(song)));
        if (!targets.length) return;
        const ok = window.confirm(t('确定从曲库中移除选中的 {v0} 首歌曲吗？', { v0: targets.length }));
        if (!ok) return;
        for (const song of targets) {
            // Keep the existing safe backend delete path; each item reports its own failure.
            await handleDelete(song);
        }
        clearSelected();
    };

    const sourceOptions = Array.from(new Set(songs.map(s => s.source || 'local'))).sort();
    const languageOptions = Array.from(new Set(songs.map(s => s.language || '').filter(Boolean))).sort();
    const moodOptions = Array.from(new Set(songs.flatMap(s => s.moods || []).filter(Boolean))).sort();
    const duplicateKeyCounts = songs.reduce((acc, song) => {
        const key = song.duplicate_key || '';
        if (key) acc.set(key, (acc.get(key) || 0) + 1);
        return acc;
    }, new Map<string, number>());

    const filtered = songs.filter(s => {
        const q = searchQuery.trim().toLowerCase();
        const matchesQuery = !q ||
            (s.title || '').toLowerCase().includes(q) ||
            (s.artist || '').toLowerCase().includes(q) ||
            (s.album || '').toLowerCase().includes(q) ||
            (s.moods || []).some(m => (m || '').toLowerCase().includes(q)) ||
            (s.themes || []).some(t => (t || '').toLowerCase().includes(q)) ||
            (s.genres || []).some(g => (g || '').toLowerCase().includes(q)) ||
            (s.scenarios || []).some(sc => (sc || '').toLowerCase().includes(q)) ||
            (s.vibe || '').toLowerCase().includes(q);
        const matchesSource = sourceFilter === 'all' || (s.source || 'local') === sourceFilter;
        const matchesLanguage = languageFilter === 'all' || (s.language || '') === languageFilter;
        const matchesMood = moodFilter === 'all' || (s.moods || []).includes(moodFilter);
        const missingCount = (s.missing_fields || []).length;
        const matchesQuality = qualityFilter === 'all'
            || (qualityFilter === 'ready' && missingCount === 0)
            || (qualityFilter === 'missing' && missingCount > 0)
            || (qualityFilter === 'low' && (s.quality_score ?? 1) < 0.8)
            || (qualityFilter === 'duplicate' && (duplicateKeyCounts.get(s.duplicate_key || '') || 0) > 1);
        return matchesQuery && matchesSource && matchesLanguage && matchesMood && matchesQuality;
    });

    const duplicateGroups = Array.from(duplicateKeyCounts.entries()).filter(([, count]) => count > 1);

    const sourceLabel = (src: string) => {
        switch (src) {
            case 'online': return { text: t('联网'), color: '#3b82f6' };
            case 'mtg': return { text: 'MTG', color: '#8b5cf6' };
            default: return { text: t('本地'), color: theme.colors.primary.accent };
        }
    };

    const missingLabel = (field: string) => {
        const labels: Record<string, string> = {
            audio: t('音频'),
            cover: t('封面'),
            lyrics: t('歌词'),
            language: t('语言'),
            release_year: t('发行年'),
            muq_embedding: 'MuQ',
            m2d_embedding: 'M2D',
            omar_embedding: 'OMAR',
        };
        return labels[field] || field;
    };

    const qualityBadge = (song: LibrarySong) => {
        const score = typeof song.quality_score === 'number' ? song.quality_score : 0;
        const missingCount = (song.missing_fields || []).length;
        const color = score >= 0.92 ? '#86efac' : score >= 0.8 ? '#fde68a' : '#fca5a5';
        const background = score >= 0.92 ? 'rgba(34,197,94,0.12)' : score >= 0.8 ? 'rgba(250,204,21,0.10)' : 'rgba(248,113,113,0.10)';
        const border = score >= 0.92 ? 'rgba(34,197,94,0.24)' : score >= 0.8 ? 'rgba(250,204,21,0.18)' : 'rgba(248,113,113,0.22)';
        return (
            <span style={{ fontSize: '0.7rem', padding: '0.15rem 0.5rem', borderRadius: '9999px', color, background, border: `1px solid ${border}`, whiteSpace: 'nowrap', flexShrink: 0 }}>
                {t('质量 {v0}%{v1}', { v0: Math.round(score * 100), v1: missingCount ? t(' · 缺 {v0}', { v0: missingCount }) : '' })}
            </span>
        );
    };

    const vectorBadge = (label: string, ok?: boolean) => (
        <span key={label} style={{
            fontSize: '0.72rem',
            padding: '0.18rem 0.5rem',
            borderRadius: '9999px',
            background: ok ? 'rgba(34,197,94,0.12)' : 'rgba(239,68,68,0.1)',
            color: ok ? '#86efac' : '#fca5a5',
            border: `1px solid ${ok ? 'rgba(34,197,94,0.24)' : 'rgba(239,68,68,0.2)'}`,
        }}>{label} {ok ? '✓' : t('缺')}</span>
    );

    const tagInputStyle = {
        width: '100%',
        padding: '0.45rem 0.55rem',
        background: 'rgba(255,255,255,0.045)',
        border: `1px solid ${theme.colors.border.default}`,
        borderRadius: theme.borderRadius.sm,
        color: theme.colors.text.primary,
        fontSize: '0.78rem',
        outline: 'none',
    };

    const renderTagInput = (label: string, key: keyof typeof tagDraft, placeholder: string) => (
        <label style={{ display: 'grid', gap: '0.3rem', fontSize: '0.74rem', color: theme.colors.text.muted }}>
            {label}
            <input
                value={tagDraft[key]}
                onChange={e => setTagDraft(prev => ({ ...prev, [key]: e.target.value }))}
                placeholder={placeholder}
                style={tagInputStyle}
            />
        </label>
    );

    return (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', padding: '1rem', color: theme.colors.text.primary, minHeight: '100%' }}>
            {/* 返回按钮 */}
            <button onClick={() => router.back()} style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', background: 'none', border: 'none', color: theme.colors.text.secondary, cursor: 'pointer', fontSize: '0.9rem', fontWeight: 500, padding: '0.25rem 0', width: 'fit-content', transition: 'color 0.2s' }}
                onMouseEnter={e => (e.currentTarget.style.color = '#fff')}
                onMouseLeave={e => (e.currentTarget.style.color = theme.colors.text.secondary)}
            >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><polyline points="15 18 9 12 15 6" /></svg>
                {t('返回')}
            </button>

            {/* Header */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '1.5rem', marginBottom: '0.5rem' }}>
                <div style={{ width: '100px', height: '100px', borderRadius: theme.borderRadius.md, background: 'linear-gradient(135deg, #8b5cf6 0%, #6d28d9 100%)', display: 'flex', alignItems: 'center', justifyContent: 'center', boxShadow: theme.shadows.md }}>
                    <svg width="42" height="42" viewBox="0 0 24 24" fill="white" stroke="none">
                        <path d="M12 3v10.55c-.59-.34-1.27-.55-2-.55-2.21 0-4 1.79-4 4s1.79 4 4 4 4-1.79 4-4V7h4V3h-6z" />
                    </svg>
                </div>
                <div>
                    <p style={{ margin: 0, fontSize: '0.8rem', fontWeight: 600, letterSpacing: '0.05em', color: theme.colors.text.muted }}>{t('知识图谱')}</p>
                    <h1 style={{ margin: '0.2rem 0', fontSize: '2.5rem', fontWeight: 800, letterSpacing: '-0.02em' }}>{t('我的曲库')}</h1>
                    <p style={{ margin: 0, fontSize: '0.9rem', color: theme.colors.text.secondary }}>
                        {loading ? t('加载中...') : t('曲库 {v0} 首 · 临时候选 {v1} 首', { v0: tierCounts.library, v1: tierCounts.candidate })}
                    </p>
                </div>
            </div>

            {/* 曲库 / 临时候选。联网推荐时后台抓来的缓存单独一层，不混进曲库规模。 */}
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem', alignItems: 'center' }}>
                {([
                    { key: 'library' as CatalogTier, label: t('我的曲库'), count: tierCounts.library },
                    { key: 'candidate' as CatalogTier, label: t('临时候选'), count: tierCounts.candidate },
                    { key: 'all' as CatalogTier, label: t('全部'), count: tierCounts.library + tierCounts.candidate },
                ]).map(option => {
                    const active = tier === option.key;
                    return (
                        <button
                            key={option.key}
                            onClick={() => { setTier(option.key); setSelectedKeys(new Set()); setSelectedSong(null); }}
                            style={{
                                padding: '0.4rem 0.9rem', borderRadius: theme.borderRadius.sm,
                                border: `1px solid ${active ? theme.colors.primary.accent : theme.colors.border.default}`,
                                background: active ? 'rgba(29,185,84,0.16)' : 'rgba(255,255,255,0.04)',
                                color: active ? theme.colors.primary.accent : theme.colors.text.secondary,
                                cursor: 'pointer', fontSize: '0.8rem', fontWeight: active ? 700 : 500,
                            }}
                        >
                            {option.label} {option.count}
                        </button>
                    );
                })}
                {tier === 'candidate' && (
                    <>
                        <span style={{ fontSize: '0.76rem', color: theme.colors.text.muted, maxWidth: '46rem' }}>
                            {t('这些是推荐时联网抓来、为了能立刻试听而临时缓存的歌，你没有下载或入库过。MP3 超过缓存期会自动释放，节点保留在这一层。')}
                        </span>
                        <button onClick={runPurge} disabled={purging} style={{
                            padding: '0.4rem 0.8rem', borderRadius: theme.borderRadius.sm,
                            border: '1px solid rgba(240,96,96,0.5)', background: 'rgba(240,96,96,0.08)',
                            color: '#f06060', cursor: purging ? 'not-allowed' : 'pointer', fontSize: '0.78rem',
                        }}>
                            {purging ? t('清理中...') : t('清理已过期的临时候选')}
                        </button>
                    </>
                )}
            </div>

            {/* Search and filters */}
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.75rem', alignItems: 'center' }}>
                <div style={{ position: 'relative', minWidth: '260px', flex: '1 1 320px', maxWidth: '460px' }}>
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke={theme.colors.text.muted} strokeWidth="2" style={{ position: 'absolute', left: '0.85rem', top: '50%', transform: 'translateY(-50%)' }}>
                        <circle cx="11" cy="11" r="8" /><line x1="21" y1="21" x2="16.65" y2="16.65" />
                    </svg>
                    <input
                        type="text"
                        placeholder={t('搜索歌名、歌手、专辑、标签')}
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
                <select value={sourceFilter} onChange={e => setSourceFilter(e.target.value)} style={{ padding: '0.65rem 0.85rem', background: 'rgba(255,255,255,0.05)', border: `1px solid ${theme.colors.border.default}`, borderRadius: theme.borderRadius.sm, color: theme.colors.text.primary }}>
                    <option value="all">{t('全部来源')}</option>
                    {sourceOptions.map(source => <option key={source} value={source}>{sourceLabel(source).text}</option>)}
                </select>
                <select value={languageFilter} onChange={e => setLanguageFilter(e.target.value)} style={{ padding: '0.65rem 0.85rem', background: 'rgba(255,255,255,0.05)', border: `1px solid ${theme.colors.border.default}`, borderRadius: theme.borderRadius.sm, color: theme.colors.text.primary }}>
                    <option value="all">{t('全部语言')}</option>
                    {languageOptions.map(language => <option key={language} value={language}>{language}</option>)}
                </select>
                <select value={moodFilter} onChange={e => setMoodFilter(e.target.value)} style={{ padding: '0.65rem 0.85rem', background: 'rgba(255,255,255,0.05)', border: `1px solid ${theme.colors.border.default}`, borderRadius: theme.borderRadius.sm, color: theme.colors.text.primary }}>
                    <option value="all">{t('全部情绪')}</option>
                    {moodOptions.map(mood => <option key={mood} value={mood}>{mood}</option>)}
                </select>
                <select value={qualityFilter} onChange={e => setQualityFilter(e.target.value)} style={{ padding: '0.65rem 0.85rem', background: 'rgba(255,255,255,0.05)', border: `1px solid ${theme.colors.border.default}`, borderRadius: theme.borderRadius.sm, color: theme.colors.text.primary }}>
                    <option value="all">{t('全部质量')}</option>
                    <option value="ready">{t('资料完整')}</option>
                    <option value="missing">{t('待补资料')}</option>
                    <option value="low">{t('低质量优先')}</option>
                    <option value="duplicate">{t('疑似重复')}</option>
                </select>
                <span style={{ fontSize: '0.78rem', color: theme.colors.text.muted }}>{t('显示 {v0} / {v1}', { v0: filtered.length, v1: total })}</span>
                {duplicateGroups.length > 0 && (
                    <span style={{ fontSize: '0.78rem', color: '#fde68a' }}>{t('疑似重复组 {v0}', { v0: duplicateGroups.length })}</span>
                )}
            </div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem', alignItems: 'center', minHeight: '2rem' }}>
                <button onClick={selectFiltered} disabled={filtered.length === 0} style={{ padding: '0.45rem 0.7rem', borderRadius: theme.borderRadius.sm, border: `1px solid ${theme.colors.border.default}`, background: 'rgba(255,255,255,0.04)', color: theme.colors.text.secondary, cursor: filtered.length ? 'pointer' : 'not-allowed', fontSize: '0.78rem' }}>
                    {t('全选当前结果')}
                </button>
                <button onClick={clearSelected} disabled={selectedKeys.size === 0} style={{ padding: '0.45rem 0.7rem', borderRadius: theme.borderRadius.sm, border: `1px solid ${theme.colors.border.default}`, background: 'rgba(255,255,255,0.04)', color: selectedKeys.size ? theme.colors.text.secondary : theme.colors.text.muted, cursor: selectedKeys.size ? 'pointer' : 'not-allowed', fontSize: '0.78rem' }}>
                    {t('清空选择')}
                </button>
                <button onClick={handleBulkDelete} disabled={selectedKeys.size === 0} style={{ padding: '0.45rem 0.7rem', borderRadius: theme.borderRadius.sm, border: '1px solid rgba(248,113,113,0.28)', background: selectedKeys.size ? 'rgba(248,113,113,0.12)' : 'rgba(255,255,255,0.03)', color: selectedKeys.size ? '#fca5a5' : theme.colors.text.muted, cursor: selectedKeys.size ? 'pointer' : 'not-allowed', fontSize: '0.78rem' }}>
                    {t('批量移除')}{selectedKeys.size ? ` (${selectedKeys.size})` : ''}
                </button>
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
                        {searchQuery ? t('没有匹配的歌曲') : t('曲库为空')}
                    </h3>
                    <p style={{ margin: 0, fontSize: '0.9rem', color: theme.colors.text.muted, maxWidth: '24rem' }}>
                        {searchQuery ? t('试试其他关键词') : t('通过 AI 对话获取新歌后，在待入库页面确认入库即可添加到这里。')}
                    </p>
                </div>
            ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
                    {filtered.map((song) => {
                        const src = sourceLabel(song.source);
                        const key = songKey(song);
                        const isSelected = selectedKeys.has(key);
                        const isDeleting = deleting === key;
                        const audioUrl = resolveOptionalMediaUrl(song.audio_url);
                        const coverUrl = resolveOptionalMediaUrl(song.cover_url);
                        const lyricsUrl = resolveOptionalMediaUrl(song.lrc_url);
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
                                        if (audioUrl) {
                                            playSong({
                                                title: song.title, artist: song.artist,
                                                preview_url: audioUrl,
                                                coverUrl,
                                                lrc_url: lyricsUrl,
                                            });
                                        }
                                    }}
                            >
                                <input
                                    type="checkbox"
                                    aria-label={t('选择 {v0}', { v0: song.title })}
                                    checked={isSelected}
                                    onChange={e => { e.stopPropagation(); toggleSelected(song); }}
                                    onClick={e => e.stopPropagation()}
                                    style={{ width: '16px', height: '16px', accentColor: theme.colors.primary.accent, flexShrink: 0 }}
                                />
                                {/* Cover */}
                                <div style={{
                                    width: '46px', height: '46px', borderRadius: '6px', flexShrink: 0,
                                    background: coverUrl
                                        ? `url(${coverUrl}) center/cover, linear-gradient(135deg, #333, #222)`
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
                                    {song.genres?.slice(0, 1).map(g => (
                                        <span key={g} style={{ fontSize: '0.7rem', padding: '0.15rem 0.45rem', borderRadius: '9999px', background: 'rgba(59,130,246,0.12)', color: '#93c5fd', whiteSpace: 'nowrap' }}>{g}</span>
                                    ))}
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
                                {qualityBadge(song)}

                                <button title={t('详情')} aria-label={t('查看 {v0} 详情', { v0: song.title })} onClick={e => { e.stopPropagation(); setSelectedSong(song); }}
                                    style={{ background: 'rgba(255,255,255,0.05)', border: `1px solid ${theme.colors.border.default}`, color: theme.colors.text.secondary, cursor: 'pointer', padding: '0.35rem 0.6rem', borderRadius: theme.borderRadius.sm, fontSize: '0.76rem' }}>
                                    {t('详情')}
                                </button>

                                {/* Play */}
                                <button title={audioUrl ? t('播放') : t('暂无音源')} aria-label={audioUrl ? t('播放 {v0}', { v0: song.title }) : t('{v0} 暂无音源', { v0: song.title })}
                                    onClick={e => {
                                        e.stopPropagation();
                                        if (audioUrl) {
                                            playSong({
                                                title: song.title, artist: song.artist,
                                                preview_url: audioUrl,
                                                coverUrl,
                                                lrc_url: lyricsUrl,
                                            });
                                        }
                                    }}
                                    disabled={!audioUrl}
                                    style={{ background: 'none', border: 'none', color: audioUrl ? theme.colors.primary.accent : theme.colors.text.muted, cursor: audioUrl ? 'pointer' : 'not-allowed', padding: '0.4rem', display: 'flex', opacity: audioUrl ? 1 : 0.35 }}>
                                    <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3" /></svg>
                                </button>

                                {/* Delete */}
                                <button title={t('从曲库移除')} aria-label={t('从曲库移除 {v0}', { v0: song.title })} onClick={e => { e.stopPropagation(); handleDelete(song); }}
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

            {selectedSong && (
                <div style={{ border: `1px solid ${theme.colors.border.default}`, borderRadius: theme.borderRadius.md, background: 'rgba(255,255,255,0.03)', padding: '1rem', display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                        <div style={{ flex: 1, minWidth: 0 }}>
                            <div style={{ fontSize: '1rem', fontWeight: 700, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{selectedSong.title}</div>
                            <div style={{ fontSize: '0.84rem', color: theme.colors.text.secondary }}>{selectedSong.artist}{selectedSong.album ? ` · ${selectedSong.album}` : ''}</div>
                        </div>
                        <button onClick={() => setSelectedSong(null)} style={{ background: 'none', border: `1px solid ${theme.colors.border.default}`, color: theme.colors.text.secondary, cursor: 'pointer', borderRadius: theme.borderRadius.sm, padding: '0.35rem 0.6rem' }}>{t('关闭')}</button>
                    </div>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: '0.65rem', fontSize: '0.82rem', color: theme.colors.text.secondary }}>
                        <div>{t('来源：')}{sourceLabel(selectedSong.source).text}</div>
                        <div>{t('语言：')}{selectedSong.language || t('未标注')}</div>
                        <div>{t('发行年：')}{selectedSong.release_year || t('未补全')}</div>
                        <div>{t('格式：')}{selectedSong.format || t('未知')}</div>
                        <div>{t('时长：')}{selectedSong.duration ? `${Math.round(selectedSong.duration / 1000)}s` : t('未知')}</div>
                        <div>{t('标签来源：')}{selectedSong.tag_source || t('未记录')}</div>
                        <div>{t('音源保留：')}{selectedSong.audio_retention === 'saved' ? t('长期保存') : selectedSong.audio_retention === 'temporary' ? t('临时缓存') : t('未记录')}</div>
                        <div>{t('入库状态：')}{selectedSong.acquire_status || selectedSong.audio_status || t('正常')}</div>
                        <div>{t('质量分：')}{Math.round((selectedSong.quality_score ?? 0) * 100)}%</div>
                        <div>{t('去重键：')}{selectedSong.duplicate_key || t('未生成')}</div>
                    </div>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem' }}>
                        {vectorBadge('MuQ', selectedSong.vector_coverage?.muq)}
                        {vectorBadge('M2D', selectedSong.vector_coverage?.m2d)}
                        {vectorBadge('OMAR', selectedSong.vector_coverage?.omar)}
                        {(selectedSong.missing_fields || []).slice(0, 8).map(field => (
                            <span key={field} style={{ fontSize: '0.72rem', padding: '0.18rem 0.5rem', borderRadius: '9999px', background: 'rgba(250,204,21,0.1)', color: '#fde68a', border: '1px solid rgba(250,204,21,0.18)' }}>
 {t('待补：')}{missingLabel(field)}
                            </span>
                        ))}
                    </div>
                    {selectedSong.source === 'online' && selectedSong.audio_retention !== 'saved' && selectedSong.audio_url && (
                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '0.75rem', padding: '0.75rem', border: '1px solid rgba(251,191,36,0.22)', borderRadius: theme.borderRadius.sm, background: 'rgba(251,191,36,0.08)' }}>
                            <div style={{ fontSize: '0.78rem', color: '#fde68a', lineHeight: 1.5 }}>
                                {t('这首来自联网临时入库。长期保存后，退出应用或清理临时缓存时不会释放 MP3 文件。')}
                            </div>
                            <button
                                onClick={retainSelectedAudio}
                                disabled={retainingAudio}
                                style={{ background: retainingAudio ? 'rgba(255,255,255,0.08)' : 'rgba(251,191,36,0.18)', border: '1px solid rgba(251,191,36,0.35)', borderRadius: theme.borderRadius.sm, color: '#fde68a', cursor: retainingAudio ? 'wait' : 'pointer', padding: '0.45rem 0.75rem', fontWeight: 700, fontSize: '0.76rem', whiteSpace: 'nowrap' }}
                            >
                                {retainingAudio ? t('保存中...') : t('长期保存音源')}
                            </button>
                        </div>
                    )}
                    {(duplicateKeyCounts.get(selectedSong.duplicate_key || '') || 0) > 1 && (
                        <div style={{ padding: '0.65rem 0.75rem', border: '1px solid rgba(250,204,21,0.18)', borderRadius: theme.borderRadius.sm, background: 'rgba(250,204,21,0.06)', color: '#fde68a', fontSize: '0.76rem' }}>
                            {t('疑似重复：同一标准化键下有 {v0} 首。建议人工确认版本、Live、Remaster 或翻唱后再删除。', { v0: duplicateKeyCounts.get(selectedSong.duplicate_key || '') ?? 0 })}
                        </div>
                    )}
                    {!!selectedSong.knowledge_cards?.length && (
                        <div style={{ display: 'grid', gap: '0.55rem', padding: '0.75rem', border: `1px solid ${theme.colors.border.default}`, borderRadius: theme.borderRadius.sm, background: 'rgba(255,255,255,0.025)' }}>
                            <div style={{ fontSize: '0.78rem', fontWeight: 700, color: theme.colors.text.primary }}>{t('知识卡摘要')}</div>
                            {selectedSong.knowledge_cards.slice(0, 2).map((card, index) => (
                                <div key={card.key || index} style={{ display: 'grid', gap: '0.25rem', fontSize: '0.76rem', color: theme.colors.text.secondary, lineHeight: 1.55 }}>
                                    <div>{card.summary}</div>
                                    <div style={{ color: theme.colors.text.muted }}>
 {t('来源：')}{card.source || 'knowledge'}{card.confidence ? t(' · 置信度 {v0}%', { v0: Math.round(card.confidence * 100) }) : ''}
                                        {card.source_url && (
                                            <a href={card.source_url} target="_blank" rel="noreferrer" style={{ marginLeft: '0.5rem', color: theme.colors.primary.accent }}>{t('查看来源')}</a>
                                        )}
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '0.6rem', padding: '0.75rem', border: `1px solid ${theme.colors.border.default}`, borderRadius: theme.borderRadius.sm, background: 'rgba(255,255,255,0.025)' }}>
                        {renderTagInput(t('流派，最多 5 个'), 'genres', 'Indie, Folk')}
                        {renderTagInput(t('情绪，最多 5 个'), 'moods', 'Peaceful, Dreamy')}
                        {renderTagInput(t('主题，最多 5 个'), 'themes', 'Healing, Rainy')}
                        {renderTagInput(t('场景，最多 5 个'), 'scenarios', 'Late Night, Study')}
                        {renderTagInput(t('语言'), 'language', 'Chinese')}
                        <div style={{ display: 'flex', alignItems: 'end' }}>
                            <button
                                onClick={saveSelectedTags}
                                disabled={savingTags}
                                style={{ width: '100%', background: savingTags ? 'rgba(255,255,255,0.08)' : theme.colors.primary.accent, border: 'none', borderRadius: theme.borderRadius.sm, color: savingTags ? theme.colors.text.muted : '#000', cursor: savingTags ? 'wait' : 'pointer', padding: '0.5rem 0.75rem', fontWeight: 700, fontSize: '0.78rem' }}
                            >
                                {savingTags ? t('保存中...') : t('保存标签')}
                            </button>
                        </div>
                    </div>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem' }}>
                        {[...(selectedSong.genres || []), ...(selectedSong.moods || []), ...(selectedSong.themes || []), ...(selectedSong.scenarios || [])].slice(0, 20).map(tag => (
                            <span key={tag} style={{ fontSize: '0.72rem', padding: '0.18rem 0.5rem', borderRadius: '9999px', background: 'rgba(255,255,255,0.06)', color: theme.colors.text.secondary }}>{tag}</span>
                        ))}
                    </div>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '0.35rem', fontSize: '0.76rem', color: theme.colors.text.muted }}>
                        <div>{t('音频：')}{selectedSong.audio_url || t('无')}</div>
                        <div>{t('歌词：')}{selectedSong.lrc_url || t('无')}</div>
                        <div>ID：{selectedSong.music_id || t('无')}</div>
                    </div>
                </div>
            )}
        </div>
    );
}
