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
import {
    fetchPendingSongs,
    ingestPendingSongs,
    deletePendingSong,
    fetchIngestJobs,
    retryIngestJob,
    PendingSong,
    IngestJob,
} from '@/lib/api';

export default function PendingPage() {
    const [songs, setSongs] = useState<PendingSong[]>([]);
    const [selected, setSelected] = useState<Set<string>>(new Set());
    const [loading, setLoading] = useState(true);
    const [ingesting, setIngesting] = useState(false);
    const [jobs, setJobs] = useState<IngestJob[]>([]);
    const [jobCounts, setJobCounts] = useState<Record<string, number>>({});
    const [retryingJob, setRetryingJob] = useState<string | null>(null);
    const [searchQuery, setSearchQuery] = useState('');
    const [formatFilter, setFormatFilter] = useState('all');
    const { playSong } = usePlayer();
    const { showToast } = useLibrary();
    const router = useRouter();

    const loadSongs = useCallback(async () => {
        setLoading(true);
        const data = await fetchPendingSongs();
        setSongs(data);
        setLoading(false);
    }, []);

    const loadJobs = useCallback(async () => {
        const data = await fetchIngestJobs(12);
        setJobs(data.jobs);
        setJobCounts(data.counts);
    }, []);

    useEffect(() => { loadSongs(); loadJobs(); }, [loadSongs, loadJobs]);

    const toggleSelect = (song: PendingSong) => {
        if (song.valid === false) return;
        setSelected(prev => {
            const next = new Set(prev);
            if (next.has(song.file_basename)) next.delete(song.file_basename);
            else next.add(song.file_basename);
            return next;
        });
    };

    const formats = Array.from(new Set(songs.map(s => s.format).filter(Boolean))).sort();
    const filteredSongs = songs.filter(song => {
        const q = searchQuery.trim().toLowerCase();
        const matchesQuery = !q ||
            (song.title || '').toLowerCase().includes(q) ||
            (song.artist || '').toLowerCase().includes(q) ||
            (song.album || '').toLowerCase().includes(q);
        const matchesFormat = formatFilter === 'all' || song.format === formatFilter;
        return matchesQuery && matchesFormat;
    });
    const validFilteredSongs = filteredSongs.filter(song => song.valid !== false);
    const allFilteredSelected = validFilteredSongs.length > 0 && validFilteredSongs.every(s => selected.has(s.file_basename));
    const invalidJobCount = jobs.filter(job => job.valid === false).length;
    const invalidPendingCount = songs.filter(song => song.valid === false).length;

    const toggleSelectAll = () => {
        if (allFilteredSelected) {
            setSelected(new Set());
        } else {
            setSelected(new Set(validFilteredSongs.map(s => s.file_basename)));
        }
    };

    const handleIngest = async () => {
        const toIngest = songs.filter(s => selected.has(s.file_basename) && s.valid !== false);
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
            release_year: s.release_year,
            source_platform: s.source_platform,
            source_id: s.source_id,
            metadata_source: s.metadata_source,
        })));
        setIngesting(false);
        if (result.success) {
            const jobText = result.job_id ? `，后台任务 ${result.job_id}` : '';
            showToast(`✅ 已写入 ${result.ingested} 首歌曲${jobText}`);
            setSelected(new Set());
            loadSongs();
            loadJobs();
        } else {
            showToast('❌ 入库失败，请重试');
        }
    };

    const handleRetryJob = async (job: IngestJob) => {
        setRetryingJob(job.job_id);
        const result = await retryIngestJob(job.job_id);
        setRetryingJob(null);
        if (result.success) {
            showToast('✅ 已重新加入入库队列');
            loadJobs();
        } else {
            showToast('❌ 重试失败，请查看后端日志');
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
                        {loading ? '加载中...' : `共 ${songs.length} 首待确认，当前显示 ${filteredSongs.length} 首，可入库 ${songs.length - invalidPendingCount} 首，已选 ${selected.size} 首`}
                    </p>
                </div>
            </div>

            {!loading && songs.length > 0 && (
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.75rem', alignItems: 'center' }}>
                    <div style={{ position: 'relative', minWidth: '260px', flex: '1 1 280px', maxWidth: '420px' }}>
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke={theme.colors.text.muted} strokeWidth="2" style={{ position: 'absolute', left: '0.85rem', top: '50%', transform: 'translateY(-50%)' }}>
                            <circle cx="11" cy="11" r="8" /><line x1="21" y1="21" x2="16.65" y2="16.65" />
                        </svg>
                        <input
                            type="text"
                            placeholder="搜索歌名、歌手、专辑"
                            value={searchQuery}
                            onChange={e => setSearchQuery(e.target.value)}
                            style={{ width: '100%', padding: '0.65rem 0.85rem 0.65rem 2.5rem', background: 'rgba(255,255,255,0.05)', border: `1px solid ${theme.colors.border.default}`, borderRadius: theme.borderRadius.sm, color: theme.colors.text.primary, fontSize: '0.88rem', outline: 'none' }}
                        />
                    </div>
                    <select
                        value={formatFilter}
                        onChange={e => setFormatFilter(e.target.value)}
                        style={{ padding: '0.65rem 0.85rem', background: 'rgba(255,255,255,0.05)', border: `1px solid ${theme.colors.border.default}`, borderRadius: theme.borderRadius.sm, color: theme.colors.text.primary, fontSize: '0.88rem', outline: 'none' }}
                    >
                        <option value="all">全部格式</option>
                        {formats.map(format => <option key={format} value={format}>{format.toUpperCase()}</option>)}
                    </select>
                    <span style={{ fontSize: '0.78rem', color: theme.colors.text.muted }}>
                        状态：获取音源 → 元数据入库 → 标签/向量分析
                    </span>
                    {invalidPendingCount > 0 && (
                        <span style={{ fontSize: '0.78rem', color: '#fca5a5' }}>
                            {invalidPendingCount} 首缺音频，需重新获取后才能入库
                        </span>
                    )}
                </div>
            )}

            {jobs.length > 0 && (
                <div style={{ display: 'grid', gap: '0.65rem', padding: '0.85rem 1rem', borderRadius: theme.borderRadius.md, background: 'rgba(255,255,255,0.025)', border: `1px solid ${theme.colors.border.default}` }}>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem', alignItems: 'center' }}>
                        <span style={{ fontSize: '0.82rem', fontWeight: 700 }}>入库增强队列</span>
                        {['pending', 'processing', 'failed', 'done'].map(status => (
                            <span key={status} style={{ fontSize: '0.72rem', padding: '0.15rem 0.5rem', borderRadius: '999px', background: 'rgba(255,255,255,0.05)', color: theme.colors.text.secondary }}>
                                {status}: {jobCounts[status] || 0}
                            </span>
                        ))}
                        {invalidJobCount > 0 && (
                            <span style={{ fontSize: '0.72rem', padding: '0.15rem 0.5rem', borderRadius: '999px', background: 'rgba(248,113,113,0.10)', color: '#fca5a5', border: '1px solid rgba(248,113,113,0.25)' }}>
                                无效: {invalidJobCount}
                            </span>
                        )}
                        <button onClick={loadJobs} style={{ marginLeft: 'auto', background: 'rgba(255,255,255,0.05)', border: `1px solid ${theme.colors.border.default}`, borderRadius: theme.borderRadius.sm, color: theme.colors.text.secondary, cursor: 'pointer', padding: '0.28rem 0.65rem', fontSize: '0.74rem' }}>
                            刷新
                        </button>
                    </div>
                    <div style={{ display: 'grid', gap: '0.35rem' }}>
                        {jobs.slice(0, 5).map(job => {
                            const isRetrying = retryingJob === job.job_id;
                            const isInvalid = job.valid === false;
                            const statusColor = isInvalid || job.status === 'failed' ? '#f87171' : job.status === 'done' ? theme.colors.primary.accent : job.status === 'processing' ? '#60a5fa' : '#f59e0b';
                            const detail = job.validation_error || job.error || '';
                            return (
                                <div key={job.job_id} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.76rem', color: theme.colors.text.secondary }}>
                                    <span style={{ color: statusColor, width: '5.8rem' }}>{isInvalid ? 'invalid' : job.status}</span>
                                    <span title={detail} style={{ flex: 1, minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{job.job_id} · {job.song_count} 首{detail ? ` · ${detail}` : ''}</span>
                                    {job.status === 'failed' && !isInvalid && (
                                        <button disabled={isRetrying} onClick={() => handleRetryJob(job)} style={{ background: 'rgba(248,113,113,0.1)', border: '1px solid rgba(248,113,113,0.35)', borderRadius: theme.borderRadius.sm, color: '#fca5a5', cursor: isRetrying ? 'wait' : 'pointer', padding: '0.22rem 0.55rem', fontSize: '0.72rem' }}>
                                            {isRetrying ? '重试中' : '重试'}
                                        </button>
                                    )}
                                </div>
                            );
                        })}
                    </div>
                </div>
            )}

            {/* Song List */}
            {!loading && filteredSongs.length === 0 ? (
                <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '1rem', padding: '4rem', borderRadius: theme.borderRadius.lg, backgroundColor: 'rgba(255,255,255,0.02)', border: `1px dashed ${theme.colors.border.default}`, textAlign: 'center' }}>
                    <div style={{ width: '64px', height: '64px', borderRadius: '50%', backgroundColor: 'rgba(255,255,255,0.05)', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '0.5rem' }}>
                        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke={theme.colors.text.muted} strokeWidth="2">
                            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><polyline points="7 10 12 15 17 10" /><line x1="12" y1="15" x2="12" y2="3" />
                        </svg>
                    </div>
                    <h3 style={{ margin: 0, fontSize: '1.2rem', fontWeight: 600 }}>{songs.length === 0 ? '暂无待入库歌曲' : '没有匹配的待入库歌曲'}</h3>
                    <p style={{ margin: 0, fontSize: '0.9rem', color: theme.colors.text.muted, maxWidth: '24rem' }}>
                        {songs.length === 0 ? '通过 AI 对话获取新歌后，歌曲会先下载到这里等待你确认入库。' : '调整搜索关键词或格式筛选后再试。'}
                    </p>
                </div>
            ) : (
                <>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
                        {filteredSongs.map((song) => {
                            const isSelected = selected.has(song.file_basename);
                            const isInvalid = song.valid === false;
                            return (
                                <div key={song.file_basename}
                                    style={{
                                        display: 'flex', alignItems: 'center', gap: '0.75rem',
                                        padding: '0.75rem 1rem', borderRadius: theme.borderRadius.md,
                                        backgroundColor: isSelected ? 'rgba(245,158,11,0.08)' : 'rgba(255,255,255,0.02)',
                                        border: isInvalid
                                            ? '1px solid rgba(248,113,113,0.25)'
                                            : isSelected ? '1px solid rgba(245,158,11,0.3)' : '1px solid transparent',
                                        transition: 'all 0.2s',
                                        cursor: isInvalid ? 'not-allowed' : 'pointer',
                                        opacity: isInvalid ? 0.68 : 1,
                                    }}
                                    onMouseEnter={e => { if (!isSelected) e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.05)'; }}
                                    onMouseLeave={e => { if (!isSelected) e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.02)'; }}
                                    onClick={() => toggleSelect(song)}
                                >
                                    {/* Checkbox */}
                                    <div style={{
                                        width: '20px', height: '20px', borderRadius: '4px', flexShrink: 0,
                                        border: isInvalid
                                            ? '2px solid rgba(248,113,113,0.45)'
                                            : isSelected ? '2px solid #f59e0b' : `2px solid ${theme.colors.border.focus}`,
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
                                        {!!song.missing_assets?.length && (
                                            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.25rem', marginTop: '0.25rem' }}>
                                                {song.missing_assets.map(asset => (
                                                    <span key={asset} style={{ fontSize: '0.66rem', padding: '0.08rem 0.35rem', borderRadius: '999px', background: 'rgba(248,113,113,0.10)', color: '#fca5a5', border: '1px solid rgba(248,113,113,0.22)' }}>
                                                        缺 {asset}
                                                    </span>
                                                ))}
                                            </div>
                                        )}
                                        {song.acquire_status === 'failed' && (
                                            <div style={{ marginTop: '0.25rem', fontSize: '0.72rem', color: '#fca5a5' }}>
                                                音源获取失败{song.acquire_error ? `：${song.acquire_error}` : '，可稍后重新保存'}
                                            </div>
                                        )}
                                        {song.audio_retention && song.acquire_status !== 'failed' && (
                                            <div style={{ marginTop: '0.2rem', fontSize: '0.7rem', color: theme.colors.text.muted }}>
                                                {song.audio_retention === 'saved' ? '音源：长期保存' : '音源：临时缓存（点赞、收藏或保存音源后长期保留）'}
                                            </div>
                                        )}
                                        {song.is_trial && (
                                            <div style={{ marginTop: '0.2rem', fontSize: '0.7rem', color: '#fbbf24' }}>
                                                当前来源仅提供试听片段，不标记为完整音源
                                            </div>
                                        )}
                                    </div>

                                    {/* Time */}
                                    <div style={{ fontSize: '0.75rem', color: theme.colors.text.muted, whiteSpace: 'nowrap', paddingRight: '0.5rem' }}>
                                        {song.acquired_at ? new Date(song.acquired_at).toLocaleDateString('zh-CN') : ''}
                                    </div>

                                    <span style={{ fontSize: '0.72rem', padding: '0.16rem 0.5rem', borderRadius: '9999px', color: isInvalid ? '#fca5a5' : ingesting && isSelected ? '#60a5fa' : '#f59e0b', border: `1px solid ${isInvalid ? 'rgba(248,113,113,0.35)' : ingesting && isSelected ? 'rgba(96,165,250,0.35)' : 'rgba(245,158,11,0.35)'}`, whiteSpace: 'nowrap' }}>
                                        {isInvalid ? '缺音频' : ingesting && isSelected ? '分析中' : '待入库'}
                                    </span>

                                    {/* Play */}
                                    <button title={isInvalid ? '缺音频，无法试听' : '试听'} aria-label={isInvalid ? `${song.title} 缺音频` : `试听 ${song.title}`} disabled={isInvalid} onClick={e => { e.stopPropagation(); if (!isInvalid) playSong({ title: song.title, artist: song.artist, preview_url: `http://localhost:8501${song.audio_url}`, coverUrl: `http://localhost:8501${song.cover_url}`, lrc_url: `http://localhost:8501${song.lrc_url}` }); }}
                                        style={{ background: 'none', border: 'none', color: isInvalid ? theme.colors.text.muted : theme.colors.primary.accent, cursor: isInvalid ? 'not-allowed' : 'pointer', padding: '0.4rem', borderRadius: '50%', display: 'flex', transition: 'transform 0.2s', opacity: isInvalid ? 0.4 : 1 }}
                                        onMouseEnter={e => (e.currentTarget.style.transform = 'scale(1.15)')}
                                        onMouseLeave={e => (e.currentTarget.style.transform = 'scale(1)')}
                                    >
                                        <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3" /></svg>
                                    </button>

                                    {/* Delete */}
                                    <button title="删除" aria-label={`删除 ${song.title}`} onClick={e => { e.stopPropagation(); handleDelete(song); }}
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
                                {allFilteredSelected ? '取消全选' : '全选当前'}
                            </button>

                            <span style={{ fontSize: '0.82rem', color: theme.colors.text.muted }}>
                                已选 {selected.size} / 当前可入库 {validFilteredSongs.length}
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
