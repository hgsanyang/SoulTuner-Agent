'use client';

import { theme } from '@/styles/theme';
import { usePlayer } from '@/context/PlayerContext';
import { useLibrary } from '@/context/LibraryContext';
import { sendUserEvent, acquireSong } from '@/lib/api';
import { useState } from 'react';

interface SongCardProps {
  title: string;
  artist: string;
  genre?: string;
  mood?: string;
  reason?: string;
  preview_url?: string;
  cover_url?: string;
  lrc_url?: string;
  song_id?: string;
  platform?: string;
  onRemove?: () => void;  // 从当前结果列表中删除
}

export default function SongCard({ title, artist, genre, mood, reason, preview_url, cover_url, lrc_url, song_id, platform, onRemove }: SongCardProps) {
  const { currentSong, isPlaying, playSong, togglePlay: globalToggle, queue, addToQueue, removeFromQueue } = usePlayer();
  const { isLiked, toggleLike, collections, addToCollection, showToast } = useLibrary();
  const [showFolderPicker, setShowFolderPicker] = useState(false);
  const [isHovered, setIsHovered] = useState(false);
  const [acquireState, setAcquireState] = useState<'idle' | 'loading' | 'done' | 'error'>('idle');

  const handleAcquire = async (e: React.MouseEvent) => {
    e.stopPropagation();
    if (acquireState === 'loading' || acquireState === 'done') return;
    setAcquireState('loading');
    try {
      await acquireSong({ title, artist, song_id, platform });
      setAcquireState('done');
      showToast('✅ 已成功加入本地曲库');
    } catch (err: any) {
      setAcquireState('error');
      showToast(`❌ ${err.message || '加入本地失败'}`);
      setTimeout(() => setAcquireState('idle'), 3000);
    }
  };

  const isThisActive = currentSong?.title === title && currentSong?.artist === artist;
  const isThisPlaying = isThisActive && isPlaying;
  const liked = isLiked(title, artist);
  const inQueue = queue.some(s => s.title === title && s.artist === artist);

  const togglePlay = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (!preview_url) return;
    if (isThisActive) globalToggle();
    else playSong({ title, artist, genre, preview_url, coverUrl: cover_url, lrc_url }, undefined);
  };

  const handleLike = (e: React.MouseEvent) => {
    e.stopPropagation();
    toggleLike({ title, artist, genre, preview_url, coverUrl: cover_url, lrc_url });
  };

  const handleDislike = (e: React.MouseEvent) => {
    e.stopPropagation();
    sendUserEvent('dislike', title, artist);
    showToast('👎 已标记为不喜欢');
    onRemove?.();
  };

  const handleQueueToggle = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (inQueue) {
      removeFromQueue(title, artist);
      showToast('已从播放列表移除');
    } else {
      addToQueue({ title, artist, genre, preview_url, coverUrl: cover_url, lrc_url });
      showToast('✚ 已加入播放列表');
    }
  };

  const handleRemove = (e: React.MouseEvent) => {
    e.stopPropagation();
    onRemove?.();
  };

  return (
    <div
      style={{
        padding: '1rem 1.25rem',
        marginBottom: '0.75rem',
        backgroundColor: isHovered ? 'rgba(255,255,255,0.06)' : theme.colors.background.card,
        borderRadius: theme.borderRadius.md,
        border: `1px solid ${isHovered ? theme.colors.border.focus : theme.colors.border.default}`,
        boxShadow: isHovered ? theme.shadows.md : theme.shadows.sm,
        transition: 'all 0.18s',
        position: 'relative',
      }}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => { setIsHovered(false); setShowFolderPicker(false); }}
    >
      {/* 主行：播放按钮 + 歌曲信息 + 操作按钮 */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
        {/* 播放按钮 */}
        <div
          onClick={togglePlay}
          style={{
            width: '44px', height: '44px', borderRadius: theme.borderRadius.md,
            backgroundColor: isThisActive
              ? 'rgba(29, 185, 84, 0.2)'
              : preview_url ? 'rgba(29, 185, 84, 0.1)' : 'rgba(255, 255, 255, 0.05)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            flexShrink: 0,
            boxShadow: preview_url ? 'inset 0 0 0 1px rgba(29, 185, 84, 0.25)' : 'inset 0 0 0 1px rgba(255,255,255,0.1)',
            cursor: preview_url ? 'pointer' : 'default',
            opacity: preview_url ? 1 : 0.45,
            transition: 'all 0.15s',
          }}
          title={preview_url ? (isThisPlaying ? '暂停' : '播放') : '暂无试听'}
        >
          {isThisPlaying ? (
            // 暂停图标（两个竖条）
            <svg width="18" height="18" viewBox="0 0 24 24" fill={theme.colors.primary.accent} stroke="none">
              <rect x="6" y="4" width="4" height="16" /><rect x="14" y="4" width="4" height="16" />
            </svg>
          ) : (
            // 播放图标（三角形）
            <svg width="19" height="19" viewBox="0 0 24 24" fill={preview_url ? theme.colors.primary.accent : 'rgba(255,255,255,0.35)'} stroke="none" style={{ marginLeft: '2px' }}>
              <polygon points="5 3 19 12 5 21 5 3" />
            </svg>
          )}
        </div>

        {/* 歌曲信息 */}
        <div style={{ flex: 1, minWidth: 0 }}>
          <h3 style={{
            fontSize: '1rem', fontWeight: 700, color: isThisActive ? theme.colors.primary.accent : theme.colors.text.primary,
            margin: '0 0 0.2rem 0', letterSpacing: '-0.01em',
            overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
          }}>
            {title}
          </h3>
          <p style={{ margin: 0, fontSize: '0.87rem', color: theme.colors.text.secondary, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            {artist}
          </p>
        </div>

        {/* ── Hover 操作区 ── */}
        <div style={{
          display: 'flex', alignItems: 'center', gap: '0.25rem', flexShrink: 0,
          opacity: isHovered ? 1 : 0,
          transform: isHovered ? 'translateX(0)' : 'translateX(8px)',
          transition: 'opacity 0.18s, transform 0.18s',
        }}>

          {/* 1. 加入/移出播放列表 */}
          <button
            onClick={handleQueueToggle}
            title={inQueue ? '从播放列表移除' : '加入播放列表'}
            style={actionBtnStyle(inQueue ? '#1DB954' : undefined)}
            onMouseEnter={e => (e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.14)')}
            onMouseLeave={e => (e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.06)')}
          >
            {inQueue ? (
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#1DB954" strokeWidth="2.2" strokeLinecap="round">
                <line x1="8" y1="6" x2="21" y2="6" /><line x1="8" y1="12" x2="21" y2="12" /><line x1="8" y1="18" x2="21" y2="18" />
                <polyline stroke="#1DB954" points="3 9 4.5 10.5 7 8" strokeWidth="2" />
              </svg>
            ) : (
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round">
                <line x1="8" y1="6" x2="21" y2="6" /><line x1="8" y1="12" x2="21" y2="12" /><line x1="8" y1="18" x2="21" y2="18" />
                <line x1="3" y1="12" x2="3" y2="12" /><line x1="3" y1="6" x2="3" y2="6" /><line x1="3" y1="18" x2="3" y2="18" />
                <line x1="1" y1="12" x2="5" y2="12" /><line x1="3" y1="10" x2="3" y2="14" />
              </svg>
            )}
          </button>

          {/* 1.5 加入本地（数据飞轮） */}
          <button
            onClick={handleAcquire}
            title={acquireState === 'done' ? '已加入本地' : acquireState === 'loading' ? '正在下载...' : '加入本地曲库'}
            style={actionBtnStyle(acquireState === 'done' ? '#1DB954' : acquireState === 'loading' ? '#f0a500' : undefined)}
            onMouseEnter={e => acquireState === 'idle' && (e.currentTarget.style.backgroundColor = 'rgba(29,185,84,0.22)')}
            onMouseLeave={e => (e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.06)')}
          >
            {acquireState === 'done' ? (
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#1DB954" strokeWidth="2.5" strokeLinecap="round"><polyline points="20 6 9 17 4 12" /></svg>
            ) : acquireState === 'loading' ? (
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#f0a500" strokeWidth="2" strokeLinecap="round" style={{ animation: 'spin 1s linear infinite' }}>
                <path d="M21 12a9 9 0 1 1-6.22-8.56" />
              </svg>
            ) : (
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="7 10 12 15 17 10" />
                <line x1="12" y1="15" x2="12" y2="3" />
              </svg>
            )}
          </button>

          {/* 2. 喜欢/取消喜欢 */}
          <button
            onClick={handleLike}
            title={liked ? '取消喜欢' : '添加到喜欢'}
            style={actionBtnStyle(liked ? '#e91e63' : undefined)}
            onMouseEnter={e => (e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.14)')}
            onMouseLeave={e => (e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.06)')}
          >
            <svg width="15" height="15" viewBox="0 0 24 24" fill={liked ? '#e91e63' : 'none'} stroke={liked ? '#e91e63' : 'currentColor'} strokeWidth="2">
              <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z" />
            </svg>
          </button>

          {/* 3. 👎 不喜欢 */}
          <button
            onClick={handleDislike}
            title="不喜欢这首歌"
            style={actionBtnStyle()}
            onMouseEnter={e => (e.currentTarget.style.backgroundColor = 'rgba(255,80,80,0.18)')}
            onMouseLeave={e => (e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.06)')}
          >
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="rgba(255,120,120,0.7)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3zm7-13h2.67A2.31 2.31 0 0 1 22 4v7a2.31 2.31 0 0 1-2.33 2H17" />
            </svg>
          </button>

          {/* 4. 收藏到歌单 */}
          <div style={{ position: 'relative' }}>
            <button
              onClick={e => { e.stopPropagation(); setShowFolderPicker(prev => !prev); }}
              title="收藏到歌单"
              style={actionBtnStyle(showFolderPicker ? '#fff' : undefined)}
              onMouseEnter={e => (e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.14)')}
              onMouseLeave={e => (e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.06)')}
            >
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke={showFolderPicker ? '#fff' : 'currentColor'} strokeWidth="2">
                <path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z" />
              </svg>
            </button>

            {/* 歌单选择下拉 */}
            {showFolderPicker && (
              <div style={{
                position: 'absolute', right: 0, bottom: 'calc(100% + 6px)',
                backgroundColor: 'rgba(20,20,20,0.97)',
                border: `1px solid ${theme.colors.border.focus}`,
                borderRadius: theme.borderRadius.md,
                boxShadow: '0 8px 32px rgba(0,0,0,0.7)',
                minWidth: '190px', zIndex: 200,
                overflow: 'hidden', backdropFilter: 'blur(16px)',
              }}
                onClick={e => e.stopPropagation()}
              >
                <div style={{ padding: '0.5rem 0.85rem', fontSize: '0.75rem', color: theme.colors.text.muted, borderBottom: `1px solid rgba(255,255,255,0.07)`, fontWeight: 600, letterSpacing: '0.06em' }}>
                  收藏到歌单
                </div>
                {collections.length === 0 ? (
                  <div style={{ padding: '1rem', fontSize: '0.85rem', color: theme.colors.text.muted, textAlign: 'center' }}>暂无歌单</div>
                ) : (
                  collections.map(col => (
                    <button key={col.id}
                      onClick={e => { e.stopPropagation(); addToCollection(col.id, { title, artist, genre, preview_url }); setShowFolderPicker(false); }}
                      style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', width: '100%', padding: '0.65rem 0.85rem', background: 'none', border: 'none', cursor: 'pointer', color: theme.colors.text.primary, fontSize: '0.88rem', textAlign: 'left', transition: 'background-color 0.12s' }}
                      onMouseEnter={e => (e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.08)')}
                      onMouseLeave={e => (e.currentTarget.style.backgroundColor = 'transparent')}
                    >
                      <div style={{ width: '26px', height: '26px', borderRadius: '4px', backgroundColor: col.coverColor, flexShrink: 0 }} />
                      <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{col.name}</span>
                    </button>
                  ))
                )}
              </div>
            )}
          </div>

          {/* 4. 从结果删除（仅当 onRemove 存在时显示） */}
          {onRemove && (
            <button
              onClick={handleRemove}
              title="从推荐结果中移除"
              style={actionBtnStyle()}
              onMouseEnter={e => (e.currentTarget.style.backgroundColor = 'rgba(255,80,80,0.18)')}
              onMouseLeave={e => (e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.06)')}
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="rgba(255,100,100,0.8)" strokeWidth="2.5" strokeLinecap="round">
                <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
              </svg>
            </button>
          )}
        </div>
      </div>

      {/* 标签行（genre/mood） */}
      {(genre || mood) && (
        <div style={{ display: 'flex', gap: '0.5rem', marginTop: '0.7rem', flexWrap: 'wrap' }}>
          {genre && (
            <span style={{ padding: '0.25rem 0.7rem', fontSize: '0.72rem', backgroundColor: 'rgba(255,255,255,0.06)', color: theme.colors.text.secondary, borderRadius: theme.borderRadius.full, border: `1px solid ${theme.colors.border.default}` }}>
              {genre}
            </span>
          )}
          {mood && (
            <span style={{ padding: '0.25rem 0.7rem', fontSize: '0.72rem', backgroundColor: 'rgba(29, 185, 84, 0.08)', color: theme.colors.primary.accent, borderRadius: theme.borderRadius.full, border: '1px solid rgba(29,185,84,0.22)' }}>
              {mood}
            </span>
          )}
          {/* 播放列表角标 */}
          {inQueue && (
            <span style={{ padding: '0.25rem 0.6rem', fontSize: '0.7rem', backgroundColor: 'rgba(29,185,84,0.12)', color: theme.colors.primary.accent, borderRadius: theme.borderRadius.full, border: '1px solid rgba(29,185,84,0.2)' }}>
              ▶ 播放列表
            </span>
          )}
        </div>
      )}

      {/* 推荐理由 */}
      {reason && (
        <p style={{ margin: '0.6rem 0 0', fontSize: '0.87rem', color: theme.colors.text.muted, lineHeight: '1.6', paddingTop: '0.5rem', borderTop: `1px dashed ${theme.colors.border.default}` }}>
          {reason}
        </p>
      )}
    </div>
  );
}

/** 统一的操作按钮样式 */
function actionBtnStyle(activeColor?: string): React.CSSProperties {
  return {
    background: 'none',
    backgroundColor: 'rgba(255,255,255,0.06)',
    border: 'none',
    cursor: 'pointer',
    color: activeColor || 'rgba(255,255,255,0.65)',
    padding: '0.4rem',
    borderRadius: '6px',
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    transition: 'background-color 0.15s, transform 0.1s',
    width: '32px', height: '32px',
  };
}
