'use client';

import { useState } from 'react';
import { theme } from '@/styles/theme';
import MusicShareCard from './MusicShareCard';
import { generateMusicCard, MusicCardResponse } from '@/lib/api';
import { JourneySegment } from '@/lib/api';
import { usePlayer, Song } from '@/context/PlayerContext';

export type JourneySegmentStatus = 'pending' | 'active' | 'complete';

export interface JourneySegmentState extends JourneySegment {
  status: JourneySegmentStatus;
  songs: any[];
}

interface JourneySegmentsProps {
  segments: JourneySegmentState[];
  activeSegmentId: number | null;
}

const MOOD_COLORS: Record<string, string> = {
  '放松': '#10b981',
  '专注': '#3b82f6',
  '活力': '#f59e0b',
  '平静': '#a78bfa',
  '浪漫': '#ec4899',
  '疗愈': '#14b8a6',
  '开心': '#f97316',
  '悲伤': '#6366f1',
};

const MOOD_EMOJIS: Record<string, string> = {
  '放松': '🌿',
  '专注': '🎯',
  '活力': '⚡',
  '平静': '🌙',
  '浪漫': '💕',
  '疗愈': '🌸',
  '开心': '🎉',
  '悲伤': '🌧️',
};

const moodHints: Record<string, string> = {
  放松: '适合慢慢放空，像黄昏散步。',
  专注: '减少干扰，保持节奏进入心流。',
  活力: '节奏更明显，适合通勤或夜跑。',
  平静: '旋律更柔和，适合睡前或独处。',
  浪漫: '适合约会、夜景和小惊喜。',
  疗愈: '轻声安慰，帮你慢慢恢复能量。',
  开心: '明亮旋律，加一点小小的庆祝。',
  悲伤: '陪你把故事听完，再慢慢走出来。',
};

function formatTimeRange(startMinutes?: number, durationMinutes?: number): string {
  if (startMinutes == null || durationMinutes == null) return '';
  const start = Math.max(0, startMinutes);
  const end = Math.max(start, start + durationMinutes);
  const toLabel = (m: number) => {
    if (m >= 60) {
      const h = Math.floor(m / 60);
      const mm = Math.round(m % 60).toString().padStart(2, '0');
      return `${h}:${mm}`;
    }
    return `${Math.round(m)}'`;
  };
  return `${toLabel(start)} → ${toLabel(end)}`;
}

function toPlayerSong(song: any, coverUrl?: string): Song {
  return {
    title: song.title || song.name || '未命名歌曲',
    artist:
      song.artist ||
      song.artist_name ||
      (Array.isArray(song.artists) ? song.artists.join(', ') : '未知艺术家'),
    genre: song.genre,
    preview_url: song.preview_url || song.audio_url,
    coverUrl: song.cover_url || song.coverUrl || coverUrl,
    lrc_url: song.lrc_url,
  };
}

export default function JourneySegments({ segments, activeSegmentId }: JourneySegmentsProps) {
  const { currentSong, isPlaying, playSong, togglePlay: globalToggle, addAllToQueue } = usePlayer();
  const [shareCardSong, setShareCardSong] = useState<{
    title: string;
    artist: string;
    mood?: string;
    segmentIndex?: number;
  } | null>(null);
  const [shareCardData, setShareCardData] = useState<MusicCardResponse | null>(null);
  const [shareCardLoading, setShareCardLoading] = useState(false);
  const [shareCardError, setShareCardError] = useState<string | null>(null);

  if (!segments.length) {
    return (
      <div
        style={{
          padding: '3rem 2rem',
          borderRadius: theme.borderRadius.lg,
          border: `1px dashed ${theme.colors.border.default}`,
          textAlign: 'center',
          color: theme.colors.text.muted,
          fontSize: '0.9rem',
        }}
      >
        <div style={{ fontSize: '2.5rem', marginBottom: '0.75rem', opacity: 0.4 }}>🎵</div>
        在左侧输入故事或设置情绪曲线，点击生成开始你的音乐旅程
      </div>
    );
  }

  // Collect all songs from all segments for "Play All"
  const allSongs = segments.flatMap((seg) =>
    (seg.songs || []).map((s) => toPlayerSong(s))
  );
  const hasAnySongs = allSongs.some((s) => s.preview_url);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
      {/* Play All button */}
      {hasAnySongs && (
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            padding: '0.75rem 1rem',
            borderRadius: theme.borderRadius.lg,
            background: 'linear-gradient(135deg, rgba(29,185,84,0.12) 0%, rgba(29,185,84,0.04) 100%)',
            border: '1px solid rgba(29,185,84,0.2)',
          }}
        >
          <span style={{ fontSize: '0.85rem', color: theme.colors.text.secondary }}>
            ✨ 共 {allSongs.length} 首歌曲已生成
          </span>
          <button
            type="button"
            onClick={() => {
              const playable = allSongs.filter((s) => s.preview_url);
              if (playable.length > 0) {
                playSong(playable[0], playable);
              }
            }}
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '0.4rem',
              padding: '0.5rem 1.1rem',
              borderRadius: theme.borderRadius.full,
              border: 'none',
              background: 'linear-gradient(135deg, #1db954, #179342)',
              color: '#fff',
              cursor: 'pointer',
              fontSize: '0.85rem',
              fontWeight: 700,
              boxShadow: '0 4px 12px rgba(29,185,84,0.3)',
              transition: 'all 0.2s',
            }}
            onMouseEnter={(e) => (e.currentTarget.style.transform = 'scale(1.03)')}
            onMouseLeave={(e) => (e.currentTarget.style.transform = 'scale(1)')}
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="#fff" stroke="none">
              <polygon points="5 3 19 12 5 21 5 3" />
            </svg>
            播放全部
          </button>
        </div>
      )}

      {/* Segment cards */}
      {segments.map((segment) => {
        const moodColor = MOOD_COLORS[segment.mood] || '#1db954';
        const moodEmoji = MOOD_EMOJIS[segment.mood] || '🎵';

        return (
          <div
            key={segment.segment_id}
            id={`journey-segment-${segment.segment_id}`}
            style={{
              borderRadius: theme.borderRadius.lg,
              border: `1px solid ${segment.segment_id === activeSegmentId ? moodColor + '40' : theme.colors.border.default}`,
              backgroundColor: theme.colors.background.card,
              overflow: 'hidden',
              transition: 'border-color 0.3s, box-shadow 0.3s',
              boxShadow:
                segment.segment_id === activeSegmentId
                  ? `0 0 20px ${moodColor}15`
                  : 'none',
            }}
          >
            {/* Segment header */}
            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                padding: '1rem 1.25rem',
                borderBottom: `1px solid ${theme.colors.border.default}`,
                background: `linear-gradient(120deg, ${moodColor}10, transparent)`,
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem' }}>
                <span
                  style={{
                    fontSize: '1.5rem',
                    width: '2.5rem',
                    height: '2.5rem',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    borderRadius: theme.borderRadius.md,
                    backgroundColor: moodColor + '15',
                  }}
                >
                  {moodEmoji}
                </span>
                <div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <span style={{ fontSize: '0.75rem', color: theme.colors.text.muted }}>
                      第 {segment.segment_id + 1} 章
                    </span>
                    <span
                      style={{
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: '0.25rem',
                        fontSize: '0.7rem',
                        padding: '0.1rem 0.5rem',
                        borderRadius: theme.borderRadius.full,
                        backgroundColor:
                          segment.status === 'complete'
                            ? 'rgba(16,185,129,0.12)'
                            : segment.status === 'active'
                            ? 'rgba(59,130,246,0.12)'
                            : 'rgba(255,255,255,0.05)',
                        color:
                          segment.status === 'complete'
                            ? '#10b981'
                            : segment.status === 'active'
                            ? '#3b82f6'
                            : theme.colors.text.muted,
                      }}
                    >
                      <span
                        style={{
                          width: 6,
                          height: 6,
                          borderRadius: '50%',
                          backgroundColor: 'currentColor',
                          display: 'inline-block',
                        }}
                      />
                      {segment.status === 'complete' ? '已完成' : segment.status === 'active' ? '生成中' : '等待'}
                    </span>
                  </div>
                  <h4 style={{ margin: '0.15rem 0 0', fontSize: '1.15rem', color: theme.colors.text.primary }}>
                    {segment.mood}
                  </h4>
                  <p style={{ margin: '0.1rem 0 0', fontSize: '0.78rem', color: theme.colors.text.muted }}>
                    {moodHints[segment.mood] || '这一段会围绕这种情绪慢慢展开。'}
                  </p>
                </div>
              </div>
              <div style={{ textAlign: 'right', flexShrink: 0 }}>
                <p style={{ margin: 0, color: theme.colors.text.secondary, fontSize: '0.82rem' }}>
                  {segment.duration?.toFixed(0)} 分钟 · {segment.total_songs ?? segment.songs?.length ?? 0} 首
                </p>
                {segment.start_time != null && segment.duration != null && (
                  <p style={{ margin: '0.15rem 0 0', fontSize: '0.72rem', color: theme.colors.text.muted }}>
                    {formatTimeRange(segment.start_time, segment.duration)}
                  </p>
                )}
                {/* Play chapter button */}
                {segment.songs?.length > 0 && (
                  <button
                    type="button"
                    onClick={() => {
                      const queue = segment.songs!.map((s) => toPlayerSong(s));
                      if (queue.length > 0 && queue[0].preview_url) {
                        playSong(queue[0], queue);
                      }
                    }}
                    style={{
                      marginTop: '0.4rem',
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: '0.3rem',
                      padding: '0.3rem 0.7rem',
                      borderRadius: theme.borderRadius.full,
                      border: `1px solid ${moodColor}35`,
                      backgroundColor: `${moodColor}10`,
                      color: moodColor,
                      cursor: 'pointer',
                      fontSize: '0.75rem',
                      fontWeight: 600,
                      transition: 'all 0.15s',
                    }}
                    onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = `${moodColor}20`)}
                    onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = `${moodColor}10`)}
                  >
                    <svg width="10" height="10" viewBox="0 0 24 24" fill={moodColor} stroke="none">
                      <polygon points="5 3 19 12 5 21 5 3" />
                    </svg>
                    播放本章
                  </button>
                )}
              </div>
            </div>

            {/* Description + songs */}
            <div style={{ padding: '1rem 1.25rem' }}>
              {segment.description && (
                <p style={{ margin: '0 0 0.75rem', color: theme.colors.text.secondary, fontSize: '0.88rem' }}>
                  {segment.description}
                </p>
              )}

              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                {segment.songs?.length ? (
                  segment.songs.map((song, index) => {
                    const ps = toPlayerSong(song);
                    const isThisActive = currentSong?.title === ps.title && currentSong?.artist === ps.artist;
                    const isThisPlaying = isThisActive && isPlaying;
                    const hasAudio = !!ps.preview_url;
                    const coverUrl = ps.coverUrl;

                    return (
                      <div
                        key={`${segment.segment_id}-${index}-${song.title}`}
                        style={{
                          display: 'flex',
                          alignItems: 'center',
                          gap: '0.65rem',
                          padding: '0.6rem 0.7rem',
                          borderRadius: theme.borderRadius.md,
                          border: `1px solid ${isThisActive ? moodColor + '40' : theme.colors.border.default}`,
                          backgroundColor: isThisActive ? `${moodColor}08` : theme.colors.background.main,
                          transition: 'all 0.18s',
                          cursor: hasAudio ? 'pointer' : 'default',
                        }}
                        onClick={() => {
                          if (!hasAudio) return;
                          if (isThisActive) globalToggle();
                          else playSong(ps);
                        }}
                      >
                        {/* Cover thumbnail */}
                        <div
                          style={{
                            width: 44,
                            height: 44,
                            borderRadius: theme.borderRadius.sm,
                            flexShrink: 0,
                            overflow: 'hidden',
                            backgroundColor: moodColor + '15',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            position: 'relative',
                          }}
                        >
                          {coverUrl ? (
                            <img
                              src={coverUrl}
                              alt=""
                              style={{
                                width: '100%',
                                height: '100%',
                                objectFit: 'cover',
                              }}
                            />
                          ) : (
                            <span style={{ fontSize: '1.2rem', opacity: 0.5 }}>{moodEmoji}</span>
                          )}
                          {/* Play overlay */}
                          {hasAudio && (
                            <div
                              style={{
                                position: 'absolute',
                                inset: 0,
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                backgroundColor: 'rgba(0,0,0,0.4)',
                                opacity: isThisActive ? 1 : 0,
                                transition: 'opacity 0.15s',
                              }}
                            >
                              {isThisPlaying ? (
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="#fff" stroke="none">
                                  <rect x="6" y="4" width="4" height="16" />
                                  <rect x="14" y="4" width="4" height="16" />
                                </svg>
                              ) : (
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="#fff" stroke="none">
                                  <polygon points="5 3 19 12 5 21 5 3" />
                                </svg>
                              )}
                            </div>
                          )}
                        </div>

                        {/* Song info */}
                        <div style={{ flex: 1, minWidth: 0 }}>
                          <p
                            style={{
                              margin: 0,
                              fontWeight: 600,
                              fontSize: '0.88rem',
                              color: isThisActive ? moodColor : theme.colors.text.primary,
                              overflow: 'hidden',
                              textOverflow: 'ellipsis',
                              whiteSpace: 'nowrap',
                            }}
                          >
                            {index + 1}. {ps.title}
                          </p>
                          <p
                            style={{
                              margin: '0.15rem 0 0',
                              color: theme.colors.text.muted,
                              fontSize: '0.78rem',
                              overflow: 'hidden',
                              textOverflow: 'ellipsis',
                              whiteSpace: 'nowrap',
                            }}
                          >
                            {ps.artist}
                          </p>
                        </div>

                        {/* AI card button */}
                        <button
                          type="button"
                          onClick={async (e) => {
                            e.stopPropagation();
                            const base = {
                              title: ps.title,
                              artist: ps.artist,
                              mood: segment.mood,
                              segmentIndex: segment.segment_id,
                            };
                            setShareCardSong(base);
                            setShareCardLoading(true);
                            setShareCardError(null);
                            setShareCardData(null);
                            try {
                              const data = await generateMusicCard({
                                title: base.title,
                                artist: base.artist,
                                mood: base.mood,
                                segmentLabel: `第 ${segment.segment_id + 1} 章 · ${segment.mood}`,
                              });
                              setShareCardData(data);
                            } catch (err: any) {
                              setShareCardError(err?.message || '生成卡片文案失败');
                            } finally {
                              setShareCardLoading(false);
                            }
                          }}
                          style={{
                            padding: '0.3rem 0.6rem',
                            borderRadius: theme.borderRadius.full,
                            border: `1px solid ${theme.colors.border.default}`,
                            backgroundColor: 'transparent',
                            cursor: 'pointer',
                            fontSize: '0.72rem',
                            color: theme.colors.text.muted,
                            flexShrink: 0,
                            transition: 'all 0.15s',
                          }}
                          onMouseEnter={(e) => {
                            e.currentTarget.style.borderColor = moodColor;
                            e.currentTarget.style.color = moodColor;
                          }}
                          onMouseLeave={(e) => {
                            e.currentTarget.style.borderColor = theme.colors.border.default;
                            e.currentTarget.style.color = theme.colors.text.muted;
                          }}
                        >
                          AI 卡片
                        </button>
                      </div>
                    );
                  })
                ) : (
                  <div
                    style={{
                      padding: '0.85rem',
                      borderRadius: theme.borderRadius.md,
                      border: `1px dashed ${theme.colors.border.default}`,
                      color: theme.colors.text.muted,
                      fontSize: '0.85rem',
                      textAlign: 'center',
                    }}
                  >
                    {segment.status === 'active' ? '⏳ 正在检索歌曲...' : '等待生成...'}
                  </div>
                )}
              </div>
            </div>
          </div>
        );
      })}

      {/* Share card modal */}
      {shareCardSong && (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            backgroundColor: 'rgba(0,0,0,0.6)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 40,
            padding: '1rem',
            backdropFilter: 'blur(4px)',
          }}
          onClick={() => setShareCardSong(null)}
        >
          <div
            style={{ position: 'relative' }}
            onClick={(e) => e.stopPropagation()}
          >
            {shareCardLoading && (
              <div
                style={{
                  position: 'absolute',
                  inset: -16,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  backgroundColor: 'rgba(15,23,42,0.6)',
                  zIndex: 10,
                  color: '#e5e7eb',
                  fontSize: '0.9rem',
                  borderRadius: '32px',
                }}
              >
                正在生成卡片文案...
              </div>
            )}
            <MusicShareCard
              title={shareCardSong.title}
              artist={shareCardSong.artist}
              mood={shareCardSong.mood}
              segmentLabel={`第 ${
                typeof shareCardSong.segmentIndex === 'number'
                  ? shareCardSong.segmentIndex + 1
                  : '?'
              } 章`}
              headline={shareCardData?.headline}
              subline={shareCardData?.subline}
              hashtags={shareCardData?.hashtags}
            />
            {shareCardError && (
              <div
                style={{
                  marginTop: '0.75rem',
                  backgroundColor: 'rgba(248,113,113,0.15)',
                  borderRadius: theme.borderRadius.md,
                  padding: '0.5rem 0.9rem',
                  color: '#fecaca',
                  fontSize: '0.82rem',
                }}
              >
                {shareCardError}
              </div>
            )}
            <button
              type="button"
              onClick={() => setShareCardSong(null)}
              style={{
                position: 'absolute',
                top: 8,
                right: 8,
                padding: '0.25rem 0.6rem',
                borderRadius: theme.borderRadius.full,
                border: 'none',
                backgroundColor: 'rgba(15,23,42,0.75)',
                color: '#e5e7eb',
                fontSize: '0.78rem',
                cursor: 'pointer',
              }}
            >
              关闭
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
