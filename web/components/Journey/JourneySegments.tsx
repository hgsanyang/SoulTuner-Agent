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

const statusColors: Record<JourneySegmentStatus, string> = {
  pending: '#94a3b8',
  active: '#3b82f6',
  complete: '#10b981',
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
      const mm = Math.round(m % 60)
        .toString()
        .padStart(2, '0');
      return `${h}:${mm}`;
    }
    return `${Math.round(m)}'`;
  };

  return `${toLabel(start)} → ${toLabel(end)}`;
}

/** 将后端返回的 song 对象映射为 PlayerContext 所需的 Song */
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
          padding: '2rem',
          borderRadius: theme.borderRadius.lg,
          border: `1px dashed ${theme.colors.border.default}`,
          textAlign: 'center',
          color: theme.colors.text.secondary,
        }}
      >
        暂无旅程片段，生成后会在这里显示每个阶段及对应歌曲。
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      {segments.map((segment) => (
        <div
          key={segment.segment_id}
          style={{
            borderRadius: theme.borderRadius.lg,
            border: `1px solid ${theme.colors.border.default}`,
            backgroundColor: theme.colors.background.card,
            boxShadow:
              segment.segment_id === activeSegmentId
                ? '0 12px 25px rgba(59, 130, 246, 0.15)'
                : '0 8px 18px rgba(15, 23, 42, 0.08)',
            overflow: 'hidden',
          }}
        >
          <div
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              padding: '1.25rem 1.5rem',
              borderBottom: `1px solid ${theme.colors.border.default}`,
              background: 'linear-gradient(120deg, rgba(37,99,235,0.08), rgba(14,165,233,0.05))',
            }}
          >
            <div>
              <p
                style={{
                  margin: 0,
                  fontSize: '0.85rem',
                  color: theme.colors.text.secondary,
                }}
              >
                第 {segment.segment_id + 1} 章 · 情绪片段
              </p>
              <h4
                style={{
                  margin: '0.35rem 0 0',
                  fontSize: '1.35rem',
                  color: theme.colors.text.primary,
                }}
              >
                {segment.mood}
              </h4>
              <p
                style={{
                  margin: '0.25rem 0 0',
                  fontSize: '0.9rem',
                  color: theme.colors.text.secondary,
                }}
              >
                {moodHints[segment.mood] || '这一段会围绕这种情绪慢慢展开。'}
              </p>
            </div>
            <div style={{ textAlign: 'right' }}>
              <span
                style={{
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: '0.35rem',
                  color: statusColors[segment.status],
                  fontWeight: 600,
                }}
              >
                <span
                  style={{
                    width: '0.75rem',
                    height: '0.75rem',
                    borderRadius: '999px',
                    backgroundColor: statusColors[segment.status],
                    display: 'inline-block',
                  }}
                />
                {segment.status === 'pending'
                  ? '等待生成'
                  : segment.status === 'active'
                  ? '生成中'
                  : '已完成'}
              </span>
              <p
                style={{
                  margin: '0.35rem 0 0',
                  color: theme.colors.text.secondary,
                  fontSize: '0.95rem',
                }}
              >
                {segment.duration?.toFixed(1)} 分钟 ·{' '}
                {segment.total_songs ?? segment.songs?.length ?? 0} 首歌曲
                {segment.start_time != null && segment.duration != null && (
                  <span style={{ marginLeft: '0.45rem', fontSize: '0.85rem' }}>
                    ({formatTimeRange(segment.start_time, segment.duration)})
                  </span>
                )}
              </p>
            </div>
          </div>

          <div style={{ padding: '1.25rem 1.5rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
              <p style={{ margin: 0, color: theme.colors.text.primary }}>
                {segment.description}
              </p>
              {/* 播放全部按钮 */}
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
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: '0.35rem',
                    padding: '0.4rem 0.85rem',
                    borderRadius: theme.borderRadius.full,
                    border: '1px solid rgba(29, 185, 84, 0.35)',
                    backgroundColor: 'rgba(29, 185, 84, 0.1)',
                    color: '#1DB954',
                    cursor: 'pointer',
                    fontSize: '0.82rem',
                    fontWeight: 600,
                    flexShrink: 0,
                    transition: 'all 0.15s',
                  }}
                  onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = 'rgba(29,185,84,0.2)')}
                  onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = 'rgba(29,185,84,0.1)')}
                >
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="#1DB954" stroke="none">
                    <polygon points="5 3 19 12 5 21 5 3" />
                  </svg>
                  播放本章
                </button>
              )}
            </div>

            <div
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
                gap: '0.85rem',
              }}
            >
              {segment.songs?.length ? (
                segment.songs.map((song, index) => {
                  const ps = toPlayerSong(song);
                  const isThisActive = currentSong?.title === ps.title && currentSong?.artist === ps.artist;
                  const isThisPlaying = isThisActive && isPlaying;
                  const hasAudio = !!ps.preview_url;

                  return (
                    <div
                      key={`${segment.segment_id}-${index}-${song.title}`}
                      style={{
                        padding: '0.85rem',
                        borderRadius: theme.borderRadius.md,
                        border: `1px solid ${isThisActive ? 'rgba(29,185,84,0.4)' : theme.colors.border.default}`,
                        backgroundColor: isThisActive ? 'rgba(29,185,84,0.06)' : theme.colors.background.main,
                        transition: 'all 0.18s',
                      }}
                    >
                      {/* 歌曲行：播放按钮 + 信息 */}
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem' }}>
                        {/* 播放按钮 */}
                        <div
                          onClick={() => {
                            if (!hasAudio) return;
                            if (isThisActive) globalToggle();
                            else playSong(ps);
                          }}
                          style={{
                            width: 36,
                            height: 36,
                            borderRadius: theme.borderRadius.md,
                            backgroundColor: isThisActive
                              ? 'rgba(29,185,84,0.2)'
                              : hasAudio
                              ? 'rgba(29,185,84,0.1)'
                              : 'rgba(255,255,255,0.05)',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            flexShrink: 0,
                            cursor: hasAudio ? 'pointer' : 'default',
                            opacity: hasAudio ? 1 : 0.4,
                            boxShadow: hasAudio
                              ? 'inset 0 0 0 1px rgba(29,185,84,0.25)'
                              : 'inset 0 0 0 1px rgba(255,255,255,0.1)',
                            transition: 'all 0.15s',
                          }}
                          title={hasAudio ? (isThisPlaying ? '暂停' : '播放') : '暂无试听'}
                        >
                          {isThisPlaying ? (
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="#1DB954" stroke="none">
                              <rect x="6" y="4" width="4" height="16" />
                              <rect x="14" y="4" width="4" height="16" />
                            </svg>
                          ) : (
                            <svg
                              width="16"
                              height="16"
                              viewBox="0 0 24 24"
                              fill={hasAudio ? '#1DB954' : 'rgba(255,255,255,0.35)'}
                              stroke="none"
                              style={{ marginLeft: '2px' }}
                            >
                              <polygon points="5 3 19 12 5 21 5 3" />
                            </svg>
                          )}
                        </div>

                        {/* 歌曲信息 */}
                        <div style={{ flex: 1, minWidth: 0 }}>
                          <p
                            style={{
                              margin: 0,
                              fontWeight: 600,
                              color: isThisActive ? '#1DB954' : theme.colors.text.primary,
                              overflow: 'hidden',
                              textOverflow: 'ellipsis',
                              whiteSpace: 'nowrap',
                            }}
                          >
                            {index + 1}. {ps.title}
                          </p>
                          <p
                            style={{
                              margin: '0.2rem 0 0',
                              color: theme.colors.text.secondary,
                              fontSize: '0.85rem',
                              overflow: 'hidden',
                              textOverflow: 'ellipsis',
                              whiteSpace: 'nowrap',
                            }}
                          >
                            {ps.artist}
                          </p>
                        </div>
                      </div>

                      {/* AI 卡片按钮 */}
                      <button
                        type="button"
                        onClick={async () => {
                          const base = {
                            title: ps.title,
                            artist: ps.artist,
                            mood: segment.mood,
                            segmentIndex: segment.segment_id,
                          } as const;

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
                            setShareCardError(err?.message || '生成卡片文案失败，请稍后重试。');
                          } finally {
                            setShareCardLoading(false);
                          }
                        }}
                        style={{
                          marginTop: '0.5rem',
                          padding: '0.35rem 0.75rem',
                          borderRadius: theme.borderRadius.full,
                          border: `1px solid ${theme.colors.border.default}`,
                          backgroundColor: theme.colors.background.hover,
                          cursor: 'pointer',
                          fontSize: '0.8rem',
                          color: theme.colors.text.secondary,
                        }}
                      >
                        生成 AI 音乐卡片
                      </button>
                    </div>
                  );
                })
              ) : (
                <div
                  style={{
                    padding: '1rem',
                    borderRadius: theme.borderRadius.md,
                    border: `1px dashed ${theme.colors.border.default}`,
                    color: theme.colors.text.secondary,
                    fontSize: '0.95rem',
                  }}
                >
                  正在获取歌曲...
                </div>
              )}
            </div>
          </div>
        </div>
      ))}

      {shareCardSong && (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            backgroundColor: 'rgba(15,23,42,0.55)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 40,
            padding: '1rem',
          }}
          onClick={() => setShareCardSong(null)}
        >
          <div
            style={{ position: 'relative' }}
            onClick={(e) => {
              e.stopPropagation();
            }}
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
                }}
              >
                正在向 WAN 大模型请求卡片文案...
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
              关闭预览
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

