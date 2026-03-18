'use client';

import { theme } from '@/styles/theme';
import { JourneySegmentState, JourneySegmentStatus } from './JourneySegments';

interface JourneyTimelineProps {
  segments: JourneySegmentState[];
  activeSegmentId: number | null;
  onSegmentClick?: (segmentId: number) => void;
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

export default function JourneyTimeline({ segments, activeSegmentId, onSegmentClick }: JourneyTimelineProps) {
  if (!segments.length) return null;

  const totalDuration = segments.reduce((sum, s) => sum + (s.duration || 1), 0);

  return (
    <div
      style={{
        borderRadius: theme.borderRadius.lg,
        border: `1px solid ${theme.colors.border.default}`,
        backgroundColor: theme.colors.background.card,
        padding: '1rem 1.25rem',
        marginBottom: '1rem',
      }}
    >
      {/* Header */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: '0.75rem',
        }}
      >
        <span style={{ fontSize: '0.82rem', color: theme.colors.text.muted, fontWeight: 600 }}>
          ⏱️ 旅程时间轴
        </span>
        <span style={{ fontSize: '0.78rem', color: theme.colors.text.muted }}>
          {Math.round(totalDuration)} 分钟 · {segments.length} 个阶段
        </span>
      </div>

      {/* Timeline bar */}
      <div
        style={{
          display: 'flex',
          gap: '3px',
          borderRadius: theme.borderRadius.full,
          overflow: 'hidden',
          height: '8px',
          backgroundColor: 'rgba(255,255,255,0.05)',
        }}
      >
        {segments.map((seg) => {
          const pct = ((seg.duration || 1) / totalDuration) * 100;
          const color = MOOD_COLORS[seg.mood] || '#1db954';
          const isActive = seg.segment_id === activeSegmentId;

          return (
            <div
              key={seg.segment_id}
              onClick={() => onSegmentClick?.(seg.segment_id)}
              style={{
                width: `${pct}%`,
                backgroundColor: seg.status === 'pending' ? 'rgba(255,255,255,0.08)' : color,
                borderRadius: '4px',
                cursor: 'pointer',
                transition: 'all 0.3s ease',
                opacity: seg.status === 'pending' ? 0.4 : 1,
                animation: isActive ? 'timelinePulse 1.5s ease-in-out infinite' : 'none',
                boxShadow: isActive ? `0 0 12px ${color}60` : 'none',
              }}
            />
          );
        })}
      </div>

      {/* Labels */}
      <div
        style={{
          display: 'flex',
          gap: '3px',
          marginTop: '0.5rem',
        }}
      >
        {segments.map((seg) => {
          const pct = ((seg.duration || 1) / totalDuration) * 100;
          const color = MOOD_COLORS[seg.mood] || '#1db954';
          const emoji = MOOD_EMOJIS[seg.mood] || '🎵';

          return (
            <div
              key={seg.segment_id}
              style={{
                width: `${pct}%`,
                textAlign: 'center',
                minWidth: 0,
              }}
            >
              <div
                style={{
                  fontSize: '0.72rem',
                  color: seg.status === 'complete' ? color : theme.colors.text.muted,
                  fontWeight: 600,
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap',
                }}
              >
                {emoji} {seg.mood}
              </div>
              <div
                style={{
                  fontSize: '0.65rem',
                  color: theme.colors.text.muted,
                  opacity: 0.7,
                }}
              >
                {seg.songs?.length || 0}首
              </div>
            </div>
          );
        })}
      </div>

      {/* Pulse animation */}
      <style>{`
        @keyframes timelinePulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }
      `}</style>
    </div>
  );
}
