'use client';

import { theme } from '@/styles/theme';

interface JourneySummaryProps {
  loading: boolean;
  thinkingMessage: string;
  journeyTitle?: string;
  meta?: {
    total_segments?: number;
    total_duration?: number;
    total_songs?: number;
  } | null;
  error?: string | null;
}

const summaryItems = [
  { key: 'total_segments', label: '旅程片段', icon: '📊' },
  { key: 'total_duration', label: '总时长 (min)', icon: '⏱️' },
  { key: 'total_songs', label: '歌曲数量', icon: '🎵' },
] as const;

export default function JourneySummary({
  loading,
  thinkingMessage,
  journeyTitle,
  meta,
  error,
}: JourneySummaryProps) {
  return (
    <section
      style={{
        padding: '1.15rem 1.25rem',
        borderRadius: theme.borderRadius.lg,
        background: 'linear-gradient(135deg, rgba(29,185,84,0.08) 0%, rgba(17,24,39,1) 40%, rgba(31,41,55,1) 100%)',
        border: `1px solid ${theme.colors.border.default}`,
        boxShadow: theme.shadows.sm,
      }}
    >
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: '0.75rem',
          alignItems: 'center',
        }}
      >
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <h3 style={{ margin: 0, fontSize: '1.1rem', color: theme.colors.text.primary }}>
              {error ? '❌ 生成失败' : loading ? '🎵 生成中' : meta ? '✅ 旅程就绪' : '🎵 准备就绪'}
            </h3>
            {loading && (
              <span
                style={{
                  backgroundColor: 'rgba(59,130,246,0.15)',
                  padding: '0.2rem 0.6rem',
                  borderRadius: theme.borderRadius.full,
                  fontSize: '0.72rem',
                  color: '#60a5fa',
                  fontWeight: 600,
                }}
              >
                流式生成中
              </span>
            )}
          </div>
          <p
            style={{
              marginTop: '0.25rem',
              marginBottom: 0,
              color: theme.colors.text.muted,
              fontSize: '0.82rem',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
            }}
          >
            {error
              ? error
              : loading
              ? thinkingMessage || '正在生成音乐旅程...'
              : meta
              ? `旅程「${journeyTitle || '音乐旅程'}」已生成完毕`
              : '填写故事或设置情绪曲线开始创作'}
          </p>
        </div>
      </div>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(3, 1fr)',
          gap: '0.65rem',
          marginTop: '0.85rem',
        }}
      >
        {summaryItems.map((item) => (
          <div
            key={item.key}
            style={{
              padding: '0.75rem',
              borderRadius: theme.borderRadius.md,
              backgroundColor: 'rgba(255, 255, 255, 0.04)',
              border: '1px solid rgba(255,255,255,0.06)',
            }}
          >
            <p style={{ margin: 0, fontSize: '0.72rem', color: theme.colors.text.muted }}>
              {item.icon} {item.label}
            </p>
            <strong style={{ fontSize: '1.5rem', display: 'block', marginTop: '0.2rem', color: theme.colors.text.primary }}>
              {meta?.[item.key] ?? '--'}
            </strong>
          </div>
        ))}
      </div>
    </section>
  );
}
