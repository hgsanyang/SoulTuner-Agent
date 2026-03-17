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
  { key: 'total_segments', label: '旅程片段' },
  { key: 'total_duration', label: '总时长 (min)' },
  { key: 'total_songs', label: '歌曲数量' },
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
        marginBottom: '2rem',
        padding: '1.5rem',
        borderRadius: theme.borderRadius.lg,
        background: 'linear-gradient(135deg, #111827, #1f2937)',
        color: '#fff',
        boxShadow: '0 15px 35px rgba(15, 23, 42, 0.3)',
      }}
    >
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: '1rem',
          alignItems: 'center',
        }}
      >
        <div>
          <h3 style={{ margin: 0, fontSize: '1.25rem' }}>生成状态</h3>
          <p style={{ marginTop: '0.35rem', color: 'rgba(255,255,255,0.75)' }}>
            {error
              ? `❌ ${error}`
              : loading
              ? thinkingMessage || '正在生成音乐旅程...'
              : meta
              ? `旅程「${journeyTitle || '你的音乐旅程'}」已完成，下面可以预览每个阶段`
              : '准备就绪，填写故事或情绪曲线开始创作'}
          </p>
          {journeyTitle && !error && (
            <p
              style={{
                marginTop: '0.2rem',
                color: 'rgba(255,255,255,0.65)',
                fontSize: '0.9rem',
              }}
            >
              当前故事：{journeyTitle}
            </p>
          )}
        </div>
        {loading && (
          <span
            style={{
              backgroundColor: 'rgba(255,255,255,0.12)',
              padding: '0.4rem 0.75rem',
              borderRadius: '999px',
              fontSize: '0.9rem',
            }}
          >
            流式生成中...
          </span>
        )}
      </div>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))',
          gap: '1rem',
          marginTop: '1.25rem',
        }}
      >
        {summaryItems.map((item) => (
          <div
            key={item.key}
            style={{
              padding: '1rem',
              borderRadius: theme.borderRadius.md,
              backgroundColor: 'rgba(255, 255, 255, 0.08)',
              border: '1px solid rgba(148, 163, 184, 0.2)',
              minHeight: '96px',
              backdropFilter: 'blur(4px)',
            }}
          >
            <p style={{ margin: 0, fontSize: '0.85rem', color: 'rgba(255,255,255,0.8)' }}>
              {item.label}
            </p>
            <strong style={{ fontSize: '1.75rem', display: 'block', marginTop: '0.35rem', color: '#ffffff' }}>
              {meta?.[item.key] ?? '--'}
            </strong>
          </div>
        ))}
      </div>
    </section>
  );
}

