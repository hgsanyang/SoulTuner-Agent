'use client';

import { theme } from '@/styles/theme';

interface WelcomeScreenProps {
  title?: string;
  description?: string;
  badgeLabel?: string;
  subtitle?: string;
  onPrimaryAction?: () => void;
  onSecondaryAction?: () => void;
}

const metricCards = [
  {
    label: '智能理解',
    value: '心情 / 场景 / 流派',
    detail: '基于大模型自动解析你的自然语言需求',
  },
  {
    label: '推荐来源',
    value: 'Spotify + 网络',
    detail: '综合在线乐库与本地示例库进行多路推荐',
  },
  {
    label: '交互体验',
    value: 'SSE 流式输出',
    detail: '推荐说明与歌曲逐步流式返回，实时刷新界面',
  },
];

export default function WelcomeScreen({
  title = '音乐推荐 Agent',
  description = '用一句自然语言描述你的心情、场景或喜欢的歌手，AI 会自动理解你的需求，联动 Spotify / 网络搜索与本地数据，为你生成解释清晰的个性化音乐推荐。',
  badgeLabel = 'SSE 流式推荐 · 实时响应',
  subtitle = '推荐页支持流式 AI 讲解与逐首推荐，搜索页提供按关键词与流派的快速歌曲发现。',
  onPrimaryAction,
  onSecondaryAction,
}: WelcomeScreenProps) {
  return (
    <section
      style={{
        width: '100%',
        padding: '3rem 1.5rem 4rem',
        background: 'transparent',
      }}
    >
      <div
        style={{
          width: '100%',
          maxWidth: `${theme.layout.contentMaxWidth}px`,
          margin: '0 auto',
          display: 'flex',
          flexDirection: 'column',
          gap: '2rem',
          alignItems: 'center',
          textAlign: 'center',
        }}
      >
        <div
          style={{
            padding: '0.4rem 1rem',
            borderRadius: theme.borderRadius.full,
            backgroundColor: 'rgba(29, 185, 84, 0.1)',
            border: '1px solid rgba(29, 185, 84, 0.2)',
            color: theme.colors.primary.accent,
            fontSize: '0.85rem',
            display: 'inline-flex',
            alignItems: 'center',
            gap: '0.5rem',
            boxShadow: '0 6px 12px rgba(29, 185, 84, 0.08)',
          }}
        >
          <span
            style={{
              width: '8px',
              height: '8px',
              borderRadius: '50%',
              backgroundColor: theme.colors.primary.accent,
              boxShadow: `0 0 10px ${theme.colors.primary.accent}`,
            }}
          />
          {badgeLabel}
        </div>

        <div style={{ maxWidth: '720px' }}>
          <h1
            style={{
              margin: '1rem 0 0.5rem',
              fontSize: '2.5rem',
              lineHeight: 1.3,
              fontWeight: 800,
              color: theme.colors.text.primary,
              letterSpacing: '-0.02em',
            }}
          >
            {title}
          </h1>
          <p
            style={{
              margin: '0 auto',
              fontSize: '1.05rem',
              lineHeight: 1.75,
              color: theme.colors.text.secondary,
              maxWidth: '56ch',
            }}
          >
            {description}
          </p>
        </div>

        <div
          style={{
            position: 'relative',
            display: 'flex',
            flexDirection: 'column',
            gap: '1rem',
            alignItems: 'center',
            width: '100%',
            maxWidth: '640px',
            backgroundColor: 'rgba(255,255,255,0.02)',
            borderRadius: '28px',
            border: `1px solid ${theme.colors.border.default}`,
            padding: '2.5rem 1.5rem',
            boxShadow: theme.shadows.md,
            overflow: 'hidden',
          }}
        >
          <div
            style={{
              position: 'absolute',
              inset: 0,
              borderRadius: 'inherit',
              background: 'radial-gradient(circle at 50% 0%, rgba(29, 185, 84, 0.1), transparent 70%)',
              pointerEvents: 'none',
            }}
          />

          <div
            style={{
              display: 'flex',
              gap: '1rem',
              flexWrap: 'wrap',
              justifyContent: 'center',
              position: 'relative',
              zIndex: 1,
            }}
          >
            <button
              type="button"
              onClick={onPrimaryAction}
              style={{
                borderRadius: theme.borderRadius.full,
                border: 'none',
                padding: '0.9rem 2rem',
                fontWeight: 700,
                fontSize: '1rem',
                cursor: 'pointer',
                color: '#000',
                background: theme.colors.primary.accent,
                boxShadow: '0 8px 24px rgba(29, 185, 84, 0.3)',
                transition: 'transform 0.2s',
              }}
              onMouseOver={(e) => (e.currentTarget.style.transform = 'scale(1.05)')}
              onMouseOut={(e) => (e.currentTarget.style.transform = 'scale(1)')}
            >
              开始智能推荐
            </button>
            <button
              type="button"
              onClick={onSecondaryAction}
              style={{
                borderRadius: theme.borderRadius.full,
                border: `1px solid ${theme.colors.border.default}`,
                padding: '0.9rem 1.8rem',
                fontWeight: 500,
                fontSize: '1rem',
                cursor: 'pointer',
                color: theme.colors.text.primary,
                backgroundColor: 'rgba(255,255,255,0.05)',
                transition: 'background-color 0.2s',
              }}
              onMouseOver={(e) => (e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.1)')}
              onMouseOut={(e) => (e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.05)')}
            >
              查看使用指南
            </button>
          </div>
          <p
            style={{
              margin: 0,
              fontSize: '0.88rem',
              color: theme.colors.text.secondary,
              zIndex: 1,
            }}
          >
            {subtitle}
          </p>
        </div>

        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
            gap: '1rem',
            width: '100%',
            maxWidth: '960px',
            marginTop: '1rem',
          }}
        >
          {metricCards.map((metric) => (
            <div
              key={metric.label}
              style={{
                padding: '1.25rem',
                borderRadius: '20px',
                border: `1px solid ${theme.colors.border.default}`,
                backgroundColor: 'rgba(255,255,255,0.02)',
                textAlign: 'left' as const,
              }}
            >
              <p
                style={{
                  margin: 0,
                  fontSize: '0.78rem',
                  color: theme.colors.primary.accent,
                  letterSpacing: '0.08em',
                  textTransform: 'uppercase',
                  fontWeight: 600,
                }}
              >
                {metric.label}
              </p>
              <p
                style={{
                  margin: '0.5rem 0 0.3rem',
                  fontSize: '1.25rem',
                  fontWeight: 700,
                  color: theme.colors.text.primary,
                }}
              >
                {metric.value}
              </p>
              <p
                style={{
                  margin: 0,
                  fontSize: '0.88rem',
                  color: theme.colors.text.secondary,
                  lineHeight: 1.6,
                }}
              >
                {metric.detail}
              </p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

