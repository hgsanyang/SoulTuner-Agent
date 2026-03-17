'use client';
// 首页的页面设计
import { theme } from '@/styles/theme';

interface ProductIntroProps {
  onPrimaryAction?: () => void;
  onSecondaryAction?: () => void;
  onQuickExampleSelect?: (prompt: string) => void;
}

const quickExamples = [
  { title: '给加班夜写代码的人推荐稳态节奏', meta: '心情：专注 / 场景：深夜' },
  { title: '我想在雨天窗边听些治愈的独立民谣', meta: '联动心情 + 流派' },
  { title: '根据周杰伦帮我找一些同样浪漫的中文 R&B', meta: '歌手 + 风格' },
];

export default function ProductIntro({ onPrimaryAction, onSecondaryAction, onQuickExampleSelect }: ProductIntroProps) {
  return (
    <section
      style={{
        flex: 1,
        width: '100%',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '0 1.5rem',
        minHeight: 'calc(100vh - 7rem)', // Account for padding
      }}
    >
      <div
        style={{
          width: '100%',
          maxWidth: `${theme.layout.contentMaxWidth}px`,
          borderRadius: theme.borderRadius.lg,
          backgroundColor: 'rgba(36, 36, 36, 0.2)',
          backdropFilter: 'blur(12px)',
          padding: '4rem 3rem',
          color: theme.colors.text.primary,
          position: 'relative',
          overflow: 'hidden',
          boxShadow: theme.shadows.lg,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          textAlign: 'center',
          border: `1px solid ${theme.colors.border.default}`,
        }}
      >
        {/* Glow effect */}
        <div
          style={{
            position: 'absolute',
            top: '-50%',
            left: '50%',
            transform: 'translateX(-50%)',
            width: '80%',
            height: '100%',
            background: 'radial-gradient(circle, rgba(29, 185, 84, 0.15) 0%, rgba(0,0,0,0) 70%)',
            pointerEvents: 'none',
          }}
        />

        <span
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: '0.5rem',
            padding: '0.4rem 1rem',
            borderRadius: theme.borderRadius.full,
            backgroundColor: 'rgba(29, 185, 84, 0.1)',
            border: '1px solid rgba(29, 185, 84, 0.2)',
            fontSize: '0.85rem',
            color: theme.colors.primary.accent,
            marginBottom: '2rem',
          }}
        >
          <span
            style={{
              display: 'inline-block',
              width: '8px',
              height: '8px',
              borderRadius: '50%',
              backgroundColor: theme.colors.primary.accent,
              boxShadow: `0 0 10px ${theme.colors.primary.accent}`,
            }}
          />
          AI Music Agent 已就绪
        </span>

        <h1
          style={{
            margin: '0',
            fontSize: 'clamp(2.5rem, 5vw, 4rem)',
            lineHeight: 1.1,
            fontWeight: 800,
            letterSpacing: '-0.03em',
            zIndex: 1,
          }}
        >
          听懂你的每一刻情绪
          <br />
          <span style={{ color: theme.colors.primary.accent }}>探索未知的音乐旅程</span>
        </h1>

        <p
          style={{
            marginTop: '1.5rem',
            maxWidth: '42rem',
            fontSize: '1.1rem',
            lineHeight: 1.6,
            color: theme.colors.text.secondary,
            zIndex: 1,
          }}
        >
          基于 LangGraph 与多模态搜索，用自然语言即可生成专属你的沉浸式歌单，联动 Spotify 或本地音频网，发现契合当下的绝佳旋律。
        </p>

        <div
          style={{
            marginTop: '2.5rem',
            display: 'flex',
            flexWrap: 'wrap',
            gap: '1rem',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1,
          }}
        >
          <button
            type="button"
            onClick={onPrimaryAction}
            style={{
              borderRadius: theme.borderRadius.full,
              border: 'none',
              padding: '1rem 2.5rem',
              fontWeight: 700,
              fontSize: '1.05rem',
              cursor: 'pointer',
              color: '#000',
              backgroundColor: theme.colors.primary.accent,
              boxShadow: '0 8px 24px rgba(29, 185, 84, 0.3)',
              transition: 'transform 0.2s, background-color 0.2s',
            }}
            onMouseOver={(e) => {
              e.currentTarget.style.transform = 'scale(1.05)';
              e.currentTarget.style.backgroundColor = '#1ed760';
            }}
            onMouseOut={(e) => {
              e.currentTarget.style.transform = 'scale(1)';
              e.currentTarget.style.backgroundColor = theme.colors.primary.accent;
            }}
          >
            开启智能推荐
          </button>
        </div>

        {/* Quick Inspiration Pills Integrated */}
        <div style={{ marginTop: '4rem', zIndex: 1, width: '100%', maxWidth: '48rem' }}>
          <p style={{ fontSize: '0.9rem', color: theme.colors.text.muted, marginBottom: '1rem', fontWeight: 600 }}>
            快速灵感注入
          </p>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.8rem', justifyContent: 'center' }}>
            {quickExamples.map((example) => (
              <button
                key={example.title}
                type="button"
                onClick={() => onQuickExampleSelect?.(example.title)}
                style={{
                  borderRadius: theme.borderRadius.full,
                  padding: '0.7rem 1.2rem',
                  backgroundColor: 'rgba(255,255,255,0.05)',
                  border: `1px solid ${theme.colors.border.default}`,
                  color: theme.colors.text.primary,
                  fontSize: '0.9rem',
                  cursor: 'pointer',
                  transition: 'background-color 0.2s, border-color 0.2s',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem',
                }}
                onMouseOver={(e) => {
                  e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.1)';
                  e.currentTarget.style.borderColor = theme.colors.border.focus;
                }}
                onMouseOut={(e) => {
                  e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.05)';
                  e.currentTarget.style.borderColor = theme.colors.border.default;
                }}
              >
                <span>{example.title}</span>
                <span style={{ fontSize: '0.75rem', color: theme.colors.text.muted }}>· {example.meta}</span>
              </button>
            ))}
          </div>
        </div>

      </div>
    </section>
  );
}
