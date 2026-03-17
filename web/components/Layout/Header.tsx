'use client';

import { theme } from '@/styles/theme';

interface HeaderProps {
  onMenuToggle?: () => void;
  isMobile?: boolean;
}

export default function Header({ onMenuToggle, isMobile = false }: HeaderProps) {
  return (
    <header
      style={{
        position: isMobile ? 'relative' : 'sticky',
        top: isMobile ? 0 : '1.25rem',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        gap: '1.25rem',
        padding: isMobile ? '0.9rem 1rem' : '1rem 1.5rem',
        borderRadius: theme.borderRadius.lg,
        backgroundColor: 'rgba(36, 36, 36, 0.4)',
        backdropFilter: 'blur(16px)',
        border: `1px solid ${theme.colors.border.default}`,
        boxShadow: theme.shadows.sm,
        zIndex: 5,
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
        {onMenuToggle && (
          <button
            type="button"
            aria-label="打开导航"
            onClick={onMenuToggle}
            style={{
              width: '40px',
              height: '40px',
              borderRadius: theme.borderRadius.md,
              border: `1px solid ${theme.colors.border.default}`,
              backgroundColor: theme.colors.background.card,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              cursor: 'pointer',
            }}
          >
            <span
              style={{
                display: 'flex',
                flexDirection: 'column',
                gap: '5px',
              }}
            >
              {[0, 1, 2].map((line) => (
                <span
                  key={line}
                  style={{
                    width: '22px',
                    height: '2px',
                    borderRadius: '9999px',
                    backgroundColor: theme.colors.text.primary,
                  }}
                />
              ))}
            </span>
          </button>
        )}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
          <span
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '0.35rem',
              padding: '0.25rem 0.7rem',
              borderRadius: theme.borderRadius.full,
              backgroundColor: theme.colors.primary[100],
              color: theme.colors.primary[700],
              fontSize: '0.78rem',
              fontWeight: 500,
              width: 'fit-content',
            }}
          >
            <span
              style={{
                width: '6px',
                height: '6px',
                borderRadius: '50%',
                backgroundColor: theme.colors.primary[700],
              }}
            />
            Beta
          </span>
          <div>
            <h1
              style={{
                fontSize: isMobile ? '1.2rem' : '1.4rem',
                color: theme.colors.text.primary,
                margin: 0,
                fontWeight: 600,
              }}
            >
              Deep Search Studio
            </h1>
            <p
              style={{
                margin: '0.2rem 0 0',
                color: theme.colors.text.muted,
                fontSize: '0.9rem',
              }}
            >
              音乐体验工作台 · 自然语言编排 · 快速穿梭每个场景
            </p>
          </div>
        </div>
      </div>
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '0.75rem',
          flexWrap: 'wrap',
          justifyContent: isMobile ? 'flex-end' : 'flex-start',
        }}
      >
        <a
          href="/recommendations"
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: '0.5rem',
            padding: '0.65rem 1.4rem',
            borderRadius: theme.borderRadius.full,
            backgroundColor: theme.colors.text.primary,
            color: theme.colors.background.main,
            fontWeight: 600,
            fontSize: '0.9rem',
            textDecoration: 'none',
          }}
        >
          立即使用
          <svg
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M7 17L17 7" />
            <path d="M7 7h10v10" />
          </svg>
        </a>
      </div>
    </header>
  );
}

