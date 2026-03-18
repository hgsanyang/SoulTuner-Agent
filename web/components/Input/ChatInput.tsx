'use client';

import { useState, FormEvent } from 'react';
import { theme } from '@/styles/theme';

interface ChatInputProps {
  onSubmit: (value: string) => void;
  onAbort?: () => void;         // 新增：中止当前搜索的回调
  placeholder?: string;
  disabled?: boolean;
  isLoading?: boolean;          // 新增：是否正在搜索中（用于切换中止按钮）
  isMobile?: boolean;
}

const quickPrompts = ['晨跑的鼓点', '办公室保持专注', '串联周末的晚风'];

export default function ChatInput({
  onSubmit,
  onAbort,
  placeholder = '输入你的问题...',
  disabled = false,
  isLoading = false,
  isMobile = false,
}: ChatInputProps) {
  const [value, setValue] = useState('');

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (value.trim() && !disabled) {
      onSubmit(value.trim());
      setValue('');
    }
  };

  const handleAbort = (e: React.MouseEvent) => {
    e.preventDefault();
    onAbort?.();
  };

  return (
    <form
      onSubmit={handleSubmit}
      style={{
        margin: '0 auto',
        padding: isMobile ? '0 0.25rem' : '0 1rem',
        backgroundColor: 'transparent',
        display: 'flex',
        flexDirection: 'column',
        gap: '0.65rem',
        width: '100%',
        maxWidth: isMobile ? '520px' : '640px',
        zIndex: 50,
      }}
    >
      <div
        style={{
          display: 'flex',
          gap: isMobile ? '0.5rem' : '0.65rem',
          alignItems: 'center',
          flexDirection: isMobile ? 'column' : 'row',
        }}
      >
        <div
          style={{
            flex: 1,
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
            backgroundColor: 'rgba(255, 255, 255, 0.06)',
            borderRadius: theme.borderRadius.full,
            border: `1px solid ${isLoading ? 'rgba(255, 120, 50, 0.5)' : theme.colors.border.focus}`,
            padding: isMobile ? '0.35rem 0.35rem 0.35rem 1rem' : '0.4rem 0.4rem 0.4rem 1.5rem',
            boxShadow: isLoading ? '0 8px 32px rgba(255,80,0,0.15)' : '0 8px 32px rgba(0, 0, 0, 0.4)',
            backdropFilter: 'blur(12px)',
            transition: 'border-color 0.3s, box-shadow 0.3s',
          }}
        >
          {/* 搜索中：显示脉冲动画点 */}
          {isLoading && (
            <div style={{ display: 'flex', gap: '3px', alignItems: 'center', flexShrink: 0, marginLeft: '0.25rem' }}>
              {[0, 1, 2].map(i => (
                <span
                  key={i}
                  style={{
                    width: '5px', height: '5px', borderRadius: '50%',
                    backgroundColor: 'rgba(255, 140, 60, 0.85)',
                    display: 'inline-block',
                    animation: `pulse 1.2s ease-in-out ${i * 0.2}s infinite`,
                  }}
                />
              ))}
              <style>{`
                @keyframes pulse {
                  0%, 80%, 100% { transform: scale(0.6); opacity: 0.4; }
                  40% { transform: scale(1); opacity: 1; }
                }
              `}</style>
            </div>
          )}
          <input
            type="text"
            value={value}
            onChange={(e) => setValue(e.target.value)}
            placeholder={isLoading ? '搜索中...输入新问题可直接切换' : placeholder}
            disabled={disabled}
            style={{
              flex: 1,
              padding: '0.65rem 0',
              fontSize: '1.05rem',
              minHeight: '40px',
              border: 'none',
              backgroundColor: 'transparent',
              color: theme.colors.text.primary,
              outline: 'none',
              opacity: disabled ? 0.5 : 1,
            }}
          />

          {/* 搜索中显示「中止」按钮，否则显示「发送」按钮 */}
          {isLoading ? (
            <button
              type="button"
              onClick={handleAbort}
              title="中止当前搜索"
              style={{
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                width: '40px', height: '40px',
                borderRadius: '50%',
                border: 'none',
                backgroundColor: 'rgba(255, 80, 30, 0.85)',
                cursor: 'pointer',
                flexShrink: 0,
                transition: 'background-color 0.2s, transform 0.1s',
              }}
              onMouseEnter={e => {
                e.currentTarget.style.backgroundColor = 'rgba(255, 50, 10, 1)';
                e.currentTarget.style.transform = 'scale(1.08)';
              }}
              onMouseLeave={e => {
                e.currentTarget.style.backgroundColor = 'rgba(255, 80, 30, 0.85)';
                e.currentTarget.style.transform = 'scale(1)';
              }}
            >
              {/* 方形停止图标 */}
              <svg width="14" height="14" viewBox="0 0 16 16" fill="white" stroke="none">
                <rect x="2" y="2" width="12" height="12" rx="2" />
              </svg>
            </button>
          ) : (
            <button
              type="submit"
              disabled={!value.trim()}
              title="发送"
              style={{
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                width: '40px', height: '40px',
                borderRadius: '50%',
                border: 'none',
                backgroundColor: value.trim() ? theme.colors.primary.accent : 'rgba(255,255,255,0.08)',
                cursor: value.trim() ? 'pointer' : 'default',
                flexShrink: 0,
                transition: 'background-color 0.2s, transform 0.1s',
              }}
              onMouseEnter={e => { if (value.trim()) e.currentTarget.style.transform = 'scale(1.08)'; }}
              onMouseLeave={e => { e.currentTarget.style.transform = 'scale(1)'; }}
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <line x1="22" y1="2" x2="11" y2="13" />
                <polygon points="22 2 15 22 11 13 2 9 22 2" />
              </svg>
            </button>
          )}
        </div>
      </div>
      {!isMobile && (
        <span
          style={{
            fontSize: '0.78rem',
            color: theme.colors.text.muted,
            textAlign: 'right',
            marginTop: '0.2rem',
          }}
        >
          {isLoading ? '点击 ■ 中止搜索，或直接输入新问题' : '提示：直接输入自然语言，Shift + Enter 换行'}
        </span>
      )}
    </form>
  );
}
