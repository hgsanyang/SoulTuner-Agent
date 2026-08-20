'use client';

import { useState, useRef, useLayoutEffect, FormEvent, KeyboardEvent } from 'react';
import { useLang } from '@/context/LanguageContext';
import { theme } from '@/styles/theme';

interface ChatInputProps {
  onSubmit: (value: string) => void;
  onAbort?: () => void;         // 新增：中止当前搜索的回调
  placeholder?: string;
  disabled?: boolean;
  isLoading?: boolean;          // 新增：是否正在搜索中（用于切换中止按钮）
  isMobile?: boolean;
}

// 点一下就填进输入框。以前这三条只是个声明，整个文件没有第二处引用——
// placeholder 里的例句浏览器从不支持 Tab 补全，所以"想要那句话"的唯一出路
// 是让它可点。
const quickPrompts = ['晨跑的鼓点', '办公室保持专注', '串联周末的晚风'];

const MAX_ROWS = 6;

export default function ChatInput({
  onSubmit,
  onAbort,
  placeholder = '输入你的问题...',
  disabled = false,
  isLoading = false,
  isMobile = false,
}: ChatInputProps) {
  const { t } = useLang();
  const [value, setValue] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // 高度跟着内容走。先归零再读 scrollHeight，否则删字时高度只增不减。
  useLayoutEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = 'auto';
    const lineHeight = 24;
    el.style.height = `${Math.min(el.scrollHeight, lineHeight * MAX_ROWS)}px`;
  }, [value]);

  const submit = () => {
    if (value.trim() && !disabled) {
      onSubmit(value.trim());
      setValue('');
    }
  };

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    submit();
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    // 输入法组字期间的 Enter 是在选词，不是发送。isComposing 不判会让中文
    // 用户每打一个词就误发一次。
    if (e.key !== 'Enter' || e.nativeEvent.isComposing) return;
    if (e.shiftKey) return;      // 换行交给 textarea 自己
    e.preventDefault();
    submit();
  };

  const applyQuickPrompt = (prompt: string) => {
    setValue(prompt);
    textareaRef.current?.focus();
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
          {/* textarea 而不是 input：input 在物理上装不下换行符，之前底下那句
              "Shift + Enter 换行"承诺的是这个元素做不到的能力。 */}
          <textarea
            ref={textareaRef}
            rows={1}
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={isLoading ? t('搜索中...输入新问题可直接切换') : placeholder}
            disabled={disabled}
            style={{
              flex: 1,
              padding: '0.65rem 0',
              fontSize: '1.05rem',
              lineHeight: '24px',
              minHeight: '40px',
              maxHeight: `${24 * MAX_ROWS}px`,
              border: 'none',
              backgroundColor: 'transparent',
              color: theme.colors.text.primary,
              outline: 'none',
              opacity: disabled ? 0.5 : 1,
              resize: 'none',
              overflowY: 'auto',
              fontFamily: 'inherit',
            }}
          />

          {/* 搜索中显示「中止」按钮，否则显示「发送」按钮 */}
          {isLoading ? (
            <button
              type="button"
              onClick={handleAbort}
              title={t("中止当前搜索")}
              aria-label={t("中止当前搜索")}
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
              title={t("发送")}
              aria-label={t("发送")}
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
      {/* 空输入框时给几个能点的例子。placeholder 里的例句点不了也补不了，
          想用只能自己重打一遍——这才是"提示没用"的真正原因。 */}
      {!value && !isLoading && !disabled && (
        <div style={{
          display: 'flex', flexWrap: 'wrap', gap: '0.4rem',
          justifyContent: 'center', marginTop: '0.1rem',
        }}>
          {quickPrompts.map(prompt => (
            <button
              key={prompt}
              type="button"
              onClick={() => applyQuickPrompt(prompt)}
              style={{
                padding: '0.3rem 0.75rem',
                borderRadius: theme.borderRadius.full,
                border: `1px solid ${theme.colors.border.default}`,
                background: 'rgba(255,255,255,0.04)',
                color: theme.colors.text.secondary,
                fontSize: '0.78rem',
                cursor: 'pointer',
                transition: 'background-color 0.2s, color 0.2s',
              }}
              onMouseEnter={e => {
                e.currentTarget.style.background = 'rgba(255,255,255,0.10)';
                e.currentTarget.style.color = theme.colors.text.primary;
              }}
              onMouseLeave={e => {
                e.currentTarget.style.background = 'rgba(255,255,255,0.04)';
                e.currentTarget.style.color = theme.colors.text.secondary;
              }}
            >
              {t(prompt)}
            </button>
          ))}
        </div>
      )}

      {/* 只在正在搜索、或输入框还空着的时候提示。第一次有用，第一百次是噪音。 */}
      {!isMobile && (isLoading || !value) && (
        <span
          style={{
            fontSize: '0.78rem',
            color: theme.colors.text.muted,
            textAlign: 'right',
            marginTop: '0.2rem',
          }}
        >
          {isLoading ? t('点击 ■ 中止搜索，或直接输入新问题') : t('Enter 发送，Shift + Enter 换行')}
        </span>
      )}
    </form>
  );
}
