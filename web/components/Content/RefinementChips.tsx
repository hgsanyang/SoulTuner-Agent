'use client';

import type { RefinementOption } from '@/lib/api';

/**
 * 非阻塞微调方向 chips：显示在最新 assistant 回复下方。
 * chips 内容完全由后端 LLM 基于本轮上下文生成；点击后作为普通
 * 自然语言 follow-up 提交，不走任何固定状态机。
 * 视觉上使用单一克制的青绿色 pill，与整组反馈的矩形按钮区分。
 */
export default function RefinementChips({
  options,
  disabled,
  onSelect,
}: {
  options: RefinementOption[];
  disabled?: boolean;
  onSelect: (prompt: string) => void;
}) {
  if (!options || options.length === 0) return null;
  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem', marginTop: '0.75rem' }}>
      {options.map(option => (
        <button
          key={`${option.label}-${option.prompt}`}
          onClick={() => onSelect(option.prompt || option.label)}
          disabled={disabled}
          title={option.reason || option.prompt}
          style={{
            padding: '0.42rem 0.78rem',
            borderRadius: '999px',
            border: '1px solid rgba(91, 214, 170, 0.34)',
            backgroundColor: 'rgba(42, 170, 130, 0.14)',
            color: 'rgba(238, 255, 248, 0.9)',
            fontSize: '0.81rem',
            fontWeight: 500,
            cursor: disabled ? 'not-allowed' : 'pointer',
            transition: 'all 0.18s ease',
            whiteSpace: 'normal',
            wordBreak: 'break-word',
            textAlign: 'left',
          }}
          onMouseEnter={e => {
            if (!disabled) {
              e.currentTarget.style.backgroundColor = 'rgba(54, 190, 146, 0.24)';
              e.currentTarget.style.borderColor = 'rgba(112, 235, 190, 0.48)';
            }
          }}
          onMouseLeave={e => {
            e.currentTarget.style.backgroundColor = 'rgba(42, 170, 130, 0.14)';
            e.currentTarget.style.borderColor = 'rgba(91, 214, 170, 0.34)';
          }}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}
