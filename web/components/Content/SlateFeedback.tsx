'use client';

import { useState } from 'react';
import type { SlateFeedbackRating } from '@/lib/api';
import { theme } from '@/styles/theme';

/**
 * 整组推荐反馈：绑定当前 exposure_id，显示在最新 assistant 回复下方。
 * 主操作收束为「整体合适 / 部分合适 / 不太合适」；
 * 选择"部分/不太合适"后渐进展开原因 chips 与可选备注（progressive disclosure）。
 * 视觉上使用矩形按钮，与微调方向的 pill chips 区分。
 */

const PRIMARY_OPTIONS: { rating: SlateFeedbackRating; label: string }[] = [
  { rating: 'great', label: '整体合适' },
  { rating: 'partial', label: '部分合适' },
  { rating: 'off', label: '不太合适' },
];

const REASON_OPTIONS = ['太吵', '太悲伤', '太热门', '太冷门', '重复太多', '场景不合', '语言/年代不准', '其他'];

export default function SlateFeedback({
  exposureId,
  songCount,
  submittedRating,
  onSubmit,
}: {
  exposureId: string;
  songCount: number;
  submittedRating?: string;
  onSubmit: (rating: SlateFeedbackRating, reasons: string[], note: string) => Promise<boolean>;
}) {
  const [expandedRating, setExpandedRating] = useState<SlateFeedbackRating | null>(null);
  const [reasons, setReasons] = useState<string[]>([]);
  const [note, setNote] = useState('');
  const [submitting, setSubmitting] = useState(false);

  if (!exposureId || songCount === 0) return null;

  if (submittedRating) {
    const label = PRIMARY_OPTIONS.find(o => o.rating === submittedRating)?.label || '已记录';
    return (
      <div style={{
        marginTop: '0.7rem',
        padding: '0.5rem 0.75rem',
        borderRadius: '0.6rem',
        border: '1px solid rgba(29,185,84,0.25)',
        backgroundColor: 'rgba(29,185,84,0.08)',
        color: 'rgba(210,245,225,0.85)',
        fontSize: '0.78rem',
        display: 'inline-flex',
        alignItems: 'center',
        gap: '0.4rem',
      }}>
        ✓ 已记录反馈：{label}
      </div>
    );
  }

  const toggleReason = (reason: string) => {
    setReasons(prev => prev.includes(reason) ? prev.filter(r => r !== reason) : [...prev, reason]);
  };

  const submit = async (rating: SlateFeedbackRating) => {
    if (submitting) return;
    setSubmitting(true);
    try {
      await onSubmit(rating, reasons, note);
    } finally {
      setSubmitting(false);
    }
  };

  const handlePrimary = (rating: SlateFeedbackRating) => {
    if (rating === 'great') {
      // 正向反馈无需原因，直接提交
      void submit('great');
      return;
    }
    setExpandedRating(prev => (prev === rating ? null : rating));
  };

  return (
    <div style={{
      marginTop: '0.85rem',
      padding: '0.75rem 0.85rem',
      borderRadius: '0.75rem',
      border: `1px solid ${theme.colors.border.default}`,
      backgroundColor: 'rgba(255,255,255,0.03)',
    }}>
      <div style={{ color: theme.colors.text.muted, fontSize: '0.76rem', marginBottom: '0.55rem' }}>
        这组 {songCount} 首推荐怎么样？你的反馈会用于改进之后的推荐。
      </div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
        {PRIMARY_OPTIONS.map(option => {
          const active = expandedRating === option.rating;
          return (
            <button
              key={option.rating}
              onClick={() => handlePrimary(option.rating)}
              disabled={submitting}
              style={{
                padding: '0.45rem 0.9rem',
                borderRadius: '0.55rem',
                border: active ? '1px solid rgba(120,180,255,0.45)' : '1px solid rgba(255,255,255,0.14)',
                backgroundColor: active ? 'rgba(70,130,220,0.16)' : 'rgba(255,255,255,0.06)',
                color: active ? '#fff' : 'rgba(255,255,255,0.78)',
                fontSize: '0.8rem',
                fontWeight: 550,
                cursor: submitting ? 'wait' : 'pointer',
                transition: 'all 0.18s ease',
              }}
              onMouseEnter={e => { if (!submitting && !active) e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.1)'; }}
              onMouseLeave={e => { if (!active) e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.06)'; }}
            >
              {option.label}
            </button>
          );
        })}
      </div>

      {expandedRating && (
        <div style={{ marginTop: '0.65rem' }}>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem', marginBottom: '0.55rem' }}>
            {REASON_OPTIONS.map(reason => {
              const selected = reasons.includes(reason);
              return (
                <button
                  key={reason}
                  onClick={() => toggleReason(reason)}
                  disabled={submitting}
                  style={{
                    padding: '0.32rem 0.6rem',
                    borderRadius: '0.45rem',
                    border: selected ? '1px solid rgba(120,180,255,0.44)' : '1px solid rgba(255,255,255,0.10)',
                    backgroundColor: selected ? 'rgba(70,130,220,0.16)' : 'rgba(255,255,255,0.045)',
                    color: selected ? 'rgba(238,246,255,0.95)' : 'rgba(255,255,255,0.6)',
                    fontSize: '0.74rem',
                    cursor: submitting ? 'wait' : 'pointer',
                  }}
                >
                  {reason}
                </button>
              );
            })}
          </div>
          {reasons.includes('其他') && (
            <input
              value={note}
              onChange={event => setNote(event.target.value)}
              placeholder="可选：具体哪里不对？"
              maxLength={240}
              style={{
                width: '100%',
                boxSizing: 'border-box',
                padding: '0.5rem 0.65rem',
                borderRadius: '0.5rem',
                border: '1px solid rgba(255,255,255,0.10)',
                backgroundColor: 'rgba(0,0,0,0.22)',
                color: theme.colors.text.primary,
                outline: 'none',
                fontSize: '0.76rem',
                marginBottom: '0.55rem',
              }}
            />
          )}
          <button
            onClick={() => void submit(expandedRating)}
            disabled={submitting}
            style={{
              padding: '0.42rem 1.1rem',
              borderRadius: '0.55rem',
              border: '1px solid rgba(120,180,255,0.4)',
              backgroundColor: 'rgba(70,130,220,0.22)',
              color: '#fff',
              fontSize: '0.78rem',
              fontWeight: 600,
              cursor: submitting ? 'wait' : 'pointer',
            }}
          >
            {submitting ? '提交中…' : '提交反馈'}
          </button>
        </div>
      )}
    </div>
  );
}
