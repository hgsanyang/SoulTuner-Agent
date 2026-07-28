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

/**
 * 存的是 slug 不是按钮文字：改个中文标签就不该让历史数据分裂成两类。
 * 与逐首反馈的 OffReason 是两套词表 —— 这里评判的是「整组」（重复/太冷门
 * 不可能用来说一首歌），后端 schemas/feedback_events.py 的 SlateReason 是唯一真值。
 */
const REASON_OPTIONS: { value: string; label: string }[] = [
  { value: 'too_loud', label: '太吵' },
  { value: 'too_sad', label: '太悲伤' },
  { value: 'too_mainstream', label: '太热门' },
  { value: 'too_obscure', label: '太冷门' },
  { value: 'too_repetitive', label: '重复太多' },
  { value: 'scene_mismatch', label: '场景不合' },
  { value: 'wrong_language_or_era', label: '语言/年代不准' },
  { value: 'other', label: '其他' },
];

export type SlatePickSong = { musicId: string; title: string; artist?: string };
export type SlatePicks = { best: string[]; worst: string[] };

const MAX_PICKS = 3;

export default function SlateFeedback({
  exposureId,
  songCount,
  songs = [],
  submittedRating,
  onSubmit,
}: {
  exposureId: string;
  songCount: number;
  /** 用于「最符合/最不符合」挑选；缺 musicId 的歌无法归因，不显示。 */
  songs?: SlatePickSong[];
  submittedRating?: string;
  onSubmit: (rating: SlateFeedbackRating, reasons: string[], note: string, picks: SlatePicks) => Promise<boolean>;
}) {
  const [expandedRating, setExpandedRating] = useState<SlateFeedbackRating | null>(null);
  const [reasons, setReasons] = useState<string[]>([]);
  const [note, setNote] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [best, setBest] = useState<string[]>([]);
  const [worst, setWorst] = useState<string[]>([]);

  // 只有带 musicId 的歌能被服务端对回曝光记录并归因；其余不显示。
  const pickable = songs.filter(s => s.musicId && s.title);

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

  /** 一首歌不能既是最符合又是最不符合；选进一边就从另一边移除。 */
  const togglePick = (which: 'best' | 'worst', musicId: string) => {
    const [list, setList] = which === 'best' ? [best, setBest] as const : [worst, setWorst] as const;
    const [other, setOther] = which === 'best' ? [worst, setWorst] as const : [best, setBest] as const;
    if (list.includes(musicId)) {
      setList(list.filter(id => id !== musicId));
      return;
    }
    if (list.length >= MAX_PICKS) return;
    if (other.includes(musicId)) setOther(other.filter(id => id !== musicId));
    setList([...list, musicId]);
  };

  const submit = async (rating: SlateFeedbackRating) => {
    if (submitting) return;
    setSubmitting(true);
    try {
      await onSubmit(rating, reasons, note, { best, worst });
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
        <span style={{ opacity: 0.75 }}>（选一个之后可以补充原因、挑出最合适/最不合适的歌，也可以自己写）</span>
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

      {expandedRating && pickable.length > 0 && (
        <div style={{ marginTop: '0.65rem' }}>
          {(['best', 'worst'] as const).map(which => {
            const selected = which === 'best' ? best : worst;
            const full = selected.length >= MAX_PICKS;
            return (
              <div key={which} style={{ marginBottom: '0.5rem' }}>
                <div style={{ color: theme.colors.text.muted, fontSize: '0.72rem', marginBottom: '0.3rem' }}>
                  {which === 'best' ? '哪几首最符合？' : '哪几首最不符合？'}
                  <span style={{ opacity: 0.6 }}>（可选，最多 {MAX_PICKS} 首；没选的算「未知」，不当负样本）</span>
                </div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.35rem' }}>
                  {pickable.map(song => {
                    const on = selected.includes(song.musicId);
                    const disabled = submitting || (!on && full);
                    const accent = which === 'best' ? '29,185,84' : '255,120,120';
                    return (
                      <button
                        key={`${which}-${song.musicId}`}
                        onClick={() => togglePick(which, song.musicId)}
                        disabled={disabled}
                        title={song.artist ? `${song.title} — ${song.artist}` : song.title}
                        style={{
                          padding: '0.28rem 0.55rem',
                          borderRadius: '999px',
                          border: `1px solid ${on ? `rgba(${accent},0.55)` : 'rgba(255,255,255,0.14)'}`,
                          backgroundColor: on ? `rgba(${accent},0.18)` : 'rgba(255,255,255,0.05)',
                          color: on ? `rgb(${accent})` : 'rgba(255,255,255,0.7)',
                          fontSize: '0.74rem',
                          maxWidth: '13rem',
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap',
                          cursor: disabled ? 'default' : 'pointer',
                          opacity: disabled && !on ? 0.45 : 1,
                          transition: 'all 0.15s ease',
                        }}
                      >
                        {song.title}
                      </button>
                    );
                  })}
                </div>
              </div>
            );
          })}
        </div>
      )}

      {expandedRating && (
        <div style={{ marginTop: '0.65rem' }}>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem', marginBottom: '0.55rem' }}>
            {REASON_OPTIONS.map(({ value, label }) => {
              const selected = reasons.includes(value);
              return (
                <button
                  key={value}
                  onClick={() => toggleReason(value)}
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
                  {label}
                </button>
              );
            })}
          </div>
          {reasons.includes('other') && (
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
