'use client';

import { useState } from 'react';
import { useLang } from '@/context/LanguageContext';
import { SONG_OFF_REASON_LABELS, SongOffReason, sendSongFeedback } from '@/lib/api';
import { theme } from '@/styles/theme';

/**
 * Per-song CONTEXT feedback.
 *
 * This is deliberately separate from the heart / collect / dislike buttons:
 * those are the long-term TASTE channel ("我喜欢这首歌"), while this answers
 * "这首适合我此刻要的吗" for THIS slate. A song can be a favourite and still be
 * wrong for tonight, so the two are recorded on independent channels.
 *
 * Nothing is submitted unless the user actually picks something — an untouched
 * song stays UNKNOWN and never becomes a negative sample.
 */
export default function SongFeedbackPanel({
    exposureId,
    musicId,
    title,
    artist,
    sessionId,
    scene,
    onClose,
}: {
    exposureId: string;
    musicId?: string;
    title: string;
    artist?: string;
    sessionId?: string;
    scene?: string;
    onClose: () => void;
}) {
  const { t } = useLang();
    const [fit, setFit] = useState<'fits' | 'partial' | 'off' | null>(null);
    const [reasons, setReasons] = useState<SongOffReason[]>([]);
    const [note, setNote] = useState('');
    const [state, setState] = useState<'idle' | 'sending' | 'done' | 'error'>('idle');
    const [error, setError] = useState('');

    const toggleReason = (r: SongOffReason) =>
        setReasons(prev => (prev.includes(r) ? prev.filter(x => x !== r) : [...prev, r]));

    /** Reasons only mean anything under "不符合" — clear them when leaving it,
     *  otherwise a user who switches back to 很符合 writes a self-contradicting
     *  record (fits + "太吵"). */
    const chooseFit = (v: 'fits' | 'partial' | 'off') => {
        const next = fit === v ? null : v;
        setFit(next);
        if (next !== 'off') setReasons([]);
    };

    const canSubmit = Boolean(fit) || reasons.length > 0 || note.trim().length > 0;

    const submit = async () => {
        if (!canSubmit || state === 'sending') return;
        setState('sending');
        try {
            // rank/policy are NOT sent: the server backfills them from its own
            // exposure record so the browser cannot restate what the policy did.
            const result = await sendSongFeedback({
                exposureId,
                musicId,
                title,
                artist,
                contextFit: fit ?? undefined,
                offReasons: reasons,
                note: note.trim(),
                sessionId,
                scene,
            });
            // The endpoint answers 200 with {success:false} when the write fails,
            // so resp.ok alone would show "已记录" for feedback nobody stored.
            if (!result?.success) throw new Error(result?.error || t('服务端未确认'));
            setState('done');
            setTimeout(onClose, 900);
        } catch (e: any) {
            setError(e?.message || t('提交失败'));
            setState('error');
        }
    };

    const chip = (active: boolean): React.CSSProperties => ({
        padding: '0.32rem 0.7rem',
        borderRadius: '999px',
        border: `1px solid ${active ? '#1DB954' : 'rgba(255,255,255,0.18)'}`,
        background: active ? 'rgba(29,185,84,0.18)' : 'rgba(255,255,255,0.05)',
        color: active ? '#1DB954' : theme.colors.text.secondary,
        fontSize: '0.78rem',
        cursor: 'pointer',
        transition: 'all 0.12s',
    });

    return (
        <div
            onClick={e => e.stopPropagation()}
            style={{
                marginTop: '0.6rem',
                padding: '0.8rem',
                borderRadius: '10px',
                background: 'rgba(255,255,255,0.04)',
                border: '1px solid rgba(255,255,255,0.10)',
                display: 'flex',
                flexDirection: 'column',
                gap: '0.6rem',
            }}
        >
            <div style={{ fontSize: '0.8rem', color: theme.colors.text.secondary }}>
                {t('这首适合你此刻要的吗？')}<span style={{ opacity: 0.6 }}>{t('（只评这一次，不影响你对它的长期喜好）')}</span>
            </div>

            <div style={{ display: 'flex', gap: '0.4rem', flexWrap: 'wrap' }}>
                {([['fits', t('很符合')], ['partial', t('一般')], ['off', t('不符合')]] as const).map(([v, label]) => (
                    <button key={v} onClick={() => chooseFit(v)} style={chip(fit === v)}>
                        {label}
                    </button>
                ))}
            </div>

            {fit === 'off' && (
                <div style={{ display: 'flex', gap: '0.4rem', flexWrap: 'wrap' }}>
                    {(Object.keys(SONG_OFF_REASON_LABELS) as SongOffReason[]).map(r => (
                        <button key={r} onClick={() => toggleReason(r)} style={chip(reasons.includes(r))}>
                            {SONG_OFF_REASON_LABELS[r]}
                        </button>
                    ))}
                </div>
            )}

            <textarea
                value={note}
                onChange={e => setNote(e.target.value.slice(0, 500))}
                placeholder={t("想具体说点什么？（可选，比如「前奏太长」「人声太靠前」）")}
                rows={2}
                style={{
                    width: '100%',
                    resize: 'vertical',
                    padding: '0.5rem 0.6rem',
                    borderRadius: '8px',
                    border: '1px solid rgba(255,255,255,0.14)',
                    background: 'rgba(0,0,0,0.25)',
                    color: theme.colors.text.primary,
                    fontSize: '0.82rem',
                    outline: 'none',
                }}
            />

            <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                <button
                    onClick={submit}
                    disabled={!canSubmit || state === 'sending' || state === 'done'}
                    style={{
                        padding: '0.36rem 0.9rem',
                        borderRadius: '8px',
                        border: 'none',
                        cursor: canSubmit && state === 'idle' ? 'pointer' : 'default',
                        background: state === 'done' ? '#1DB954' : canSubmit ? 'rgba(29,185,84,0.85)' : 'rgba(255,255,255,0.10)',
                        color: canSubmit || state === 'done' ? '#0b0b0b' : theme.colors.text.secondary,
                        fontSize: '0.82rem',
                        fontWeight: 600,
                    }}
                >
                    {state === 'done' ? t('已记录') : state === 'sending' ? t('提交中…') : t('提交')}
                </button>
                <button
                    onClick={onClose}
                    style={{ background: 'none', border: 'none', color: theme.colors.text.secondary, fontSize: '0.8rem', cursor: 'pointer' }}
                >
                    {t('取消')}
                </button>
                {state === 'error' && <span style={{ color: '#ff6b6b', fontSize: '0.78rem' }}>{error}</span>}
            </div>
        </div>
    );
}
