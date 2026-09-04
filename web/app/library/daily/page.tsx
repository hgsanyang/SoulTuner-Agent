'use client';
/* eslint-disable @next/next/no-img-element -- QR data URIs must render directly in the browser. */

/**
 * 网易云日推 → 本地曲库对账
 *
 * 只读元数据：歌名、歌手、专辑。不经过任何音频。
 * 每首歌标出它在本地处于哪一层：已在曲库 / 只是临时缓存 / 本地还没有。
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import { useRouter } from 'next/navigation';
import { useLang } from '@/context/LanguageContext';
import { useLibrary } from '@/context/LibraryContext';
import { theme } from '@/styles/theme';
import {
  fetchNeteaseAccount,
  fetchNeteaseDaily,
  startNeteaseQrLogin,
  checkNeteaseQrLogin,
  logoutNetease,
  type NeteaseDailyResult,
  type NeteaseDailySong,
} from '@/lib/api';

type Shelf = 'in_library' | 'in_candidates' | 'missing';

export default function DailyRecommendationsPage() {
  const { t } = useLang();
  const { showToast } = useLibrary();
  const router = useRouter();

  const [account, setAccount] = useState<{ logged_in: boolean; nickname?: string; stale_session?: boolean }>({ logged_in: false });
  const [daily, setDaily] = useState<NeteaseDailyResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [qr, setQr] = useState<{ key: string; image: string } | null>(null);
  const [qrStatus, setQrStatus] = useState('');
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  const load = useCallback(async () => {
    setLoading(true);
    const who = await fetchNeteaseAccount();
    setAccount(who);
    setDaily(who.logged_in ? await fetchNeteaseDaily(30) : null);
    setLoading(false);
  }, []);

  useEffect(() => { load(); }, [load]);
  // 离开页面时一定要停掉轮询，否则它会一直打后端直到刷新。
  useEffect(() => stopPolling, [stopPolling]);

  const beginLogin = async () => {
    stopPolling();
    setQrStatus(t('正在生成二维码...'));
    const result = await startNeteaseQrLogin();
    if (!result.success || !result.key || !result.qr_image) {
      setQrStatus('');
      showToast(`❌ ${result.error || t('无法生成二维码')}`);
      return;
    }
    setQr({ key: result.key, image: result.qr_image });
    setQrStatus(t('用网易云音乐 App 扫码'));

    pollRef.current = setInterval(async () => {
      const check = await checkNeteaseQrLogin(result.key!);
      if (check.status === 'scanned') {
        setQrStatus(t('已扫码，请在手机上确认'));
      } else if (check.status === 'confirmed') {
        stopPolling();
        setQr(null);
        setQrStatus('');
        showToast(t('✅ 网易云账号已连接'));
        load();
      } else if (check.status === 'expired') {
        stopPolling();
        setQr(null);
        setQrStatus(t('二维码已过期，请重新生成'));
      }
    }, 2000);
  };

  const disconnect = async () => {
    if (!window.confirm(t('断开后将无法读取日推，需要重新扫码。继续？'))) return;
    await logoutNetease();
    setDaily(null);
    setAccount({ logged_in: false });
    showToast(t('已断开网易云账号'));
  };

  const shelfOf = (song: NeteaseDailySong): Shelf => {
    if (!daily) return 'missing';
    const id = song.song_id;
    if (daily.in_library.some(s => s.song_id === id)) return 'in_library';
    if (daily.in_candidates.some(s => s.song_id === id)) return 'in_candidates';
    return 'missing';
  };

  const SHELF_STYLE: Record<Shelf, { label: string; color: string; bg: string }> = {
    in_library: { label: t('已在曲库'), color: '#1db954', bg: 'rgba(29,185,84,0.14)' },
    in_candidates: { label: t('临时缓存'), color: '#60a5fa', bg: 'rgba(96,165,250,0.14)' },
    missing: { label: t('本地没有'), color: '#f0a040', bg: 'rgba(240,160,64,0.14)' },
  };

  const card = {
    border: `1px solid ${theme.colors.border.default}`,
    borderRadius: theme.borderRadius.md,
    background: 'rgba(255,255,255,0.03)',
    padding: '1rem 1.1rem',
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem', padding: '1rem', color: theme.colors.text.primary, minHeight: '100%' }}>
      <button onClick={() => router.back()} style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', background: 'none', border: 'none', color: theme.colors.text.secondary, cursor: 'pointer', fontSize: '0.9rem', fontWeight: 500, width: 'fit-content' }}>
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><polyline points="15 18 9 12 15 6" /></svg>
        {t('返回')}
      </button>

      <div>
        <p style={{ margin: 0, fontSize: '0.8rem', fontWeight: 600, letterSpacing: '0.05em', color: theme.colors.text.muted }}>{t('网易云')}</p>
        <h1 style={{ margin: '0.2rem 0', fontSize: '2.2rem', fontWeight: 800, letterSpacing: '-0.02em' }}>{t('每日推荐')}</h1>
        <p style={{ margin: 0, fontSize: '0.88rem', color: theme.colors.text.secondary, maxWidth: '46rem' }}>
          {t('读取你自己账号的日推列表，并标出每首歌在本地处于哪一层。只读歌名歌手等元数据，不涉及音频文件。')}
        </p>
      </div>

      {/* 账号状态 */}
      <div style={{ ...card, display: 'flex', flexWrap: 'wrap', gap: '0.9rem', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ fontSize: '0.86rem', color: theme.colors.text.secondary }}>
          {loading ? t('加载中...')
            : account.logged_in
              ? t('已连接：{v0}', { v0: account.nickname || t('（未获取昵称）') })
              : account.stale_session
                ? t('会话已失效，需要重新扫码')
                : t('未连接网易云账号')}
        </div>
        <div style={{ display: 'flex', gap: '0.5rem' }}>
          {account.logged_in ? (
            <>
              <button onClick={load} style={btn()}>{t('刷新日推')}</button>
              <button onClick={disconnect} style={{ ...btn(), color: '#f06060', borderColor: 'rgba(240,96,96,0.4)' }}>{t('断开连接')}</button>
            </>
          ) : (
            <button onClick={beginLogin} style={{ ...btn(), color: theme.colors.primary.accent, borderColor: theme.colors.primary.accent }}>
              {t('扫码连接')}
            </button>
          )}
        </div>
      </div>

      {/* 二维码 */}
      {(qr || qrStatus) && (
        <div style={{ ...card, display: 'flex', gap: '1.2rem', alignItems: 'center', flexWrap: 'wrap' }}>
          {qr && (
            // 后端直接返回 data: URI，二维码不落盘也不经第三方
            <img src={qr.image} alt={t('网易云登录二维码')} width={180} height={180}
                 style={{ borderRadius: theme.borderRadius.sm, background: '#fff', padding: '0.4rem' }} />
          )}
          <div style={{ fontSize: '0.86rem', color: theme.colors.text.secondary, maxWidth: '24rem' }}>
            <p style={{ margin: '0 0 0.5rem' }}>{qrStatus}</p>
            <p style={{ margin: 0, fontSize: '0.78rem', color: theme.colors.text.muted }}>
              {t('登录凭证只保存在后端本机文件里，不会出现在页面、响应体或日志中。')}
            </p>
          </div>
        </div>
      )}

      {/* 对账结果 */}
      {daily && daily.counts.total > 0 && (
        <>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.6rem' }}>
            {(['in_library', 'in_candidates', 'missing'] as Shelf[]).map(shelf => (
              <div key={shelf} style={{
                padding: '0.45rem 0.9rem', borderRadius: theme.borderRadius.sm,
                background: SHELF_STYLE[shelf].bg, color: SHELF_STYLE[shelf].color,
                fontSize: '0.82rem', fontWeight: 600,
              }}>
                {SHELF_STYLE[shelf].label} {daily.counts[shelf]}
              </div>
            ))}
            <div style={{ padding: '0.45rem 0.9rem', fontSize: '0.82rem', color: theme.colors.text.muted }}>
              {t('共 {v0} 首', { v0: daily.counts.total })}
            </div>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
            {daily.songs.map(song => {
              const shelf = shelfOf(song);
              const style = SHELF_STYLE[shelf];
              return (
                <div key={song.song_id} style={{
                  display: 'flex', alignItems: 'center', gap: '0.9rem',
                  padding: '0.7rem 0.9rem',
                  borderRadius: theme.borderRadius.sm,
                  background: 'rgba(255,255,255,0.025)',
                  border: `1px solid ${theme.colors.border.default}`,
                }}>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ fontSize: '0.9rem', fontWeight: 600, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {song.title}
                    </div>
                    <div style={{ fontSize: '0.78rem', color: theme.colors.text.muted, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {song.artist}{song.album ? ` · ${song.album}` : ''}
                    </div>
                  </div>
                  <span style={{
                    flexShrink: 0, padding: '0.22rem 0.6rem',
                    borderRadius: theme.borderRadius.sm,
                    background: style.bg, color: style.color,
                    fontSize: '0.74rem', fontWeight: 600,
                  }}>
                    {style.label}
                  </span>
                </div>
              );
            })}
          </div>

          {daily.counts.missing > 0 && (
            <p style={{ fontSize: '0.8rem', color: theme.colors.text.muted, margin: 0, maxWidth: '48rem' }}>
              {t('标为「本地没有」的歌，可以在推荐里直接搜歌名让联网通道去取；能取到多少取决于你账号对这首歌的权限，取不到的会如实报「音源获取失败」，不会绕过。')}
            </p>
          )}
        </>
      )}

      {account.logged_in && daily && daily.counts.total === 0 && !loading && (
        <div style={{ ...card, fontSize: '0.86rem', color: theme.colors.text.muted }}>
          {t('今天没有拿到日推。可能是接口变了或者今天还没生成 —— 这是补充源，取不到不影响其他功能。')}
        </div>
      )}
    </div>
  );

  function btn() {
    return {
      padding: '0.45rem 0.9rem',
      borderRadius: theme.borderRadius.sm,
      border: `1px solid ${theme.colors.border.default}`,
      background: 'rgba(255,255,255,0.04)',
      color: theme.colors.text.secondary,
      cursor: 'pointer',
      fontSize: '0.8rem',
    } as const;
  }
}
