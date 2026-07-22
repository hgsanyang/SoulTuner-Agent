'use client';

import { useCallback, useState } from 'react';
import { fetchCatalogDiagnostics, type CatalogDiagnostics } from '@/lib/api';
import { theme } from '@/styles/theme';

/**
 * 曲库诊断抽屉：开发/管理工具，与用户反馈分离。
 * 由工具栏的「诊断」入口打开，覆盖式抽屉展示曲库分布与最近曝光来源。
 */
export default function DiagnosticsDrawer({
  open,
  onClose,
}: {
  open: boolean;
  onClose: () => void;
}) {
  const [loading, setLoading] = useState(false);
  const [report, setReport] = useState<CatalogDiagnostics | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const data = await fetchCatalogDiagnostics(50);
      setReport(data);
    } catch (err: any) {
      setReport({ success: false, error: err?.message || '曲库诊断失败' });
    } finally {
      setLoading(false);
    }
  }, []);

  if (!open) return null;

  return (
    <div
      onClick={onClose}
      style={{
        position: 'fixed', inset: 0, zIndex: 60,
        backgroundColor: 'rgba(0,0,0,0.45)',
        display: 'flex', justifyContent: 'flex-end',
      }}
    >
      <div
        onClick={e => e.stopPropagation()}
        style={{
          width: 'min(420px, 92vw)',
          height: '100%',
          overflowY: 'auto',
          backgroundColor: 'rgba(24,24,24,0.98)',
          borderLeft: `1px solid ${theme.colors.border.default}`,
          padding: '1.1rem 1.2rem',
          boxSizing: 'border-box',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.9rem' }}>
          <h3 style={{ margin: 0, fontSize: '0.95rem', color: theme.colors.text.primary }}>曲库诊断（开发工具）</h3>
          <button
            onClick={onClose}
            style={{
              border: '1px solid rgba(255,255,255,0.15)', borderRadius: '0.45rem',
              backgroundColor: 'transparent', color: 'rgba(255,255,255,0.65)',
              padding: '0.25rem 0.6rem', fontSize: '0.76rem', cursor: 'pointer',
            }}
          >
            关闭
          </button>
        </div>

        {!report && !loading && (
          <button
            onClick={load}
            style={{
              padding: '0.45rem 0.9rem', borderRadius: '0.55rem',
              border: '1px solid rgba(91,214,170,0.3)',
              backgroundColor: 'rgba(42,170,130,0.14)',
              color: 'rgba(238,255,248,0.9)', fontSize: '0.8rem', cursor: 'pointer',
            }}
          >
            读取曲库分布
          </button>
        )}

        {loading && <div style={{ color: theme.colors.text.muted, fontSize: '0.8rem' }}>正在读取曲库分布...</div>}

        {!loading && report && (
          <div style={{ color: 'rgba(255,255,255,0.72)', fontSize: '0.78rem', lineHeight: 1.6 }}>
            {report.success ? (
              <>
                <div style={{ color: '#fff', fontWeight: 650, marginBottom: '0.45rem' }}>
                  曲库 {report.catalog?.total_songs || 0} 首 · 可播放 {report.catalog?.playable_songs || 0} 首 · 最近曝光 {report.recent_recommendations?.exposures || 0} 批
                </div>
                <div>语言：{(report.catalog?.top?.languages || []).slice(0, 4).map(item => `${item.label} ${Math.round(item.ratio * 100)}%`).join(' / ') || '暂无'}</div>
                <div>流派：{(report.catalog?.top?.genres || []).slice(0, 5).map(item => `${item.label} ${item.count}`).join(' / ') || '暂无'}</div>
                <div>最近来源：{(report.recent_recommendations?.top_recall_sources || []).slice(0, 5).map(item => `${item.label} ${item.count}`).join(' / ') || '暂无曝光'}</div>
                {(report.warnings || []).length > 0 && (
                  <div style={{ marginTop: '0.5rem', color: 'rgba(255,220,160,0.92)' }}>
                    {(report.warnings || []).slice(0, 3).map(w => w.message).join('；')}
                  </div>
                )}
                <button
                  onClick={load}
                  style={{
                    marginTop: '0.8rem', padding: '0.35rem 0.75rem', borderRadius: '0.5rem',
                    border: '1px solid rgba(255,255,255,0.14)', backgroundColor: 'rgba(255,255,255,0.06)',
                    color: 'rgba(255,255,255,0.7)', fontSize: '0.74rem', cursor: 'pointer',
                  }}
                >
                  刷新
                </button>
              </>
            ) : (
              <div style={{ color: '#ff8f8f' }}>{report.error || '曲库诊断失败'}</div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
