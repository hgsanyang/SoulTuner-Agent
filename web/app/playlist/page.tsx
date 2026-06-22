'use client';

// 此页面使用 useSearchParams，不能静态预渲染
export const dynamic = 'force-dynamic';

import { useCallback, useEffect, useState } from 'react';
import { usePathname, useRouter, useSearchParams } from 'next/navigation';
import MainLayout from '@/components/Layout/MainLayout';
import ThinkingIndicator from '@/components/Content/ThinkingIndicator';
import ResultsDisplay from '@/components/Content/ResultsDisplay';
import { getMockRecommendations, mockDelay } from '@/lib/mockData';
import { theme } from '@/styles/theme';

function PlaylistWelcome({ onPromptClick }: { onPromptClick: (prompt: string) => void }) {
  const prompts = [
    '创建一个适合夜跑的电子歌单',
    '做一张周末做饭听的轻快中文歌单',
    '整理一组写代码时不抢注意力的器乐',
    '来一套不太大众的华语独立歌单',
  ];

  return (
    <div style={{ maxWidth: '820px', margin: '0 auto', padding: '3rem 1.5rem 7rem', color: theme.colors.text.primary }}>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
        <p style={{ margin: 0, color: theme.colors.text.muted, fontSize: '0.82rem', fontWeight: 700, letterSpacing: '0.08em' }}>
          PLAYLIST BUILDER
        </p>
        <h1 style={{ margin: 0, fontSize: 'clamp(2rem, 7vw, 4.5rem)', lineHeight: 1, fontWeight: 800 }}>
          风格编排器
        </h1>
        <p style={{ margin: 0, maxWidth: '38rem', color: theme.colors.text.secondary, fontSize: '1rem', lineHeight: 1.7 }}>
          用一句话定制你的私人歌单，SoulTuner 会按场景、语言、情绪走向和避雷偏好组织曲目。
        </p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '0.75rem', marginTop: '2rem' }}>
        {prompts.map(prompt => (
          <button
            key={prompt}
            onClick={() => onPromptClick(prompt)}
            style={{
              minHeight: '74px',
              padding: '0.85rem 1rem',
              borderRadius: theme.borderRadius.md,
              border: `1px solid ${theme.colors.border.default}`,
              backgroundColor: 'rgba(255,255,255,0.045)',
              color: theme.colors.text.primary,
              cursor: 'pointer',
              textAlign: 'left',
              fontSize: '0.92rem',
              lineHeight: 1.45,
              transition: 'border-color 0.18s, background-color 0.18s, transform 0.18s',
            }}
            onMouseEnter={e => {
              e.currentTarget.style.borderColor = theme.colors.border.focus;
              e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.075)';
              e.currentTarget.style.transform = 'translateY(-1px)';
            }}
            onMouseLeave={e => {
              e.currentTarget.style.borderColor = theme.colors.border.default;
              e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.045)';
              e.currentTarget.style.transform = 'translateY(0)';
            }}
          >
            {prompt}
          </button>
        ))}
      </div>
    </div>
  );
}

export default function PlaylistPage() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<{ response?: string; recommendations?: any[] } | null>(null);
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const seedPrompt = searchParams?.get('prompt');

  const handleSubmit = useCallback(async (value: string) => {
    setLoading(true);
    setResult(null);

    try {
      await mockDelay(1800);

      const mockData = getMockRecommendations(value);
      setResult({
        response: `已为你创建歌单：${value}\n\n${mockData.response}\n\n歌单已保存，你可以随时查看和编辑。`,
        recommendations: mockData.recommendations,
      });
    } catch (error) {
      setResult({
        response: '创建歌单失败，请稍后重试',
      });
      console.error(error);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (!seedPrompt) return;
    handleSubmit(seedPrompt);
    router.replace(pathname);
  }, [seedPrompt, handleSubmit, router, pathname]);

  return (
    <MainLayout
      onInputSubmit={handleSubmit}
      inputPlaceholder="例如：创建一个适合运动的歌单"
      inputDisabled={loading}
    >
      {!result && !loading && <PlaylistWelcome onPromptClick={handleSubmit} />}
      {loading && <ThinkingIndicator />}
      {result && <ResultsDisplay response={result.response} songs={result.recommendations} />}
    </MainLayout>
  );
}

