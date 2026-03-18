'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { usePathname, useRouter, useSearchParams } from 'next/navigation';
import MainLayout from '@/components/Layout/MainLayout';
import WelcomeScreen from '@/components/Content/WelcomeScreen';
import ThinkingIndicator from '@/components/Content/ThinkingIndicator';
import ResultsDisplay from '@/components/Content/ResultsDisplay';
import { searchMusic } from '@/lib/api';

export default function SearchPage() {
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<any[]>([]);
  const [error, setError] = useState<string | null>(null);
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const seedPrompt = searchParams?.get('prompt');

  const handleSubmit = useCallback(async (value: string) => {
    const trimmed = value.trim();
    if (!trimmed) return;

    setLoading(true);
    setResults([]);
    setError(null);

    try {
      const parts = trimmed.split(/流派|genre/i);
      const query = parts[0].trim();
      const genre = parts[1]?.trim() || undefined;

      const data = await searchMusic(query, genre);
      const songs = data?.songs || data?.results || [];
      setResults(songs);
    } catch (err: any) {
      console.error('搜索失败', err);
      setError(err?.message || '搜索失败，请稍后重试');
      setResults([]);
    } finally {
      setLoading(false);
    }
  }, []);

  const seedHandledRef = useRef<string | null>(null);
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const prompt = params.get('prompt');
    if (!prompt || seedHandledRef.current === prompt) return;
    seedHandledRef.current = prompt;
    const timer = setTimeout(() => {
      handleSubmit(prompt);
      window.history.replaceState({}, '', pathname);
    }, 150);
    return () => clearTimeout(timer);
  }, []);  // eslint-disable-line react-hooks/exhaustive-deps

  const hasResults = results.length > 0;

  return (
    <MainLayout
      onInputSubmit={handleSubmit}
      inputPlaceholder="例如：根据周杰伦在网上为我推荐一些相似风格的歌曲；或 周杰伦 流派：流行"
      inputDisabled={loading}
    >
      {!hasResults && !loading && !error && <WelcomeScreen onPromptClick={handleSubmit} />}
      {loading && <ThinkingIndicator message="正在从网上为你查找和推荐歌曲..." />}
      {error && (
        <div
          style={{
            padding: '1rem',
            margin: '1rem',
            backgroundColor: '#fee2e2',
            color: '#991b1b',
            borderRadius: '0.5rem',
            border: '1px solid #fecaca',
          }}
        >
          {error}
        </div>
      )}
      {hasResults && <ResultsDisplay songs={results} />}
    </MainLayout>
  );
}

