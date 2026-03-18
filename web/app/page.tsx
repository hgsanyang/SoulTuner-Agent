'use client';

import { useRouter } from 'next/navigation';
import ProductIntro from '@/components/Landing/ProductIntro';
import { theme } from '@/styles/theme';

export default function Home() {
  const router = useRouter();

  const navigateWithPrompt = (prompt: string) => {
    sessionStorage.setItem('seed_prompt', prompt);
    router.push('/recommendations');
  };

  return (
    <div
      style={{
        minHeight: '100vh',
        backgroundColor: 'transparent',
        display: 'flex',
        justifyContent: 'center',
        padding: '3rem 1.5rem 4rem',
      }}
    >
      <main
        style={{
          width: '100%',
          maxWidth: `${theme.layout.maxWidth}px`,
        }}
      >
        <ProductIntro
          onPrimaryAction={() => router.push('/recommendations')}
          onSecondaryAction={() => router.push('/search')}
          onQuickExampleSelect={navigateWithPrompt}
        />
      </main>
    </div>
  );
}

