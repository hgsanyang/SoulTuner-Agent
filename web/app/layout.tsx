import type { Metadata } from 'next';
import './globals.css';
import StarryBackground from '@/components/Layout/StarryBackground';
import Providers from '@/components/Providers';

export const metadata: Metadata = {
  title: 'Deep Search · 音乐体验工作台',
  description: '用自然语言规划歌单、检索声音、串联旅程的音乐平台',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="zh-CN">
      <body style={{ margin: 0, padding: 0 }}>
        <Providers>
          <StarryBackground />
          {children}
        </Providers>
      </body>
    </html>
  );
}

