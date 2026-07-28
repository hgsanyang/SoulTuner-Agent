import type { Metadata } from 'next';
import './globals.css';
import StarryBackground from '@/components/Layout/StarryBackground';
import Providers from '@/components/Providers';

export const metadata: Metadata = {
  // Metadata is rendered on the server before any client language choice is
  // known, so it uses English -- the repo default. A visitor from GitHub sees
  // an English tab title; the in-page language switch handles the rest.
  title: 'SoulTuner · natural-language music recommendations',
  description: 'A natural-language music recommendation agent',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  // LanguageProvider updates this on mount to whatever the user picked;
  // "en" is what the server renders, matching the repo default.
  return (
    <html lang="en">
      <body style={{ margin: 0, padding: 0 }}>
        <Providers>
          <StarryBackground />
          {children}
        </Providers>
      </body>
    </html>
  );
}

