'use client';
import React from 'react';
import { PlayerProvider } from '@/context/PlayerContext';
import { LibraryProvider } from '@/context/LibraryContext';
import { AppSessionProvider } from '@/context/AppSessionContext';
import { LanguageProvider } from '@/context/LanguageContext';
import GlobalPlayer from '@/components/Player/GlobalPlayer';

export default function Providers({ children }: { children: React.ReactNode }) {
    return (
        <LanguageProvider>
            <AppSessionProvider>
                <PlayerProvider>
                    <LibraryProvider>
                        {children}
                        <GlobalPlayer />
                    </LibraryProvider>
                </PlayerProvider>
            </AppSessionProvider>
        </LanguageProvider>
    );
}
