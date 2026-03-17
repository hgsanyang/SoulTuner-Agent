'use client';
import React from 'react';
import { PlayerProvider } from '@/context/PlayerContext';
import { LibraryProvider } from '@/context/LibraryContext';
import GlobalPlayer from '@/components/Player/GlobalPlayer';

export default function Providers({ children }: { children: React.ReactNode }) {
    return (
        <PlayerProvider>
            <LibraryProvider>
                {children}
                <GlobalPlayer />
            </LibraryProvider>
        </PlayerProvider>
    );
}
