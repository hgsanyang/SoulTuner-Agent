export const mockDelay = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

export function getMockRecommendations(query: string) {
    return {
        response: `为你推荐了适合 "${query}" 的音乐。这些歌曲节奏鲜明，能够帮助你更好地进入状态。`,
        recommendations: [
            {
                id: '1',
                title: 'Bohemian Rhapsody',
                artist: 'Queen',
                album: 'A Night at the Opera',
                coverUrl: 'https://i.scdn.co/image/ab67616d0000b273e8b066f70c206551210d902b',
                previewUrl: '',
                durationMs: 354000,
                popularity: 85,
                reason: '经典之作，旋律跌宕起伏。',
            },
            {
                id: '2',
                title: 'Shape of You',
                artist: 'Ed Sheeran',
                album: '÷ (Divide)',
                coverUrl: 'https://i.scdn.co/image/ab67616d0000b273ba5db46f4b838ef6027e6f96',
                previewUrl: '',
                durationMs: 233000,
                popularity: 88,
                reason: '节奏轻快，非常适合当前的氛围。',
            },
            {
                id: '3',
                title: 'Blinding Lights',
                artist: 'The Weeknd',
                album: 'After Hours',
                coverUrl: 'https://i.scdn.co/image/ab67616d0000b2738863bc11d2aa12b54f5aeb36',
                previewUrl: '',
                durationMs: 200000,
                popularity: 92,
                reason: '合成器流行风格，充满动感。',
            },
        ],
    };
}
