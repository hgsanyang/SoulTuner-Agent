'use client';

import React, { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';
import { Lang, LANG_STORAGE_KEY, defaultLang, readStoredLang, translate } from '@/lib/i18n';

type LanguageValue = {
    lang: Lang;
    setLang: (next: Lang) => void;
    t: (source: string) => string;
};

const LanguageContext = createContext<LanguageValue | null>(null);

export function LanguageProvider({ children }: { children: React.ReactNode }) {
    // Start from the build-time default on both server and first client render;
    // reading localStorage during render would make the two disagree and React
    // would blow away the markup. The stored choice is applied in an effect.
    const [lang, setLangState] = useState<Lang>(defaultLang);

    useEffect(() => { setLangState(readStoredLang()); }, []);

    const setLang = useCallback((next: Lang) => {
        setLangState(next);
        try { window.localStorage.setItem(LANG_STORAGE_KEY, next); } catch { /* private mode */ }
        document.documentElement.lang = next === 'zh' ? 'zh-CN' : 'en';
    }, []);

    useEffect(() => {
        document.documentElement.lang = lang === 'zh' ? 'zh-CN' : 'en';
    }, [lang]);

    const value = useMemo<LanguageValue>(() => ({
        lang,
        setLang,
        t: (source: string) => translate(source, lang),
    }), [lang, setLang]);

    return <LanguageContext.Provider value={value}>{children}</LanguageContext.Provider>;
}

/**
 * Usable outside the provider on purpose: components render in tests and in
 * isolated stories without it, and a missing provider should not crash the page
 * — it should just render the Chinese source.
 */
export function useLang(): LanguageValue {
    const ctx = useContext(LanguageContext);
    if (ctx) return ctx;
    return { lang: 'zh', setLang: () => {}, t: (source: string) => source };
}
