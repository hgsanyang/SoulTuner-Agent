/**
 * Language switching, keyed by the Chinese source text.
 *
 * Why source-text-as-key instead of invented ids like `slate.great`:
 *  - 489 unique strings would need 489 invented names, each a chance to pick a
 *    bad one and each needing a lookup to read the code.
 *  - A missing entry falls back to the Chinese source, so a half-finished
 *    dictionary means "partly translated", never a blank or a raw key on screen.
 *  - The call sites stay readable: t('整体合适') says what it renders.
 *
 * Default language: NEXT_PUBLIC_DEFAULT_LANG, 'en' when unset (a visitor from
 * GitHub should land in English). A local .env.local can set zh. Whatever the
 * user picks in Settings wins and is remembered.
 */

export type Lang = 'zh' | 'en';

export const LANG_STORAGE_KEY = 'soultuner_lang';

export const LANGUAGES: { value: Lang; label: string }[] = [
    { value: 'zh', label: '中文' },
    { value: 'en', label: 'English' },
];

/** Chinese source -> English. Anything missing renders as the Chinese source. */
export const EN: Record<string, string> = {};

export function defaultLang(): Lang {
    const configured = (process.env.NEXT_PUBLIC_DEFAULT_LANG || 'en').trim().toLowerCase();
    return configured === 'zh' ? 'zh' : 'en';
}

export function readStoredLang(): Lang {
    if (typeof window === 'undefined') return defaultLang();
    const stored = window.localStorage.getItem(LANG_STORAGE_KEY);
    return stored === 'zh' || stored === 'en' ? stored : defaultLang();
}

/**
 * Translate one source string.
 *
 * Interpolation is intentionally absent: call sites build their own strings
 * from template literals, and a translation layer that also owned formatting
 * would need every one of those rewritten. `t()` handles the fixed parts.
 */
export function translate(source: string, lang: Lang): string {
    if (lang === 'zh') return source;
    return EN[source] ?? source;
}
