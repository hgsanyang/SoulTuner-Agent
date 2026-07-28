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

import { EN_DICT } from './i18n-en';

export type Lang = 'zh' | 'en';

export const LANG_STORAGE_KEY = 'soultuner_lang';

export const LANGUAGES: { value: Lang; label: string }[] = [
    { value: 'zh', label: '中文' },
    { value: 'en', label: 'English' },
];

/** Chinese source -> English. Anything missing renders as the Chinese source. */
export const EN: Record<string, string> = EN_DICT;

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
 * Translate one source string, optionally filling {name} placeholders.
 *
 * Interpolation has to live here, not at the call site: word order differs.
 * "这组 15 首推荐怎么样？" is "How was this set of 15?" — the number lands in a
 * different place, so splitting the sentence around the value in JSX and
 * translating the pieces produces broken English. The whole sentence is one
 * key with a placeholder instead.
 *
 *   t('这组 {n} 首推荐怎么样？', { n: songCount })
 *
 * A placeholder with no matching value is left as-is rather than blanked, so a
 * mistake shows up as `{n}` on screen instead of a silently missing number.
 */
export function translate(
    source: string,
    lang: Lang,
    vars?: Record<string, string | number>,
): string {
    const text = lang === 'zh' ? source : (EN[source] ?? source);
    if (!vars) return text;
    return text.replace(/\{(\w+)\}/g, (whole, name) =>
        name in vars ? String(vars[name]) : whole);
}
