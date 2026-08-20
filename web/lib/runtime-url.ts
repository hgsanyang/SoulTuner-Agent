const LOCAL_BACKEND_HOSTS = new Set(['localhost', '127.0.0.1', '::1']);

function configuredPublicBackend(): string {
  return (process.env.NEXT_PUBLIC_API_URL || '').trim().replace(/\/$/, '');
}

function isLocalBackend(url: URL): boolean {
  return LOCAL_BACKEND_HOSTS.has(url.hostname.toLowerCase()) && url.port === '8501';
}

/**
 * Convert the historical http://localhost:8501 URLs into the browser-visible
 * public endpoint.  In the default configuration that endpoint is the current
 * origin, where Next.js proxies both /api and /static to FastAPI.
 *
 * Absolute third-party URLs are deliberately left untouched: cover art and
 * licensed preview providers may live outside the SoulTuner deployment.
 */
export function resolveBackendUrl(value: string | URL): string {
  const raw = value instanceof URL ? value.toString() : String(value || '').trim();
  if (!raw) return raw;
  if (/^(data|blob):/i.test(raw)) return raw;

  const publicBackend = configuredPublicBackend();
  if (raw.startsWith('/')) {
    return publicBackend ? `${publicBackend}${raw}` : raw;
  }

  try {
    const parsed = new URL(raw);
    if (!isLocalBackend(parsed)) return raw;
    const path = `${parsed.pathname}${parsed.search}${parsed.hash}`;
    return publicBackend ? `${publicBackend}${path}` : path;
  } catch {
    const path = raw.startsWith('/') ? raw : `/${raw}`;
    return publicBackend ? `${publicBackend}${path}` : path;
  }
}

export function resolveOptionalMediaUrl(value?: string | null): string | undefined {
  const clean = String(value || '').trim();
  return clean ? resolveBackendUrl(clean) : undefined;
}
