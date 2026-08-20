/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Keep the deployable artifact small and let a single public Next.js process
  // front the internal FastAPI service in ModelScope/Docker deployments.
  output: 'standalone',
  // 如果需要代理到后端 API
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        // 容器里 localhost 是前端容器自己，必须走 compose 服务名 backend。
        // 默认留 localhost，这样宿主机 npm run dev 也能用。后端是 FastAPI，
        // 不是 Streamlit（旧注释遗留）。
        destination: `${process.env.BACKEND_INTERNAL_URL || 'http://localhost:8501'}/api/:path*`,
      },
      {
        // Audio, covers and lyrics stay same-origin in the browser.  Besides
        // avoiding CORS, this is what keeps HTTP Range requests working when
        // only the Next.js port is public (for example a ModelScope Studio).
        source: '/static/:path*',
        destination: `${process.env.BACKEND_INTERNAL_URL || 'http://localhost:8501'}/static/:path*`,
      },
    ];
  },
  // 图片优化配置
  images: {
    remotePatterns: [
      { protocol: 'http', hostname: 'localhost' },
      { protocol: 'http', hostname: '127.0.0.1' },
    ],
    unoptimized: true,
  },
};

module.exports = nextConfig;


