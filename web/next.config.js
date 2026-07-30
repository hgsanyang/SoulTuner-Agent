/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Docker 生产构建时跳过 ESLint（CI 已经单独检查）
  eslint: {
    ignoreDuringBuilds: true,
  },
  // 允许 useSearchParams() 在不包裹 Suspense 的情况下使用（客户端页面）
  experimental: {
    missingSuspenseWithCSRBailout: false,
  },
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
    ];
  },
  // 图片优化配置
  images: {
    domains: ['localhost'],
    unoptimized: process.env.NODE_ENV === 'development',
  },
};

module.exports = nextConfig;


