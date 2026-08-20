# web/

基于 Next.js 16 + TypeScript 构建的音乐推荐系统前端。

## 技术栈

- **Next.js 16** — App Router
- **TypeScript** — 类型安全
- **React 18** — UI 框架

## 快速开始

```bash
npm install
npm run dev
# 访问 http://localhost:3003
```

## 项目结构

```
web/
├── app/                    # Next.js App Router
│   ├── layout.tsx          # 根布局（全局音频播放器）
│   ├── page.tsx            # 首页
│   ├── library/            # 曲库页面
│   │   ├── staging/        # 待入库暂存区
│   │   ├── my-library/     # 我的曲库
│   │   ├── favorites/      # 喜欢的歌
│   │   └── saved/          # 收藏的歌
│   └── globals.css         # 全局样式
├── components/
│   ├── Chat/               # 对话消息与歌曲卡片
│   ├── Navigation/         # 侧边栏导航
│   ├── Player/             # 全局音频播放器
│   ├── Settings/           # ⚙️ 运行时设置面板
│   └── Profile/            # 👤 用户画像面板
├── lib/                    # API 客户端与工具函数
├── public/                 # 静态资源
├── next.config.js          # Next.js 配置（API 代理等）
└── package.json            # 依赖管理
```

## 与后端集成

前端通过 `next.config.js` 中的 `rewrites` 将同源 `/api/*` 与 `/static/*`
代理到 FastAPI。推荐、SSE、音频、封面和歌词因此只需要一个公开入口；音频的
`Range` 请求也会原样转发，支持拖动进度和按需加载。浏览器不再依赖访问者机器上的
`localhost:8501`。

本机开发默认代理到 `http://localhost:8501`。容器构建必须把
`BACKEND_INTERNAL_URL` 设为容器内地址（Compose 为 `http://backend:8501`；
单容器创空间为 `http://127.0.0.1:8501`）。除非前后端确实位于不同公开域名，
`NEXT_PUBLIC_API_URL` 应保持为空，使用同源访问。

**依赖**：后端 API（FastAPI :8501）
