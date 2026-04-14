# web/

基于 Next.js 14 + TypeScript 构建的音乐推荐系统前端。

## 技术栈

- **Next.js 14** — App Router
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
│   ├── Profile/            # 👤 用户画像面板
│   └── MusicJourney/       # 🗺️ 音乐旅程组件
├── lib/                    # API 客户端与工具函数
├── public/                 # 静态资源
├── next.config.js          # Next.js 配置（API 代理等）
└── package.json            # 依赖管理
```

## 与后端集成

前端通过 `next.config.js` 中的 `rewrites` 将 `/api/*` 代理到后端 `http://localhost:8501`。

**依赖**：后端 API（FastAPI :8501）
