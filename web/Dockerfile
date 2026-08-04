# ============================================================
# SoulTuner-Agent Next.js 前端 Dockerfile
# 多阶段构建：builder 编译 → runner 仅运行产物
# ============================================================

# ---- Stage 1: 构建 ----
FROM node:22-alpine AS builder
WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .

# 构建时需要的环境变量。
# NEXT_PUBLIC_API_URL 是浏览器侧用的：浏览器跑在宿主机上，所以是 localhost。
# BACKEND_INTERNAL_URL 是服务端 rewrite 用的：容器里的 localhost 指向前端容器
# 自己，必须走 compose 服务名。**必须在这里设**——next.config 的 rewrites 会被
# next build 冻进 .next/routes-manifest.json，运行时再给环境变量已经晚了。
ENV NEXT_PUBLIC_API_URL=http://localhost:8501
ENV BACKEND_INTERNAL_URL=http://backend:8501
RUN npm run build

# ---- Stage 2: 运行 ----
FROM node:22-alpine AS runner
WORKDIR /app

ENV NODE_ENV=production

# 仅复制必要文件
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json ./
COPY --from=builder /app/public ./public
COPY --from=builder /app/next.config.js ./

EXPOSE 3003

CMD ["npx", "next", "start", "-p", "3003"]
