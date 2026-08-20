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
# NEXT_PUBLIC_API_URL 默认为空，浏览器统一使用当前站点的 /api 与 /static；
# 这既适用于本机，也适用于只公开一个端口的 ModelScope Studio。
# BACKEND_INTERNAL_URL 是服务端 rewrite 用的：容器里的 localhost 指向前端容器
# 自己，必须走 compose 服务名。**必须在这里设**——next.config 的 rewrites 会被
# next build 冻进 .next/routes-manifest.json，运行时再给环境变量已经晚了。
ARG NEXT_PUBLIC_API_URL=
ARG BACKEND_INTERNAL_URL=http://backend:8501
ENV NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL}
ENV BACKEND_INTERNAL_URL=${BACKEND_INTERNAL_URL}
RUN npm run build

# ---- Stage 2: 运行 ----
FROM node:22-alpine AS runner
WORKDIR /app

ENV NODE_ENV=production
ARG BACKEND_INTERNAL_URL=http://backend:8501
ENV BACKEND_INTERNAL_URL=${BACKEND_INTERNAL_URL}

# 仅复制必要文件
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
COPY --from=builder /app/public ./public

EXPOSE 3003

ENV PORT=3003
ENV HOSTNAME=0.0.0.0
CMD ["node", "server.js"]
