# Build context is web/:  docker build -f web/Dockerfile web/
ARG NODE_IMAGE=registry.access.redhat.com/ubi8/nodejs-20:latest
FROM ${NODE_IMAGE} AS deps

USER 0
WORKDIR /app
COPY package.json package-lock.json* ./
# `npm ci` fails loudly if the lockfile disagrees with package.json, which is
# what you want when the registry is an internal mirror.
RUN npm ci --no-audit --no-fund


FROM ${NODE_IMAGE} AS builder

USER 0
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .

# Leave NEXT_PUBLIC_API_BASE_URL empty. Empty means the UI uses relative URLs,
# which the server-side proxy in app/api/v1/[...path]/route.ts forwards to
# RAG_API_URL -- a RUNTIME variable, so one image runs in every environment.
#
# Setting it here still works and makes the browser call the API directly, but
# it is inlined into the client bundle at build time: the value becomes part of
# the image, and the API then needs its own public Route plus a CORS entry for
# each UI hostname. Only do that if something requires bypassing the proxy.
ARG NEXT_PUBLIC_API_BASE_URL=""
ENV NEXT_PUBLIC_API_BASE_URL=${NEXT_PUBLIC_API_BASE_URL} \
    NEXT_TELEMETRY_DISABLED=1

RUN npm run build


FROM ${NODE_IMAGE} AS runner

LABEL org.opencontainers.image.title="rag-web"

USER 0
WORKDIR /app
ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=3000 \
    HOSTNAME=0.0.0.0

# The standalone output bundles only the modules actually imported.
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
COPY --from=builder /app/public ./public

RUN chgrp -R 0 /app && chmod -R g=u /app

USER 1001
EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD node -e "require('http').get('http://localhost:3000/',r=>process.exit(r.statusCode<500?0:1)).on('error',()=>process.exit(1))"

CMD ["node", "server.js"]
