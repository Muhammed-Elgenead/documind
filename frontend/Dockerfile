# Frontend images.
#   dev:  Vite dev server with hot reload (docker-compose.yml)
#   prod: static build served by nginx, proxying /api to the backend (docker-compose.prod.yml)
#
# On Vercel none of this is used: Vercel builds the static site itself (see vercel.json).

FROM node:22-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

FROM deps AS dev
COPY . .
EXPOSE 5173
CMD ["npm", "run", "dev"]

FROM deps AS build
COPY . .
# "/api" = same origin, proxied by nginx below. Set a full URL to call the API directly.
ARG VITE_API_URL=/api
ENV VITE_API_URL=${VITE_API_URL}
RUN npm run build

FROM nginx:1.27-alpine AS prod
# The image renders templates with envsubst at start-up (only for variables that are set,
# so nginx's own $variables are left alone).
ENV BACKEND_URL=http://backend:8000
COPY nginx.conf.template /etc/nginx/templates/default.conf.template
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 8080
