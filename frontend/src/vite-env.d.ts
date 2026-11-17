/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Backend base URL in production builds, e.g. https://documind-api.up.railway.app */
  readonly VITE_API_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
