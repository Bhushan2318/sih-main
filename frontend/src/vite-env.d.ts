/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_BASE_URL?: string;
  readonly VITE_WS_URL?: string;
  /** "false" hides the upload -> retrain panel. Set on a serving-only host,
      where a retrain would exceed the box's memory. */
  readonly VITE_ENABLE_UPLOAD?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
