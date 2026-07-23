// NOTE: the browser's /api/v1/* calls are proxied to the API server-side by the
// route handler in app/api/v1/[...path]/route.ts, NOT by a rewrite here.
// Rewrites are evaluated at build time and frozen into routes-manifest.json, so
// a rewrite reading process.env would bake the build machine's value into the
// image and silently ignore the ConfigMap. Verified, not assumed.

/** @type {import('next').NextConfig} */
const nextConfig = {
  // Produces .next/standalone with only the needed node_modules, which keeps the
  // runtime image small and avoids shipping the full dependency tree.
  output: "standalone",
  reactStrictMode: true,

  // Next phones home with anonymous usage data by default. There is no config
  // key for this -- it is controlled by NEXT_TELEMETRY_DISABLED, which is set in
  // the Dockerfile for both the build and runtime stages. An air-gapped
  // deployment must make no outbound connections at all, and the attempt itself
  // would show up in egress monitoring as a finding.

  // No next/image anywhere in this app, so the image optimizer is dead weight.
  images: { unoptimized: true },

  // sharp carries unpatched high-severity libvips CVEs and would fail an image
  // scan. It cannot simply be skipped at install time (`--omit=optional` would
  // also drop @next/swc-*, and the build would then try to DOWNLOAD the SWC
  // binary -- fatal if images are ever built inside the network). So: install
  // everything, and exclude sharp from the traced standalone output instead.
  // Safe only because images.unoptimized is set above.
  outputFileTracingExcludes: {
    "*": ["node_modules/sharp/**", "node_modules/@img/**"],
  },

  async headers() {
    return [
      {
        source: "/:path*",
        headers: [
          { key: "X-Content-Type-Options", value: "nosniff" },
          { key: "X-Frame-Options", value: "DENY" },
          { key: "Referrer-Policy", value: "no-referrer" },
        ],
      },
    ];
  },
};

export default nextConfig;
