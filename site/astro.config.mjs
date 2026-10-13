import { defineConfig } from "astro/config";

// Served from https://<user>.github.io/<repo>/ on GitHub Pages. With a custom
// domain, set SITE_URL to it and BASE_PATH to "/".
export default defineConfig({
  site: process.env.SITE_URL ?? "https://ryanvickr.github.io",
  base: process.env.BASE_PATH ?? "/PersonalWebsite",
  trailingSlash: "ignore",
  build: { assets: "_assets" },
});
