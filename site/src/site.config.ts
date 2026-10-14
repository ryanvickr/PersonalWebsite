// Everything personal lives here. Edit this file (and src/content/) to make
// the site yours; nothing else needs to change.
export const site = {
  name: "Ryan Vickramasinghe",
  handle: "ryanvickr",
  title: "Ryan Vickramasinghe: Software Engineer",
  description:
    "Software engineer writing fast C++ for markets and machine learning. Includes a transformer inference engine running live in your browser.",
  headline: "I write fast C++ for markets and machine learning.",
  tagline: "C++ · low-latency systems · ML inference",
  intro:
    "Software engineer focused on performance-critical systems. I care about memory layout, cache behaviour and measuring before optimising, and I'm looking for roles in trading and AI infrastructure.",
  links: {
    github: "https://github.com/ryanvickr",
    linkedin: "https://www.linkedin.com/in/ryanvickr", // TODO: confirm URL
    email: "ryanvickr@gmail.com",
    resume: "Ryan-Vickramasinghe-Resume.pdf", // relative to the site base; drop a PDF in public/
  },
  sourceRepo: "https://github.com/ryanvickr/PersonalWebsite",
};

// Models offered in the demo. Files are produced by tools/export.py into
// public/models/ (CI does this; see .github/workflows/deploy.yml).
export const models = [
  {
    id: "stories15M",
    file: "models/stories15M.tlm",
    label: "TinyStories 15M",
    detail: "6 layers · 288 dim · 32k vocab",
    sizeMB: 17.5,
  },
  {
    id: "stories260K",
    file: "models/stories260K.tlm",
    label: "TinyStories 260K",
    detail: "5 layers · 64 dim · 512 vocab",
    sizeMB: 0.3,
  },
] as const;

export const skills = [
  { group: "Languages", items: ["C++20/23", "Python", "TypeScript", "SQL"] },
  { group: "Performance", items: ["SIMD (AVX2, WASM)", "Cache-aware data layout", "Lock-free concurrency", "perf / VTune / flamegraphs"] },
  { group: "ML systems", items: ["Transformer inference", "Quantization", "PyTorch", "NumPy"] },
  { group: "Tooling", items: ["CMake", "Linux", "Emscripten / WebAssembly", "GitHub Actions"] },
];
