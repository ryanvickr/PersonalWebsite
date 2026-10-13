import { defineCollection } from "astro:content";
import { glob } from "astro/loaders";
import { z } from "astro/zod";

const projects = defineCollection({
  loader: glob({ pattern: "**/*.md", base: "./src/content/projects" }),
  schema: z.object({
    title: z.string(),
    summary: z.string(),
    stack: z.array(z.string()),
    metric: z.object({ value: z.string(), label: z.string() }).optional(),
    repo: z.url().optional(),
    order: z.number().default(100),
    featured: z.boolean().default(false),
    placeholder: z.boolean().default(false),
  }),
});

const experience = defineCollection({
  loader: glob({ pattern: "**/*.md", base: "./src/content/experience" }),
  schema: z.object({
    role: z.string(),
    org: z.string(),
    start: z.string(),
    end: z.string().default("Present"),
    location: z.string().optional(),
    highlights: z.array(z.string()).default([]),
    order: z.number(),
    placeholder: z.boolean().default(false),
  }),
});

export const collections = { projects, experience };
