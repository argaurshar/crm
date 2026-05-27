#!/usr/bin/env node
import { GoogleGenAI } from "@google/genai";
import { readFile, writeFile, mkdir, stat } from "node:fs/promises";
import { resolve, dirname, extname, basename } from "node:path";

const MODEL = process.env.GEMINI_IMAGE_MODEL || "gemini-2.5-flash-image-preview";
const OUTPUT_DIR = process.env.GEMINI_IMAGE_OUTPUT_DIR || "./out";

const MIME_BY_EXT = {
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".webp": "image/webp",
  ".gif": "image/gif",
};

function usage() {
  console.log(`Usage:
  node edit.js <image> <prompt-or-prompt-file> [more-images...] [--out <path>]

Examples:
  node edit.js room.jpg prompt.txt
  node edit.js room.jpg "make the sofa green velvet"
  node edit.js a.jpg b.jpg prompt.txt --out result.png

Env:
  GEMINI_API_KEY            (required)
  GEMINI_IMAGE_MODEL        (default: gemini-2.5-flash-image-preview)
  GEMINI_IMAGE_OUTPUT_DIR   (default: ./out)
`);
}

async function isFile(p) {
  try { return (await stat(p)).isFile(); } catch { return false; }
}

async function loadImagePart(path) {
  const abs = resolve(path);
  const buf = await readFile(abs);
  const ext = extname(abs).toLowerCase();
  return { inlineData: { data: buf.toString("base64"), mimeType: MIME_BY_EXT[ext] || "image/png" } };
}

async function main() {
  const argv = process.argv.slice(2);
  if (argv.length < 2 || argv.includes("-h") || argv.includes("--help")) {
    usage();
    process.exit(argv.length < 2 ? 1 : 0);
  }

  let outPath;
  const outIdx = argv.indexOf("--out");
  if (outIdx !== -1) {
    outPath = argv[outIdx + 1];
    argv.splice(outIdx, 2);
  }

  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    console.error("GEMINI_API_KEY env var is required.");
    process.exit(1);
  }

  const last = argv[argv.length - 1];
  let prompt;
  let images;
  if (await isFile(last) && extname(last).toLowerCase() === ".txt") {
    prompt = (await readFile(resolve(last), "utf8")).trim();
    images = argv.slice(0, -1);
  } else if (await isFile(last) && MIME_BY_EXT[extname(last).toLowerCase()]) {
    console.error("Last argument looks like an image — provide the prompt as the final argument (string or .txt file).");
    process.exit(1);
  } else {
    prompt = last;
    images = argv.slice(0, -1);
  }

  if (!images.length) {
    console.error("At least one input image is required.");
    process.exit(1);
  }
  if (!prompt) {
    console.error("Prompt is empty.");
    process.exit(1);
  }

  const ai = new GoogleGenAI({ apiKey });
  const parts = [];
  for (const p of images) parts.push(await loadImagePart(p));
  parts.push({ text: prompt });

  process.stderr.write(`→ ${MODEL} | ${images.length} image(s) | prompt ${prompt.length} chars\n`);

  const response = await ai.models.generateContent({
    model: MODEL,
    contents: [{ role: "user", parts }],
  });

  const outParts = response.candidates?.[0]?.content?.parts || [];
  const imgPart = outParts.find((p) => p.inlineData?.data);
  const text = outParts.map((p) => p.text).filter(Boolean).join("\n");

  if (!imgPart) {
    console.error(`No image returned. Model text:\n${text || "(empty)"}`);
    process.exit(2);
  }

  const outMime = imgPart.inlineData.mimeType || "image/png";
  const outExt = outMime === "image/jpeg" ? ".jpg" : outMime === "image/webp" ? ".webp" : ".png";

  let finalPath;
  if (outPath) {
    finalPath = resolve(outPath);
  } else {
    const stamp = new Date().toISOString().replace(/[:.]/g, "-");
    const base = basename(images[0], extname(images[0]));
    finalPath = resolve(OUTPUT_DIR, `${base}-${stamp}${outExt}`);
  }
  await mkdir(dirname(finalPath), { recursive: true });
  await writeFile(finalPath, Buffer.from(imgPart.inlineData.data, "base64"));

  if (text) process.stderr.write(`\nModel commentary:\n${text}\n`);
  process.stdout.write(`${finalPath}\n`);
}

main().catch((err) => {
  console.error(`Error: ${err.message}`);
  process.exit(1);
});
