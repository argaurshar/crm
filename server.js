#!/usr/bin/env node
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { GoogleGenAI } from "@google/genai";
import { readFile, writeFile, mkdir } from "node:fs/promises";
import { resolve, dirname, extname, basename } from "node:path";
import { tmpdir } from "node:os";

const MODEL = process.env.GEMINI_IMAGE_MODEL || "gemini-2.5-flash-image-preview";
const OUTPUT_DIR = process.env.GEMINI_IMAGE_OUTPUT_DIR || tmpdir();

const apiKey = process.env.GEMINI_API_KEY;
if (!apiKey) {
  console.error("GEMINI_API_KEY env var is required.");
  process.exit(1);
}

const ai = new GoogleGenAI({ apiKey });

const MIME_BY_EXT = {
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".webp": "image/webp",
  ".gif": "image/gif",
};

async function loadImagePart(path) {
  const abs = resolve(path);
  const buf = await readFile(abs);
  const ext = extname(abs).toLowerCase();
  const mimeType = MIME_BY_EXT[ext] || "image/png";
  return { inlineData: { data: buf.toString("base64"), mimeType } };
}

async function runImageTool({ prompt, input_images = [], output_path }) {
  const parts = [];
  for (const p of input_images) parts.push(await loadImagePart(p));
  parts.push({ text: prompt });

  const response = await ai.models.generateContent({
    model: MODEL,
    contents: [{ role: "user", parts }],
  });

  const candidate = response.candidates?.[0];
  const outParts = candidate?.content?.parts || [];
  const imgPart = outParts.find((p) => p.inlineData?.data);
  if (!imgPart) {
    const txt = outParts.map((p) => p.text).filter(Boolean).join("\n");
    throw new Error(
      `Model returned no image. Text response: ${txt || "(empty)"}`,
    );
  }

  const outMime = imgPart.inlineData.mimeType || "image/png";
  const outExt = outMime === "image/jpeg" ? ".jpg" : outMime === "image/webp" ? ".webp" : ".png";

  let finalPath;
  if (output_path) {
    finalPath = resolve(output_path);
  } else {
    const stamp = new Date().toISOString().replace(/[:.]/g, "-");
    const base = input_images[0] ? basename(input_images[0], extname(input_images[0])) : "gemini";
    finalPath = resolve(OUTPUT_DIR, `${base}-${stamp}${outExt}`);
  }
  await mkdir(dirname(finalPath), { recursive: true });
  await writeFile(finalPath, Buffer.from(imgPart.inlineData.data, "base64"));

  const textCommentary = outParts.map((p) => p.text).filter(Boolean).join("\n");
  return { path: finalPath, mimeType: outMime, text: textCommentary };
}

const TOOLS = [
  {
    name: "edit_image",
    description:
      "Edit one or more input images with Gemini 2.5 Flash Image (Nano Banana). Provide a prompt describing the change and absolute paths to source images. Returns the path to the saved output image.",
    inputSchema: {
      type: "object",
      properties: {
        prompt: { type: "string", description: "Instruction describing the edit." },
        input_images: {
          type: "array",
          items: { type: "string" },
          description: "Absolute paths to input image files (png/jpg/webp).",
          minItems: 1,
        },
        output_path: {
          type: "string",
          description: "Optional absolute path where the result should be written.",
        },
      },
      required: ["prompt", "input_images"],
    },
  },
  {
    name: "generate_image",
    description:
      "Generate a new image from a text prompt using Gemini 2.5 Flash Image (Nano Banana).",
    inputSchema: {
      type: "object",
      properties: {
        prompt: { type: "string", description: "Description of the image to generate." },
        output_path: {
          type: "string",
          description: "Optional absolute path where the result should be written.",
        },
      },
      required: ["prompt"],
    },
  },
];

const server = new Server(
  { name: "gemini-image-mcp", version: "0.1.0" },
  { capabilities: { tools: {} } },
);

server.setRequestHandler(ListToolsRequestSchema, async () => ({ tools: TOOLS }));

server.setRequestHandler(CallToolRequestSchema, async (req) => {
  const { name, arguments: args } = req.params;
  try {
    if (name === "edit_image" || name === "generate_image") {
      const result = await runImageTool({
        prompt: args.prompt,
        input_images: args.input_images || [],
        output_path: args.output_path,
      });
      const summary = `Saved: ${result.path}${result.text ? `\n\nModel commentary:\n${result.text}` : ""}`;
      return { content: [{ type: "text", text: summary }] };
    }
    return {
      content: [{ type: "text", text: `Unknown tool: ${name}` }],
      isError: true,
    };
  } catch (err) {
    return {
      content: [{ type: "text", text: `Error: ${err.message}` }],
      isError: true,
    };
  }
});

const transport = new StdioServerTransport();
await server.connect(transport);
