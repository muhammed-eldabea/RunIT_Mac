/**
 * RunIT Engine Development Tools Extension for PI Agent
 *
 * Provides custom tools for building, testing, and debugging
 * the bare-metal LLM inference engine on Apple Silicon.
 */
import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";
import { Type } from "@sinclair/typebox";
import { execFileSync, ExecFileSyncOptions } from "node:child_process";
import { readdirSync } from "node:fs";
import { join } from "node:path";

function runCommand(
  command: string,
  args: string[],
  cwd: string,
  timeoutMs = 300000
): string {
  const opts: ExecFileSyncOptions = {
    cwd,
    timeout: timeoutMs,
    encoding: "utf-8" as const,
    maxBuffer: 10 * 1024 * 1024,
    stdio: ["pipe", "pipe", "pipe"],
  };
  try {
    const result = execFileSync(command, args, opts);
    return typeof result === "string" ? result : result.toString("utf-8");
  } catch (err: any) {
    const stdout = err.stdout ? err.stdout.toString() : "";
    const stderr = err.stderr ? err.stderr.toString() : "";
    throw new Error(`${stdout}\n${stderr}`.trim() || err.message);
  }
}

export default function (pi: ExtensionAPI) {
  // ── Quick Build Tool ──────────────────────────────────────
  pi.registerTool({
    name: "runit-build",
    label: "Build Engine",
    description:
      "Build the RunIT inference engine workspace. Pass 'release' for optimized build, 'check' for fast type-checking only, or a crate name to build a specific crate.",
    parameters: Type.Object({
      mode: Type.Optional(
        Type.String({
          description:
            'Build mode: "debug" (default), "release", "check", or a crate name like "bare-metal-kernels"',
        })
      ),
    }),
    async execute(_toolCallId, params, _signal, _onUpdate, ctx) {
      const mode = params.mode || "debug";
      let args: string[];

      if (mode === "release") {
        args = ["build", "--release"];
      } else if (mode === "check") {
        args = ["check"];
      } else if (mode === "debug") {
        args = ["build"];
      } else {
        args = ["build", "-p", mode];
      }

      try {
        const output = runCommand("cargo", args, ctx.cwd);
        return {
          content: [
            { type: "text", text: `Build succeeded (${mode}):\n${output}` },
          ],
          details: {},
        };
      } catch (err: any) {
        return {
          content: [
            { type: "text", text: `Build failed (${mode}):\n${err.message}` },
          ],
          details: {},
        };
      }
    },
  });

  // ── Quick Test Tool ───────────────────────────────────────
  pi.registerTool({
    name: "runit-test",
    label: "Run Tests",
    description:
      "Run tests for the RunIT engine. Optionally specify a crate or test name filter.",
    parameters: Type.Object({
      crate: Type.Optional(
        Type.String({
          description:
            'Crate to test: "bare-metal-engine", "bare-metal-kernels", "bare-metal-gguf", or omit for all',
        })
      ),
      filter: Type.Optional(
        Type.String({ description: "Test name filter (passed to cargo test)" })
      ),
      nocapture: Type.Optional(
        Type.Boolean({
          description: "Show stdout/stderr from tests (default: true)",
        })
      ),
    }),
    async execute(_toolCallId, params, _signal, _onUpdate, ctx) {
      const args = ["test"];
      if (params.crate) {
        args.push("-p", params.crate);
      }
      args.push("--");
      if (params.filter) {
        args.push(params.filter);
      }
      if (params.nocapture !== false) {
        args.push("--nocapture");
      }

      try {
        const output = runCommand("cargo", args, ctx.cwd);
        return {
          content: [{ type: "text", text: `Tests passed:\n${output}` }],
          details: {},
        };
      } catch (err: any) {
        return {
          content: [
            { type: "text", text: `Test failures:\n${err.message}` },
          ],
          details: {},
        };
      }
    },
  });

  // ── Shader Compiler Check Tool ────────────────────────────
  pi.registerTool({
    name: "runit-check-shaders",
    label: "Check Metal Shaders",
    description:
      "Compile all Metal shaders to check for syntax errors without building the full project. Fast feedback loop for shader development.",
    parameters: Type.Object({}),
    async execute(_toolCallId, _params, _signal, _onUpdate, ctx) {
      const shaderDir = join(ctx.cwd, "crates", "kernels", "shaders");
      const results: string[] = [];

      try {
        const files = readdirSync(shaderDir).filter((f) =>
          f.endsWith(".metal")
        );

        for (const file of files) {
          const shaderPath = join(shaderDir, file);
          try {
            runCommand(
              "xcrun",
              ["metal", "-c", shaderPath, "-o", "/dev/null", "-std=metal3.0"],
              ctx.cwd,
              30000
            );
            results.push(`pass: ${file}`);
          } catch (err: any) {
            results.push(`FAIL: ${file}:\n${err.message}`);
          }
        }

        const allPassed = results.every((r) => r.startsWith("pass"));
        return {
          content: [
            {
              type: "text",
              text: `Metal Shader Check (${allPassed ? "ALL PASSED" : "FAILURES"}):\n${results.join("\n")}`,
            },
          ],
          details: {},
        };
      } catch (err: any) {
        return {
          content: [
            {
              type: "text",
              text: `Failed to check shaders: ${err.message}`,
            },
          ],
          details: {},
        };
      }
    },
  });

  // ── Quick Generate Tool ───────────────────────────────────
  pi.registerTool({
    name: "runit-generate",
    label: "Run Inference",
    description:
      "Run a quick inference test with the engine. Builds in release mode and generates text.",
    parameters: Type.Object({
      model: Type.String({
        description: "Path to the GGUF model file",
      }),
      prompt: Type.String({
        description: "The prompt to generate from",
      }),
      max_tokens: Type.Optional(
        Type.Number({
          description: "Maximum tokens to generate (default: 50)",
        })
      ),
      temperature: Type.Optional(
        Type.Number({ description: "Sampling temperature (default: 0.7)" })
      ),
    }),
    async execute(_toolCallId, params, _signal, _onUpdate, ctx) {
      const maxTokens = String(params.max_tokens || 50);
      const temp = String(params.temperature ?? 0.7);

      const args = [
        "run",
        "--release",
        "--bin",
        "generate",
        "--",
        "--model",
        params.model,
        "--prompt",
        params.prompt,
        "--max-tokens",
        maxTokens,
        "--temperature",
        temp,
      ];

      try {
        const output = runCommand("cargo", args, ctx.cwd, 600000);
        return {
          content: [{ type: "text", text: `Generation result:\n${output}` }],
          details: {},
        };
      } catch (err: any) {
        return {
          content: [
            { type: "text", text: `Generation failed:\n${err.message}` },
          ],
          details: {},
        };
      }
    },
  });

  // ── Model Inspector Tool ──────────────────────────────────
  pi.registerTool({
    name: "runit-inspect",
    label: "Inspect Model",
    description:
      "Inspect a GGUF model file to see architecture, tensor inventory, quantization types, and memory footprint.",
    parameters: Type.Object({
      model: Type.String({
        description: "Path to the GGUF model file to inspect",
      }),
    }),
    async execute(_toolCallId, params, _signal, _onUpdate, ctx) {
      const args = [
        "run",
        "--release",
        "--bin",
        "inspect",
        "--",
        "--model",
        params.model,
      ];

      try {
        const output = runCommand("cargo", args, ctx.cwd, 120000);
        return {
          content: [{ type: "text", text: `Model inspection:\n${output}` }],
          details: {},
        };
      } catch (err: any) {
        return {
          content: [
            { type: "text", text: `Inspection failed:\n${err.message}` },
          ],
          details: {},
        };
      }
    },
  });

  // ── Session Start: Show project context ───────────────────
  pi.on("session_start", async (_event, ctx) => {
    ctx.ui.notify(
      "RunIT Engine tools loaded — 5 custom tools available",
      "info"
    );
    ctx.ui.setStatus("runit", "RunIT Engine | Rust + Metal | Apple Silicon");
  });

  // ── Command: Quick status ─────────────────────────────────
  pi.registerCommand("runit", {
    description:
      "Show RunIT engine project status (git branch, changed files)",
    handler: async (_args, ctx) => {
      try {
        const gitStatus = runCommand(
          "git",
          ["status", "--short"],
          ctx.cwd,
          10000
        );
        const branch = runCommand(
          "git",
          ["branch", "--show-current"],
          ctx.cwd,
          10000
        ).trim();

        let status = `Branch: ${branch}\n`;
        status += gitStatus
          ? `Changed files:\n${gitStatus}`
          : "Working tree clean\n";

        ctx.ui.notify(status, "info");
      } catch (err: any) {
        ctx.ui.notify(`Error: ${err.message}`, "error");
      }
    },
  });
}
