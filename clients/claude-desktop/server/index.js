#!/usr/bin/env node
// Alexandria MCP bridge for Claude Desktop.
//
// Claude Desktop only speaks stdio to config-defined servers, but Alexandria is a
// remote streamable-HTTP server guarded by a static bearer token (no OAuth, no TLS).
// This wrapper runs mcp-remote to proxy stdio <-> the remote HTTP endpoint.
//
// It runs mcp-remote IN-PROCESS (dynamic import), not as a child process. Under
// Claude Desktop's built-in Node, process.execPath is the Electron binary, so
// re-spawning it does NOT run node on a script — the child silently fails to start.
// Importing keeps everything in this one process, whose stdin/stdout are already the
// pipes Desktop expects.
//
// The token arrives in AUTH_TOKEN (injected by Desktop from user_config). mcp-remote
// expands ${AUTH_TOKEN} in header values from the environment, so the real token never
// appears in any process's argv. The literal ${AUTH_TOKEN} below lives in JS source,
// which Desktop does not run variable substitution over (that only applies to
// manifest.json), so it reaches mcp-remote intact.

const { pathToFileURL } = require('node:url');

const url = process.env.ALEXANDRIA_URL || 'http://your-server:8765/mcp';

if (!process.env.AUTH_TOKEN) {
  console.error('[alexandria] AUTH_TOKEN is not set — set the token in the extension settings.');
  process.exit(1);
}

const proxyPath = require.resolve('mcp-remote/dist/proxy.js');

// mcp-remote reads process.argv.slice(2).
process.argv = [
  process.argv[0],
  proxyPath,
  url,
  '--transport', 'http-only', // Alexandria has no SSE endpoint
  '--allow-http',             // remote endpoint is plain HTTP (e.g. over a Tailscale/WireGuard mesh), no TLS
  '--header', 'Authorization: Bearer ${AUTH_TOKEN}', // expanded from env by mcp-remote
];

// pathToFileURL is required for Windows paths (drive letter / backslashes) to import cleanly.
import(pathToFileURL(proxyPath).href).catch((err) => {
  console.error('[alexandria] failed to start mcp-remote:', (err && err.stack) || err);
  process.exit(1);
});
