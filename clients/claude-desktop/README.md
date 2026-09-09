# Alexandria — Claude Desktop extension (`.mcpb`)

Connects **Claude Desktop** to a remote Alexandria MCP server.

## Why an extension (and not a plain config entry)

Claude Desktop's `claude_desktop_config.json` only launches **stdio** servers, and it
has no Node.js on `PATH` of its own to run one with. Alexandria's `mcp-http` transport
is a remote **streamable-HTTP** server authenticated with a **static bearer token** over
plain HTTP (TLS is terminated by a proxy or skipped inside a Tailscale/WireGuard mesh) —
so the "Add custom connector" UI doesn't fit either, since that path expects OAuth over
HTTPS.

This extension bridges the gap: it bundles [`mcp-remote`](https://www.npmjs.com/package/mcp-remote)
and runs it on Claude Desktop's **built-in Node** to proxy stdio ⇄ the remote endpoint.
`server/index.js` runs `mcp-remote` **in-process** (dynamic `import`) rather than spawning
a child — under the built-in runtime `process.execPath` is the Electron binary, so
re-spawning it would not run the script.

The bearer token is stored by Desktop as a `sensitive` config field and injected via the
`AUTH_TOKEN` environment variable. `mcp-remote` expands `${AUTH_TOKEN}` in the request
header from the environment, so the token never appears in any process's argv.

## Configuration (in Claude Desktop)

After installing, the extension prompts for:

- **Auth token** — contents of `~/.local/share/alexandria/auth_token` on the server.
- **Server URL** — defaults to `http://your-server:8765/mcp`; change for your host. The host must
  be reachable from the machine running Claude Desktop (e.g. Tailscale MagicDNS must
  resolve it).

## Build

Requires Node.js and the [`@anthropic-ai/mcpb`](https://github.com/anthropics/mcpb) packer.

```sh
cd clients/claude-desktop
npm --prefix server install          # fetch mcp-remote into server/node_modules
npx @anthropic-ai/mcpb pack . alexandria.mcpb
```

`node_modules/` and the built `*.mcpb` are gitignored — the bundle is reproducible from
this source.

## Install

Double-click the resulting `alexandria.mcpb`, or in Claude Desktop go to
**Settings → Extensions → Install extension…** and select it. Then enter the token and
enable it.

## Verify the bridge without Desktop

`index.js` is a normal stdio MCP server, so you can drive it with any Node:

```sh
cd clients/claude-desktop/server
npm install
AUTH_TOKEN=<token> ALEXANDRIA_URL=http://your-server:8765/mcp \
  node -e '/* pipe a JSON-RPC initialize line into: */ require("child_process")' \
  # or simply: printf '%s\n' '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"probe","version":"1"}}}' | node index.js
```

A healthy run returns `serverInfo` naming `alexandria`.
