# Security policy

## The threat model

Lamplighter runs a FastAPI server on a daemon thread **inside your Jupyter
kernel**, bound to `127.0.0.1`, with **no authentication**. That is deliberate:
it is a single-user local tool that drives the kernel it lives in, and anything
it can do, you could already do from a notebook cell.

Two consequences are worth stating plainly, because they are the boundary:

- **Anyone who can reach the port can drive the kernel** — start training runs,
  read the registered-data listing, download trained weights. Keep it on
  localhost. For a remote kernel use the SSH tunnel `sess.open()` prints;
  `Lamplighter(host=...)` warns loudly for exactly this reason.
- **Your browser can reach that port on behalf of any page you visit.** The
  network is not the boundary a browser respects, so the server checks the
  `Origin` header on the WebSocket handshake and the `Host` header on every HTTP
  request, and answers only to loopback (see `lamplighter/backend/origins.py`).
  This is not authentication; it is what makes the localhost model true.

Everything the app executes is code you can read — the generated sources in the
Show code panels — run with your own privileges in your own kernel.

**Loading a checkpoint executes code.** A `.pt` written by Lamplighter carries
the model's own source so old runs stay rebuildable, and restoring, resuming,
previewing or evaluating one runs it. Weights are read with
`weights_only=True`, but the stored source is not data. Treat a checkpoint like
a Python file: only load ones you produced or trust.

**The MCP server executes the setup code it is sent.** `lamplighter-mcp`'s one
tool takes Python source, runs it in a subprocess with your privileges, and
checks the objects it built. That is its function, not a side effect: it
exists so a coding agent can hand over construction code, and it carries the
same trust level as that agent's shell access. It speaks stdio only — no port,
no network listener. Wire it only into agent hosts you already trust to run
code on your machine.

## Out of scope

Reports that reduce to "the server has no authentication", "a checkpoint can
execute code", or "the MCP tool executes the setup code" describe documented
design, not vulnerabilities. So does anything requiring an attacker who
already has local code execution as your user.

## In scope

- Any way for a **remote page or host** to reach the API, the WebSocket, or the
  kernel — an `Origin`/`Host` check that can be bypassed, a DNS-rebinding path,
  a route that answers when it should not.
- **Injection into generated code**: a node name, parameter, or registered
  variable that escapes into the emitted source as executable code rather than
  as a value.
- **Path traversal** — a checkpoint name, data root, or download path that
  escapes the directory it should be confined to.
- Anything that makes Lamplighter execute code the user did not supply.

## Reporting

Please report privately via GitHub's **Report a vulnerability** button on the
[Security tab](https://github.com/scott-kirila/Lamplighter/security), rather
than opening a public issue.

Include what `lamplighter.diagnostics()` prints, and the smallest reproduction
you can manage. This is a solo project — expect a first response within a week,
and a fix or a clear "won't fix, here's why" for anything confirmed in scope.
