const http = require('http');
const { spawn } = require('child_process');

const RESEND_API_KEY = process.env.RESEND_API_KEY;
const LISTEN_PORT = parseInt(process.env.PORT || '3000', 10);
const UPSTREAM_PORT = parseInt(process.env.UPSTREAM_PORT || '3001', 10);

if (!RESEND_API_KEY) {
  console.error('RESEND_API_KEY is not set');
  process.exit(1);
}

// 1. 启动 resend-mcp 作为子进程
const upstream = spawn(
  'npx',
  ['resend-mcp', '--http', '--host', '127.0.0.1', '--port', String(UPSTREAM_PORT)],
  {
    stdio: 'inherit',
    env: { ...process.env },
  }
);

upstream.on('exit', (code) => {
  console.error(`resend-mcp exited with code ${code}`);
  process.exit(code || 1);
});

// 2. 等 upstream 起来
function waitForUpstream(retries = 30, delayMs = 500) {
  return new Promise((resolve, reject) => {
    const tryConnect = (n) => {
      const req = http.request(
        { host: '127.0.0.1', port: UPSTREAM_PORT, path: '/health', method: 'GET' },
        (res) => {
          res.resume();
          resolve();
        }
      );
      req.on('error', () => {
        if (n <= 0) return reject(new Error('upstream not ready'));
        setTimeout(() => tryConnect(n - 1), delayMs);
      });
      req.end();
    };
    tryConnect(retries);
  });
}

// 3. 反向代理
const server = http.createServer((req, res) => {
  const headers = { ...req.headers };
  // 强制覆盖 Authorization
  headers['authorization'] = `Bearer ${RESEND_API_KEY}`;
  // 避免上游 Host 校验问题
  headers['host'] = `127.0.0.1:${UPSTREAM_PORT}`;

  const options = {
    hostname: '127.0.0.1',
    port: UPSTREAM_PORT,
    path: req.url,
    method: req.method,
    headers,
  };

  const proxyReq = http.request(options, (proxyRes) => {
    res.writeHead(proxyRes.statusCode, proxyRes.headers);
    proxyRes.pipe(res);
  });

  proxyReq.on('error', (err) => {
    console.error('proxy error:', err.message);
    res.writeHead(502, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: 'Bad Gateway', detail: err.message }));
  });

  req.pipe(proxyReq);
});

waitForUpstream()
  .then(() => {
    server.listen(LISTEN_PORT, '0.0.0.0', () => {
      console.log(`Proxy listening on 0.0.0.0:${LISTEN_PORT}`);
      console.log(`Upstream resend-mcp on 127.0.0.1:${UPSTREAM_PORT}`);
    });
  })
  .catch((err) => {
    console.error('Failed to start:', err.message);
    process.exit(1);
  });