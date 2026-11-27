// Cloudflare Pages 反向代理：把 pages.dev 的请求转发到本机 Tailscale Funnel
// 上游地址在部署前写入 UPSTREAM。
const UPSTREAM = "https://ai-labeling-system.tailf8a030.ts.net";

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const target = new URL(url.pathname + url.search, UPSTREAM);

    const headers = new Headers(request.headers);
    headers.set("Host", target.host);
    headers.delete("cf-connecting-ip");
    headers.delete("cf-ray");
    headers.delete("cf-ipcountry");
    headers.delete("cdn-loop");

    const init = {
      method: request.method,
      headers,
      redirect: "follow",
    };
    if (request.method !== "GET" && request.method !== "HEAD") {
      init.body = request.body;
      init.duplex = "half";
    }

    const resp = await fetch(target, init);
    const outHeaders = new Headers(resp.headers);
    outHeaders.delete("content-encoding");
    outHeaders.delete("content-length");

    return new Response(resp.body, {
      status: resp.status,
      statusText: resp.statusText,
      headers: outHeaders,
    });
  },
};
