/* Acurix Edge — visualizador de bancada (AEX-BENCH-VIEW-001). Sem framework, sem CDN.
   Só textContent/createElement (nunca HTML montado a partir de dados). Consulta /api/status 2 vezes por segundo.
   (c) 2026 Luís Eduardo de Sousa Barbosa. */
"use strict";

(function () {
  const POLL_MS = 500;
  const FETCH_TIMEOUT_MS = 1500;
  const $ = (id) => document.getElementById(id);
  let stopped = false;

  function el(tag, cls, text) {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }

  function setDot(node, kind) {
    node.className = "ax-dot" + (kind === "ok" ? "" : " ax-dot--" + kind);
  }

  function limitBar(bar) {
    const root = el("div", "ax-limit" + (bar.attention ? " ax-limit--review" : ""));
    const head = el("div", "ax-limit__head");
    head.appendChild(el("span", null, bar.label));
    head.appendChild(el("b", null, bar.value_text));
    root.appendChild(head);
    const track = el("div", "ax-limit__track");
    if (bar.band_pct) {
      const band = el("div", "ax-limit__band");
      band.style.left = bar.band_pct[0] + "%";
      band.style.width = Math.max(0, bar.band_pct[1] - bar.band_pct[0]) + "%";
      track.appendChild(band);
    }
    for (const t of bar.thresholds_pct) {
      const th = el("div", "ax-limit__threshold");
      th.style.left = t + "%";
      track.appendChild(th);
    }
    const value = el("div", "ax-limit__value");
    value.style.left = bar.value_pct + "%";
    track.appendChild(value);
    root.appendChild(track);
    const scale = el("div", "ax-limit__scale");
    scale.appendChild(el("span", null, bar.scale_text[0]));
    if (bar.detail) scale.appendChild(el("span", "bv-limit-detail", bar.detail));
    scale.appendChild(el("span", null, bar.scale_text[1]));
    root.appendChild(scale);
    return root;
  }

  function trailStep(step) {
    const cls = "ax-trail__step" + (step.kind === "ok" ? "" : " ax-trail__step--" + step.kind);
    const row = el("div", cls);
    row.appendChild(el("span", "ax-trail__mark"));
    const body = el("div");
    body.appendChild(el("div", "ax-trail__name" + (step.kind === "skip" ? " bv-trail-skip-name" : ""), step.name));
    body.appendChild(el("div", "ax-trail__detail", step.detail));
    row.appendChild(body);
    row.appendChild(el("span", "ax-trail__value", step.value));
    return row;
  }

  function render(s) {
    const st = s.strip;
    $("cam").textContent = st.camera;
    setDot($("cam-dot"), st.camera_dot);
    $("sys").textContent = st.system;
    setDot($("sys-dot"), st.system_dot);
    $("lat").textContent = st.latency_text;
    $("fps").textContent = st.fps_text;
    $("ram").textContent = st.ram_text;
    $("env").textContent = st.envelope_text;

    const banner = $("banner");
    if (s.banner) {
      $("banner-title").textContent = s.banner.title;
      $("banner-text").textContent = s.banner.text;
      $("banner-code").textContent = s.banner.code;
      banner.classList.remove("bv-hidden");
    } else {
      banner.classList.add("bv-hidden");
    }

    const v = s.verdict;
    const verdict = $("verdict");
    verdict.textContent = v.value;
    verdict.className = "ax-verdict ax-verdict--hero ax-verdict--" + (v.modifier === "review" ? "review" : "none");
    $("verdict-phrase").textContent = v.phrase;
    $("verdict-reason").textContent = v.reason ? "Motivo: " + v.reason : "";

    const q = s.quality;
    $("q-status").textContent = q.status;
    $("q-bars").replaceChildren(...q.bars.map(limitBar));
    const refused = q.valid === false;
    $("media").classList.toggle("bv-media--refused", refused);
    const refusal = $("refusal");
    refusal.textContent = refused ? "Imagem recusada: " + q.reasons.join("; ") : "";
    refusal.classList.toggle("bv-hidden", !refused);

    const roi = $("roi");
    roi.style.left = s.roi_pct.x + "%";
    roi.style.top = s.roi_pct.y + "%";
    roi.style.width = s.roi_pct.w + "%";
    roi.style.height = s.roi_pct.h + "%";

    const sh = s.shadow;
    $("s-bar").replaceChildren(...(sh.bar ? [limitBar(sh.bar)] : []));
    $("s-text").textContent = sh.text;

    $("trail").replaceChildren(...s.trail.map(trailStep));
    $("image-note").textContent = s.image_note;
  }

  function showBanner(kind, title, text) {
    $("banner-title").textContent = title;
    $("banner-text").textContent = text;
    $("banner-code").textContent = "";
    $("banner").className = "ax-banner ax-banner--" + kind + " bv-banner";
  }

  function showStopped(message) {
    stopped = true;
    showBanner("p3", "Visualizador encerrado", message);
    $("stop").disabled = true;
  }

  // Never stops polling by itself (review M1): a failed or slow request only marks the data as
  // stale; the page keeps trying, so a later FAULT is still shown.
  let lastOk = Date.now();

  async function poll() {
    if (stopped) return;
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), FETCH_TIMEOUT_MS);
    try {
      const resp = await fetch("/api/status", { cache: "no-store", signal: ctrl.signal });
      if (!resp.ok) throw new Error("HTTP " + resp.status);
      const data = await resp.json();
      lastOk = Date.now();
      $("banner").className = "ax-banner ax-banner--p1 bv-banner bv-hidden";
      render(data);
    } catch (err) {
      const secs = Math.round((Date.now() - lastOk) / 1000);
      showBanner("p2", "Sem atualização há " + secs + " s",
        "O servidor local não responde. Os dados na tela podem estar desatualizados; tentando de novo.");
    } finally {
      clearTimeout(timer);
    }
    if (!stopped) setTimeout(poll, POLL_MS);
  }

  $("stop").addEventListener("click", async () => {
    if (!window.confirm("Encerrar o visualizador e liberar a câmera?")) return;
    stopped = true;
    try {
      await fetch("/api/shutdown", { method: "POST", headers: { "X-Acurix-Action": "shutdown" } });
    } catch (err) { /* o servidor pode fechar antes de responder */ }
    showStopped("Encerrado pelo botão. A câmera foi liberada; pode fechar esta aba.");
  });

  poll();
})();
