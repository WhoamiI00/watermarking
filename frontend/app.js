// Frontend controller for the self-recovery watermarking demo.
//
// Talks to FastAPI routes:
//   POST /api/embed
//   POST /api/attack

const state = {
  coverFile: null,
  donorFile: null,
  sessionId: null,
  selectedAttack: null,
};

// --------------------------------------------------------------
// helpers
// --------------------------------------------------------------
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => [...document.querySelectorAll(sel)];

function setStatus(el, text, cls = "") {
  el.textContent = text || "";
  el.className = "status " + cls;
}

function renderImg(el, dataUrl, alt = "") {
  if (!dataUrl) { el.innerHTML = ""; return; }
  el.innerHTML = `<img src="${dataUrl}" alt="${alt}"/>`;
}

function metric(k, v, sub) {
  return `<div class="metric"><div class="k">${k}</div><div class="v">${v}</div>${sub ? `<div class="sub">${sub}</div>` : ""}</div>`;
}

function fmtDb(x) {
  if (x === null || x === undefined) return "&ndash;";
  if (!Number.isFinite(x) || x > 300) return "&infin; dB";
  return `${x.toFixed(2)} dB`;
}

function updatePaperTable(key, value) {
  const row = document.querySelector(`#paper-table tr[data-ref="${key}"]`);
  if (row) row.children[2].innerHTML = value;
}

// --------------------------------------------------------------
// attack params UI
// --------------------------------------------------------------
const ATTACK_SPECS = {
  mean: {
    desc: "Preserve the 64&times;64 block's mean but flip ±&delta; on a checker pattern.",
    params: [
      ["y0", "row", 224, "px"], ["x0", "col", 224, "px"],
      ["size", "size", 64, "px"], ["delta", "&delta;", 10, ""],
    ],
    hot: true,
  },
  collage: {
    desc: "Paste a region from another watermarked image at the same coordinates.",
    params: [
      ["y0", "row", 96, "px"], ["x0", "col", 96, "px"],
      ["h", "height", 120, "px"], ["w", "width", 120, "px"],
    ],
    needsDonor: true,
    hot: true,
  },
  rectangular: {
    desc: "Paint a solid-colour rectangle.",
    params: [
      ["y0", "row", 200, "px"], ["x0", "col", 200, "px"],
      ["h", "height", 96, "px"], ["w", "width", 96, "px"],
      ["value", "fill", 128, "0..255"],
    ],
  },
  gaussian: { desc: "Gaussian blur.", params: [["sigma", "&sigma;", 0.3, ""], ["ksize", "k", 3, "odd"]] },
  median:   { desc: "Median filter.", params: [["ksize", "k", 3, "odd"]] },
  white_noise: { desc: "Additive white Gaussian noise. \"0.01\" means stddev = 1% of dynamic range (matches paper's Table 2).",
                 params: [["variance", "std on [0,1]", 0.01, ""]] },
  salt_pepper: { desc: "Salt &amp; pepper impulse noise. Density is the fraction of pixels corrupted (half -> 0, half -> 255).",
                 params: [["density", "density", 0.002, ""]] },
  shear:    { desc: "Zero-out a bottom strip (1/32 of height).", params: [["fraction", "fraction", 0.03125, ""]] },
  jpeg:     { desc: "Round-trip JPEG compression.", params: [["quality", "quality", 50, "1..95"]] },
  rescale:  { desc: "Downscale then upscale.", params: [["factor", "factor", 0.7, ""]] },
  rotation: { desc: "Rotate then undo-rotate.", params: [["angle_deg", "angle", 30, "deg"]] },
  none:     { desc: "Sanity run with no attack.", params: [] },
};

function renderAttackParams(attack) {
  const box = $("#attack-params");
  box.innerHTML = "";
  const spec = ATTACK_SPECS[attack];
  if (!spec) return;
  if (spec.desc) {
    const p = document.createElement("p");
    p.className = "note";
    p.innerHTML = spec.desc;
    p.style.gridColumn = "1 / -1";
    box.appendChild(p);
  }
  spec.params.forEach(([key, label, def, suffix]) => {
    const lbl = document.createElement("label");
    lbl.innerHTML = `<span>${label}${suffix ? " (" + suffix + ")" : ""}</span>
                     <input data-param="${key}" type="number" step="any" value="${def}" />`;
    box.appendChild(lbl);
  });
  $("#donor-upload").style.display = spec.needsDonor ? "flex" : "none";
}

function readParams() {
  const obj = {};
  $$("#attack-params input[data-param]").forEach((inp) => {
    const raw = inp.value;
    const v = Number(raw);
    obj[inp.dataset.param] = Number.isFinite(v) ? (Number.isInteger(v) ? parseInt(raw, 10) : v) : raw;
  });
  return obj;
}

// --------------------------------------------------------------
// file drop
// --------------------------------------------------------------
function wireFileDrop(drop, inputEl, onChange) {
  const setFile = (f) => {
    if (!f) return;
    onChange(f);
    drop.classList.add("has-file");
    drop.querySelector("strong").textContent = f.name;
  };
  inputEl.addEventListener("change", (e) => setFile(e.target.files[0]));
  ["dragenter", "dragover"].forEach((evt) => drop.addEventListener(evt, (e) => {
    e.preventDefault(); drop.classList.add("has-file");
  }));
  drop.addEventListener("dragleave", () => drop.classList.remove("has-file"));
  drop.addEventListener("drop", (e) => {
    e.preventDefault();
    setFile(e.dataTransfer.files[0]);
  });
}

// --------------------------------------------------------------
// step 1 -- embed
// --------------------------------------------------------------
$("#btn-embed").addEventListener("click", async () => {
  if (!state.coverFile) {
    setStatus($("#status-embed"), "Please upload a cover image first.", "err");
    return;
  }
  setStatus($("#status-embed"), "Embedding watermark -- this takes a few seconds on 512x512...", "working");

  const form = new FormData();
  form.append("image", state.coverFile);
  form.append("key", $("#key").value);
  form.append("gamma", $("#gamma").value);
  form.append("target_size", $("#target-size").value);

  try {
    const r = await fetch("/api/embed", { method: "POST", body: form });
    if (!r.ok) throw new Error(await r.text());
    const d = await r.json();

    state.sessionId = d.session_id;
    renderImg($("#img-original"), d.original);
    renderImg($("#img-decomp"), d.decomposition);
    renderImg($("#img-watermarked"), d.watermarked);
    renderImg($("#img-restored"), d.restored_no_attack);

    $("#metrics-embed").innerHTML = [
      metric("PSNR (watermarked)", fmtDb(d.psnr_watermarked), "paper: 41.34 dB"),
      metric("NC (reversibility)", d.nc_reversibility.toFixed(5), "paper: 1.0000"),
      metric("Homogeneous leaves", d.blocks_homogeneous, ""),
      metric("Non-homogeneous leaves", d.blocks_nonhomogeneous, ""),
      metric("Recovery bits", d.recovery_bits_total.toLocaleString(), ""),
    ].join("");

    updatePaperTable("psnr_watermarked", fmtDb(d.psnr_watermarked));
    updatePaperTable("nc", d.nc_reversibility.toFixed(5));

    setStatus($("#status-embed"), "Watermarked image ready. You can now launch an attack below.", "ok");
    $("#step-attack").classList.remove("locked");
  } catch (err) {
    setStatus($("#status-embed"), "Failed: " + err.message, "err");
  }
});

// --------------------------------------------------------------
// step 2 -- attack buttons
// --------------------------------------------------------------
$$(".attack-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    $$(".attack-btn").forEach((b) => b.classList.remove("selected"));
    btn.classList.add("selected");
    state.selectedAttack = btn.dataset.attack;
    renderAttackParams(state.selectedAttack);
    $("#btn-attack").disabled = false;
  });
});

$("#btn-attack").addEventListener("click", async () => {
  if (!state.sessionId || !state.selectedAttack) return;
  setStatus($("#status-attack"), "Running attack + recovery...", "working");

  const form = new FormData();
  form.append("session_id", state.sessionId);
  form.append("attack", state.selectedAttack);
  form.append("attack_params", JSON.stringify(readParams()));
  if (state.selectedAttack === "collage" && state.donorFile) {
    form.append("donor", state.donorFile);
  }

  try {
    const r = await fetch("/api/attack", { method: "POST", body: form });
    if (!r.ok) throw new Error(await r.text());
    const d = await r.json();

    renderImg($("#img-wm2"), d.watermarked);
    renderImg($("#img-attacked"), d.attacked);
    renderImg($("#img-detected"), d.detected_mask);
    renderImg($("#img-recovered"), d.recovered);

    const globalNote = d.global_attack_detected
      ? '<div class="metric" style="grid-column:1/-1;background:rgba(247,185,85,.08);border-color:#7c5b20"><div class="k">Heads up</div><div class="v" style="font-size:14px">Global distortion detected -- surgical repair was skipped (it would overwrite real content with flat fill). Use "attacked" as the output.</div></div>'
      : "";

    $("#metrics-attack").innerHTML = [
      globalNote,
      metric("Attacked vs. watermarked", fmtDb(d.psnr_attacked_vs_watermarked)),
      metric("Recovered vs. watermarked", fmtDb(d.psnr_recovered_vs_watermarked)),
      d.psnr_recovered_vs_original != null
        ? metric("Recovered vs. original", fmtDb(d.psnr_recovered_vs_original))
        : "",
      metric("Tampered leaves repaired", `${d.n_blocks_repaired}`),
      metric("TPR", d.tpr.toFixed(3), "correctly detected / tampered"),
      metric("TNR", d.tnr.toFixed(3), "false positives / tampered (paper Eq.)"),
    ].filter(Boolean).join("");

    // Push the *correct* score into the paper table: for localized
    // attacks we compare against the recovered image, for global ones
    // against the attacked image (recovery was skipped by design).
    const key = {
      mean: "mean", collage: "collage",
      gaussian: "gaussian03", median: "median3",
      white_noise: "noise001", salt_pepper: "salt002",
      shear: "shear", jpeg: "jpeg50", rescale: "shrink", rotation: "rot30",
    }[state.selectedAttack];
    if (key) {
      const score = d.global_attack_detected
        ? d.psnr_attacked_vs_watermarked
        : d.psnr_recovered_vs_watermarked;
      updatePaperTable(key, fmtDb(score));
    }

    setStatus($("#status-attack"), "Done. Scroll to see results.", "ok");
  } catch (err) {
    setStatus($("#status-attack"), "Failed: " + err.message, "err");
  }
});

// --------------------------------------------------------------
// wire up file inputs
// --------------------------------------------------------------
wireFileDrop($("#drop-cover"), $("#file-cover"), (f) => { state.coverFile = f; });
$("#file-donor").addEventListener("change", (e) => { state.donorFile = e.target.files[0]; });

// preselect the first attack
(function bootstrap() {
  const btn = document.querySelector('.attack-btn[data-attack="rectangular"]');
  if (btn) btn.click();
})();
