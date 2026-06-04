const state = {
  sessionId: null,
  chunks: [],
  counts: [],
  detailLevel: "普通",
  activeChunkIndex: -1,
  loopStart: null,
  loopEnd: null,
  loopEnabled: false,
  t0: 0,
  t1: 0,
  clipNames: {},
  namingChunkIndex: -1,
  menuVisible: true,
  activeTab: "section",
  originalVideoUrl: null,
  boneVideoUrl: null,
  viewMode: "normal",
};

const appShell = document.getElementById("appShell");
const videoStage = document.getElementById("videoStage");
const video = document.getElementById("danceVideo");
const videoFile = document.getElementById("videoFile");
const videoFileMenu = document.getElementById("videoFileMenu");
const placeholder = document.getElementById("videoPlaceholder");
const loadingOverlay = document.getElementById("loadingOverlay");
const loadingText = document.getElementById("loadingText");
const errorBox = document.getElementById("errorBox");

const pinLayer = document.getElementById("pinLayer");
const timeline = document.getElementById("timeline");
const progress = document.getElementById("progress");
const thumb = document.getElementById("thumb");
const startTimeLabel = document.getElementById("startTimeLabel");
const endTimeLabel = document.getElementById("endTimeLabel");
const activeClipName = document.getElementById("activeClipName");

const playBtn = document.getElementById("playBtn");
const prevBtn = document.getElementById("prevBtn");
const nextBtn = document.getElementById("nextBtn");
const chunkInfo = document.getElementById("chunkInfo");

const clipNameDialog = document.getElementById("clipNameDialog");
const clipNameInput = document.getElementById("clipNameInput");
const clipNameSaveBtn = document.getElementById("clipNameSaveBtn");
const clipNameCancelBtn = document.getElementById("clipNameCancelBtn");
const clipNameClearBtn = document.getElementById("clipNameClearBtn");

let progressRafId = null;
let pinClickTimerId = null;

function removeLegacyCountUi() {
  document.getElementById("metronomeDisplay")?.remove();
  document.getElementById("countRow")?.remove();
  document.getElementById("setCountOneBtn")?.remove();
  document.querySelector('[data-pane="metronome"]')?.remove();
  document.querySelector('[data-tab="metronome"]')?.remove();
}

function setMenuVisible(visible) {
  state.menuVisible = Boolean(visible);
  appShell.classList.toggle("menu-visible", state.menuVisible);
  appShell.classList.toggle("menu-hidden", !state.menuVisible);
}

function switchTab(tabName) {
  state.activeTab = tabName;

  document.querySelectorAll(".menu-tab").forEach(btn => {
    btn.classList.toggle("active", btn.dataset.tab === tabName);
  });

  document.querySelectorAll(".tab-pane").forEach(pane => {
    pane.classList.toggle("active", pane.dataset.pane === tabName);
  });
}

function playVideo() {
  const promise = video.play();
  if (promise && typeof promise.catch === "function") {
    promise.catch(() => {});
  }
}

function setVideoRate(rate) {
  const nextRate = Number(rate);
  if (!Number.isFinite(nextRate) || nextRate <= 0) return;

  const wasPaused = video.paused;
  try {
    video.defaultPlaybackRate = nextRate;
    video.playbackRate = nextRate;
  } catch (err) {
    showError("このブラウザでは指定した再生速度を使えません。");
    return;
  }

  if (!wasPaused) {
    playVideo();
  }
}

async function regenerateBoneVideo() {
  if (!state.sessionId) return false;

  const res = await fetch(`/api/regenerate-bone/${state.sessionId}`, {
    method: "POST",
  });
  const data = await res.json();
  if (!res.ok || !data.ok) {
    throw new Error(data.error || "ボーン表示用の動画を作成できませんでした。");
  }
  state.boneVideoUrl = data.bone_video_url;
  return true;
}

function setViewMode(mode, options = {}) {
  const nextMode = mode === "bone" ? "bone" : "normal";
  if (nextMode === "bone" && !state.boneVideoUrl) {
    regenerateBoneVideo()
      .then(() => setViewMode("bone", { retried: true }))
      .catch(err => showError(err.message));
    return;
  }

  const nextUrl = nextMode === "bone" ? state.boneVideoUrl : state.originalVideoUrl;
  if (!nextUrl) {
    state.viewMode = nextMode;
    syncViewModeButtons();
    return;
  }
  const nextHref = new URL(nextUrl, window.location.href).href;
  if (video.currentSrc === nextHref || video.src === nextHref) {
    state.viewMode = nextMode;
    syncViewModeButtons();
    return;
  }

  const currentTime = video.currentTime || 0;
  const currentRate = video.playbackRate || 1;
  const wasPaused = video.paused;

  state.viewMode = nextMode;
  syncViewModeButtons();

  video.addEventListener("loadedmetadata", () => {
    video.currentTime = Math.min(currentTime, Math.max(video.duration - 0.05, 0));
    try {
      video.defaultPlaybackRate = currentRate;
      video.playbackRate = currentRate;
    } catch (err) {
      // 再生速度の復元に失敗しても表示切替は続ける。
    }
    if (!wasPaused) {
      playVideo();
    }
    updateProgress();
  }, { once: true });

  video.addEventListener("error", () => {
    if (nextMode === "bone" && !options.retried) {
      regenerateBoneVideo()
        .then(() => setViewMode("bone", { retried: true }))
        .catch(err => showError(err.message));
      return;
    }
    showError("ボーン表示用の動画を読み込めませんでした。");
    setViewMode("normal", { retried: true });
  }, { once: true });

  video.src = nextUrl;
  video.load();
}

function syncViewModeButtons() {
  document.querySelectorAll(".view-mode-btn").forEach(btn => {
    btn.classList.toggle("active", btn.dataset.viewMode === state.viewMode);
  });
}

function startProgressMonitor() {
  if (progressRafId !== null) return;

  const tick = () => {
    updateProgress();
    progressRafId = window.requestAnimationFrame(tick);
  };

  progressRafId = window.requestAnimationFrame(tick);
}

function stopProgressMonitor() {
  if (progressRafId === null) return;
  window.cancelAnimationFrame(progressRafId);
  progressRafId = null;
}

function showLoading(text = "解析中...") {
  loadingText.textContent = text;
  loadingOverlay.classList.remove("hidden");
}

function hideLoading() {
  loadingOverlay.classList.add("hidden");
}

function showError(message) {
  errorBox.textContent = message;
  errorBox.classList.remove("hidden");
}

function clearError() {
  errorBox.classList.add("hidden");
  errorBox.textContent = "";
}

function clipStorageKey(sessionId = state.sessionId) {
  return sessionId ? `idol-pin-loop:clip-names:${sessionId}` : null;
}

function getClipKey(ch) {
  const start = Number(ch.start_sec || 0).toFixed(3);
  const end = Number(ch.end_sec || 0).toFixed(3);
  return `${ch.chunk_id || "clip"}:${start}:${end}`;
}

function loadClipNames(sessionId) {
  const key = clipStorageKey(sessionId);
  if (!key) return {};

  try {
    const raw = window.localStorage.getItem(key);
    return raw ? JSON.parse(raw) : {};
  } catch (err) {
    return {};
  }
}

function saveClipNames() {
  const key = clipStorageKey();
  if (!key) return;

  try {
    window.localStorage.setItem(key, JSON.stringify(state.clipNames));
  } catch (err) {
    // localStorageが使えない環境でも、同じ画面内ではstate上で動作させる。
  }
}

function getClipName(ch) {
  return String(ch?.name || "").trim();
}

function getChunkDisplayName(ch) {
  if (!ch) return "";
  return getClipName(ch) || `区間 ${ch.chunk_id}`;
}

function setClipName(idx, name) {
  if (idx < 0 || idx >= state.chunks.length) return;

  const ch = state.chunks[idx];
  const key = getClipKey(ch);
  const nextName = String(name || "").trim();
  ch.name = nextName;

  if (nextName) {
    state.clipNames[key] = nextName;
  } else {
    delete state.clipNames[key];
  }

  saveClipNames();
  renderPins();
  updateProgress();
}

function openClipNameDialog(idx) {
  if (idx < 0 || idx >= state.chunks.length) return;
  state.namingChunkIndex = idx;
  clipNameInput.value = getClipName(state.chunks[idx]);
  clipNameDialog.classList.remove("hidden");
  window.setTimeout(() => {
    clipNameInput.focus();
    clipNameInput.select();
  }, 0);
}

function closeClipNameDialog() {
  clipNameDialog.classList.add("hidden");
  state.namingChunkIndex = -1;
}

function saveClipNameFromDialog() {
  setClipName(state.namingChunkIndex, clipNameInput.value);
  closeClipNameDialog();
}

function mmss(t) {
  t = Math.max(0, Number(t || 0));
  const m = Math.floor(t / 60);
  const s = String(Math.floor(t % 60)).padStart(2, "0");
  return `${m}:${s}`;
}

function pct(t) {
  const denom = Math.max(state.t1 - state.t0, 0.001);
  return Math.max(0, Math.min(100, ((t - state.t0) / denom) * 100));
}

async function analyzeVideo(file) {
  clearError();
  showLoading("動画を解析中... 少し時間がかかります");

  const form = new FormData();
  form.append("video", file);
  form.append("detail_level", state.detailLevel);

  try {
    const res = await fetch("/api/analyze", {
      method: "POST",
      body: form,
    });
    const data = await res.json();
    if (!res.ok || !data.ok) {
      throw new Error(data.error || "解析に失敗しました");
    }
    applyAnalysisResult(data);
  } catch (err) {
    showError(err.message);
  } finally {
    hideLoading();
  }
}

async function rechunk(detailLevel) {
  if (!state.sessionId) return;

  clearError();
  showLoading("区切りを更新中...");

  const form = new FormData();
  form.append("detail_level", detailLevel);

  try {
    const res = await fetch(`/api/rechunk/${state.sessionId}`, {
      method: "POST",
      body: form,
    });
    const data = await res.json();
    if (!res.ok || !data.ok) {
      throw new Error(data.error || "区切りの更新に失敗しました");
    }
    applyAnalysisResult(data, { keepVideo: true });
  } catch (err) {
    showError(err.message);
  } finally {
    hideLoading();
  }
}

function applyAnalysisResult(data, options = {}) {
  state.sessionId = data.session_id;
  state.detailLevel = data.detail_level;
  state.clipNames = loadClipNames(state.sessionId);
  state.chunks = (data.chunks || []).map(ch => ({
    ...ch,
    name: state.clipNames[getClipKey(ch)] || ch.name || "",
  }));
  state.counts = data.counts || [];
  state.activeChunkIndex = -1;
  state.loopEnabled = false;
  state.originalVideoUrl = data.video_url;
  state.boneVideoUrl = data.bone_video_url || null;

  if (!options.keepVideo) {
    state.viewMode = "normal";
    syncViewModeButtons();
    video.src = state.originalVideoUrl;
    video.load();
    placeholder.classList.add("hidden");
    appShell.classList.add("has-video");
    setMenuVisible(true);
  }

  state.t0 = 0;
  state.t1 = Number(data.duration || 0);
  if (!state.t1 && state.chunks.length > 0) {
    state.t1 = Math.max(...state.chunks.map(c => Number(c.end_sec)));
  } else if (!state.t1 && state.counts.length > 0) {
    state.t1 = Math.max(...state.counts.map(c => Number(c.end_sec)));
  }

  startTimeLabel.textContent = mmss(state.t0);
  endTimeLabel.textContent = mmss(state.t1);

  renderPins();

  if (state.chunks.length > 0) {
    if (Number.isFinite(options.preserveTime)) {
      const preservedTime = Math.max(0, Number(options.preserveTime));
      const activeIdx = state.chunks.findIndex(ch =>
        preservedTime >= Number(ch.start_sec) &&
        preservedTime < Number(ch.end_sec)
      );
      video.currentTime = preservedTime;
      activateChunk(activeIdx >= 0 ? activeIdx : 0, false, false);
    } else {
      activateChunk(0, false);
      video.currentTime = state.chunks[0].start_sec;
    }
  }

  updateProgress();
}

function renderPins() {
  pinLayer.innerHTML = "";
  for (const [idx, ch] of state.chunks.entries()) {
    const name = getClipName(ch);
    const pinPct = pct(Number(ch.end_sec));
    const pin = document.createElement("div");
    pin.className = `pin${name ? " has-name" : ""}${pinPct > 82 ? " label-left" : ""}`;
    pin.tabIndex = 0;
    pin.style.left = `calc(14px + (100% - 28px) * ${pinPct / 100})`;
    pin.title = name
      ? `${name}: ${mmss(ch.start_sec)}〜${mmss(ch.end_sec)}`
      : `Chunk ${ch.chunk_id}: ${mmss(ch.start_sec)}〜${mmss(ch.end_sec)}`;

    const icon = document.createElement("span");
    icon.className = "pin-icon";
    icon.textContent = "📍";
    pin.appendChild(icon);

    if (name) {
      const label = document.createElement("span");
      label.className = "pin-label";
      label.textContent = name;
      pin.appendChild(label);
    }

    pin.addEventListener("click", (e) => {
      e.stopPropagation();
      if (pinClickTimerId !== null) {
        window.clearTimeout(pinClickTimerId);
      }
      pinClickTimerId = window.setTimeout(() => {
        activateChunk(idx, true);
        pinClickTimerId = null;
      }, 180);
    });
    pin.addEventListener("dblclick", (e) => {
      e.stopPropagation();
      e.preventDefault();
      if (pinClickTimerId !== null) {
        window.clearTimeout(pinClickTimerId);
        pinClickTimerId = null;
      }
      openClipNameDialog(idx);
    });
    pin.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        openClipNameDialog(idx);
      }
    });
    pin.classList.toggle("active", idx === state.activeChunkIndex);
    pinLayer.appendChild(pin);
  }
}

function activateChunk(idx, shouldPlay, shouldSeek = true) {
  if (idx < 0 || idx >= state.chunks.length) return;

  state.activeChunkIndex = idx;
  const ch = state.chunks[idx];

  state.loopStart = Number(ch.start_sec);
  state.loopEnd = Number(ch.end_sec);
  state.loopEnabled = true;

  if (shouldSeek) {
    video.currentTime = state.loopStart;
  }

  document.querySelectorAll(".pin").forEach((pin, i) => {
    pin.classList.toggle("active", i === idx);
  });

  updateActiveClipName();

  chunkInfo.textContent = "";

  if (shouldPlay) {
    playVideo();
  }
}

function updateProgress() {
  const p = pct(video.currentTime);
  progress.style.width = `calc((100% - 28px) * ${p / 100})`;
  thumb.style.left = `calc(14px + (100% - 28px) * ${p / 100})`;

  updateActiveClipName();

  if (state.loopEnabled && state.loopEnd !== null && video.currentTime >= state.loopEnd) {
    const wasPaused = video.paused;
    video.currentTime = state.loopStart;
    if (!wasPaused) {
      playVideo();
    }
  }
}

function updateActiveClipName() {
  const selectedChunk = state.chunks[state.activeChunkIndex];

  if (selectedChunk) {
    activeClipName.textContent = getChunkDisplayName(selectedChunk);
    activeClipName.classList.remove("hidden");
  } else {
    activeClipName.textContent = "";
    activeClipName.classList.add("hidden");
  }
}

function handleVideoFileChange(e) {
  const file = e.target.files?.[0];
  if (file) analyzeVideo(file);
}

videoFile.addEventListener("change", handleVideoFileChange);
videoFileMenu.addEventListener("change", handleVideoFileChange);

clipNameSaveBtn.addEventListener("click", saveClipNameFromDialog);
clipNameCancelBtn.addEventListener("click", closeClipNameDialog);
clipNameClearBtn.addEventListener("click", () => {
  setClipName(state.namingChunkIndex, "");
  closeClipNameDialog();
});
clipNameInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    e.preventDefault();
    saveClipNameFromDialog();
  }
  if (e.key === "Escape") {
    e.preventDefault();
    closeClipNameDialog();
  }
});
clipNameDialog.addEventListener("click", (e) => {
  if (e.target === clipNameDialog) {
    closeClipNameDialog();
  }
});

document.querySelectorAll(".detail-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".detail-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    state.detailLevel = btn.dataset.detail;
    if (state.sessionId) rechunk(state.detailLevel);
  });
});

document.querySelectorAll(".speed-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".speed-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    setVideoRate(btn.dataset.rate);
  });
});

document.querySelectorAll(".mirror-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".mirror-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    video.classList.toggle("mirrored", btn.dataset.mirror === "on");
  });
});

document.querySelectorAll(".view-mode-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    setViewMode(btn.dataset.viewMode);
  });
});

document.querySelectorAll(".menu-tab").forEach(btn => {
  btn.addEventListener("click", () => {
    switchTab(btn.dataset.tab);
    setMenuVisible(true);
  });
});

videoStage.addEventListener("click", (e) => {
  if (state.menuVisible && e.target.closest("[data-menu-control]")) return;
  if (!video.src) return;
  setMenuVisible(!state.menuVisible);
});

timeline.addEventListener("click", (e) => {
  const rect = timeline.getBoundingClientRect();
  const ratio = Math.max(0, Math.min(1, (e.clientX - rect.left - 14) / Math.max(rect.width - 28, 1)));
  video.currentTime = state.t0 + ratio * (state.t1 - state.t0);
  state.loopEnabled = false;
});

playBtn.addEventListener("click", () => {
  if (!video.src) return;

  if (video.paused) {
    if (state.loopEnabled && state.loopStart !== null && state.loopEnd !== null) {
      if (video.currentTime < state.loopStart || video.currentTime >= state.loopEnd) {
        video.currentTime = state.loopStart;
      }
    }
    playVideo();
  } else {
    video.pause();
  }
});

prevBtn.addEventListener("click", () => {
  if (!state.chunks.length) return;
  const nextIdx = state.activeChunkIndex <= 0 ? 0 : state.activeChunkIndex - 1;
  activateChunk(nextIdx, true);
});

nextBtn.addEventListener("click", () => {
  if (!state.chunks.length) return;
  const nextIdx = state.activeChunkIndex < 0
    ? 0
    : Math.min(state.chunks.length - 1, state.activeChunkIndex + 1);
  activateChunk(nextIdx, true);
});

video.addEventListener("timeupdate", updateProgress);
video.addEventListener("play", () => {
  playBtn.textContent = "⏸";
  startProgressMonitor();
});
video.addEventListener("pause", () => {
  playBtn.textContent = "▶";
  stopProgressMonitor();
  updateProgress();
});

// 初期表示
removeLegacyCountUi();
setMenuVisible(true);
