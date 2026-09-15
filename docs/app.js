const CONTEXT_LABELS = {
  div_game: "分區內對戰",
  qb_change_home: "主隊QB異動",
  qb_change_away: "客隊QB異動",
  cold_game: "低溫比賽",
  windy_game: "強風比賽",
  playoff: "季後賽",
};

const MODEL_LABELS = { elo: "Elo 基準模型", advanced: "進階整合模型", market: "Vegas 市場盤口" };

const WEEKDAY_ZH = {
  Sunday: "週日", Monday: "週一", Tuesday: "週二", Wednesday: "週三",
  Thursday: "週四", Friday: "週五", Saturday: "週六",
};

// NFL kickoff times are published in US Eastern time (America/New_York),
// which shifts between EDT (-04:00) and EST (-05:00) with US daylight
// saving. Rather than hardcoding those transition dates, ask the browser's
// own timezone database what the correct offset is for the game's date.
function nyOffsetString(dateStr) {
  const [y, m, d] = dateStr.split("-").map(Number);
  const ref = new Date(Date.UTC(y, m - 1, d, 12));
  try {
    const parts = new Intl.DateTimeFormat("en-US", {
      timeZone: "America/New_York",
      timeZoneName: "longOffset",
    }).formatToParts(ref);
    const tz = parts.find((p) => p.type === "timeZoneName")?.value;
    if (tz && tz.startsWith("GMT")) return tz.replace("GMT", "") || "+00:00";
  } catch {
    /* Intl.DateTimeFormat with longOffset unsupported — fall through */
  }
  return null;
}

function kickoffInstant(g) {
  if (!g.date || !g.gametime) return null;
  const offset = nyOffsetString(g.date);
  if (!offset) return null;
  const dt = new Date(`${g.date}T${g.gametime}:00${offset}`);
  return Number.isNaN(dt.getTime()) ? null : dt;
}

function formatKickoff(g) {
  if (!g.date) return "";
  const [, m, d] = g.date.split("-");
  const md = `${Number(m)}月${Number(d)}日`;
  const wd = WEEKDAY_ZH[g.weekday] ? `（${WEEKDAY_ZH[g.weekday]}）` : "";
  if (!g.gametime) return `${md}${wd}（時間未定）`;

  const etLine = `${md}${wd} ${g.gametime} 美東時間`;
  const instant = kickoffInstant(g);
  if (!instant) return etLine;

  const twStr = instant.toLocaleString("zh-TW", {
    timeZone: "Asia/Taipei", month: "numeric", day: "numeric",
    hour: "2-digit", minute: "2-digit", hour12: false, weekday: "short",
  });
  return `${etLine}<br><span class="kickoff-tw">台灣時間 ${twStr}</span>`;
}

let SITE_DATA = null;

async function loadData() {
  const res = await fetch("data.json", { cache: "no-store" });
  if (!res.ok) throw new Error(`failed to load data.json: ${res.status}`);
  return res.json();
}

function fmtPct(x) {
  return `${(x * 100).toFixed(1)}%`;
}

function teamLabel(code, nameZh) {
  return `${nameZh}（${code}）`;
}

function findTeam(teams, code) {
  return teams.find((t) => t.code === code);
}

function renderUpdatedAt(iso) {
  const el = document.getElementById("updated-at");
  try {
    const d = new Date(iso);
    el.textContent = `模型與資料更新時間：${d.toLocaleString("zh-TW", { dateStyle: "medium", timeStyle: "short" })}`;
  } catch {
    el.textContent = `模型與資料更新時間：${iso}`;
  }
}

function populateTeamSelects(teams) {
  const homeSel = document.getElementById("home-select");
  const awaySel = document.getElementById("away-select");
  const sorted = [...teams].sort((a, b) => a.name_zh.localeCompare(b.name_zh, "zh-Hant"));
  for (const sel of [homeSel, awaySel]) {
    sel.innerHTML = "";
    for (const t of sorted) {
      const opt = document.createElement("option");
      opt.value = t.code;
      opt.textContent = teamLabel(t.code, t.name_zh);
      opt.title = t.name;
      sel.appendChild(opt);
    }
  }
}

function probBarHtml(homeCode, awayCode, homeProb, awayProb) {
  const homePct = Math.max(homeProb * 100, 4);
  const awayPct = Math.max(awayProb * 100, 4);
  return `
    <div class="prob-bar">
      <div class="home-seg" style="width:${homePct}%">${homeCode} ${fmtPct(homeProb)}</div>
      <div class="away-seg" style="width:${awayPct}%">${awayCode} ${fmtPct(awayProb)}</div>
    </div>`;
}

function contextBadgesHtml(context, restDiffLabel) {
  if (!context) return "";
  const badges = [];
  for (const [key, label] of Object.entries(CONTEXT_LABELS)) {
    if (context[key]) badges.push(`<span class="badge">${label}</span>`);
  }
  if (restDiffLabel) badges.push(`<span class="badge">${restDiffLabel}</span>`);
  return badges.join("");
}

function populateWeekSelect(upcoming) {
  const weekSel = document.getElementById("week-select");
  const weeks = [...new Set(upcoming.map((g) => `${g.season}|${g.week}`))];
  weeks.sort((a, b) => {
    const [sa, wa] = a.split("|");
    const [sb, wb] = b.split("|");
    if (sa !== sb) return Number(sa) - Number(sb);
    return Number(wa) - Number(wb);
  });
  weekSel.innerHTML = "";
  for (const key of weeks) {
    const [season, week] = key.split("|");
    const opt = document.createElement("option");
    opt.value = key;
    opt.textContent = `${season} 賽季 第 ${week} 週`;
    weekSel.appendChild(opt);
  }
  if (weeks.length) weekSel.value = weeks[0];
}

function renderGamesForWeek(upcoming, weekKey) {
  const [season, week] = weekKey.split("|");
  const games = upcoming
    .filter((g) => String(g.season) === season && String(g.week) === week)
    .sort((a, b) => {
      const dateCmp = (a.date || "").localeCompare(b.date || "");
      if (dateCmp !== 0) return dateCmp;
      return (a.gametime || "").localeCompare(b.gametime || "");
    });

  const grid = document.getElementById("games-grid");
  grid.innerHTML = "";
  if (!games.length) {
    grid.innerHTML = `<p class="section-note">這週沒有排定的比賽資料。</p>`;
    return;
  }

  for (const g of games) {
    const restDiff = g.context ? g.context.rest_diff : 0;
    let restLabel = "";
    if (restDiff >= 2) restLabel = `${g.home_team} 多休 ${restDiff.toFixed(0)} 天`;
    else if (restDiff <= -2) restLabel = `${g.away_team} 多休 ${Math.abs(restDiff).toFixed(0)} 天`;

    const marketLine = g.market_spread !== undefined
      ? `｜市場盤口：${g.home_team} ${g.market_spread > 0 ? "-" : "+"}${Math.abs(g.market_spread).toFixed(1)}`
      : "";

    const card = document.createElement("div");
    card.className = "game-card";
    card.innerHTML = `
      <div class="kickoff">${formatKickoff(g)}</div>
      <div class="matchup"><span>${g.away_team}</span><span>@</span><span>${g.home_team}</span></div>
      <div class="matchup-zh">${g.away_name_zh ?? g.away_name} @ ${g.home_name_zh ?? g.home_name}</div>
      ${probBarHtml(g.home_team, g.away_team, g.home_win_prob, g.away_win_prob)}
      <div class="score-line">預測比分：${g.home_team} ${g.predicted_home_score.toFixed(1)} - ${g.predicted_away_score.toFixed(1)} ${g.away_team}</div>
      <div class="spread-line">模型分差：${g.home_team} ${g.predicted_margin >= 0 ? "+" : ""}${g.predicted_margin.toFixed(1)}${marketLine}</div>
      <div style="margin-top:8px">${contextBadgesHtml(g.context, restLabel)}</div>
    `;
    grid.appendChild(card);
  }
}

function renderRankings(rankings) {
  const body = document.getElementById("rankings-body");
  body.innerHTML = "";
  const max = Math.max(...rankings.map((r) => r.elo));
  const min = Math.min(...rankings.map((r) => r.elo));
  const span = Math.max(max - min, 1);

  for (const r of rankings) {
    const pct = ((r.elo - min) / span) * 100;
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${r.rank}</td>
      <td title="${r.name}">${teamLabel(r.team, r.name_zh)}</td>
      <td>${r.elo.toFixed(1)}</td>
      <td class="rating-bar-cell"><div class="rating-bar-track"><div class="rating-bar-fill" style="width:${pct}%"></div></div></td>
    `;
    body.appendChild(tr);
  }
}

function renderPerformance(performance) {
  const body = document.getElementById("perf-body");
  body.innerHTML = "";
  for (const key of ["elo", "advanced", "market"]) {
    const r = performance[key];
    const tr = document.createElement("tr");
    if (!r || r.error) {
      tr.innerHTML = `<td>${MODEL_LABELS[key]}</td><td colspan="4">${r && r.error ? r.error : "無資料"}</td>`;
    } else {
      tr.innerHTML = `
        <td>${MODEL_LABELS[key]}</td>
        <td>${r.games}</td>
        <td>${fmtPct(r.accuracy)}</td>
        <td>${r.brier_score.toFixed(4)}</td>
        <td>${r.spread_mae.toFixed(2)} 分</td>
      `;
    }
    body.appendChild(tr);
  }
}

// ---- client-side replica of the advanced model (logistic + linear) ----

function standardizedDot(model, values) {
  let z = model.intercept;
  for (let i = 0; i < values.length; i++) {
    z += ((values[i] - model.mean[i]) / model.std[i]) * model.coef[i];
  }
  return z;
}

function clientPredict(clientModel, homeCode, awayCode, opts) {
  const homeRating = clientModel.ratings[homeCode] ?? 1500;
  const awayRating = clientModel.ratings[awayCode] ?? 1500;
  const advantage = opts.neutral ? 0 : clientModel.home_advantage;
  const eloDiff = homeRating + advantage - awayRating;

  const featureValues = {
    elo_diff: eloDiff,
    rest_diff: opts.homeRest - opts.awayRest,
    div_game: opts.div ? 1 : 0,
    qb_change_home: opts.homeQbChange ? 1 : 0,
    qb_change_away: opts.awayQbChange ? 1 : 0,
    cold_game: opts.cold ? 1 : 0,
    windy_game: opts.windy ? 1 : 0,
    playoff: opts.playoff ? 1 : 0,
  };
  const vector = clientModel.feature_names.map((name) => featureValues[name] ?? 0);

  const winZ = standardizedDot(clientModel.win_model, vector);
  const homeWinProb = 1 / (1 + Math.exp(-winZ));
  const predictedMargin = standardizedDot(clientModel.margin_model, vector);
  const homeScore = Math.max((clientModel.avg_total_points + predictedMargin) / 2, 0);
  const awayScore = Math.max((clientModel.avg_total_points - predictedMargin) / 2, 0);

  return { homeRating, awayRating, homeWinProb, predictedMargin, homeScore, awayScore };
}

function wireUpPredictor(data) {
  const homeSel = document.getElementById("home-select");
  const awaySel = document.getElementById("away-select");
  const resultBox = document.getElementById("predictor-result");

  // default to an interesting matchup: the top two teams in the rankings
  if (data.power_rankings.length >= 2) {
    homeSel.value = data.power_rankings[0].team;
    awaySel.value = data.power_rankings[1].team;
  }

  document.getElementById("predict-btn").addEventListener("click", () => {
    const homeCode = homeSel.value;
    const awayCode = awaySel.value;
    if (homeCode === awayCode) {
      resultBox.hidden = false;
      resultBox.innerHTML = `<p class="section-note">請選擇兩支不同的球隊。</p>`;
      return;
    }

    const opts = {
      neutral: document.getElementById("opt-neutral").checked,
      div: document.getElementById("opt-div").checked,
      playoff: document.getElementById("opt-playoff").checked,
      homeQbChange: document.getElementById("opt-home-qb").checked,
      awayQbChange: document.getElementById("opt-away-qb").checked,
      cold: document.getElementById("opt-cold").checked,
      windy: document.getElementById("opt-windy").checked,
      homeRest: Number(document.getElementById("home-rest").value) || 7,
      awayRest: Number(document.getElementById("away-rest").value) || 7,
    };

    const r = clientPredict(data.client_model, homeCode, awayCode, opts);
    const homeName = findTeam(data.teams, homeCode)?.name_zh ?? homeCode;
    const awayName = findTeam(data.teams, awayCode)?.name_zh ?? awayCode;
    const favorite = r.homeWinProb >= 0.5 ? homeName : awayName;

    resultBox.hidden = false;
    resultBox.innerHTML = `
      <div class="result-teams">${awayName} (${awayCode}) @ ${homeName} (${homeCode})</div>
      <div class="result-meta">Elo 評等　${homeCode}: ${r.homeRating.toFixed(1)}　${awayCode}: ${r.awayRating.toFixed(1)}</div>
      ${probBarHtml(homeCode, awayCode, r.homeWinProb, 1 - r.homeWinProb)}
      <div class="score-line">預測比分：${homeCode} ${r.homeScore.toFixed(1)} - ${r.awayScore.toFixed(1)} ${awayCode}</div>
      <div class="spread-line">預測分差：${homeCode} ${r.predictedMargin >= 0 ? "+" : ""}${r.predictedMargin.toFixed(1)}</div>
      <div class="result-meta" style="margin-top:6px">預測勝方：<strong>${favorite}</strong></div>
    `;
  });
}

// ---------------------------------------------------------- history table --

const HISTORY_PAGE_SIZE = 25;
let HISTORY_STATE = null; // { all: [...], filtered: [...], page: 0 }

function pickCell(homeCode, awayCode, homeProb, correct) {
  if (homeProb === null || homeProb === undefined) {
    return `<span class="pred-na">尚無資料</span>`;
  }
  const pick = homeProb >= 0.5 ? homeCode : awayCode;
  const prob = homeProb >= 0.5 ? homeProb : 1 - homeProb;
  const icon = correct === null || correct === undefined
    ? `<span class="pred-na">—</span>`
    : correct
      ? `<span class="pred-correct">✓</span>`
      : `<span class="pred-wrong">✗</span>`;
  return `${pick} ${fmtPct(prob)} ${icon}`;
}

function marketCell(homeCode, awayCode, homeProb, spread, correct) {
  if (homeProb === null || homeProb === undefined) {
    return `<span class="pred-na">無盤口資料</span>`;
  }
  const base = pickCell(homeCode, awayCode, homeProb, correct);
  const spreadStr = spread !== null && spread !== undefined
    ? `（${homeCode} ${spread > 0 ? "-" : "+"}${Math.abs(spread).toFixed(1)}）`
    : "";
  return `${base}${spreadStr}`;
}

function applyHistoryFilters() {
  const seasonSel = document.getElementById("history-season-select");
  const filterSel = document.getElementById("history-filter-select");
  const season = seasonSel.value;
  const mode = filterSel.value;

  let rows = HISTORY_STATE.all;
  if (season !== "all") rows = rows.filter((r) => String(r.season) === season);
  if (mode === "adv-wrong") rows = rows.filter((r) => r.adv_correct === false);
  else if (mode === "adv-right") rows = rows.filter((r) => r.adv_correct === true);
  else if (mode === "upset") rows = rows.filter((r) => r.market_correct === false);

  HISTORY_STATE.filtered = rows;
  HISTORY_STATE.page = 0;
  renderHistoryPage();
}

function renderHistoryPage() {
  const { filtered, page } = HISTORY_STATE;
  const totalPages = Math.max(Math.ceil(filtered.length / HISTORY_PAGE_SIZE), 1);
  const clampedPage = Math.min(page, totalPages - 1);
  HISTORY_STATE.page = clampedPage;

  const start = clampedPage * HISTORY_PAGE_SIZE;
  const pageRows = filtered.slice(start, start + HISTORY_PAGE_SIZE);

  const body = document.getElementById("history-body");
  body.innerHTML = pageRows.map((r) => `
    <tr>
      <td>${r.date}</td>
      <td>
        ${r.away_team} @ ${r.home_team}
        <span class="team-cell-zh">${r.away_name_zh} @ ${r.home_name_zh}</span>
      </td>
      <td>${r.home_score}-${r.away_score}</td>
      <td>${pickCell(r.home_team, r.away_team, r.elo_home_win_prob, r.elo_correct)}</td>
      <td>${pickCell(r.home_team, r.away_team, r.adv_home_win_prob, r.adv_correct)}</td>
      <td>${marketCell(r.home_team, r.away_team, r.market_home_win_prob, r.market_spread, r.market_correct)}</td>
    </tr>
  `).join("");

  document.getElementById("history-count").textContent = `共 ${filtered.length} 場`;
  document.getElementById("history-page-label").textContent = `第 ${clampedPage + 1} / ${totalPages} 頁`;
  document.getElementById("history-prev").disabled = clampedPage <= 0;
  document.getElementById("history-next").disabled = clampedPage >= totalPages - 1;
}

function populateHistorySeasons(rows) {
  const seasonSel = document.getElementById("history-season-select");
  const seasons = [...new Set(rows.map((r) => r.season))].sort((a, b) => b - a);
  seasonSel.innerHTML = `<option value="all">全部賽季</option>` +
    seasons.map((s) => `<option value="${s}">${s} 賽季</option>`).join("");
}

function wireUpHistory() {
  const loadBtn = document.getElementById("load-history-btn");
  loadBtn.addEventListener("click", async () => {
    loadBtn.disabled = true;
    loadBtn.textContent = "載入中…";
    try {
      const res = await fetch("backtest_history.json", { cache: "no-store" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const payload = await res.json();
      const newestFirst = payload.games.slice().reverse(); // exported oldest-first; show recent games by default
      HISTORY_STATE = { all: newestFirst, filtered: newestFirst, page: 0 };

      populateHistorySeasons(payload.games);
      document.getElementById("history-controls").hidden = false;
      document.getElementById("history-table").hidden = false;
      document.getElementById("history-pager").hidden = false;
      loadBtn.hidden = true;

      document.getElementById("history-season-select").addEventListener("change", applyHistoryFilters);
      document.getElementById("history-filter-select").addEventListener("change", applyHistoryFilters);
      document.getElementById("history-prev").addEventListener("click", () => {
        HISTORY_STATE.page -= 1;
        renderHistoryPage();
      });
      document.getElementById("history-next").addEventListener("click", () => {
        HISTORY_STATE.page += 1;
        renderHistoryPage();
      });

      applyHistoryFilters();
    } catch (err) {
      loadBtn.disabled = false;
      loadBtn.textContent = "載入失敗，點此重試";
      console.error(err);
    }
  });
}

// ------------------------------------------------------- manual trigger ---

const TRIGGER_OWNER = "willy0929716513-debug";
const TRIGGER_REPO = "football";
const TRIGGER_WORKFLOW = "update-predictions.yml";
const TRIGGER_TOKEN_KEY = "nfl_predict_gh_token";

function setTriggerStatus(message, kind) {
  const el = document.getElementById("trigger-status");
  el.textContent = message;
  el.className = `trigger-status${kind ? ` ${kind}` : ""}`;
}

function wireUpTrigger() {
  const tokenInput = document.getElementById("trigger-token");
  const triggerBtn = document.getElementById("trigger-btn");
  const clearBtn = document.getElementById("trigger-clear-btn");

  if (localStorage.getItem(TRIGGER_TOKEN_KEY)) {
    tokenInput.placeholder = "已儲存權杖（留空並按下方按鈕即可使用已儲存的權杖）";
  }

  triggerBtn.addEventListener("click", async () => {
    const typed = tokenInput.value.trim();
    let savedNewToken = false;
    if (typed) {
      try {
        localStorage.setItem(TRIGGER_TOKEN_KEY, typed);
        savedNewToken = true;
      } catch {
        /* localStorage unavailable (private browsing etc.) — fall through and use the typed value once */
      }
    }
    const token = typed || localStorage.getItem(TRIGGER_TOKEN_KEY);
    if (!token) {
      setTriggerStatus("請先輸入 GitHub 權杖。", "err");
      return;
    }

    triggerBtn.disabled = true;
    triggerBtn.textContent = "觸發中…";
    setTriggerStatus(savedNewToken ? "已儲存權杖於本機瀏覽器，觸發中…" : "觸發中…");

    try {
      const res = await fetch(
        `https://api.github.com/repos/${TRIGGER_OWNER}/${TRIGGER_REPO}/actions/workflows/${TRIGGER_WORKFLOW}/dispatches`,
        {
          method: "POST",
          headers: {
            Authorization: `Bearer ${token}`,
            Accept: "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
          },
          body: JSON.stringify({ ref: "main" }),
        },
      );

      if (res.status === 204) {
        tokenInput.value = "";
        setTriggerStatus("已成功觸發更新！請稍候約 1-2 分鐘後重新整理頁面查看最新資料。", "ok");
      } else {
        let detail = `HTTP ${res.status}`;
        try {
          const body = await res.json();
          if (body && body.message) detail = body.message;
        } catch {
          /* response body wasn't JSON — keep the HTTP status as the detail */
        }
        setTriggerStatus(`觸發失敗：${detail}（請確認權杖是否有效、且已授權此 repo 的 Actions 讀寫權限）`, "err");
      }
    } catch (err) {
      console.error(err);
      setTriggerStatus("觸發失敗：網路或瀏覽器攔截了這個請求，請改用上方「GitHub Actions 頁面」手動觸發。", "err");
    } finally {
      triggerBtn.disabled = false;
      triggerBtn.textContent = "觸發更新";
    }
  });

  clearBtn.addEventListener("click", () => {
    localStorage.removeItem(TRIGGER_TOKEN_KEY);
    tokenInput.value = "";
    tokenInput.placeholder = "貼上 GitHub Personal Access Token（僅存於本機瀏覽器）";
    setTriggerStatus("已清除本機儲存的權杖。");
  });
}

async function main() {
  try {
    SITE_DATA = await loadData();
  } catch (err) {
    document.getElementById("updated-at").textContent = "資料載入失敗，請稍後再試。";
    console.error(err);
    return;
  }

  renderUpdatedAt(SITE_DATA.generated_at);
  populateTeamSelects(SITE_DATA.teams);
  populateWeekSelect(SITE_DATA.upcoming);
  renderRankings(SITE_DATA.power_rankings);
  renderPerformance(SITE_DATA.model_performance);
  wireUpPredictor(SITE_DATA);
  wireUpHistory();
  wireUpTrigger();

  const weekSel = document.getElementById("week-select");
  const rerenderGames = () => renderGamesForWeek(SITE_DATA.upcoming, weekSel.value);
  weekSel.addEventListener("change", rerenderGames);
  rerenderGames();
}

main();
