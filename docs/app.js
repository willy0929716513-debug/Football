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

  const weekSel = document.getElementById("week-select");
  const rerenderGames = () => renderGamesForWeek(SITE_DATA.upcoming, weekSel.value);
  weekSel.addEventListener("change", rerenderGames);
  rerenderGames();
}

main();
