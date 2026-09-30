const CONTEXT_LABELS = {
  div_game: "分區內對戰",
  qb_change_home: "主隊QB異動",
  qb_change_away: "客隊QB異動",
  cold_game: "低溫比賽",
  windy_game: "強風比賽",
  playoff: "季後賽",
};

const MODEL_LABELS = {
  elo: "Elo 基準模型", advanced: "進階整合模型", market: "Vegas 市場盤口",
  blend: "整合預測（模型+盤口）",
};

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

  const etLine = `美東時間 ${md}${wd} ${g.gametime}`;
  const instant = kickoffInstant(g);
  if (!instant) return etLine;

  // Taiwan time first and most prominent — this site's primary audience —
  // with the US Eastern kickoff time (as officially published by the NFL)
  // shown underneath for reference.
  const twStr = instant.toLocaleString("zh-TW", {
    timeZone: "Asia/Taipei", month: "numeric", day: "numeric",
    hour: "2-digit", minute: "2-digit", hour12: false, weekday: "short",
  });
  return `台灣時間 ${twStr}<br><span class="kickoff-et">${etLine}</span>`;
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

function formatUpdatedAt(iso) {
  try {
    return new Date(iso).toLocaleString("zh-TW", { dateStyle: "medium", timeStyle: "short" });
  } catch {
    return iso;
  }
}

function renderUpdatedAt(iso) {
  const el = document.getElementById("updated-at");
  el.textContent = `模型與資料更新時間：${formatUpdatedAt(iso)}`;
}

function renderPredictionsUpdatedAt(iso) {
  const el = document.getElementById("predictions-updated-at");
  el.textContent = `這份預測資料更新於：${formatUpdatedAt(iso)}（每天台灣時間晚上 6 點自動更新）`;
}

// Headline stat at the very top of the page: the "本場推薦" (auto-pick
// between 讓分/不讓分) strategy's overall historical hit rate, taken straight
// from the same honest backtest numbers shown lower down in the performance
// table (recommendation_summary_from_history in advanced_model.py) — just
// surfaced up front so it doesn't require scrolling to find.
function renderHeroRecommendationStat(performance) {
  const el = document.getElementById("hero-recommendation-stat");
  if (!el) return;
  const r = performance && performance.recommendation;
  if (!r || r.error) {
    el.textContent = "總推薦勝率：尚無足夠資料可回測";
    return;
  }
  el.innerHTML = `總推薦勝率：<strong>${fmtPct(r.accuracy)}</strong>（回測 ${r.games} 場，2015 年至今）`;
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

function marginSentence(homeCode, awayCode, margin, label) {
  const rounded = Math.round(Math.abs(margin));
  if (rounded === 0) return `${label}：勢均力敵，難分軒輊`;
  const winner = margin >= 0 ? homeCode : awayCode;
  return `${label}：${winner} 會贏 ${rounded} 分`;
}

function scoreLineHtml(homeCode, awayCode, homeScore, awayScore) {
  return `<div class="score-line">預測比分：${homeCode} ${Math.round(homeScore)} - ${Math.round(awayScore)} ${awayCode}</div>`;
}

function marginLineHtml(homeCode, awayCode, predictedMargin) {
  return `<div class="spread-line">${marginSentence(homeCode, awayCode, predictedMargin, "模型預測")}</div>`;
}

// Traditional point-spread ("讓分") notation: the favored team shown with a
// negative number (how many points they must win by to "cover"), the
// underdog with the same magnitude as a positive number. Built from the
// real market_spread (positive = home favored), never the model's own
// margin — this line is specifically the real quoted odds.
function formatSpread(homeCode, awayCode, marketSpread) {
  if (marketSpread === undefined || marketSpread === null) return null;
  if (marketSpread === 0) return `${homeCode} 拿捏（pk） / ${awayCode} 拿捏（pk）`;
  const favCode = marketSpread > 0 ? homeCode : awayCode;
  const dogCode = marketSpread > 0 ? awayCode : homeCode;
  const mag = Math.abs(marketSpread).toFixed(1).replace(/\.0$/, "");
  return `${favCode} -${mag} / ${dogCode} +${mag}`;
}

function spreadLineHtml(homeCode, awayCode, marketSpread) {
  const spread = formatSpread(homeCode, awayCode, marketSpread);
  if (!spread) return "";
  return `<div class="spread-line">讓分（Vegas 真實盤口）：${spread}</div>`;
}

function blendLineHtml(homeCode, awayCode, blendHomeWinProb) {
  if (blendHomeWinProb === undefined || blendHomeWinProb === null) return "";
  const pick = blendHomeWinProb >= 0.5 ? homeCode : awayCode;
  const prob = blendHomeWinProb >= 0.5 ? blendHomeWinProb : 1 - blendHomeWinProb;
  return `<div class="blend-line">整合預測（模型 + 真實賠率）：<strong>${pick}</strong> 獲勝機率 <strong>${fmtPct(prob)}</strong></div>`;
}

function favoriteLineHtml(homeCode, homeNameZh, awayCode, awayNameZh, homeProb) {
  const homeFavored = homeProb >= 0.5;
  const code = homeFavored ? homeCode : awayCode;
  const nameZh = homeFavored ? homeNameZh : awayNameZh;
  const prob = homeFavored ? homeProb : 1 - homeProb;
  return `<div class="favorite-line">預測勝方：<strong>${nameZh}（${code}）</strong> — 獲勝機率 <strong>${fmtPct(prob)}</strong></div>`;
}

// "本場推薦" — automatically picks whichever bet type we're more confident
// about: the straight-up ("不讓分") moneyline pick, or the against-the-
// spread ("讓分") pick. Both confidences are computed server-side (see
// pick_recommendation() in advanced_model.py) and whichever is higher wins;
// this line surfaces that choice up front, with the underlying favourite/
// spread/blend lines still shown below for full transparency.
function recommendationLineHtml(g) {
  if (!g.recommendation_type) return "";
  const isAts = g.recommendation_type === "ats";
  const pick = g.recommendation_team;
  const confidence = fmtPct(g.recommendation_confidence);
  const typeLabel = isAts ? "讓分" : "不讓分";
  let detail;
  if (isAts) {
    const spreadStr = formatSpread(g.home_team, g.away_team, g.market_spread);
    detail = `${pick} 讓分過盤（盤口 ${spreadStr}）`;
  } else {
    detail = `${pick} 直接獲勝`;
  }
  return `
    <div class="recommend-line">
      本場推薦（${typeLabel}）：<strong>${detail}</strong> — 信心 <strong>${confidence}</strong>
    </div>`;
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

  renderParlay(games);

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

    const card = document.createElement("div");
    card.className = "game-card";
    card.innerHTML = `
      <div class="kickoff">${formatKickoff(g)}</div>
      <div class="matchup"><span>${g.away_team}</span><span>@</span><span>${g.home_team}</span></div>
      <div class="matchup-zh">${g.away_name_zh ?? g.away_name} @ ${g.home_name_zh ?? g.home_name}</div>
      ${probBarHtml(g.home_team, g.away_team, g.home_win_prob, g.away_win_prob)}
      ${recommendationLineHtml(g)}
      ${favoriteLineHtml(g.home_team, g.home_name_zh ?? g.home_name, g.away_team, g.away_name_zh ?? g.away_name, g.home_win_prob)}
      ${blendLineHtml(g.home_team, g.away_team, g.blend_home_win_prob)}
      ${scoreLineHtml(g.home_team, g.away_team, g.predicted_home_score, g.predicted_away_score)}
      ${marginLineHtml(g.home_team, g.away_team, g.predicted_margin)}
      ${spreadLineHtml(g.home_team, g.away_team, g.market_spread)}
      <div style="margin-top:8px">${contextBadgesHtml(g.context, restLabel)}</div>
    `;
    grid.appendChild(card);
  }
}

// ------------------------------------------------------------- 串關 (parlay) --

// Kept in sync with DEFAULT_PARLAY_LEGS / DEFAULT_PARLAY_MIN_CONFIDENCE in
// advanced_model.py, so the live weekly parlay shown here (computed
// client-side from the already-exported recommendation fields) matches the
// exact same rule the honest historical backtest in `model_performance.parlay`
// is scored against.
const PARLAY_LEGS = 3;
const PARLAY_MIN_CONFIDENCE = 0.6;

function pickParlayLegs(games, legs = PARLAY_LEGS, minConfidence = PARLAY_MIN_CONFIDENCE) {
  return games
    .filter((g) => g.recommendation_confidence !== undefined && g.recommendation_confidence !== null
      && g.recommendation_confidence >= minConfidence)
    .sort((a, b) => b.recommendation_confidence - a.recommendation_confidence)
    .slice(0, legs);
}

function combinedParlayProbability(legs) {
  return legs.reduce((acc, g) => acc * g.recommendation_confidence, 1);
}

function parlayLegDetail(g) {
  if (g.recommendation_type === "ats") {
    const spreadStr = formatSpread(g.home_team, g.away_team, g.market_spread);
    return `${g.recommendation_team} 讓分過盤（${spreadStr}）`;
  }
  return `${g.recommendation_team} 直接獲勝`;
}

function renderParlay(games) {
  const body = document.getElementById("parlay-body");
  const label = document.getElementById("parlay-legs-label");
  if (label) label.textContent = String(PARLAY_LEGS);
  if (!body) return;

  const legs = pickParlayLegs(games);
  if (legs.length < PARLAY_LEGS) {
    body.innerHTML = `<p class="section-note">這週信心達 ${fmtPct(PARLAY_MIN_CONFIDENCE)} 以上的比賽不到 ${PARLAY_LEGS} 場，暫不建議湊成串關。</p>`;
    return;
  }

  const combined = combinedParlayProbability(legs);
  const fairOdds = (1 / combined).toFixed(2);
  const legsHtml = legs.map((g) => {
    const typeLabel = g.recommendation_type === "ats" ? "讓分" : "不讓分";
    return `<li><strong>${g.away_team} @ ${g.home_team}</strong>（${typeLabel}）：${parlayLegDetail(g)} — 信心 ${fmtPct(g.recommendation_confidence)}</li>`;
  }).join("");

  body.innerHTML = `
    <ol class="parlay-legs">${legsHtml}</ol>
    <p class="parlay-combined">全部命中機率（假設各場獨立）：<strong>${fmtPct(combined)}</strong>　公平賠率約 <strong>${fairOdds}</strong> 倍</p>
    <p class="section-note">
      誠實提醒：串關只要有一場沒中，整組就不算贏 — 就算每一場單獨看都有六七成信心，串在一起的整體
      機率還是會掉很多。上面的機率是假設「各場比賽互相獨立」乘出來的簡化估計，實際上同一週的比賽可能有
      共通因素（例如同樣的天氣系統）。這個策略本身過去的真實命中率，請見下方「模型準確度與市場比較」。
    </p>
  `;
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
  for (const key of ["elo", "advanced", "market", "blend"]) {
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
  renderRecommendationSummary(performance.recommendation);
  renderParlaySummary(performance.parlay);
}

function renderRecommendationSummary(r) {
  const el = document.getElementById("recommendation-summary-note");
  if (!el) return;
  if (!r || r.error) {
    el.textContent = "「智慧推薦」（自動在讓分／不讓分之間選信心較高者）目前尚無足夠資料可回測。";
    return;
  }
  const atsAcc = r.ats_accuracy !== null && r.ats_accuracy !== undefined ? fmtPct(r.ats_accuracy) : "無資料";
  const mlAcc = r.moneyline_accuracy !== null && r.moneyline_accuracy !== undefined ? fmtPct(r.moneyline_accuracy) : "無資料";
  el.innerHTML = `
    「智慧推薦」白話說明：每場比賽自動比較「不讓分」（猜贏家）跟「讓分」（猜誰能過盤）兩種推薦
    各自的信心，選信心較高的那一種顯示。回測結果：共 ${r.games} 場比賽中，選了 ${r.ats_recommended}
    場「讓分」推薦（命中率 ${atsAcc}）、${r.moneyline_recommended} 場「不讓分」推薦（命中率 ${mlAcc}），
    整體命中率 <strong>${fmtPct(r.accuracy)}</strong>。誠實說：這只是「自動選擇信心較高的一種」，
    不代表能穩定打敗市場。
  `;
}

function renderParlaySummary(r) {
  const el = document.getElementById("parlay-summary-note");
  if (!el) return;
  if (!r || r.error) {
    el.textContent = `「本週串關推薦」（每週信心最高的 ${PARLAY_LEGS} 場湊成一組）目前尚無足夠資料可回測。`;
    return;
  }
  el.innerHTML = `
    「本週串關推薦」白話說明：每週從當時信心最高、達到門檻的比賽中選 ${r.avg_legs.toFixed(1)}
    場（平均）湊成一組串關，全部命中才算贏。誠實回測：${r.weeks} 週裡有 ${r.parlays_formed}
    週湊得出一組完整的串關，其中真正全部命中的有 ${r.parlays_hit} 週，命中率
    <strong>${fmtPct(r.hit_rate)}</strong>（平均事前估計的全部命中機率約 ${fmtPct(r.avg_combined_probability)}）。
    誠實說：串關本來就是「一場沒中全組泡湯」的高風險玩法，命中率一定會比單場推薦低很多，這裡如實
    呈現，不是每週都湊得出來、也不是每次湊出來都會中。
  `;
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

    resultBox.hidden = false;
    resultBox.innerHTML = `
      <div class="result-teams">${awayName} (${awayCode}) @ ${homeName} (${homeCode})</div>
      <div class="result-meta">戰力值　${homeCode}: ${r.homeRating.toFixed(1)}　${awayCode}: ${r.awayRating.toFixed(1)}</div>
      ${probBarHtml(homeCode, awayCode, r.homeWinProb, 1 - r.homeWinProb)}
      ${favoriteLineHtml(homeCode, homeName, awayCode, awayName, r.homeWinProb)}
      ${scoreLineHtml(homeCode, awayCode, r.homeScore, r.awayScore)}
      ${marginLineHtml(homeCode, awayCode, r.predictedMargin)}
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
  const spreadStr = formatSpread(homeCode, awayCode, spread);
  return spreadStr ? `${base}（讓分 ${spreadStr}）` : base;
}

function recommendationCell(r) {
  if (!r.recommendation_type) {
    return `<span class="pred-na">尚無資料</span>`;
  }
  const typeLabel = r.recommendation_type === "ats" ? "讓分" : "不讓分";
  const conf = fmtPct(r.recommendation_confidence);
  const icon = r.recommendation_correct === null || r.recommendation_correct === undefined
    ? `<span class="pred-na">—</span>`
    : r.recommendation_correct
      ? `<span class="pred-correct">✓</span>`
      : `<span class="pred-wrong">✗</span>`;
  return `${r.recommendation_team}（${typeLabel}）${conf} ${icon}`;
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
      <td>${pickCell(r.home_team, r.away_team, r.blend_home_win_prob, r.blend_correct)}</td>
      <td>${recommendationCell(r)}</td>
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
      const updatedEl = document.getElementById("history-updated-at");
      updatedEl.textContent = `這份歷史回測資料更新於：${formatUpdatedAt(payload.generated_at)}`;
      updatedEl.hidden = false;
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
const TRIGGER_REPO = "Football";
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
    document.getElementById("predictions-updated-at").textContent = "資料載入失敗，請稍後再試。";
    document.getElementById("hero-recommendation-stat").textContent = "資料載入失敗，請稍後再試。";
    console.error(err);
    return;
  }

  renderUpdatedAt(SITE_DATA.generated_at);
  renderPredictionsUpdatedAt(SITE_DATA.generated_at);
  renderHeroRecommendationStat(SITE_DATA.model_performance);
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
