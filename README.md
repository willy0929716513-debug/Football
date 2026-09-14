# Football — 美式足球（NFL）比賽預測工具

一套專業的 NFL 比賽預測工具，核心是 **Elo 戰力評等**，並疊加 **休息天數／短週賽、先發QB異動、
比賽天氣、分區內對戰、季後賽情境** 等會實際影響比賽結果的因素，透過邏輯迴歸／線性迴歸整合成
一套「進階整合模型」。所有方法論公開透明、可回測，並誠實列出與真實 Vegas 收盤盤口的準確度比較。

**線上網站（GitHub Pages）**：戰力排名、本週／整季預測、互動式對戰預測器、模型準確度公開頁 —
見下方「GitHub Pages 網站」章節。

只使用 Python 標準函式庫 + numpy，不需要任何其他第三方套件即可運作。

內建資料為 **真實歷史賽事**：1999 年至今（含目前進行中的賽季，以及已公布的完整未來賽程）共 7000+
場已完賽的 NFL 常規賽與季後賽，以及該賽季所有已公布的未來賽程，取自公開的
[nflverse/nfldata](https://github.com/nflverse/nfldata) 專案，內容包含比分、休息天數、天氣、
先發QB、教練、以及 Vegas 收盤盤口（分差／美式賠率）。已將搬遷過的球隊（如 OAK→LV、SD→LAC、
STL→LA）統一為現行代碼，讓 Elo 歷史連續不中斷。所有 32 支球隊皆附有正體中文隊名（見下方對照表）。

### 盤口資料是真實的嗎？

是。`spread_line` / `home_moneyline` / `away_moneyline` / `total_line` 等欄位是 nflverse/nfldata
官方資料字典（`nflreadr::dictionary_schedules`，[來源](https://raw.githubusercontent.com/nflverse/nflreadr/master/data-raw/dictionary_schedules.csv)）
明確定義的**真實歷史收盤盤口**（"Odds for home/away team to win the game" / "The spread line for
the game"），不是模擬或虛構數據。開發時也另外做了獨立驗證：抽查歷史上懸殊的比賽（例如 2013年
DEN vs JAX，DEN 主場美式賠率 -5000、JAX 客場 +2173），確認 `spread_line` 的正負號與方向、與該場
比賽實際的大幅懸殊賠率完全吻合（大幅領先的一方對應大幅有利的賠率與盤口），並且抽查了 2026 年已
公布的未來賽程盤口，同樣可對應到已知的真實對戰強弱關係。`market` 回測（見下方）就是直接用這些
真實盤口計算準確率，而非用模型自己的預測結果假裝是「市場」。

## 兩種模型

| 模型 | 說明 | CLI 參數 |
| --- | --- | --- |
| `elo` | 純 Elo 評等基準模型（538 NFL Elo 方法論） | `--model elo` |
| `advanced`（預設） | Elo 評等 + 休息天數 + 先發QB異動 + 天氣 + 分區／季後賽情境，透過邏輯迴歸／線性迴歸整合 | `--model advanced` |

### 進階模型考慮的因素

- **Elo 戰力評等**：逐場更新，內建勝分差修正（大勝/大敗調整幅度更大）、主場優勢加成（預設 48 分）、
  季後賽權重提升（預設 ×1.2）、賽季間回歸平均值（預設 1/3）。
- **休息天數差（rest_diff）**：主客兩隊的休息天數差，反映短週四夜賽的疲勞劣勢、待命週後的體能優勢。
- **先發QB異動**：追蹤每隊最近已知的先發QB，偵測是否臨時更換 — 這是單場比賽結果最大的擺動因子之一，
  且對於未來賽程也能用官方已公布的預期先發QB提前判斷。
- **天氣與場地**：戶外比賽的低溫（≤0°C）、強風（≥15mph）條件；室內／圓頂球場不受影響。
- **分區內對戰／季後賽情境**：分區對手較熟悉彼此、季後賽強度與變異度不同。
- **市場賠率校驗**：使用真實歷史 Vegas 收盤盤口（分差與美式賠率）作為**獨立比較基準**（不是模型輸入），
  確保評估公正透明。

不包含的因素（超出目前資料來源範圍）：即時傷兵名單細節、比賽當下即時天氣預報（僅使用比賽日已知的
歷史/預報氣象資料）、教練臨場戰術調整、球員層級進階數據（如 EPA/CPOE）。這些都是可以在未來擴充的方向，
詳見下方「已知限制」。

## 安裝

```bash
pip install -r requirements.txt          # numpy + pytest
python3 -m football_predictor.cli --help

# 或安裝為指令列工具：
pip install -e .
nflpredict --help
```

## 使用方式

### 1. 訓練模型（`train`）

```bash
nflpredict train --model advanced          # 預設，寫入 ratings_advanced.json
nflpredict train --model elo               # 純 Elo 基準模型，寫入 ratings.json
```

輸出範例（advanced）：

```
[advanced] trained on 7278 games (1999-2026)
  training accuracy : 64.7%
  training log loss : 0.6277
  spread MAE        : 10.51 points
  avg total points  : 44.3
  feature weights (standardized logistic regression coefficients):
    elo_diff        +0.697
    rest_diff       +0.061
    div_game        -0.047
    qb_change_home  -0.132
    qb_change_away  +0.109
    cold_game       +0.040
    windy_game      -0.013
    playoff         +0.005
```

可調整的 Elo 參數：`--k`（K因子，預設20）、`--home-advantage`（主場加成，預設48）、
`--playoff-boost`（季後賽K因子倍率，預設1.2）、`--revert`（賽季間回歸平均值比例，預設1/3）。

### 2. 預測比賽（`predict`）

```bash
nflpredict predict --home KC --away BUF
```

```
[advanced] Buffalo Bills (BUF) @ Kansas City Chiefs (KC)  [home field]
------------------------------------------------------------
  Elo ratings        : KC: 1488.4   BUF: 1601.0
  Win probability    : KC: 43.4%   BUF: 56.6%
  Predicted score    : KC 21.1 - 23.1 BUF
  Predicted margin   : KC -2.0
  Favorite           : Buffalo Bills
  Context factors    : (none — default/neutral context)
```

`advanced` 模型可用旗標描述比賽情境（用於「若⋯會如何」的假設情境分析）：

```bash
nflpredict predict --home KC --away BUF \
  --div --away-qb-change --home-rest 10 --away-rest 4
```

支援旗標：`--neutral`（中立場地）、`--div`（分區內對戰）、`--playoff`（季後賽）、
`--home-rest N` / `--away-rest N`（休息天數，預設7）、`--home-qb-change` / `--away-qb-change`
（先發QB臨時異動）、`--cold`（低溫戶外比賽）、`--windy`（強風戶外比賽）。加上 `--json`
可輸出機器可讀格式。

### 3. 查看戰力排名（`ratings`）

```bash
nflpredict ratings --top 10
```

### 4. 回測驗證準確度（`backtest`）

用「隨時間往前走」(walk-forward) 的方式驗證模型：每場比賽先用當下已知的評等/模型做預測、記錄結果，
再更新評等，避免用到未來資訊（look-ahead bias）。`advanced` 模型每個賽季開打前只用先前賽季的資料
重新校準一次迴歸係數，同樣不會偷看未來。

```bash
nflpredict backtest --start-season 2015 --model all
```

```
backtest: seasons >= 2015
  [elo] 3030 games
      winner accuracy : 64.3%
      Brier score     : 0.2214  (lower is better, 0.25 = coin flip)
      spread MAE      : 10.20 points
  [advanced] 3030 games
      winner accuracy : 64.1%
      Brier score     : 0.2206  (lower is better, 0.25 = coin flip)
      spread MAE      : 10.18 points
  [market] 3029 games
      winner accuracy : 66.0%
      Brier score     : 0.2116  (lower is better, 0.25 = coin flip)
      spread MAE      : 9.81 points
```

`market` 是同期真實 Vegas 收盤盤口的表現，作為誠實的對照基準：長期穩定打敗收盤盤口本身就是眾所皆知
的難題，這裡如實呈現而非誇大宣稱「打敗莊家」。

### 5. 匯出網站資料（`export-site`）

```bash
nflpredict export-site --start-season 2015
```

重新訓練 elo / advanced 兩個模型、跑三方回測比較，並將戰力排名、整季賽程預測、模型準確度、以及給
瀏覽器端互動預測器使用的模型快照，全部寫入 `docs/data.json`（GitHub Pages 網站讀取的資料檔）。

## GitHub Pages 網站

`docs/` 目錄是一個純靜態網站（HTML/CSS/JS，無需建置工具），內容包含：

- **互動式對戰預測器**：任選兩隊、開關情境因子（主客場、分區戰、季後賽、QB異動、天氣、休息天數），
  瀏覽器端即時運算勝率與預測比分（與後端訓練用的是同一套模型參數）。
- **本週／整季預測**：目前已公布賽程的每一場比賽，含勝率、預測比分、預測分差，並列出市場盤口對照。
- **戰力排名**：目前 Elo 評等排序。
- **模型準確度與市場比較**：公開的回測結果，`elo` / `advanced` / `market` 三方比較。
- **方法論頁**：列出模型考慮的所有因素與已知限制。

### 啟用方式（一次性設定）

此 repo 的 GitHub Pages 尚未啟用，需要 repo 擁有者手動開啟一次：

1. 到 repo 的 **Settings → Pages**
2. **Source** 選擇 `Deploy from a branch`
3. **Branch** 選 `main`，資料夾選 `/docs`，按下 **Save**
4. 幾分鐘後網站會發佈在 `https://<你的帳號>.github.io/football/`

啟用後，`.github/workflows/update-predictions.yml` 會自動：每週二 UTC 10:00（約在
Monday Night Football 結束後）重新抓取最新真實比賽資料、重新訓練模型、重新產生
`docs/data.json`，並自動 commit + push — 網站內容會持續保持在最新戰績與賽程之上，
不需要手動維護。也可以在 GitHub 的 Actions 頁面手動觸發（`workflow_dispatch`）。

## 球隊代碼對照表

| 代碼 | 中文隊名 | 代碼 | 中文隊名 | 代碼 | 中文隊名 | 代碼 | 中文隊名 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BUF | 水牛城比爾 | BAL | 巴爾的摩烏鴉 | HOU | 休士頓德州人 | DEN | 丹佛野馬 |
| MIA | 邁阿密海豚 | CIN | 辛辛那提孟加拉虎 | IND | 印第安納波利斯小馬 | KC | 堪薩斯城酋長 |
| NE | 新英格蘭愛國者 | CLE | 克里夫蘭布朗 | JAX | 傑克遜維爾美洲豹 | LV | 拉斯維加斯突襲者 |
| NYJ | 紐約噴射機 | PIT | 匹茲堡鋼人 | TEN | 田納西泰坦 | LAC | 洛杉磯電光 |
| DAL | 達拉斯牛仔 | CHI | 芝加哥熊 | ATL | 亞特蘭大獵鷹 | ARI | 亞利桑那紅雀 |
| NYG | 紐約巨人 | DET | 底特律雄獅 | CAR | 卡羅萊納黑豹 | LA | 洛杉磯公羊 |
| PHI | 費城老鷹 | GB | 綠灣包裝工 | NO | 紐奧良聖徒 | SF | 舊金山49人 |
| WAS | 華盛頓指揮官 | MIN | 明尼蘇達維京人 | TB | 坦帕灣海盜 | SEA | 西雅圖海鷹 |

（依 AFC/NFC 東北南西分區排列；程式內部對應表見 `football_predictor/data.py` 的 `TEAM_NAMES_ZH`。）

## 專案結構

```
football_predictor/
  elo.py             # Elo 評等系統核心邏輯
  data.py            # 比賽資料讀取（含未來賽程）、球隊代碼正規化
  features.py        # 情境特徵：休息天數、QB異動追蹤、天氣、分區/季後賽旗標
  regression.py       # 純 numpy 的邏輯迴歸（IRLS）與 ridge 線性迴歸
  model.py           # Elo 基準模型（FootballPredictor）+ 回測
  advanced_model.py  # 進階整合模型（AdvancedPredictor）+ 回測 + 市場盤口回測
  cli.py             # 命令列介面
data/
  nfl_games.csv      # 1999 年至今的真實 NFL 比賽結果與未來賽程
docs/
  index.html / style.css / app.js / data.json   # GitHub Pages 靜態網站
scripts/
  fetch_data.py      # 從 nflverse/nfldata 重新抓取並正規化資料
.github/workflows/
  update-predictions.yml   # 每週自動重新訓練並更新網站資料
tests/               # pytest 單元測試（34 個測試，涵蓋各模組）
```

## 使用自己的資料

`train` / `backtest` / `export-site` 接受任何符合以下欄位的 CSV（`data/nfl_games.csv` 的欄位）：

```
season, week, game_type, gameday, home_team, away_team, home_score, away_score,
home_rest, away_rest, div_game, roof, surface, temp, wind,
home_qb_id, away_qb_id, home_qb_name, away_qb_name, home_coach, away_coach,
spread_line, home_moneyline, away_moneyline, total_line
```

只有 `season`、`home_team`、`away_team`、`home_score`、`away_score` 是必要欄位（未賽的比賽把分數
留空即可，會被視為未來賽程），其餘欄位缺漏時會使用預設值 / 不觸發該項情境因子。`game_type` 非
`REG` 者會被視為季後賽。想更新為最新資料，執行 `python3 scripts/fetch_data.py`。

## 執行測試

```bash
pip install -r requirements.txt
python3 -m pytest
```

## 模型說明與已知限制

- 這是一個**戰績與可量化情境因子驅動**的統計模型；不包含即時傷兵名單細節、臨場天氣預報（僅用比賽日
  已知資料）、教練臨場調度、球員層級進階數據（EPA/CPOE等）。若要進一步提升準確度，可以把這些訊號
  另外整合進特徵向量。
- Elo 只反映「球隊整體實力的相對強弱」，預測比分是由評等差與情境特徵的迴歸結果換算而來，屬於統計估計，
  實際比分變異度遠大於模型給出的期望值。
- 情境特徵刻意只用連續型的 `rest_diff`，而不是額外疊加「短週賽」二元旗標：測試時發現兩者高度共線，
  會讓迴歸係數方向不穩定（甚至反直覺地翻轉），單一連續特徵反而更穩健。
- 誠實的基準比較：同期真實 Vegas 收盤盤口的勝負準確率通常會略優於本模型 —
  這是公開的統計事實（市場資訊包含了傷兵報告等本模型沒有的內部資訊），不代表模型無效，而是展示
  一套嚴謹、可回測、方法論公開透明的框架，讓使用者能自行判斷模型的可信度。
- 本工具目的為研究、分析與教育用途，**不構成博弈建議**。
