# Football — 美式足球（NFL）比賽預測工具

一套以 **Elo 評等系統** 為核心的專業 NFL 比賽預測工具，方法論參考 FiveThirtyEight
知名的 NFL Elo 模型：勝分差修正（margin-of-victory）、主場優勢加成、季後賽權重提升、
賽季間回歸平均值。只使用 Python 標準函式庫，不需安裝任何第三方套件即可運作。

內建資料為 **真實歷史賽事**：1999 年至今（含目前進行中的賽季）共 7000+ 場已完賽的
NFL 常規賽與季後賽比分，取自公開的 [nflverse/nfldata](https://github.com/nflverse/nfldata)
專案，並已將搬遷過的球隊（如 OAK→LV、SD→LAC、STL→LA）統一為現行代碼，讓 Elo 歷史
連續不中斷。

## 安裝

```bash
# 不需要安裝任何相依套件，Python 3.9+ 即可直接執行：
python3 -m football_predictor.cli --help

# 或安裝為指令列工具：
pip install -e .
nflpredict --help
```

## 使用方式

### 1. 訓練 Elo 評等（`train`）

用歷史比賽資料，依時間順序逐場更新每支球隊的 Elo 評等，並校準「評等差 → 預測分差」
的比例係數，最後把結果存成 `ratings.json`。

```bash
nflpredict train --games data/nfl_games.csv --out ratings.json
```

輸出範例：

```
trained on 7278 games (1999-2026)
  training accuracy : 64.4%
  training log loss : 0.6316
  spread scale      : 23.80 elo points / point
  avg total points  : 44.3
ratings saved to ratings.json
```

可調整的參數：

| 參數 | 說明 | 預設值 |
| --- | --- | --- |
| `--k` | Elo K 因子（每場比賽最大評等變動幅度） | 20 |
| `--home-advantage` | 主場優勢的 Elo 加成 | 48 |
| `--playoff-boost` | 季後賽的 K 因子倍率 | 1.2 |
| `--revert` | 每個新賽季開始時，評等回歸 1500 平均值的比例 | 1/3 |

### 2. 預測比賽（`predict`）

```bash
nflpredict predict --home KC --away BUF
```

```
Buffalo Bills (BUF) @ Kansas City Chiefs (KC)  [home field]
------------------------------------------------------------
  Elo ratings        : KC: 1488.4   BUF: 1601.0
  Win probability    : KC: 40.8%   BUF: 59.2%
  Predicted score    : KC 20.8 - 23.5 BUF
  Predicted margin   : KC -2.7
  Favorite           : Buffalo Bills
```

加上 `--neutral` 可模擬中立場地（如超級盃）；加上 `--json` 可輸出機器可讀格式，方便
整合進其他程式或網頁後端。

### 3. 查看戰力排名（`ratings`）

```bash
nflpredict ratings --top 10
```

### 4. 回測驗證準確度（`backtest`）

用「隨時間往前走」的方式驗證模型：每場比賽先用當下評等做預測、記錄結果，再更新評等，
避免用到未來資訊（look-ahead bias）。

```bash
nflpredict backtest --start-season 2020
```

```
backtest: seasons >= 2020, 1695 games
  winner accuracy : 63.7%
  Brier score     : 0.2219  (lower is better, 0.25 = coin flip)
  spread MAE      : 10.18 points
```

（此結果與公開的 538 NFL Elo 模型長期準確率 ~63-64% 相近，可作為模型健全度的參考。）

## 專案結構

```
football_predictor/
  elo.py     # Elo 評等系統核心邏輯
  data.py    # 比賽資料讀取、球隊代碼正規化
  model.py   # 預測模型：分差校準、比分推算、回測評估
  cli.py     # 命令列介面
data/
  nfl_games.csv   # 1999 年至今的真實 NFL 比賽結果
tests/          # pytest 單元測試
```

## 使用自己的資料

`train` / `backtest` 接受任何符合以下欄位的 CSV：

```
season, week, game_type, gameday, home_team, away_team, home_score, away_score, home_rest, away_rest, div_game
```

只有 `season`、`home_team`、`away_team`、`home_score`、`away_score` 是必要欄位，
其餘欄位缺漏時會使用預設值。`game_type` 非 `REG` 者會被視為季後賽（套用
`--playoff-boost`）。

## 執行測試

```bash
pip install pytest
python3 -m pytest
```

## 模型說明與限制

- 這是一個**分數/戰績驅動**的統計模型，不包含即時因素，例如受傷名單、天氣、
  博弈盤口、教練異動等；如需更高精準度，可將這些訊號另外整合進評等或作為
  賠率校正。
- Elo 只反映「球隊整體實力的相對強弱」，預測比分是由評等差線性換算而來，
  屬於粗略估計，實際比分變異度遠大於模型給出的期望值。
- 本工具目的為展示一套嚴謹、可回測、方法論公開透明的預測框架，適合用於研究、
  分析與教育用途；不構成博弈建議。
