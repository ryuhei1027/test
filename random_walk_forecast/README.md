# ランダムウォークによる翌日価格予測

対数価格 $X_t = \log S_t$ がランダムウォークに従うと仮定し、翌日の価格を予測します。

$$
X_{t+1} = X_t + \mu + \sigma \varepsilon_{t+1}, \qquad \varepsilon_{t+1} \sim \text{i.i.d. } N(0,1)
$$

基本モデルは $\mu = 0$（ドリフトなし）です。予想値には中央値 $S_t e^{\mu}$ を使い、予測区間は $S_t \exp(\mu \pm z_{1-\alpha/2}\,\sigma)$ で計算します。

## 実行方法

```bash
python random_walk.py data/prices_2026-10-19.xlsx -o forecast_2026-10-20.xlsx
```

## 出力（Excel）

| シート | 内容 |
|---|---|
| 予測 | 予想値、68%・95% 予測区間、モンテカルロ法による検算、ドリフトありモデルの参考値 |
| バックテスト | 1期先予測の検証結果（RMSE、MAE、95% 区間のカバー率） |
| バックテスト明細 | 各予測と実績の一覧 |
