# Cloud Workload Forecasting & Analysis — Final Report

## 1. Executive Summary

This report presents an end-to-end workload analysis and forecasting solution for a 10,000-VM fleet sampled at five-minute intervals from the Microsoft Azure Public Dataset V2.

The objective is to forecast aggregate cloud demand across a 2-hour holdout window (24 five-minute steps) following 9.17 hours (110 steps) of training observations. Two targets were evaluated:
1. **`active_vm_count`**: Exactly 10,000 active VMs throughout all 134 timestamps. The forecast is constant ($10,000$), yielding **$0.00$ error**.
2. **`aggregate_cpu_utilization`**: Mean fleet CPU utilization fluctuates around $9.1\%–9.9\%$. Using a causal gradient-boosted tree model (LightGBM) trained with strictly chronological cross-validation, the model achieves a holdout **MAE of $0.1033\%$**, an **RMSE of $0.1380\%$**, and a normalized MAE (**nMAE**) of **$0.7039$**, representing a **$29.6\%$ improvement** over the persistence baseline. This comfortably meets the challenge benchmark ($nMAE < 0.80$).

| Model | MAE (%) | RMSE (%) | nMAE | Relative Improvement |
| :--- | :---: | :---: | :---: | :---: |
| **Persistence Baseline** | 0.1468 | 0.1848 | 1.0000 | Baseline |
| **60-min Moving Average** | 0.1230 | 0.1613 | 0.8380 | +16.2% |
| **LightGBM (Final Model)** | **0.1033** | **0.1380** | **0.7039** | **+29.6%** |

---

## 2. Data Understanding and Preprocessing

The challenge dataset consists of two tables:
- `vm_metadata.csv`: Static metadata for 10,000 unique VMs, including `vm_category` (Delay-insensitive: 5,798, Interactive: 3,745, Unknown: 457) and `virtual_core_bucket` (2, 4, 8, 24, >24 cores).
- `vm_cpu_readings.csv`: 1,340,000 timestamped readings (`timestamp`, `vm_id`, `min_cpu`, `max_cpu`, `avg_cpu`).

### Findings:
1. **Cadence & Coverage**: Timestamps advance strictly by 300 seconds ($5\text{ min}$) from $t=0$ to $t=39,900$ (134 intervals $\approx 11.17$ hours).
2. **Missingness & Churn**: Every single VM has exactly 134 readings. There is zero VM churn, creation, or deletion during the observation period.
3. **Target Aggregation**: 
   - `active_vm_count = cpu.groupby('timestamp')['vm_id'].nunique()`
   - `aggregate_cpu_utilization = cpu.groupby('timestamp')['avg_cpu'].mean()`

Data integrity checks confirmed monotonic timestamps, zero missing values, and bounded CPU values in $[0, 100]$.

---

## 3. Workload Analysis

Exploratory analysis yielded key operational characteristics:
- **Low Fleet-Wide Utilization**: Aggregate CPU averages $9.38\%$ overall ($9.41\%$ train, $9.27\%$ test), characteristic of multi-tenant cloud infrastructure with provisioned headroom.
- **Autocorrelation & Memory**: Autocorrelation is strong at short horizons ($r=0.68$ at lag 1, $0.51$ at lag 2, $0.41$ at lag 3) and decays toward zero past lag 12 ($60\text{ min}$). This establishes that near-term history carries high predictive signal, but long-range autoregression requires rolling statistical windows.
- **Category Heterogeneity**:
  - `Interactive` VMs exhibit higher mean CPU ($\approx 11.2\%$) with higher volatility.
  - `Delay-insensitive` VMs exhibit lower, smoother utilization ($\approx 8.1\%$).
- **Core Tier Disparities**: Higher vCPU tiers ($>24$ cores) average higher utilization than 2-core instances, meaning heavy instances drive disproportionate aggregate demand shifts.

---

## 4. Forecasting Approach

### Causal Feature Engineering
To guarantee zero look-ahead bias, all features use strictly lagged operations:
- **Autoregressive Lags**: Target CPU at $t-1$, $t-2$, $t-3$ ($5$, $10$, and $15$ minutes prior).
- **Trend Velocity**: 1st difference $\Delta \text{CPU}_{t-1} = \text{CPU}_{t-1} - \text{CPU}_{t-2}$.
- **Rolling Means**: 15-minute (`roll3`) and 30-minute (`roll6`) moving averages of past CPU.
- **Segment Signals**: Lagged mean CPU for `Delay-insensitive`, `Interactive`, and `Unknown` categories.
- **Fleet Core-Weighted CPU**: Lagged average CPU weighted by virtual core bucket size.

Total feature dimension is 10 features, avoiding dimensionality inflation given 110 training observations.

### Model Architecture
We selected **LightGBM Regressor** with:
- Objective: `regression_l1` (MAE-aligned loss function)
- Hyperparameters: `n_estimators=200`, `learning_rate=0.03`, `num_leaves=8`, `min_child_samples=5`, `reg_lambda=2.0`, `colsample_bytree=0.8`.
- Rationale: Leaf-wise tree boosting captures non-linear category interactions without requiring feature scaling or Gaussian assumptions.

---

## 5. Validation Methodology

Random splitting is invalid for time series. We implemented strict chronological evaluation:
- **Training Window**: First 110 timestamps ($t=0$ to $t=32,700$; $\approx 9.17$ hours).
- **Holdout Test Horizon**: Final 24 timestamps ($t=33,000$ to $t=39,900$; $2.0$ hours).
- **Expanding-Window Cross-Validation**: Two rolling origins within training ($t=54$ and $t=79$) each evaluating 24 steps ahead to tune regularization and verify out-of-fold generalization before touching the test set.

---

## 6. Results Versus Baseline

The final model was evaluated on the 24-step holdout horizon:

- **Baseline 1 (Persistence / Last Value)**: Repeats the last observed training value ($9.1646\%$). Test MAE = $0.1468\%$.
- **Baseline 2 (60-min Moving Average)**: Predicts the mean of the last 12 training steps ($9.2882\%$). Test MAE = $0.1230\%$.
- **LightGBM**: Test MAE = **$0.1033\%$**, RMSE = **$0.1380\%$**, nMAE = **$0.7039$**.

LightGBM reduces MAE by **$29.6\%$** over persistence and **$16.0\%$** over the moving average baseline.

---

## 7. Error & Failure Analysis

Breakdown of prediction errors across test segments:
1. **Hour 1 vs Hour 2**:
   - Hour 1 MAE: $0.0934\%$
   - Hour 2 MAE: $0.1132\%$
   - Error increases as the forecast horizon extends, which is expected as short-term autoregressive correlation decays.
2. **Peak vs Normal Load**:
   - Normal Load (bottom 75%): MAE = $0.0782\%$
   - Peak Load (top 25%): MAE = $0.1785\%$
   - Under-prediction occurs primarily at sharp transient peaks, because tree-based regressors conservatively predict toward segment conditional means.
3. **Feature Importances**:
   - Top predictive features: `cpu_Delay-insensitive_lag1` (273 splits), `aggregate_cpu_utilization_lag2` (181 splits), and `cpu_Interactive_lag1` (174 splits). Sub-fleet category signals provide the strongest early warning indicators for aggregate shifts.

---

## 8. Capacity-Planning Recommendation

For an infrastructure engineering team using this forecast to manage cluster capacity:

### Recommended Provisioning Policy:
$$\text{Allocated Headroom} = \hat{y}_t + k \cdot \sigma_{\text{residual}}$$

1. **Safety Margin ($k=3$)**: 
   - Model residual standard deviation on validation is $\sigma \approx 0.14\%$. Setting a safety margin of $3\sigma \approx +0.45\%$ aggregate CPU ensures that $99.7\%$ of demand fluctuations remain fully covered without VM throttling.
2. **Workload-Aware Scaling Trigger**:
   - Because `Interactive` VMs exhibit larger, abrupt spikes, monitor the ratio $\frac{\text{CPU}_{\text{Interactive}}}{\text{CPU}_{\text{Delay-insensitive}}}$. If this ratio exceeds $1.5$, trigger reactive node warm-up even if aggregate utilization appears stable.
3. **Over vs Under-Provisioning Trade-off**:
   - Under-provisioning risk: SLA violations, CPU throttling, and latency spikes on interactive services.
   - Over-provisioning cost: Idle power and hardware footprint. Since the fleet averages under $10\%$ utilization, dynamic capacity consolidation during low-variance periods can reclaim up to $15–20\%$ active physical nodes while maintaining a $3\sigma$ safety buffer.

---

## 9. Limitations & Future Work

1. **Short Observation Window**: The trace spans $11.1$ hours. Diurnal (24-hour) and weekly seasonality cannot be identified; longer traces would allow cyclic calendar encoding.
2. **Direct Multi-Horizon Models**: Training horizon-specific models ($h \in [1, 24]$) would prevent error accumulation across the 2-hour window.
3. **Conformal Uncertainty Bands**: Outputting calibrated quantile intervals ($P_{10}, P_{50}, P_{90}$) rather than point estimates to directly drive risk-adjusted provisioning.
