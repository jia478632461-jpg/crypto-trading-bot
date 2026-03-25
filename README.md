# 🤖 Crypto Trading Bot

量化交易机器人 | 技术分析 + 博弈论 + 回测引擎 + 情绪分析 + 链上数据

## 模块列表

| 文件 | 功能 |
|------|------|
| `src/unified_engine.py` | 统一分析引擎（整合所有模块）|
| `src/analytics_engine.py` | 技术分析（60+指标）|
| `src/game_theory.py` | 统计模型 & 博弈论分析 |
| `src/adaptive_strategy.py` | 自适应策略（9种策略）|
| `src/sentiment_analysis.py` | 情绪分析 |
| `src/onchain_data.py` | 链上数据分析 |
| `src/backtester.py` | 回测引擎 |
| `src/risk_manager.py` | 风险管理 |
| `src/trading_engine.py` | 交易执行 |
| `src/lstm_model.py` | LSTM 模型 |

## 快速开始

```bash
pip install numpy pandas scipy scikit-learn
python src/unified_engine.py
```
