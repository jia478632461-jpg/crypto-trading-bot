"""
LSTM加密货币自动交易机器人 - 主程序
Cryptocurrency LSTM Trading Bot - Main Entry Point
"""

import os
import sys
import logging
import argparse
import yaml
from datetime import datetime
import pandas as pd

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_fetcher import BinanceDataFetcher, DataAggregator
from src.lstm_model import LSTMPricePredictor, ModelManager
from src.risk_manager import PositionManager
from src.trading_engine import TradingEngine, StrategyEngine

# 配置日志
def setup_logging(config: dict):
    """配置日志"""
    log_dir = os.path.dirname(config["logging"]["log_file"])
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, config["logging"]["level"]),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(config["logging"]["log_file"]),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(__name__)


class TradingBot:
    """交易机器人主类"""

    def __init__(self, config_path: str = "config/config.yaml"):
        # 加载配置
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        # 设置日志
        self.logger = setup_logging(self.config)

        # API密钥
        self.api_key = os.getenv("BINANCE_API_KEY")
        self.api_secret = os.getenv("BINANCE_API_SECRET")

        if not self.api_key:
            self.logger.warning("未设置BINANCE_API_KEY环境变量")

        # 初始化组件
        self.binance = BinanceDataFetcher(
            self.api_key,
            self.api_secret,
            self.config["runtime"]["testnet"]
        )

        self.data_fetcher = DataAggregator(
            self.config,
            self.api_key,
            self.api_secret
        )

        self.position_manager = PositionManager(
            self.config,
            self.binance
        )

        self.trading_engine = TradingEngine(
            self.config,
            self.binance
        )

        self.model_manager = ModelManager(
            self.config,
            self.api_key,
            self.api_secret
        )

        self.strategy_engine = StrategyEngine(
            self.config,
            self.trading_engine,
            self.position_manager,
            self.model_manager
        )

        self.running = False
        self.logger.info("交易机器人初始化完成")

    def train_models(self, symbols: list = None):
        if symbols is None:
            symbols = self.config["symbols"]["primary_pairs"]

        self.logger.info(f"开始训练 {len(symbols)} 个交易对的模型...")
        results = self.model_manager.train_all_models(symbols)

        for symbol, result in results.items():
            if result.get("status") == "success":
                self.logger.info(
                    f"{symbol} 训练完成 - 准确率: {result.get('final_accuracy', 0):.2%}"
                )
            else:
                self.logger.error(f"{symbol} 训练失败: {result.get('error')}")

    def load_models(self):
        loaded = self.model_manager.load_all_models()
        self.logger.info(f"加载了 {loaded} 个模型")

    def run_once(self):
        try:
            self.logger.info("=" * 50)
            self.logger.info(f"交易循环开始 - {datetime.now()}")

            results = self.strategy_engine.execute_strategy()

            if results["actions"]:
                for action in results["actions"]:
                    self.logger.info(f"执行动作: {action}")
            else:
                self.logger.info("本次循环无交易动作")

            summary = self.position_manager.get_portfolio_summary()
            self.logger.info(f"当前资产: {summary['total_value']:.2f} USDT")
            self.logger.info(f"持仓数量: {summary['open_positions']}")
            self.logger.info(f"未实现盈亏: {summary['unrealized_pnl']:.2f} USDT")

            risk = self.position_manager.get_risk_metrics()
            self.logger.info(f"今日交易次数: {risk.daily_trades}")

            return results

        except Exception as e:
            self.logger.error(f"交易循环执行失败: {e}", exc_info=True)
            return {"error": str(e)}

    def run(self, interval: int = None):
        if interval is None:
            interval = self.config["runtime"]["run_interval"]

        self.running = True
        self.logger.info(f"交易机器人开始运行，间隔: {interval}秒")

        if self.config["runtime"]["auto_train"]:
            if not self.model_manager.models:
                self.logger.info("开始自动训练模型...")
                self.train_models()

        while self.running:
            try:
                self.run_once()
                import time
                time.sleep(interval)
            except KeyboardInterrupt:
                self.logger.info("收到停止信号，正在关闭...")
                self.stop()
                break

        self.logger.info("交易机器人已停止")

    def stop(self):
        self.running = False
        log_dir = os.path.dirname(self.config["logging"]["trade_log_file"])
        os.makedirs(log_dir, exist_ok=True)
        self.position_manager.export_trade_history(
            self.config["logging"]["trade_log_file"]
        )

    def get_status(self) -> dict:
        return {
            "running": self.running,
            "models_loaded": len(self.model_manager.models),
            "positions": len(self.position_manager.positions),
            "portfolio": self.position_manager.get_portfolio_summary(),
            "risk": self.position_manager.get_risk_metrics().__dict__
        }

    def backtest(self, start_date: str, end_date: str):
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        self.logger.info(f"开始回测: {start_date} - {end_date}")
        self.train_models()
        results = self.strategy_engine.run_backtest(
            self.config["symbols"]["primary_pairs"], start, end
        )
        print("\n" + "=" * 50)
        print("回测结果")
        print("=" * 50)
        print(results)


def main():
    parser = argparse.ArgumentParser(description="LSTM加密货币交易机器人")
    parser.add_argument("--config", default="config/config.yaml", help="配置文件路径")
    parser.add_argument("--mode", choices=["train", "backtest", "live", "test"],
                       default="live", help="运行模式")
    parser.add_argument("--train", action="store_true", help="训练模型")
    parser.add_argument("--interval", type=int, default=60, help="运行间隔(秒)")

    args = parser.parse_args()
    bot = TradingBot(args.config)

    if args.mode == "train" or args.train:
        bot.train_models()
        print("模型训练完成!")

    elif args.mode == "backtest":
        bot.backtest(
            (datetime.now() - pd.Timedelta(days=30)).strftime("%Y-%m-%d"),
            datetime.now().strftime("%Y-%m-%d")
        )

    elif args.mode == "test":
        bot.load_models()
        results = bot.run_once()
        print(f"\n测试结果: {results}")

    elif args.mode == "live":
        bot.load_models()
        if not bot.model_manager.models:
            print("未找到已训练的模型，是否现在训练? (y/n)")
            response = input()
            if response.lower() == "y":
                bot.train_models()
            else:
                print("警告: 没有模型，机器人将以模拟模式运行")
        bot.run(interval=args.interval)


if __name__ == "__main__":
    main()
