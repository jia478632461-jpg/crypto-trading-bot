"""
LSTM模型模块
用于价格预测的LSTM神经网络
"""

import numpy as np
import pandas as pd
import logging
from typing import Tuple, Optional
import os

logger = logging.getLogger(__name__)

# TensorFlow imports
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
    HAS_TF = True
except ImportError:
    HAS_TF = False
    logger.warning("TensorFlow未安装，模型功能将不可用")

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


class LSTMPricePredictor:
    """LSTM价格预测模型"""

    def __init__(self, config: dict):
        """
        初始化LSTM模型

        Args:
            config: 配置字典
        """
        self.config = config
        self.sequence_length = config["model"]["sequence_length"]
        self.prediction_steps = config["timeframe"]["prediction_steps"]
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_columns = None

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        准备特征列

        Args:
            df: 包含价格和技术指标的DataFrame

        Returns:
            特征DataFrame
        """
        # 基础特征
        base_features = [
            "open", "high", "low", "close", "volume",
            "ma_7", "ma_25", "ma_99",
            "rsi", "macd", "signal", "macd_hist",
            "bb_position", "volume_ratio",
            "price_change", "price_change_24h", "volatility"
        ]

        # 可选特征：市场宏观特征
        if "fear_greed" in df.columns:
            base_features.extend(["fear_greed", "btc_dominance"])

        # 只保留存在的列
        self.feature_columns = [col for col in base_features if col in df.columns]

        return df[self.feature_columns].copy()

    def create_sequences(self, data: np.ndarray, sequence_length: int,
                         prediction_steps: int = 24) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建时间序列样本

        Args:
            data: 标准化后的数据
            sequence_length: 输入序列长度
            prediction_steps: 预测的未来步数

        Returns:
            X, y 训练数据
        """
        X, y = [], []
        for i in range(len(data) - sequence_length - prediction_steps):
            X.append(data[i:(i + sequence_length)])
            # 预测未来价格是涨还是跌（分类标签）
            current_price = data[i + sequence_length - 1, 3]  # close价格
            future_price = data[i + sequence_length + prediction_steps - 1, 3]
            label = 1 if future_price > current_price else 0
            y.append(label)

        return np.array(X), np.array(y)

    def build_model(self, input_shape: Tuple[int, int]) -> 'Sequential':
        """
        构建LSTM模型

        Args:
            input_shape: 输入形状 (sequence_length, n_features)

        Returns:
            Keras模型
        """
        model = Sequential(name="LSTM_Price_Predictor")

        # 第一层LSTM
        model.add(LSTM(
            units=self.config["model"]["lstm_units"][0],
            return_sequences=True,
            input_shape=input_shape
        ))
        model.add(BatchNormalization())
        model.add(Dropout(self.config["model"]["dropout_rate"]))

        # 第二层LSTM
        model.add(LSTM(
            units=self.config["model"]["lstm_units"][1],
            return_sequences=True
        ))
        model.add(BatchNormalization())
        model.add(Dropout(self.config["model"]["dropout_rate"]))

        # 第三层LSTM
        model.add(LSTM(
            units=self.config["model"]["lstm_units"][2],
            return_sequences=False
        ))
        model.add(BatchNormalization())
        model.add(Dropout(self.config["model"]["dropout_rate"]))

        # 全连接层
        model.add(Dense(self.config["model"]["dense_units"], activation="relu"))
        model.add(Dropout(0.2))

        # 输出层：二分类（涨/跌）
        model.add(Dense(1, activation="sigmoid"))

        # 编译模型
        optimizer = Adam(learning_rate=0.001)
        model.compile(
            optimizer=optimizer,
            loss="binary_crossentropy",
            metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
        )

        model.summary()
        return model

    def train(self, df: pd.DataFrame, symbol: str = None) -> dict:
        """
        训练模型

        Args:
            df: 训练数据DataFrame
            symbol: 交易对名称（用于保存）

        Returns:
            训练历史
        """
        if not HAS_TF:
            logger.error("TensorFlow未安装，无法训练模型")
            return {}

        logger.info(f"开始训练模型...")

        # 准备特征
        features_df = self.prepare_features(df)

        # 标准化数据
        scaled_data = self.scaler.fit_transform(features_df.values)
        scaled_df = pd.DataFrame(scaled_data, columns=features_df.columns)

        # 保存scaler
        self.scaled_columns = features_df.columns.tolist()

        # 创建序列
        X, y = self.create_sequences(
            scaled_df.values,
            self.sequence_length,
            self.prediction_steps
        )

        logger.info(f"训练数据形状: X={X.shape}, y={y.shape}")
        logger.info(f"正样本比例: {y.mean():.2%}")

        # 分割训练集和验证集
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.config["model"]["validation_split"],
            shuffle=False  # 时序数据不能打乱
        )

        # 构建模型
        self.model = self.build_model(input_shape=(X.shape[1], X.shape[2]))

        # 回调函数
        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=self.config["model"]["early_stopping_patience"],
                restore_best_weights=True
            )
        ]

        # 训练模型
        history = self.model.fit(
            X_train, y_train,
            epochs=self.config["model"]["epochs"],
            batch_size=self.config["model"]["batch_size"],
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )

        # 保存模型 - 每个交易对独立模型文件
        base_path = self.config["model"]["model_path"].replace(".h5", "")
        os.makedirs(os.path.dirname(base_path) or "models", exist_ok=True)
        if symbol:
            model_path = f"{base_path}_{symbol.lower()}.h5"
        else:
            model_path = f"{base_path}.h5"
        self.model.save(model_path)
        logger.info(f"模型已保存到: {model_path}")

        # 保存scaler
        import pickle
        scaler_path = model_path.replace(".h5", "_scaler.pkl")
        with open(scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)

        return {
            "history": history.history,
            "model_path": model_path,
            "final_accuracy": history.history["accuracy"][-1],
            "final_val_accuracy": history.history["val_accuracy"][-1]
        }

    def load(self, model_path: str = None) -> bool:
        """
        加载预训练模型

        Args:
            model_path: 模型路径

        Returns:
            是否加载成功
        """
        if not HAS_TF:
            return False

        if model_path is None:
            model_path = self.config["model"]["model_path"]

        if not os.path.exists(model_path):
            logger.warning(f"模型文件不存在: {model_path}")
            return False

        try:
            self.model = load_model(model_path)

            # 加载scaler
            scaler_path = model_path.replace(".h5", "_scaler.pkl")
            if os.path.exists(scaler_path):
                import pickle
                with open(scaler_path, "rb") as f:
                    self.scaler = pickle.load(f)

            logger.info(f"模型加载成功: {model_path}")
            return True
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            return False

    def predict(self, df: pd.DataFrame) -> dict:
        """
        预测价格走势

        Args:
            df: 最新市场数据DataFrame

        Returns:
            预测结果字典
        """
        if self.model is None:
            if not self.load():
                logger.error("模型未加载")
                return {"signal": "HOLD", "confidence": 0, "error": "模型未加载"}

        try:
            # 准备特征
            features_df = self.prepare_features(df)

            # 标准化
            scaled_data = self.scaler.transform(features_df.values)

            # 取最后sequence_length个时间步
            if len(scaled_data) < self.sequence_length:
                logger.warning(f"数据长度不足: {len(scaled_data)} < {self.sequence_length}")
                return {"signal": "HOLD", "confidence": 0, "error": "数据不足"}

            input_data = scaled_data[-self.sequence_length:]
            input_data = input_data.reshape(1, self.sequence_length, len(self.feature_columns))

            # 预测
            prediction = self.model.predict(input_data, verbose=0)[0][0]

            # 根据阈值生成信号
            buy_threshold = self.config["model"]["buy_threshold"]
            sell_threshold = self.config["model"]["sell_threshold"]

            if prediction >= buy_threshold:
                signal = "BUY"
                confidence = prediction
            elif prediction <= sell_threshold:
                signal = "SELL"
                confidence = 1 - prediction
            else:
                signal = "HOLD"
                confidence = 1 - abs(prediction - 0.5) * 2

            return {
                "signal": signal,
                "confidence": float(confidence),
                "probability_up": float(prediction),
                "probability_down": float(1 - prediction)
            }

        except Exception as e:
            logger.error(f"预测失败: {e}")
            return {"signal": "HOLD", "confidence": 0, "error": str(e)}


class ModelManager:
    """模型管理器 - 管理多个交易对的模型"""

    def __init__(self, config: dict, api_key: str = None, api_secret: str = None):
        self.config = config
        self.api_key = api_key
        self.api_secret = api_secret
        self.models = {}
        self.data_fetcher = None

        # 延迟导入避免循环依赖
        from src.data_fetcher import DataAggregator
        self.data_fetcher = DataAggregator(config, api_key, api_secret)

    def train_all_models(self, symbols: list = None) -> dict:
        """
        训练所有交易对的模型

        Args:
            symbols: 交易对列表

        Returns:
            训练结果
        """
        if symbols is None:
            symbols = self.config["symbols"]["primary_pairs"]

        results = {}

        for symbol in symbols:
            logger.info(f"========== 训练 {symbol} ==========")

            try:
                # 获取数据
                df = self.data_fetcher.prepare_training_data(symbol)
                if df.empty:
                    logger.warning(f"{symbol} 无数据，跳过")
                    continue

                # 训练模型
                predictor = LSTMPricePredictor(self.config)
                result = predictor.train(df, symbol)

                self.models[symbol] = predictor
                results[symbol] = {"status": "success", **result}

            except Exception as e:
                logger.error(f"{symbol} 训练失败: {e}")
                results[symbol] = {"status": "failed", "error": str(e)}

        return results

    def predict_all(self, symbol: str) -> dict:
        """
        获取所有交易对的预测

        Args:
            symbol: 可选，只预测特定交易对

        Returns:
            预测结果字典
        """
        results = {}
        symbols = [symbol] if symbol else list(self.models.keys())

        for sym in symbols:
            if sym not in self.models:
                logger.warning(f"{sym} 模型不存在")
                continue

            try:
                df = self.data_fetcher.get_latest_data(sym)
                if df.empty:
                    continue

                prediction = self.models[sym].predict(df)
                results[sym] = prediction

            except Exception as e:
                logger.error(f"{sym} 预测失败: {e}")
                results[sym] = {"signal": "HOLD", "error": str(e)}

        return results

    def load_all_models(self) -> int:
        """
        加载所有已保存的模型

        Returns:
            加载的模型数量
        """
        loaded = 0
        base_path = self.config["model"]["model_path"].replace(".h5", "")
        for symbol in self.config["symbols"]["primary_pairs"]:
            model_path = f"{base_path}_{symbol.lower()}.h5"

            predictor = LSTMPricePredictor(self.config)
            if predictor.load(model_path):
                self.models[symbol] = predictor
                loaded += 1
            else:
                logger.warning(f"模型文件不存在: {model_path}")

        logger.info(f"加载了 {loaded} 个模型")
        return loaded
