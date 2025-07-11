#!/usr/bin/env python3
"""
Modern Trading Strategies - Estrategias Modernas con backtesting.py
===================================================================

Implementación profesional de estrategias GRID, DCA y BTD usando la librería
backtesting.py estándar de la industria. Estas estrategias reemplazan las
implementaciones caseras por versiones más robustas y eficientes.

Características:
- Uso de backtesting.py para mayor profesionalismo
- Métricas automáticas (Sharpe, Calmar, etc.)
- Visualizaciones integradas
- Mayor eficiencia de cálculo
- Compatibilidad con optimizadores externos
"""

import os
import sys
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import logging

# Agregar el directorio padre al path para importar módulos del proyecto
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GridTradingStrategy(Strategy):
    """
    Estrategia de Grid Trading modernizada usando backtesting.py.
    
    Implementa un sistema de rejilla (grid) que coloca órdenes de compra y venta
    en niveles predefinidos alrededor del precio actual.
    
    Parámetros:
    - levels: Número de niveles en el grid (3-8)
    - range_percent: Porcentaje de rango del grid (2.0-15.0)
    - umbral_adx: Filtro ADX para mercados laterales (15.0-40.0)
    - umbral_volatilidad: Filtro de volatilidad mínima (0.01-0.05)
    - umbral_sentimiento: Filtro de sentimiento (-0.3-0.3)
    """
    
    # Parámetros optimizables
    levels = 4
    range_percent = 8.0
    umbral_adx = 25.0
    umbral_volatilidad = 0.02
    umbral_sentimiento = 0.0
    
    def init(self):
        """Inicializa la estrategia y calcula indicadores necesarios."""
        
        # Validar que tenemos las columnas necesarias
        required_columns = ['Open', 'High', 'Low', 'Close']
        for col in required_columns:
            if col not in self.data.df.columns:
                raise ValueError(f"Columna faltante: {col}")
        
        # Calcular indicadores técnicos si no están presentes
        self.setup_indicators()
        
        # Variables de estado del grid
        self.grid_initialized = False
        self.buy_levels = []
        self.sell_levels = []
        self.grid_center = None
        self.active_orders = {}
        
        # Tamaño de posición fijo (porcentaje del capital)
        self.position_size = 1.0 / self.levels  # Dividir capital entre niveles
        
        logger.info(f"🔧 GridStrategy inicializada: {self.levels} niveles, ±{self.range_percent}%")
    
    def setup_indicators(self):
        """Calcula indicadores técnicos necesarios."""
        try:
            # ADX (Average Directional Index)
            if 'adx' not in self.data.df.columns:
                self.adx = self.calculate_adx()
            else:
                self.adx = self.data.df['adx']
            
            # Volatilidad
            if 'volatility' not in self.data.df.columns:
                self.volatility = self.calculate_volatility()
            else:
                self.volatility = self.data.df['volatility']
            
            # Sentimiento (si está disponible)
            if 'sentiment_ma7' in self.data.df.columns:
                self.sentiment = self.data.df['sentiment_ma7']
            else:
                # Crear sentimiento neutral si no está disponible
                self.sentiment = pd.Series(0.0, index=self.data.df.index)
        
        except Exception as e:
            logger.warning(f"Error calculando indicadores: {e}")
            # Valores por defecto en caso de error
            self.adx = pd.Series(25.0, index=self.data.df.index)
            self.volatility = pd.Series(0.03, index=self.data.df.index)
            self.sentiment = pd.Series(0.0, index=self.data.df.index)
    
    def calculate_adx(self, period: int = 14) -> pd.Series:
        """Calcula ADX simplificado usando operaciones numpy básicas."""
        try:
            # Usar arrays numpy directamente
            high = np.array(self.data.High)
            low = np.array(self.data.Low) 
            close = np.array(self.data.Close)
            
            # Cálculo simplificado de ADX
            if len(high) < period:
                return pd.Series(25.0, index=self.data.df.index)
            
            # True Range básico
            tr = high - low
            
            # Media móvil simple del TR como proxy de ADX
            adx_values = np.convolve(tr, np.ones(period)/period, mode='same')
            
            # Normalizar entre 0-100
            adx_values = np.clip(adx_values * 100 / np.max(adx_values), 0, 100)
            
            return pd.Series(adx_values, index=self.data.df.index)
        
        except Exception:
            return pd.Series(25.0, index=self.data.df.index)
    
    def calculate_volatility(self, period: int = 20) -> pd.Series:
        """Calcula volatilidad usando operaciones numpy básicas."""
        try:
            close = np.array(self.data.Close)
            
            if len(close) < 2:
                return pd.Series(0.03, index=self.data.df.index)
            
            # Calcular retornos
            returns = np.diff(close) / close[:-1]
            
            # Volatilidad con ventana deslizante
            vol_values = []
            for i in range(len(close)):
                if i < period:
                    vol_values.append(0.03)  # Valor por defecto
                else:
                    window_returns = returns[i-period:i]
                    vol = np.std(window_returns) if len(window_returns) > 0 else 0.03
                    vol_values.append(vol)
            
            return pd.Series(vol_values, index=self.data.df.index)
        
        except Exception:
            return pd.Series(0.03, index=self.data.df.index)
    
    def should_operate(self, i: int) -> bool:
        """Determina si debe operar basándose en filtros técnicos."""
        try:
            # Verificar que tenemos datos suficientes
            if i < 20:  # Necesitamos al menos 20 períodos para indicadores
                return False
            
            current_adx = self.adx.iloc[i] if i < len(self.adx) else 25.0
            current_vol = self.volatility.iloc[i] if i < len(self.volatility) else 0.03
            current_sent = self.sentiment.iloc[i] if i < len(self.sentiment) else 0.0
            
            # Aplicar filtros
            adx_ok = current_adx < self.umbral_adx  # Mercado lateral
            vol_ok = current_vol > self.umbral_volatilidad  # Suficiente volatilidad
            sent_ok = current_sent > self.umbral_sentimiento  # Sentimiento positivo
            
            return adx_ok and vol_ok and sent_ok
        
        except Exception:
            return True  # Por defecto operar si hay problemas
    
    def initialize_grid(self, current_price: float):
        """Inicializa el grid alrededor del precio actual."""
        try:
            range_value = current_price * (self.range_percent / 100)
            min_price = current_price - (range_value / 2)
            max_price = current_price + (range_value / 2)
            
            # Calcular niveles de compra (por debajo del precio)
            buy_levels_count = self.levels // 2
            if buy_levels_count > 0:
                buy_step = (current_price - min_price) / buy_levels_count
                self.buy_levels = [current_price - buy_step * (i + 1) 
                                 for i in range(buy_levels_count)]
            
            # Calcular niveles de venta (por encima del precio)
            sell_levels_count = self.levels - buy_levels_count
            if sell_levels_count > 0:
                sell_step = (max_price - current_price) / sell_levels_count
                self.sell_levels = [current_price + sell_step * (i + 1) 
                                  for i in range(sell_levels_count)]
            
            self.grid_center = current_price
            self.grid_initialized = True
            
            logger.debug(f"Grid inicializado: centro={current_price:.4f}, "
                        f"compra={len(self.buy_levels)}, venta={len(self.sell_levels)}")
        
        except Exception as e:
            logger.error(f"Error inicializando grid: {e}")
    
    def next(self):
        """Lógica principal ejecutada en cada barra."""
        try:
            current_price = self.data.Close[-1]
            current_index = len(self.data) - 1
            
            # Verificar filtros técnicos
            if not self.should_operate(current_index):
                return
            
            # Inicializar grid si es necesario
            if not self.grid_initialized:
                self.initialize_grid(current_price)
                return
            
            # Lógica de grid trading
            self.execute_grid_logic(current_price)
        
        except Exception as e:
            logger.error(f"Error en next(): {e}")
    
    def execute_grid_logic(self, current_price: float):
        """Ejecuta la lógica del grid trading."""
        try:
            # Verificar niveles de compra
            for level in self.buy_levels:
                if (current_price <= level and 
                    not self.position and 
                    level not in self.active_orders):
                    
                    # Comprar en nivel de soporte
                    self.buy(size=self.position_size)
                    self.active_orders[level] = 'buy'
                    logger.debug(f"Compra en nivel {level:.4f}")
                    break
            
            # Verificar niveles de venta
            for level in self.sell_levels:
                if (current_price >= level and 
                    self.position and 
                    level not in self.active_orders):
                    
                    # Vender en nivel de resistencia
                    self.sell(size=self.position.size)
                    self.active_orders[level] = 'sell'
                    logger.debug(f"Venta en nivel {level:.4f}")
                    break
        
        except Exception as e:
            logger.error(f"Error ejecutando grid: {e}")


class DCAStrategy(Strategy):
    """
    Estrategia de Dollar Cost Averaging (DCA) modernizada.
    
    Compra de forma periódica cuando detecta tendencias alcistas
    y dips, vendiendo con objetivo de ganancia fijo.
    
    Parámetros:
    - intervalo_compra: Días entre compras (1-7)
    - monto_compra: Monto por compra como % del capital (0.1-1.0)
    - objetivo_ganancia: % de ganancia objetivo (0.05-0.30)
    - dip_threshold: % de caída para detectar dip (0.02-0.15)
    - tendencia_alcista_dias: Días para confirmar tendencia (3-14)
    - stop_loss: % de pérdida máxima (0.10-0.40)
    """
    
    # Parámetros optimizables
    intervalo_compra = 3
    monto_compra = 0.2  # 20% del capital por compra
    objetivo_ganancia = 0.15  # 15% de ganancia
    dip_threshold = 0.05  # 5% de caída
    tendencia_alcista_dias = 7
    stop_loss = 0.20  # 20% stop loss
    
    def init(self):
        """Inicializa la estrategia DCA."""
        
        # Calcular medias móviles para tendencia
        self.sma_short = self.I(lambda x: pd.Series(x).rolling(7).mean(), self.data.Close)
        self.sma_long = self.I(lambda x: pd.Series(x).rolling(21).mean(), self.data.Close)
        
        # Variables de estado
        self.last_buy_day = -self.intervalo_compra  # Permitir compra inmediata
        self.entry_prices = []  # Precios de entrada para cálculo de ganancia
        
        logger.info(f"🔧 DCAStrategy inicializada: {self.intervalo_compra}d interval, "
                   f"{self.objetivo_ganancia*100:.0f}% target")
    
    def is_bullish_trend(self) -> bool:
        """Detecta si estamos en tendencia alcista."""
        try:
            if len(self.data) < self.tendencia_alcista_dias:
                return False
            
            # Tendencia: SMA corta > SMA larga y precio creciendo
            sma_condition = self.sma_short[-1] > self.sma_long[-1]
            price_condition = self.data.Close[-1] > self.data.Close[-self.tendencia_alcista_dias]
            
            return sma_condition and price_condition
        
        except Exception:
            return False
    
    def is_dip_opportunity(self) -> bool:
        """Detecta oportunidades de compra en dips."""
        try:
            if len(self.data) < 5:
                return False
            
            # Buscar caída reciente desde máximo local
            recent_high = max(self.data.Close[-5:])
            current_price = self.data.Close[-1]
            
            dip_size = (recent_high - current_price) / recent_high
            return dip_size >= self.dip_threshold
        
        except Exception:
            return False
    
    def should_buy(self, current_day: int) -> bool:
        """Determina si debe comprar."""
        try:
            # Verificar intervalo de compra
            if current_day - self.last_buy_day < self.intervalo_compra:
                return False
            
            # Verificar condiciones de mercado
            bullish = self.is_bullish_trend()
            dip = self.is_dip_opportunity()
            
            return bullish and dip
        
        except Exception:
            return False
    
    def should_sell(self) -> bool:
        """Determina si debe vender."""
        try:
            if not self.position or not self.entry_prices:
                return False
            
            current_price = self.data.Close[-1]
            avg_entry = np.mean(self.entry_prices)
            
            # Calcular ganancia/pérdida
            pnl_pct = (current_price - avg_entry) / avg_entry
            
            # Vender si alcanzamos objetivo o stop loss
            take_profit = pnl_pct >= self.objetivo_ganancia
            stop_loss_hit = pnl_pct <= -self.stop_loss
            
            return take_profit or stop_loss_hit
        
        except Exception:
            return False
    
    def next(self):
        """Lógica principal de DCA."""
        try:
            current_day = len(self.data) - 1
            current_price = self.data.Close[-1]
            
            # Lógica de compra
            if self.should_buy(current_day):
                buy_size = self.monto_compra
                self.buy(size=buy_size)
                self.entry_prices.append(current_price)
                self.last_buy_day = current_day
                logger.debug(f"DCA compra: ${current_price:.4f}")
            
            # Lógica de venta
            elif self.should_sell():
                self.sell(size=self.position.size)
                self.entry_prices.clear()  # Reset para nuevo ciclo
                logger.debug(f"DCA venta: ${current_price:.4f}")
        
        except Exception as e:
            logger.error(f"Error en DCA next(): {e}")


class BTDStrategy(Strategy):
    """
    Estrategia Buy The Dip (BTD) modernizada.
    
    Estrategia que busca aprovechar caídas temporales del mercado
    comprando en dips y vendiendo en recuperaciones.
    
    Parámetros:
    - intervalo_venta: Días entre operaciones (1-7)
    - monto_venta: Monto por operación como % del capital (0.1-1.0)
    - objetivo_ganancia: % de ganancia objetivo (0.05-0.25)
    - rip_threshold: % de subida para detectar entrada (0.02-0.12)
    - tendencia_bajista_dias: Días para confirmar dip (3-14)
    - stop_loss: % de pérdida máxima (0.10-0.35)
    """
    
    # Parámetros optimizables
    intervalo_venta = 2
    monto_venta = 0.25  # 25% del capital por operación
    objetivo_ganancia = 0.12  # 12% de ganancia
    rip_threshold = 0.04  # 4% de subida
    tendencia_bajista_dias = 5
    stop_loss = 0.15  # 15% stop loss
    
    def init(self):
        """Inicializa la estrategia BTD."""
        
        # Calcular indicadores
        self.sma_trend = self.I(lambda x: pd.Series(x).rolling(14).mean(), self.data.Close)
        self.rsi = self.I(self.calculate_rsi, self.data.Close)
        
        # Variables de estado
        self.last_operation_day = -self.intervalo_venta
        self.entry_price = None
        
        logger.info(f"🔧 BTDStrategy inicializada: {self.intervalo_venta}d interval, "
                   f"{self.objetivo_ganancia*100:.0f}% target")
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calcula RSI para detectar sobrecompra/sobreventa."""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # Asegurar que retornamos pandas Series y manejar NaN
            if isinstance(rsi, pd.Series):
                return rsi.fillna(50)
            else:
                return pd.Series(rsi, index=prices.index).fillna(50)
        
        except Exception:
            return pd.Series(50, index=prices.index)
    
    def is_dip_confirmed(self) -> bool:
        """Confirma si estamos en un dip válido."""
        try:
            if len(self.data) < self.tendencia_bajista_dias:
                return False
            
            # Verificar que hay tendencia bajista reciente
            price_trend = (self.data.Close[-1] < 
                          self.data.Close[-self.tendencia_bajista_dias])
            
            # Verificar RSI de sobreventa
            rsi_oversold = self.rsi[-1] < 30
            
            return price_trend and rsi_oversold
        
        except Exception:
            return False
    
    def is_recovery_signal(self) -> bool:
        """Detecta señales de recuperación después del dip."""
        try:
            if len(self.data) < 3:
                return False
            
            # Buscar recuperación desde mínimo reciente
            recent_low = min(self.data.Close[-3:])
            current_price = self.data.Close[-1]
            
            recovery_size = (current_price - recent_low) / recent_low
            return recovery_size >= self.rip_threshold
        
        except Exception:
            return False
    
    def should_buy(self, current_day: int) -> bool:
        """Determina si debe comprar en el dip."""
        try:
            # Verificar intervalo
            if current_day - self.last_operation_day < self.intervalo_venta:
                return False
            
            # Verificar condiciones de mercado
            dip_confirmed = self.is_dip_confirmed()
            recovery_signal = self.is_recovery_signal()
            
            return dip_confirmed and recovery_signal and not self.position
        
        except Exception:
            return False
    
    def should_sell(self) -> bool:
        """Determina si debe vender."""
        try:
            if not self.position or self.entry_price is None:
                return False
            
            current_price = self.data.Close[-1]
            pnl_pct = (current_price - self.entry_price) / self.entry_price
            
            # Vender en objetivo o stop loss
            take_profit = pnl_pct >= self.objetivo_ganancia
            stop_loss_hit = pnl_pct <= -self.stop_loss
            
            return take_profit or stop_loss_hit
        
        except Exception:
            return False
    
    def next(self):
        """Lógica principal de BTD."""
        try:
            current_day = len(self.data) - 1
            current_price = self.data.Close[-1]
            
            # Lógica de compra
            if self.should_buy(current_day):
                self.buy(size=self.monto_venta)
                self.entry_price = current_price
                self.last_operation_day = current_day
                logger.debug(f"BTD compra: ${current_price:.4f}")
            
            # Lógica de venta
            elif self.should_sell():
                self.sell(size=self.position.size)
                self.entry_price = None
                logger.debug(f"BTD venta: ${current_price:.4f}")
        
        except Exception as e:
            logger.error(f"Error en BTD next(): {e}")


def run_modern_backtest(df: pd.DataFrame, 
                       strategy_class: type[Strategy],
                       strategy_params: Dict[str, Any],
                       commission: float = 0.001,
                       cash: float = 10000) -> Dict[str, Any]:
    """
    Ejecuta backtesting usando la librería moderna backtesting.py.
    
    Args:
        df: DataFrame con datos OHLCV (columnas: Open, High, Low, Close, Volume)
        strategy_class: Clase de estrategia a probar
        strategy_params: Parámetros para la estrategia
        commission: Comisión por operación
        cash: Capital inicial
        
    Returns:
        Diccionario con resultados del backtesting
    """
    try:
        # Validar DataFrame
        required_columns = ['Open', 'High', 'Low', 'Close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Columnas faltantes: {missing_columns}")
        
        # Crear instancia de Backtest
        bt = Backtest(
            df,
            strategy_class,
            commission=commission,
            cash=cash,
            exclusive_orders=True
        )
        
        # Ejecutar backtesting con parámetros
        results = bt.run(**strategy_params)
        
        # Convertir resultados a formato compatible
        return {
            'Return [%]': results.get('Return [%]', 0),
            'Buy & Hold Return [%]': results.get('Buy & Hold Return [%]', 0),
            'Max. Drawdown [%]': results.get('Max. Drawdown [%]', 0),
            'Sharpe Ratio': results.get('Sharpe Ratio', 0),
            'Calmar Ratio': results.get('Calmar Ratio', 0),
            'Win Rate [%]': results.get('Win Rate [%]', 0),
            '# Trades': results.get('# Trades', 0),
            'Profit Factor': results.get('Profit Factor', 0),
            'SQN': results.get('SQN', 0),
            'start': results._strategy.data.index[0] if hasattr(results, '_strategy') else None,
            'end': results._strategy.data.index[-1] if hasattr(results, '_strategy') else None,
            'duration': str(results._strategy.data.index[-1] - results._strategy.data.index[0]) if hasattr(results, '_strategy') else None,
        }
    
    except Exception as e:
        logger.error(f"Error en backtesting moderno: {e}")
        return {
            'Return [%]': 0,
            'Buy & Hold Return [%]': 0,
            'Max. Drawdown [%]': 100,
            'Sharpe Ratio': 0,
            'Calmar Ratio': 0,
            'Win Rate [%]': 0,
            '# Trades': 0,
            'Profit Factor': 0,
            'SQN': 0,
        }


def main():
    """Función principal para demostración de estrategias modernas."""
    print("🚀 MODERN STRATEGIES - Estrategias con backtesting.py")
    print("=" * 60)
    
    # Ejemplo con datos sintéticos
    import datetime
    
    # Crear datos de ejemplo
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)
    
    # Simular precios con tendencia y volatilidad
    base_price = 100
    returns = np.random.normal(0.001, 0.02, len(dates))  # 0.1% retorno medio, 2% volatilidad
    prices = [base_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # Crear DataFrame OHLCV
    df = pd.DataFrame({
        'Open': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
        'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    }, index=dates)
    
    # Asegurar coherencia OHLC
    for i in range(len(df)):
        high = max(df.iloc[i]['Open'], df.iloc[i]['High'], df.iloc[i]['Close'])
        low = min(df.iloc[i]['Open'], df.iloc[i]['Low'], df.iloc[i]['Close'])
        df.iloc[i, df.columns.get_loc('High')] = high
        df.iloc[i, df.columns.get_loc('Low')] = low
    
    print(f"📊 Dataset generado: {len(df)} días")
    print(f"📈 Precio inicial: ${df['Close'].iloc[0]:.2f}")
    print(f"📈 Precio final: ${df['Close'].iloc[-1]:.2f}")
    
    # Probar estrategias
    strategies_to_test = [
        (GridTradingStrategy, {'levels': 4, 'range_percent': 8.0}, "Grid Trading"),
        (DCAStrategy, {'intervalo_compra': 3, 'objetivo_ganancia': 0.15}, "DCA"),
        (BTDStrategy, {'intervalo_venta': 2, 'objetivo_ganancia': 0.12}, "BTD")
    ]
    
    print(f"\n🔬 PROBANDO ESTRATEGIAS MODERNAS:")
    print("=" * 60)
    
    for strategy_class, params, name in strategies_to_test:
        try:
            print(f"\n📊 {name}:")
            results = run_modern_backtest(df, strategy_class, params)
            
            print(f"   📈 Retorno: {results['Return [%]']:.2f}%")
            print(f"   📉 Max Drawdown: {results['Max. Drawdown [%]']:.2f}%")
            print(f"   📊 Sharpe Ratio: {results['Sharpe Ratio']:.3f}")
            print(f"   🎯 Win Rate: {results['Win Rate [%]']:.1f}%")
            print(f"   🔄 Trades: {results['# Trades']}")
        
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n🎉 Demostración de estrategias modernas completada!")


if __name__ == "__main__":
    main() 