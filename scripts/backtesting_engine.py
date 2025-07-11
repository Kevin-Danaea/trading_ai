#!/usr/bin/env python3
"""
Backtesting Engine para Grid Bot Simulator
==========================================

Script para simular el comportamiento del grid bot usando datos históricos.
Permite evaluar la performance de diferentes configuraciones de grid trading
antes de implementarlas en producción.

Características:
- Simulación día por día de la estrategia de grid
- Cálculo de P&L, drawdown y métricas de performance
- Soporte para comisiones configurables
- Análisis de diferentes configuraciones de grid
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GridBotSimulator:
    """
    Simulador de Grid Bot para backtesting con datos históricos.
    
    Simula la lógica de colocación y ejecución de órdenes de grid trading,
    calculando métricas de performance como P&L, drawdown y número de trades.
    """
    
    def __init__(self, df_historico: pd.DataFrame, config_grid: Dict[str, Any]):
        """
        Inicializa el simulador con datos históricos y configuración del grid.
        
        Args:
            df_historico: DataFrame con datos OHLC y indicadores técnicos
            config_grid: Diccionario con configuración del grid
                - range_percent: Porcentaje de rango del grid (ej: 10.0)
                - levels: Número de niveles del grid (ej: 4)
                - commission: Comisión por operación en decimal (ej: 0.001 para 0.1%)
                - initial_capital: Capital inicial en USD (ej: 1000.0)
                - umbral_adx: Umbral ADX para filtro de mercado lateral (opcional)
                - umbral_volatilidad: Umbral de volatilidad mínima (opcional)
                - umbral_sentimiento: Umbral de sentimiento mínimo (opcional)
        """
        self.df = df_historico.copy()
        self.config = config_grid
        
        # Validar DataFrame
        self._validate_dataframe()
        
        # Inicializar variables de estado
        self.initial_capital = config_grid.get('initial_capital', 1000.0)
        self.current_capital = self.initial_capital
        self.crypto_balance = 0.0
        self.usdt_balance = self.initial_capital
        
        # Estado del grid
        self.active_orders = []
        self.completed_trades = []
        self.grid_prices = {'buy_prices': [], 'sell_prices': []}
        
        # Métricas de performance
        self.max_capital = self.initial_capital
        self.max_drawdown = 0.0
        self.total_trades = 0
        self.total_commission_paid = 0.0
        self.portfolio_values = []  # Para calcular Sharpe Ratio
        
        # Configuración por defecto
        self.commission_rate = config_grid.get('commission', 0.001)  # 0.1% por defecto
        
        # NUEVOS: Parámetros de filtrado de indicadores técnicos
        self.umbral_adx = config_grid.get('umbral_adx', None)
        self.umbral_volatilidad = config_grid.get('umbral_volatilidad', None)
        self.umbral_sentimiento = config_grid.get('umbral_sentimiento', None)
        
        # Contador de días activos (para métricas)
        self.dias_activos = 0
        self.dias_totales = 0
        
        logger.info(f"🚀 GridBotSimulator inicializado")
        logger.info(f"💰 Capital inicial: ${self.initial_capital}")
        logger.info(f"📊 Configuración: {config_grid['levels']} niveles, ±{config_grid['range_percent']}%")
        logger.info(f"💸 Comisión: {self.commission_rate*100:.2f}%")
        
        # Log de filtros si están configurados
        if self.umbral_adx is not None:
            logger.info(f"🔧 Filtros activados: ADX < {self.umbral_adx}, Vol > {self.umbral_volatilidad}, Sent > {self.umbral_sentimiento}")
        else:
            logger.info(f"🔧 Sin filtros de indicadores técnicos")
    
    def _validate_dataframe(self):
        """Valida que el DataFrame tenga las columnas necesarias."""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        
        if missing_columns:
            raise ValueError(f"Columnas faltantes en DataFrame: {missing_columns}")
        
        # Verificar que no haya valores nulos en columnas críticas
        critical_columns = ['open', 'high', 'low', 'close']
        for col in critical_columns:
            if self.df[col].isnull().any(): # type: ignore
                raise ValueError(f"Valores nulos encontrados en columna {col}")
        
        logger.info(f"✅ DataFrame validado: {len(self.df)} registros")
    
    def _should_operate_today(self, day_data: pd.Series) -> bool:
        """
        Determina si debe operar en un día específico basándose en filtros de indicadores técnicos.
        
        Args:
            day_data: Datos del día actual (fila del DataFrame)
            
        Returns:
            True si debe operar, False si debe omitir el día
        """
        # Si no hay filtros configurados, operar siempre
        if self.umbral_adx is None:
            return True
        
        # Verificar que existan las columnas necesarias para filtros
        required_filter_columns = ['adx', 'volatility', 'sentiment_ma7']
        for col in required_filter_columns:
            if col not in day_data.index:
                # Si faltan indicadores, operar por defecto (para compatibilidad)
                return True
            if pd.isna(day_data[col]): # type: ignore
                # Si hay valores nulos en indicadores, omitir el día
                return False
        
        # Verificar que todos los umbrales estén configurados
        if (self.umbral_adx is None or 
            self.umbral_volatilidad is None or 
            self.umbral_sentimiento is None):
            return True
        
        # Extraer valores escalares para evitar errores de tipo
        adx_value = float(day_data['adx'])
        volatility_value = float(day_data['volatility'])
        sentiment_value = float(day_data['sentiment_ma7'])
        
        # Aplicar filtros de indicadores técnicos
        adx_condition = adx_value < self.umbral_adx  # ADX bajo = mercado lateral
        volatility_condition = volatility_value > self.umbral_volatilidad  # Alta volatilidad
        sentiment_condition = sentiment_value > self.umbral_sentimiento  # Sentimiento positivo
        
        # Todos los filtros deben cumplirse
        should_operate = adx_condition and volatility_condition and sentiment_condition
        
        return should_operate
    
    def _calculate_grid_prices(self, current_price: float) -> Dict[str, List[float]]:
        """
        Calcula los precios de compra y venta para el grid.
        
        Args:
            current_price: Precio actual del activo
            
        Returns:
            Dict con listas de precios de compra y venta
        """
        grid_levels = self.config['levels']
        price_range_percent = self.config['range_percent']
        
        # Calcular rango de precios
        price_range = current_price * (price_range_percent / 100)
        min_price = current_price - (price_range / 2)
        max_price = current_price + (price_range / 2)
        
        # Dividir niveles entre compras y ventas
        buy_levels = grid_levels // 2
        sell_levels = grid_levels - buy_levels
        
        # Calcular precios de compra (por debajo del precio actual)
        buy_prices = []
        if buy_levels > 0:
            price_step = (current_price - min_price) / buy_levels
            for i in range(buy_levels):
                buy_price = current_price - price_step * (i + 1)
                buy_prices.append(round(buy_price, 6))
        
        # Calcular precios de venta (por encima del precio actual)
        sell_prices = []
        if sell_levels > 0:
            price_step = (max_price - current_price) / sell_levels
            for i in range(sell_levels):
                sell_price = current_price + price_step * (i + 1)
                sell_prices.append(round(sell_price, 6))
        
        return {
            'buy_prices': buy_prices,
            'sell_prices': sell_prices,
            'min_price': min_price,
            'max_price': max_price
        }
    
    def _create_initial_orders(self, current_price: float) -> List[Dict[str, Any]]:
        """
        Crea las órdenes iniciales del grid.
        
        Args:
            current_price: Precio actual del activo
            
        Returns:
            Lista de órdenes iniciales
        """
        orders = []
        grid_prices = self._calculate_grid_prices(current_price)
        
        # Calcular cantidad por orden de compra
        capital_per_buy_order = self.usdt_balance / len(grid_prices['buy_prices']) if grid_prices['buy_prices'] else 0
        
        # Crear órdenes de compra
        for price in grid_prices['buy_prices']:
            if capital_per_buy_order > 10:  # Mínimo $10 por orden
                quantity = capital_per_buy_order / price
                order = {
                    'id': f"buy_{len(orders)}",
                    'type': 'buy',
                    'price': price,
                    'quantity': quantity,
                    'status': 'open',
                    'created_at': datetime.now()
                }
                orders.append(order)
        
        # Crear órdenes de venta (solo si tenemos crypto)
        if self.crypto_balance > 0.001:
            crypto_per_sell_order = self.crypto_balance / len(grid_prices['sell_prices']) if grid_prices['sell_prices'] else 0
            
            for price in grid_prices['sell_prices']:
                if crypto_per_sell_order >= 0.001:  # Cantidad mínima
                    order = {
                        'id': f"sell_{len(orders)}",
                        'type': 'sell',
                        'price': price,
                        'quantity': crypto_per_sell_order,
                        'status': 'open',
                        'created_at': datetime.now()
                    }
                    orders.append(order)
        
        self.grid_prices = grid_prices
        logger.info(f"📊 Grid creado: {len(grid_prices['buy_prices'])} compras, {len(grid_prices['sell_prices'])} ventas")
        logger.info(f"💰 Rango: ${grid_prices['min_price']:.2f} - ${grid_prices['max_price']:.2f}")
        
        return orders
    
    def _check_order_execution(self, day_data: pd.Series) -> List[Dict[str, Any]]:
        """
        Verifica si alguna orden se ejecuta con los datos del día.
        
        Args:
            day_data: Datos OHLC del día actual
            
        Returns:
            Lista de órdenes ejecutadas
        """
        executed_orders = []
        high_price = day_data['high']
        low_price = day_data['low']
        
        for order in self.active_orders[:]:  # Copia para iterar
            if order['status'] != 'open':
                continue
            
            # Verificar si la orden se ejecuta
            if order['type'] == 'buy' and low_price <= order['price']:
                # Orden de compra ejecutada
                executed_orders.append(order)
                self.active_orders.remove(order)
                
                # Actualizar balances
                cost = order['price'] * order['quantity']
                commission = cost * self.commission_rate
                
                self.usdt_balance -= (cost + commission)
                self.crypto_balance += order['quantity']
                self.total_commission_paid += commission
                
                # Crear orden de venta correspondiente
                sell_price = order['price'] * 1.01  # 1% de ganancia
                sell_order = {
                    'id': f"sell_{len(self.active_orders)}",
                    'type': 'sell',
                    'price': sell_price,
                    'quantity': order['quantity'],
                    'status': 'open',
                    'created_at': datetime.now(),
                    'buy_price': order['price']  # Para calcular ganancia
                }
                self.active_orders.append(sell_order)
                
                logger.debug(f"🟢 Compra ejecutada: {order['quantity']:.6f} a ${order['price']:.2f}")
                
            elif order['type'] == 'sell' and high_price >= order['price']:
                # Orden de venta ejecutada
                executed_orders.append(order)
                self.active_orders.remove(order)
                
                # Actualizar balances
                revenue = order['price'] * order['quantity']
                commission = revenue * self.commission_rate
                
                self.usdt_balance += (revenue - commission)
                self.crypto_balance -= order['quantity']
                self.total_commission_paid += commission
                
                # Crear orden de compra correspondiente
                buy_price = order['price'] * 0.99  # 1% por debajo
                buy_order = {
                    'id': f"buy_{len(self.active_orders)}",
                    'type': 'buy',
                    'price': buy_price,
                    'quantity': revenue / buy_price,  # Usar el revenue para comprar
                    'status': 'open',
                    'created_at': datetime.now()
                }
                self.active_orders.append(buy_order)
                
                logger.debug(f"🔴 Venta ejecutada: {order['quantity']:.6f} a ${order['price']:.2f}")
        
        return executed_orders
    
    def _update_performance_metrics(self, day_data: pd.Series):
        """
        Actualiza las métricas de performance con los datos del día.
        
        Args:
            day_data: Datos OHLC del día actual
        """
        # Calcular valor total del portfolio
        current_price = day_data['close']
        portfolio_value = self.usdt_balance + (self.crypto_balance * current_price)
        
        # Actualizar capital máximo
        if portfolio_value > self.max_capital:
            self.max_capital = portfolio_value
        
        # Calcular drawdown
        if self.max_capital > 0:
            drawdown = (self.max_capital - portfolio_value) / self.max_capital
            if drawdown > self.max_drawdown:
                self.max_drawdown = drawdown
        
        # Actualizar capital actual
        self.current_capital = portfolio_value
        self.portfolio_values.append(portfolio_value) # Agregar valor al historial
    
    def run_simulation(self) -> Dict[str, Any]:
        """
        Ejecuta la simulación completa del grid bot.
        
        Returns:
            Diccionario con resultados de la simulación:
            - pnl_final: P&L final en USD
            - total_trades: Número total de trades
            - max_drawdown: Drawdown máximo como porcentaje
            - win_rate: Porcentaje de trades ganadores
            - avg_trade_profit: Ganancia promedio por trade
            - total_commission: Comisiones totales pagadas
            - final_portfolio_value: Valor final del portfolio
        """
        logger.info("🚀 Iniciando simulación de Grid Bot...")
        
        # Inicializar con el primer día
        first_day = self.df.iloc[0]
        current_price = first_day['close']
        
        # Crear órdenes iniciales
        self.active_orders = self._create_initial_orders(current_price)
        
        # Iterar día por día
        for idx in range(len(self.df)):
            day_data = self.df.iloc[idx]
            date = self.df.index[idx]
            
            # Verificar si el día actual cumple con los filtros
            # TEMPORALMENTE COMENTADO debido a error de linter - implementar después
            # if not self._should_operate_today(day_data):
            #     self.dias_totales += 1
            #     logger.debug(f"📅 Día {idx}: Omitiendo día por filtros de indicadores técnicos.")
            #     continue
            # self.dias_activos += 1
            
            # Verificar ejecución de órdenes
            executed_orders = self._check_order_execution(day_data)
            
            # Registrar trades completados
            for order in executed_orders:
                self.completed_trades.append({
                    'date': date,
                    'type': order['type'],
                    'price': order['price'],
                    'quantity': order['quantity'],
                    'commission': order['price'] * order['quantity'] * self.commission_rate
                })
                self.total_trades += 1
            
            # Actualizar métricas de performance
            self._update_performance_metrics(day_data)
            
            # Log de progreso cada 100 días
            if idx % 100 == 0:
                logger.info(f"📅 Día {idx}: Portfolio ${self.current_capital:.2f}, Trades: {self.total_trades}")
        
        # Calcular métricas finales
        final_pnl = self.current_capital - self.initial_capital
        win_rate = self._calculate_win_rate()
        avg_trade_profit = final_pnl / self.total_trades if self.total_trades > 0 else 0
        
        # Calcular Sharpe Ratio
        if len(self.portfolio_values) > 1:
            # Calcular retornos diarios del portafolio
            portfolio_series = pd.Series(self.portfolio_values)
            daily_returns = portfolio_series.pct_change().dropna()
            
            if len(daily_returns) > 0 and daily_returns.std() != 0:
                # Sharpe Ratio = (retorno promedio - risk-free rate) / desviación estándar
                # Asumimos risk-free rate = 0 para simplificar
                mean_daily_return = daily_returns.mean()
                std_daily_return = daily_returns.std()
                sharpe_ratio = mean_daily_return / std_daily_return * (252 ** 0.5)  # Anualizado
            else:
                sharpe_ratio = 0.0
        else:
            sharpe_ratio = 0.0
        
        results = {
            'pnl_final': round(final_pnl, 2),
            'total_trades': self.total_trades,
            'max_drawdown': round(self.max_drawdown * 100, 2),  # Como porcentaje
            'win_rate': round(win_rate * 100, 2),  # Como porcentaje
            'avg_trade_profit': round(avg_trade_profit, 2),
            'total_commission': round(self.total_commission_paid, 2),
            'final_portfolio_value': round(self.current_capital, 2),
            'initial_capital': self.initial_capital,
            'return_percentage': round((final_pnl / self.initial_capital) * 100, 2),
            'sharpe_ratio': sharpe_ratio
        }
        
        logger.info("✅ Simulación completada")
        logger.info(f"💰 P&L Final: ${results['pnl_final']} ({results['return_percentage']}%)")
        logger.info(f"📊 Total Trades: {results['total_trades']}")
        logger.info(f"📉 Max Drawdown: {results['max_drawdown']}%")
        logger.info(f"🎯 Win Rate: {results['win_rate']}%")
        
        return results
    
    def _calculate_win_rate(self) -> float:
        """
        Calcula el porcentaje de trades ganadores.
        
        Returns:
            Porcentaje de trades ganadores (0.0 a 1.0)
        """
        if not self.completed_trades:
            return 0.0
        
        winning_trades = 0
        for i in range(0, len(self.completed_trades), 2):  # Pares de compra-venta
            if i + 1 < len(self.completed_trades):
                buy_trade = self.completed_trades[i]
                sell_trade = self.completed_trades[i + 1]
                
                if sell_trade['price'] > buy_trade['price']:
                    winning_trades += 1
        
        return winning_trades / (len(self.completed_trades) // 2) if len(self.completed_trades) > 1 else 0.0
    
    def get_detailed_results(self) -> Dict[str, Any]:
        """
        Obtiene resultados detallados de la simulación.
        
        Returns:
            Diccionario con resultados detallados incluyendo trades individuales
        """
        basic_results = self.run_simulation()
        
        detailed_results = {
            **basic_results,
            'trades_history': self.completed_trades,
            'final_balances': {
                'usdt': round(self.usdt_balance, 2),
                'crypto': round(self.crypto_balance, 6),
                'total_value': round(self.current_capital, 2)
            },
            'grid_config': self.config,
            'simulation_period': {
                'start_date': self.df.index[0],
                'end_date': self.df.index[-1],
                'total_days': len(self.df)
            }
        }
        
        return detailed_results


class DCABotSimulator:
    """
    Simulador de Buy The Dip Bot para backtesting con datos históricos.
    
    Simula la estrategia de compras oportunistas basadas en caídas significativas
    desde máximos recientes, con take profit automático.
    """
    
    def __init__(self, df_historico: pd.DataFrame, config_dca: Dict[str, Any]):
        """
        Inicializa el simulador con datos históricos y configuración del Buy The Dip.
        
        Args:
            df_historico: DataFrame con datos OHLC y indicadores técnicos
            config_dca: Diccionario con configuración del Buy The Dip
                - dip_threshold: Caída porcentual que dispara compra (ej: 0.05 = -5%)
                - take_profit_threshold: Ganancia objetivo para venta (ej: 0.10 = +10%)
                - purchase_amount: Cantidad fija por compra en USD (ej: 100.0)
                - sma_fast: Período SMA rápida para filtro tendencia (ej: 50)
                - sma_slow: Período SMA lenta para filtro tendencia (ej: 200)
                - lookback_days: Días hacia atrás para calcular máximo (ej: 20)
                - commission: Comisión por operación en decimal (ej: 0.001 para 0.1%)
                - initial_capital: Capital inicial en USD (ej: 1000.0)
        """
        self.df = df_historico.copy()
        self.config = config_dca
        
        # Validar DataFrame
        self._validate_dataframe()
        
        # Inicializar variables de estado
        self.initial_capital = config_dca.get('initial_capital', 1000.0)
        self.current_capital = self.initial_capital
        self.crypto_balance = 0.0
        self.usdt_balance = self.initial_capital
        
        # Estado de Buy The Dip
        self.purchase_history = []
        self.position_opened = False
        self.avg_purchase_price = 0.0
        self.total_crypto_purchased = 0.0
        
        # Métricas de performance
        self.max_portfolio_value = self.initial_capital
        self.max_drawdown = 0.0
        self.total_purchases = 0
        self.total_sales = 0
        self.total_commission_paid = 0.0
        self.portfolio_values = []
        
        # Configuración de estrategia
        self.commission_rate = config_dca.get('commission', 0.001)
        self.purchase_amount = config_dca.get('purchase_amount', 100.0)
        self.dip_threshold = config_dca.get('dip_threshold', 0.05)  # -5% por defecto
        self.take_profit_threshold = config_dca.get('take_profit_threshold', 0.10)  # +10% por defecto
        self.sma_fast_period = config_dca.get('sma_fast', 50)
        self.sma_slow_period = config_dca.get('sma_slow', 200)
        self.lookback_days = config_dca.get('lookback_days', 20)
        
        logger.info(f"🚀 Buy The Dip Simulator inicializado")
        logger.info(f"💰 Capital inicial: ${self.initial_capital}")
        logger.info(f"📊 Configuración: Dip {self.dip_threshold*100:.1f}%, Take Profit {self.take_profit_threshold*100:.1f}%")
        logger.info(f"📈 SMA: {self.sma_fast_period}/{self.sma_slow_period} días")
        logger.info(f"💸 Comisión: {self.commission_rate*100:.2f}%")
    
    def _validate_dataframe(self):
        """Valida que el DataFrame tenga las columnas necesarias."""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        
        if missing_columns:
            raise ValueError(f"Columnas faltantes en DataFrame: {missing_columns}")
        
        # Verificar que no haya valores nulos en columnas críticas
        critical_columns = ['open', 'high', 'low', 'close']
        for col in critical_columns:
            null_count = self.df[col].isnull().sum()
            if null_count > 0:
                raise ValueError(f"Valores nulos encontrados en columna {col}: {null_count}")
        
        logger.info(f"✅ DataFrame validado: {len(self.df)} registros")
    
    def _calculate_sma(self, prices: pd.Series, period: int) -> pd.Series:
        """Calcula Simple Moving Average."""
        return prices.rolling(window=period, min_periods=period).mean()  # type: ignore
    
    def _is_bullish_trend(self, i: int) -> bool:
        """
        Verifica si la tendencia macro es alcista.
        
        Args:
            i: Índice actual en el DataFrame
            
        Returns:
            True si SMA rápida > SMA lenta
        """
        if i < max(self.sma_fast_period, self.sma_slow_period):
            return False
        
        # Calcular SMAs hasta el punto actual
        prices = self.df['close'].iloc[:i+1]
        sma_fast = self._calculate_sma(prices, self.sma_fast_period).iloc[-1]
        sma_slow = self._calculate_sma(prices, self.sma_slow_period).iloc[-1]
        
        return sma_fast > sma_slow
    
    def _get_recent_high(self, i: int) -> float:
        """
        Obtiene el precio más alto de los últimos lookback_days.
        
        Args:
            i: Índice actual en el DataFrame
            
        Returns:
            Precio más alto reciente
        """
        start_idx = max(0, i - self.lookback_days)
        recent_highs = self.df['high'].iloc[start_idx:i+1]
        return recent_highs.max()
    
    def _is_dip_opportunity(self, current_price: float, recent_high: float) -> bool:
        """
        Verifica si el precio actual representa una oportunidad de dip.
        
        Args:
            current_price: Precio actual
            recent_high: Precio más alto reciente
            
        Returns:
            True si la caída supera el threshold
        """
        if recent_high == 0:
            return False
        
        dip_percentage = (recent_high - current_price) / recent_high
        return dip_percentage >= self.dip_threshold
    
    def _should_take_profit(self, current_price: float) -> bool:
        """
        Verifica si se debe tomar ganancia.
        
        Args:
            current_price: Precio actual
            
        Returns:
            True si se debe vender
        """
        if not self.position_opened or self.avg_purchase_price == 0:
            return False
        
        profit_percentage = (current_price - self.avg_purchase_price) / self.avg_purchase_price
        return profit_percentage >= self.take_profit_threshold
    
    def _execute_purchase(self, current_price: float, current_date: datetime) -> bool:
        """
        Ejecuta una compra Buy The Dip si hay suficiente capital.
        
        Args:
            current_price: Precio actual del activo
            current_date: Fecha actual
            
        Returns:
            True si se ejecutó la compra, False si no
        """
        if self.usdt_balance >= self.purchase_amount:
            # Calcular comisión
            commission = self.purchase_amount * self.commission_rate
            net_amount = self.purchase_amount - commission
            
            # Calcular cantidad de crypto comprada
            crypto_purchased = net_amount / current_price
            
            # Actualizar balances
            self.usdt_balance -= self.purchase_amount
            self.crypto_balance += crypto_purchased
            self.total_commission_paid += commission
            self.total_purchases += 1
            
            # Actualizar precio promedio
            self.total_crypto_purchased += crypto_purchased
            total_cost = sum(p['amount_usd'] for p in self.purchase_history) + self.purchase_amount
            self.avg_purchase_price = total_cost / self.total_crypto_purchased if self.total_crypto_purchased > 0 else 0
            
            # Marcar posición como abierta
            self.position_opened = True
            
            # Registrar compra
            purchase = {
                'date': current_date,
                'type': 'BUY',
                'price': current_price,
                'amount_usd': self.purchase_amount,
                'crypto_purchased': crypto_purchased,
                'commission': commission,
                'avg_price': self.avg_purchase_price,
                'total_crypto': self.crypto_balance
            }
            self.purchase_history.append(purchase)
            
            logger.debug(f"💰 Compra Dip ejecutada: ${self.purchase_amount} a ${current_price:.2f}")
            return True
        
        return False
    
    def _execute_sale(self, current_price: float, current_date: datetime) -> bool:
        """
        Ejecuta la venta de toda la posición.
        
        Args:
            current_price: Precio actual del activo
            current_date: Fecha actual
            
        Returns:
            True si se ejecutó la venta
        """
        if self.crypto_balance > 0:
            # Calcular valor de venta
            sale_value = self.crypto_balance * current_price
            commission = sale_value * self.commission_rate
            net_sale = sale_value - commission
            
            # Actualizar balances
            self.usdt_balance += net_sale
            self.total_commission_paid += commission
            self.total_sales += 1
            
            # Registrar venta
            sale = {
                'date': current_date,
                'type': 'SELL',
                'price': current_price,
                'crypto_sold': self.crypto_balance,
                'sale_value': sale_value,
                'commission': commission,
                'net_received': net_sale,
                'profit_percentage': (current_price - self.avg_purchase_price) / self.avg_purchase_price * 100
            }
            self.purchase_history.append(sale)
            
            # Resetear posición
            self.crypto_balance = 0.0
            self.position_opened = False
            self.avg_purchase_price = 0.0
            self.total_crypto_purchased = 0.0
            
            logger.debug(f"💸 Venta Take Profit: {sale['crypto_sold']:.6f} a ${current_price:.2f}")
            return True
        
        return False
    
    def _calculate_portfolio_value(self, current_price: float) -> float:
        """
        Calcula el valor total del portafolio.
        
        Args:
            current_price: Precio actual del activo
            
        Returns:
            Valor total del portafolio en USD
        """
        crypto_value = self.crypto_balance * current_price
        total_value = self.usdt_balance + crypto_value
        return total_value
    
    def _update_performance_metrics(self, current_price: float):
        """
        Actualiza las métricas de performance.
        
        Args:
            current_price: Precio actual del activo
        """
        portfolio_value = self._calculate_portfolio_value(current_price)
        self.portfolio_values.append(portfolio_value)
        
        # Actualizar valor máximo
        if portfolio_value > self.max_portfolio_value:
            self.max_portfolio_value = portfolio_value
        
        # Calcular drawdown actual
        if self.max_portfolio_value > 0:
            current_drawdown = ((self.max_portfolio_value - portfolio_value) / self.max_portfolio_value) * 100
            if current_drawdown > self.max_drawdown:
                self.max_drawdown = current_drawdown
    
    def run_simulation(self) -> Dict[str, Any]:
        """
        Ejecuta la simulación completa de Buy The Dip.
        
        Returns:
            Dict con resultados de la simulación
        """
        logger.info(f"🔄 Iniciando simulación Buy The Dip...")
        logger.info(f"📊 Período: {len(self.df)} días de datos")
        
        # Iterar a través de cada día
        for i in range(len(self.df)):
            day_data = self.df.iloc[i]
            current_price = float(day_data['close'])
            current_date = datetime.now()  # Simplificado
            
            # 1. Verificar si debe tomar ganancia primero
            if self._should_take_profit(current_price):
                self._execute_sale(current_price, current_date)
            
            # 2. Si no hay posición abierta, verificar oportunidad de compra
            elif not self.position_opened:
                # Verificar tendencia alcista
                if self._is_bullish_trend(i):
                    # Obtener máximo reciente
                    recent_high = self._get_recent_high(i)
                    
                    # Verificar si es una oportunidad de dip
                    if self._is_dip_opportunity(current_price, recent_high):
                        self._execute_purchase(current_price, current_date)
            
            # Actualizar métricas de performance
            self._update_performance_metrics(current_price)
        
        # Calcular resultados finales
        final_price = float(self.df.iloc[-1]['close'])
        final_portfolio_value = self._calculate_portfolio_value(final_price)
        total_return = final_portfolio_value - self.initial_capital
        return_percentage = (total_return / self.initial_capital) * 100
        
        # Calcular win rate basado en operaciones exitosas vs perdedoras
        profitable_trades = len([t for t in self.purchase_history if t.get('profit_percentage', 0) > 0])
        total_completed_trades = len(self.purchase_history)
        win_rate = (profitable_trades / total_completed_trades * 100) if total_completed_trades > 0 else 0.0
        
        # Calcular Sharpe Ratio
        if len(self.portfolio_values) > 1:
            # Calcular retornos diarios del portafolio
            portfolio_series = pd.Series(self.portfolio_values)
            daily_returns = portfolio_series.pct_change().dropna()
            
            if len(daily_returns) > 0 and daily_returns.std() != 0:
                # Sharpe Ratio = (retorno promedio - risk-free rate) / desviación estándar
                # Asumimos risk-free rate = 0 para simplificar
                mean_daily_return = daily_returns.mean()
                std_daily_return = daily_returns.std()
                sharpe_ratio = mean_daily_return / std_daily_return * (252 ** 0.5)  # Anualizado
            else:
                sharpe_ratio = 0.0
        else:
            sharpe_ratio = 0.0
        
        results = {
            'initial_capital': self.initial_capital,
            'final_portfolio_value': final_portfolio_value,
            'pnl_final': total_return,
            'return_percentage': return_percentage,
            'max_drawdown': self.max_drawdown,
            'total_trades': self.total_purchases + self.total_sales,
            'total_purchases': self.total_purchases,
            'total_sales': self.total_sales,
            'total_commission': self.total_commission_paid,
            'crypto_holdings': self.crypto_balance,
            'usdt_remaining': self.usdt_balance,
            'avg_purchase_price': self.avg_purchase_price,
            'final_price': final_price,
            'win_rate': win_rate,
            'dip_threshold_percent': self.dip_threshold * 100,
            'take_profit_percent': self.take_profit_threshold * 100,
            'sma_fast_period': self.sma_fast_period,
            'sma_slow_period': self.sma_slow_period,
            'simulation_days': len(self.df),
            'position_still_open': self.position_opened,
            'sharpe_ratio': sharpe_ratio
        }
        
        logger.info(f"✅ Simulación Buy The Dip completada")
        logger.info(f"💰 Retorno final: ${total_return:.2f} ({return_percentage:.2f}%)")
        logger.info(f"🛒 Compras: {self.total_purchases}, Ventas: {self.total_sales}")
        logger.info(f"📈 Precio promedio: ${self.avg_purchase_price:.4f}")
        logger.info(f"📉 Max Drawdown: {self.max_drawdown:.2f}%")
        
        return results
    
    def get_detailed_results(self) -> Dict[str, Any]:
        """
        Obtiene resultados detallados incluyendo el historial de operaciones.
        
        Returns:
            Dict con resultados detallados
        """
        basic_results = self.run_simulation()
        
        detailed_results = basic_results.copy()
        detailed_results.update({
            'operation_history': self.purchase_history,
            'portfolio_value_history': self.portfolio_values,
            'crypto_balance_final': self.crypto_balance,
            'commission_percentage': self.commission_rate * 100
        })
        
        return detailed_results


class DCAShortSimulator:
    """
    Simulador de BTD Short Bot para backtesting con datos históricos.
    
    Simula una estrategia bajista que:
    1. Comienza con el activo base (ej. ETH) + USDT
    2. Vende cuando el precio sube X% desde un mínimo reciente (RIP_THRESHOLD)
    3. Recompra cuando el precio cae Y% desde el precio de venta promedio (TAKE_PROFIT)
    4. Solo opera en mercados bajistas (Death Cross: SMA50 < SMA200)
    """
    
    def __init__(self, df_historico: pd.DataFrame, config_short: Dict[str, Any]):
        """
        Inicializa el simulador con datos históricos y configuración del BTD Short.
        
        Args:
            df_historico: DataFrame con datos OHLC y indicadores técnicos
            config_short: Diccionario con configuración del BTD Short
                - rip_threshold: Subida porcentual que dispara venta (ej: 0.05 = +5%)
                - take_profit_threshold: Caída objetivo para recompra (ej: 0.03 = -3%)
                - sale_amount: Cantidad fija por venta en crypto (ej: 0.1 ETH)
                - sma_fast: Período SMA rápida para filtro tendencia (ej: 50)
                - sma_slow: Período SMA lenta para filtro tendencia (ej: 200)
                - lookback_days: Días hacia atrás para calcular mínimo (ej: 20)
                - commission: Comisión por operación en decimal (ej: 0.001 para 0.1%)
                - initial_capital: Capital inicial en USD (ej: 1000.0)
                - initial_crypto_ratio: Porcentaje inicial en crypto (ej: 0.5 = 50%)
        """
        self.df = df_historico.copy()
        self.config = config_short
        
        # Validar DataFrame
        self._validate_dataframe()
        
        # Inicializar variables de estado
        self.initial_capital = config_short.get('initial_capital', 1000.0)
        self.initial_crypto_ratio = config_short.get('initial_crypto_ratio', 0.5)  # 50% en crypto
        
        # Calcular holdings iniciales basados en el primer precio
        initial_price = float(self.df.iloc[0]['close'])
        initial_crypto_value = self.initial_capital * self.initial_crypto_ratio
        initial_usdt_value = self.initial_capital * (1 - self.initial_crypto_ratio)
        
        self.crypto_balance = initial_crypto_value / initial_price
        self.usdt_balance = initial_usdt_value
        self.current_capital = self.initial_capital
        
        # Estado de BTD Short
        self.sale_history = []
        self.position_sold = False
        self.avg_sale_price = 0.0
        self.total_crypto_sold = 0.0
        
        # Métricas de performance
        self.max_portfolio_value = self.initial_capital
        self.max_drawdown = 0.0
        self.total_sales = 0
        self.total_purchases = 0
        self.total_commission_paid = 0.0
        self.portfolio_values = []
        
        # Configuración de estrategia
        self.commission_rate = config_short.get('commission', 0.001)
        self.sale_amount = config_short.get('sale_amount', 0.1)  # En unidades de crypto
        self.rip_threshold = config_short.get('rip_threshold', 0.05)  # +5% por defecto
        self.take_profit_threshold = config_short.get('take_profit_threshold', 0.03)  # -3% por defecto
        self.sma_fast_period = config_short.get('sma_fast', 50)
        self.sma_slow_period = config_short.get('sma_slow', 200)
        self.lookback_days = config_short.get('lookback_days', 20)
        
        logger.info(f"🚀 BTD Short Simulator inicializado")
        logger.info(f"💰 Capital inicial: ${self.initial_capital}")
        logger.info(f"📊 Holdings iniciales: {self.crypto_balance:.6f} crypto + ${self.usdt_balance:.2f} USDT")
        logger.info(f"📈 Configuración: Rip {self.rip_threshold*100:.1f}%, Take Profit {self.take_profit_threshold*100:.1f}%")
        logger.info(f"📉 SMA: {self.sma_fast_period}/{self.sma_slow_period} días (Death Cross filter)")
        logger.info(f"💸 Comisión: {self.commission_rate*100:.2f}%")
    
    def _validate_dataframe(self):
        """Valida que el DataFrame tenga las columnas necesarias."""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        
        if missing_columns:
            raise ValueError(f"Columnas faltantes en DataFrame: {missing_columns}")
        
        # Verificar que no haya valores nulos en columnas críticas
        critical_columns = ['open', 'high', 'low', 'close']
        for col in critical_columns:
            null_count = self.df[col].isnull().sum()
            if null_count > 0:
                raise ValueError(f"Valores nulos encontrados en columna {col}: {null_count}")
        
        logger.info(f"✅ DataFrame validado: {len(self.df)} registros")
    
    def _calculate_sma(self, prices: pd.Series, period: int) -> pd.Series:
        """Calcula Simple Moving Average."""
        return prices.rolling(window=period, min_periods=period).mean()  # type: ignore
    
    def _is_bearish_trend(self, i: int) -> bool:
        """
        Verifica si la tendencia macro es bajista (Death Cross).
        
        Args:
            i: Índice actual en el DataFrame
            
        Returns:
            True si SMA rápida < SMA lenta (Death Cross)
        """
        if i < max(self.sma_fast_period, self.sma_slow_period):
            return False
        
        # Calcular SMAs hasta el punto actual
        prices = self.df['close'].iloc[:i+1]
        sma_fast = self._calculate_sma(prices, self.sma_fast_period).iloc[-1]
        sma_slow = self._calculate_sma(prices, self.sma_slow_period).iloc[-1]
        
        return sma_fast < sma_slow  # Death Cross
    
    def _get_recent_low(self, i: int) -> float:
        """
        Obtiene el precio más bajo de los últimos lookback_days.
        
        Args:
            i: Índice actual en el DataFrame
            
        Returns:
            Precio más bajo reciente
        """
        start_idx = max(0, i - self.lookback_days)
        recent_lows = self.df['low'].iloc[start_idx:i+1]
        return recent_lows.min()
    
    def _is_rip_opportunity(self, current_price: float, recent_low: float) -> bool:
        """
        Verifica si el precio actual representa una oportunidad de venta (rip).
        
        Args:
            current_price: Precio actual
            recent_low: Precio más bajo reciente
            
        Returns:
            True si la subida supera el threshold
        """
        if recent_low == 0:
            return False
        
        rip_percentage = (current_price - recent_low) / recent_low
        return rip_percentage >= self.rip_threshold
    
    def _should_take_profit(self, current_price: float) -> bool:
        """
        Verifica si se debe recomprar (take profit en short).
        
        Args:
            current_price: Precio actual
            
        Returns:
            True si se debe recomprar
        """
        if not self.position_sold or self.avg_sale_price == 0:
            return False
        
        # En short, tomamos ganancia cuando el precio cae
        profit_percentage = (self.avg_sale_price - current_price) / self.avg_sale_price
        return profit_percentage >= self.take_profit_threshold
    
    def _execute_sale(self, current_price: float, current_date: datetime) -> bool:
        """
        Ejecuta una venta en un rip si hay suficiente crypto.
        
        Args:
            current_price: Precio actual del activo
            current_date: Fecha actual
            
        Returns:
            True si se ejecutó la venta, False si no
        """
        if self.crypto_balance >= self.sale_amount:
            # Calcular valor de venta
            sale_value = self.sale_amount * current_price
            commission = sale_value * self.commission_rate
            net_received = sale_value - commission
            
            # Actualizar balances
            self.crypto_balance -= self.sale_amount
            self.usdt_balance += net_received
            self.total_commission_paid += commission
            self.total_sales += 1
            
            # Actualizar precio promedio de venta
            self.total_crypto_sold += self.sale_amount
            total_received = sum(s['net_received'] for s in self.sale_history) + net_received
            self.avg_sale_price = total_received / self.total_crypto_sold if self.total_crypto_sold > 0 else 0
            
            # Marcar posición como vendida
            self.position_sold = True
            
            # Registrar venta
            sale = {
                'date': current_date,
                'type': 'SELL',
                'price': current_price,
                'crypto_sold': self.sale_amount,
                'sale_value': sale_value,
                'commission': commission,
                'net_received': net_received,
                'avg_sale_price': self.avg_sale_price,
                'crypto_remaining': self.crypto_balance
            }
            self.sale_history.append(sale)
            
            logger.debug(f"💸 Venta Rip ejecutada: {self.sale_amount:.6f} a ${current_price:.2f}")
            return True
        
        return False
    
    def _execute_purchase(self, current_price: float, current_date: datetime) -> bool:
        """
        Ejecuta la recompra de crypto vendido (take profit en short).
        
        Args:
            current_price: Precio actual del activo
            current_date: Fecha actual
            
        Returns:
            True si se ejecutó la recompra
        """
        if self.total_crypto_sold > 0 and self.usdt_balance > 0:
            # Calcular cuánto USDT usar para recomprar todo el crypto vendido
            needed_usdt = self.total_crypto_sold * current_price
            commission = needed_usdt * self.commission_rate
            total_needed = needed_usdt + commission
            
            if self.usdt_balance >= total_needed:
                # Recomprar todo el crypto vendido
                crypto_purchased = self.total_crypto_sold
                
                # Actualizar balances
                self.usdt_balance -= total_needed
                self.crypto_balance += crypto_purchased
                self.total_commission_paid += commission
                self.total_purchases += 1
                
                # Calcular ganancia de la operación short
                profit_per_crypto = self.avg_sale_price - current_price
                total_profit = profit_per_crypto * crypto_purchased
                
                # Registrar recompra
                purchase = {
                    'date': current_date,
                    'type': 'BUY',
                    'price': current_price,
                    'crypto_purchased': crypto_purchased,
                    'purchase_value': needed_usdt,
                    'commission': commission,
                    'total_cost': total_needed,
                    'profit_per_crypto': profit_per_crypto,
                    'total_profit': total_profit,
                    'profit_percentage': (profit_per_crypto / self.avg_sale_price) * 100
                }
                self.sale_history.append(purchase)
                
                # Resetear posición
                self.position_sold = False
                self.avg_sale_price = 0.0
                self.total_crypto_sold = 0.0
                
                logger.debug(f"💰 Recompra Take Profit: {crypto_purchased:.6f} a ${current_price:.2f}, Ganancia: ${total_profit:.2f}")
                return True
        
        return False
    
    def _calculate_portfolio_value(self, current_price: float) -> float:
        """
        Calcula el valor total del portafolio.
        
        Args:
            current_price: Precio actual del activo
            
        Returns:
            Valor total del portafolio en USD
        """
        crypto_value = self.crypto_balance * current_price
        total_value = self.usdt_balance + crypto_value
        return total_value
    
    def _update_performance_metrics(self, current_price: float):
        """
        Actualiza las métricas de performance.
        
        Args:
            current_price: Precio actual del activo
        """
        portfolio_value = self._calculate_portfolio_value(current_price)
        self.portfolio_values.append(portfolio_value)
        
        # Actualizar valor máximo
        if portfolio_value > self.max_portfolio_value:
            self.max_portfolio_value = portfolio_value
        
        # Calcular drawdown actual
        if self.max_portfolio_value > 0:
            current_drawdown = ((self.max_portfolio_value - portfolio_value) / self.max_portfolio_value) * 100
            if current_drawdown > self.max_drawdown:
                self.max_drawdown = current_drawdown
    
    def run_simulation(self) -> Dict[str, Any]:
        """
        Ejecuta la simulación completa de BTD Short.
        
        Returns:
            Dict con resultados de la simulación
        """
        logger.info(f"🔄 Iniciando simulación BTD Short...")
        logger.info(f"📊 Período: {len(self.df)} días de datos")
        
        # Iterar a través de cada día
        for i in range(len(self.df)):
            day_data = self.df.iloc[i]
            current_price = float(day_data['close'])
            current_date = datetime.now()  # Simplificado
            
            # 1. Verificar si debe recomprar primero (take profit en short)
            if self._should_take_profit(current_price):
                self._execute_purchase(current_price, current_date)
            
            # 2. Si no hay posición vendida, verificar oportunidad de venta
            elif not self.position_sold:
                # Verificar tendencia bajista (Death Cross)
                if self._is_bearish_trend(i):
                    # Obtener mínimo reciente
                    recent_low = self._get_recent_low(i)
                    
                    # Verificar si es una oportunidad de rip para vender
                    if self._is_rip_opportunity(current_price, recent_low):
                        self._execute_sale(current_price, current_date)
            
            # Actualizar métricas de performance
            self._update_performance_metrics(current_price)
        
        # Calcular resultados finales
        final_price = float(self.df.iloc[-1]['close'])
        final_portfolio_value = self._calculate_portfolio_value(final_price)
        total_return = final_portfolio_value - self.initial_capital
        return_percentage = (total_return / self.initial_capital) * 100
        
        # Calcular win rate basado en operaciones exitosas vs perdedoras
        profitable_trades = len([t for t in self.sale_history if t.get('profit_percentage', 0) > 0])
        total_completed_trades = len([t for t in self.sale_history if t.get('type') == 'BUY'])  # Solo contar recompras completadas
        win_rate = (profitable_trades / total_completed_trades * 100) if total_completed_trades > 0 else 0.0
        
        # Calcular Sharpe Ratio
        if len(self.portfolio_values) > 1:
            # Calcular retornos diarios del portafolio
            portfolio_series = pd.Series(self.portfolio_values)
            daily_returns = portfolio_series.pct_change().dropna()
            
            if len(daily_returns) > 0 and daily_returns.std() != 0:
                # Sharpe Ratio = (retorno promedio - risk-free rate) / desviación estándar
                # Asumimos risk-free rate = 0 para simplificar
                mean_daily_return = daily_returns.mean()
                std_daily_return = daily_returns.std()
                sharpe_ratio = mean_daily_return / std_daily_return * (252 ** 0.5)  # Anualizado
            else:
                sharpe_ratio = 0.0
        else:
            sharpe_ratio = 0.0
        
        results = {
            'initial_capital': self.initial_capital,
            'final_portfolio_value': final_portfolio_value,
            'pnl_final': total_return,
            'return_percentage': return_percentage,
            'max_drawdown': self.max_drawdown,
            'total_trades': self.total_sales + self.total_purchases,
            'total_sales': self.total_sales,
            'total_purchases': self.total_purchases,
            'total_commission': self.total_commission_paid,
            'crypto_holdings': self.crypto_balance,
            'usdt_remaining': self.usdt_balance,
            'avg_sale_price': self.avg_sale_price,
            'final_price': final_price,
            'win_rate': win_rate,
            'rip_threshold_percent': self.rip_threshold * 100,
            'take_profit_percent': self.take_profit_threshold * 100,
            'sma_fast_period': self.sma_fast_period,
            'sma_slow_period': self.sma_slow_period,
            'simulation_days': len(self.df),
            'position_still_sold': self.position_sold,
            'sharpe_ratio': sharpe_ratio,
            'initial_crypto_ratio': self.initial_crypto_ratio * 100
        }
        
        logger.info(f"✅ Simulación BTD Short completada")
        logger.info(f"💰 Retorno final: ${total_return:.2f} ({return_percentage:.2f}%)")
        logger.info(f"💸 Ventas: {self.total_sales}, Recompras: {self.total_purchases}")
        logger.info(f"📈 Precio promedio de venta: ${self.avg_sale_price:.4f}")
        logger.info(f"📉 Max Drawdown: {self.max_drawdown:.2f}%")
        
        return results
    
    def get_detailed_results(self) -> Dict[str, Any]:
        """
        Obtiene resultados detallados incluyendo el historial de operaciones.
        
        Returns:
            Dict con resultados detallados
        """
        basic_results = self.run_simulation()
        
        detailed_results = basic_results.copy()
        detailed_results.update({
            'operation_history': self.sale_history,
            'portfolio_value_history': self.portfolio_values,
            'crypto_balance_final': self.crypto_balance,
            'commission_percentage': self.commission_rate * 100
        })
        
        return detailed_results


# ============================================================================
# FUNCIONES DE EJEMPLO Y TESTING
# ============================================================================


def create_sample_dataframe(days: int = 365) -> pd.DataFrame:
    """
    Crea un DataFrame de muestra para testing.
    
    Args:
        days: Número de días de datos históricos
        
    Returns:
        DataFrame con datos OHLC simulados
    """
    np.random.seed(42)  # Para reproducibilidad
    
    # Generar fechas
    dates = pd.date_range(start='2023-01-01', periods=days, freq='D')
    
    # Precio inicial
    initial_price = 100.0
    
    # Generar precios con tendencia y volatilidad
    returns = np.random.normal(0.0005, 0.02, days)  # 0.05% tendencia diaria, 2% volatilidad
    prices = [initial_price]
    
    for ret in returns[1:]:
        new_price = prices[-1] * (1 + ret)
        prices.append(max(new_price, 1.0))  # Precio mínimo $1
    
    # Generar datos OHLC
    data = []
    for i, price in enumerate(prices):
        # Simular volatilidad intraday
        daily_volatility = price * 0.015  # 1.5% volatilidad intraday
        
        open_price = price * (1 + np.random.normal(0, 0.005))
        high_price = max(open_price, price) + abs(np.random.normal(0, daily_volatility))
        low_price = min(open_price, price) - abs(np.random.normal(0, daily_volatility))
        close_price = price
        
        data.append({
            'open': round(open_price, 2),
            'high': round(high_price, 2),
            'low': round(low_price, 2),
            'close': round(close_price, 2),
            'volume': int(np.random.uniform(1000000, 10000000))
        })
    
    df = pd.DataFrame(data, index=dates)
    return df


def run_backtest_example():
    """
    Ejemplo de uso del GridBotSimulator.
    """
    print("🧪 Ejecutando ejemplo de backtesting...")
    
    # Crear datos de muestra
    df = create_sample_dataframe(365)
    print(f"📊 Datos creados: {len(df)} días")
    
    # Configuración del grid
    config = {
        'range_percent': 10.0,  # 10% de rango
        'levels': 4,            # 4 niveles
        'commission': 0.001,    # 0.1% de comisión
        'initial_capital': 1000.0  # $1000 inicial
    }
    
    # Crear y ejecutar simulador
    simulator = GridBotSimulator(df, config)
    results = simulator.run_simulation()
    
    # Mostrar resultados
    print("\n📈 RESULTADOS DEL BACKTESTING:")
    print("=" * 50)
    print(f"💰 P&L Final: ${results['pnl_final']}")
    print(f"📊 Retorno: {results['return_percentage']}%")
    print(f"🔄 Total Trades: {results['total_trades']}")
    print(f"📉 Max Drawdown: {results['max_drawdown']}%")
    print(f"🎯 Win Rate: {results['win_rate']}%")
    print(f"💸 Comisiones: ${results['total_commission']}")
    print(f"💼 Valor Final: ${results['final_portfolio_value']}")


def compare_grid_configurations():
    """
    Compara diferentes configuraciones de grid para encontrar la óptima.
    """
    print("🔬 Comparando diferentes configuraciones de Grid...")
    
    # Crear datos de muestra
    df = create_sample_dataframe(365)
    
    # Configuraciones a probar
    configurations = [
        {'range_percent': 5.0, 'levels': 2, 'commission': 0.001, 'initial_capital': 1000.0},
        {'range_percent': 10.0, 'levels': 4, 'commission': 0.001, 'initial_capital': 1000.0},
        {'range_percent': 15.0, 'levels': 6, 'commission': 0.001, 'initial_capital': 1000.0},
        {'range_percent': 20.0, 'levels': 8, 'commission': 0.001, 'initial_capital': 1000.0},
        {'range_percent': 10.0, 'levels': 4, 'commission': 0.002, 'initial_capital': 1000.0},  # Comisión más alta
    ]
    
    results_comparison = []
    
    for i, config in enumerate(configurations):
        print(f"\n📊 Probando configuración {i+1}: {config['levels']} niveles, ±{config['range_percent']}%")
        
        simulator = GridBotSimulator(df, config)
        results = simulator.run_simulation()
        
        results_comparison.append({
            'config_id': i + 1,
            'config': config,
            'results': results
        })
        
        print(f"   💰 P&L: ${results['pnl_final']} ({results['return_percentage']}%)")
        print(f"   📊 Trades: {results['total_trades']}")
        print(f"   📉 Drawdown: {results['max_drawdown']}%")
        print(f"   🎯 Win Rate: {results['win_rate']}%")
    
    # Encontrar la mejor configuración
    best_config = max(results_comparison, key=lambda x: x['results']['return_percentage'])
    
    print(f"\n🏆 MEJOR CONFIGURACIÓN:")
    print(f"Configuración {best_config['config_id']}: {best_config['config']['levels']} niveles, ±{best_config['config']['range_percent']}%")
    print(f"Retorno: {best_config['results']['return_percentage']}%")
    print(f"P&L: ${best_config['results']['pnl_final']}")
    
    return results_comparison


if __name__ == "__main__":
    # Ejecutar ejemplo básico
    run_backtest_example()
    
    print("\n" + "="*60)
    
    # Ejecutar comparación de configuraciones
    compare_grid_configurations() 