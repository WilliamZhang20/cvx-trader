from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
import os

# API credentials
api_key_id = os.environ["APCA_API_KEY_ID"]
secret_key = os.environ["APCA_API_SECRET_KEY"]

# Connect to Alpaca paper account
trading_client = TradingClient(api_key_id, secret_key, paper=True)

# Get account info
account = trading_client.get_account()
print("Connected to Alpaca paper trading account.")
print(f"Account Status: {account.status}")
print(f"Buying Power: ${account.buying_power}")
print(f"Portfolio Value: ${account.portfolio_value}")

# Symbols to purge
symbols_to_sell = ['HD']

# Get all current positions
positions = trading_client.get_all_positions()

for position in positions:
    if position.symbol in symbols_to_sell:
        qty = float(position.qty)
        if qty > 0:
            print(f"Selling {qty} shares of {position.symbol}...")
            try:
                order_data = MarketOrderRequest(
                    symbol=position.symbol,
                    qty=qty,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY
                )
                order = trading_client.submit_order(order_data=order_data)
                print(f"Sell order submitted for {position.symbol}: {order.id}")
            except Exception as e:
                print(f"Error selling {position.symbol}: {e}")
