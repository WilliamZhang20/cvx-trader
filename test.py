# Simple script to test alpaca is working
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# --- KEYS ARE SET IN OS ---
# Put your API Key ID here
api_key_id = 'API_KEY' 
# Put your Secret Key here
secret_key = 'SECRET_KEY' 

# Connect to the Alpaca paper trading API
# The 'paper=True' argument is crucial for using your paper account
trading_client = TradingClient(api_key_id, secret_key, paper=True)

# Get and print your account information to confirm it works
account = trading_client.get_account()
print("Successfully connected to your paper trading account!")
print(f"Account Status: {account.status}")
print(f"Current Buying Power: ${account.buying_power}")
print(f"Portfolio Value: ${account.portfolio_value}")

# --- Place a simulated order ---
# This is an example to buy 1 share of AAPL at the current market price
order_data = MarketOrderRequest(
    symbol="AAPL",
    qty=1,
    side=OrderSide.BUY,
    time_in_force=TimeInForce.DAY
)
"""
try:
    # Submit the order to the API
    submit_order = trading_client.submit_order(order_data=order_data)
    print("\nOrder submitted successfully!")
    print(f"Order ID: {submit_order.id}")
    print(f"Symbol: {submit_order.symbol}")
    print(f"Side: {submit_order.side}")
    print(f"Status: {submit_order.status}")
except Exception as e:
    print(f"\nError submitting order: {e}")
"""
# To get your open positions
# positions = trading_client.get_all_positions()
# print("\nYour current positions:")
# for position in positions:
#    print(f"- {position.symbol}: {position.qty} shares")