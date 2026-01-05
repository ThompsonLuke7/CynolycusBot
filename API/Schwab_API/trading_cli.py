# trading_cli.py

from schwab_client import SchwabClient
import json

def choose_account(accounts):
    print("\nAvailable accounts:")
    for i, acct in enumerate(accounts):
        display = acct.get("displayAccountNumber") or acct.get("accountNumber")
        print(f"[{i}] {display}  (hash: {acct['hashValue']})")

    choice = int(input("Select account index to use: "))
    return accounts[choice]["hashValue"]


def main():
    client = SchwabClient()

    # 1) Pick account
    accounts = client.get_account_numbers()
    # print(json.dumps(accounts, indent=2))  # uncomment if you want to inspect raw
    account_hash = choose_account(accounts)
    print(f"\nUsing account hash: {account_hash}")

    while True:
        print("\n=== Schwab Trading CLI ===")
        print("1) Get quotes")
        print("2) Place MARKET order")
        print("3) Exit")
        choice = input("Select an option: ").strip()

        if choice == "1":
            symbols = input("Enter symbols (comma-separated, e.g. SPY,NVDA): ").upper()
            symbol_list = [s.strip() for s in symbols.split(",") if s.strip()]
            quotes = client.get_quotes(symbol_list)
            print("\nQuotes:")
            print(json.dumps(quotes, indent=2))

        elif choice == "2":
            symbol = input("Symbol (e.g. SPY): ").upper().strip()
            side = input("Instruction (BUY or SELL): ").upper().strip()
            qty_str = input("Quantity (integer): ").strip()

            try:
                qty = int(qty_str)
            except ValueError:
                print("Quantity must be an integer.")
                continue

            print(f"\nYou are about to place a MARKET {side} of {qty} shares of {symbol}.")
            confirm = input("Type 'YES' to confirm: ").strip()
            if confirm != "YES":
                print("Order cancelled.")
                continue

            try:
                result = client.place_order_market(account_hash, symbol, qty, side)
                print("\n✅ Order submitted:")
                print(result)
            except Exception as e:
                print("\n❌ Order failed:")
                print(e)

        elif choice == "3":
            print("Exiting CLI.")
            break
        else:
            print("Invalid choice. Try again.")


if __name__ == "__main__":
    main()
