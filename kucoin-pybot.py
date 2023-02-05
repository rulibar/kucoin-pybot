"""
Kucoin Pybot v1.1 (23-2-5)
https://github.com/rulibar/kucoin-pybot

Warning: Not yet working.
"""

import os
import time
from calendar import timegm as timegm
import numpy
import random
import json
import logging
import talib

# instance vars
# if exchange has no api_passphrase then leave it empty
asset = "ETH"; base = "BTC"
interval_mins = 30
exchange = "kucoin"
api_key = ""
api_secret = ""
api_passphrase = ""

# strategy vars
storage = dict()

# set up logger
def set_log_file():
    # Set up the log folders
    gmt = time.gmtime()
    yy = str(gmt.tm_year)[2:]; mm = str(gmt.tm_mon); dd = str(gmt.tm_mday)
    if len(mm) == 1: mm = "0" + mm
    if len(dd) == 1: dd = "0" + dd
    path = "./logs/"
    if not os.path.isdir(path): os.mkdir(path)
    path += "{}/".format(yy + mm)
    if not os.path.isdir(path): os.mkdir(path)
    # Set the log destination and format
    fileh = logging.FileHandler("./logs/{}/{}.log".format(yy + mm, yy + mm + dd), "a")
    formatter = logging.Formatter("%(levelname)s %(asctime)s - %(message)s")
    fileh.setFormatter(formatter)
    logger.handlers = [fileh]

logging.basicConfig(level=logging.INFO)
logging.Formatter.converter = time.gmtime
logger = logging.getLogger()
set_log_file()

# set up trading bot
def fix_dec(float_in):
    float_out = "{:.8f}".format(float_in)
    while float_out[-1] == "0": float_out = float_out[:-1]
    if float_out[-1] == ".": float_out = float_out[:-1]
    return float_out

def shrink_list(list_in, size):
    if len(list_in) > size: return list_in[-size:]
    return list_in

class Exchange:
    def __init__(self, exchange, api):
        self.name = str(exchange).lower()
        api_key = api[0]
        api_secret = api[1]
        api_passphrase = api[2]

        if self.name == "binance":
            from binance.client import Client
            self.client = Client(api_key, api_secret)
        elif self.name == "kucoin":
            from kucoin.client import Client
            self.client = Client(api_key, api_secret, api_passphrase)
        else: logger.error(f"Error: Unsupported exchange '{exchange}'."); exit()

        # Exchange vars
        self.positions_init_ts = 0
        self.last_acc_check = 0
        self.last_acc_check_cache = 0

        # Binance vars
        self.deposits_pending = dict()
        self.withdrawals_pending = dict()

        # KuCoin vars
        self.tickers = dict()
        if self.name == "kucoin":
            currencies = self.client.get_currencies()
            for i in range(len(currencies)):
                coin_name = currencies[i]['currency']
                coin_ticker = currencies[i]['name']
                self.tickers[coin_name] = coin_ticker

    def get_account(self, asset, base):
        data = {"asset": [asset, 0.0], "base": [base, 0.0]}

        self.last_acc_check_cache = int(self.last_acc_check)
        if self.positions_init_ts == 0:
            self.positions_init_ts = int(1000 * time.time())
            self.last_acc_check_cache = int(self.positions_init_ts)

        if self.name == "binance":
            acc = self.client.get_account()["balances"]
            self.last_acc_check = int(1000 * time.time())
            for i in range(len(acc)):
                acc_asset = acc[i]["asset"]
                if acc_asset not in {asset, base}: continue
                free = float(acc[i]["free"])
                locked = float(acc[i]["locked"])
                total = free + locked
                if acc_asset == asset: data['asset'][1] = total
                if acc_asset == base: data['base'][1] = total
        elif self.name == "kucoin":
            acc = self.client.get_accounts()
            self.last_acc_check = int(1000 * time.time())
            for i in range(len(acc)):
                if acc[i]['type'] != 'trade': continue
                acc_asset = self.tickers[acc[i]["currency"]]
                if acc_asset not in {asset, base}: continue
                free = float(acc[i]["available"])
                locked = float(acc[i]["holds"])
                total = free + locked
                if acc_asset == asset: data['asset'][1] = total
                if acc_asset == base: data['base'][1] = total

        return data

    def get_dws(self, asset, base):
        data = list()

        ts_last = int(self.last_acc_check_cache)
        if self.name == "binance":
            start_time = int(ts_last)
            for dep in self.deposits_pending: start_time = min([start_time, self.deposits_pending[dep] - 1000])
            for wit in self.withdrawals_pending: start_time = min([start_time, self.withdrawals_pending[wit] - 1000])
            if start_time == self.positions_init_ts: start_time -= 24 * 60 * 60 * 1000

            deposits = self.client.get_deposit_history(startTime = start_time)
            withdrawals = self.client.get_withdraw_history(startTime = start_time)

            # deal with differing formats
            if type(deposits) is dict:
                deposits = deposits['depositList']

            if type(withdrawals) is dict:
                withdrawals = withdrawals['withdrawList']

            for deposit in deposits:
                if 'asset' not in deposit:
                    deposit['asset'] = str(deposit['coin'])
                    deposit['creator'] = ''
                    deposit['amount'] = float(deposit['amount'])

            for withdrawal in withdrawals:
                if 'asset' not in withdrawal:
                    withdrawal['asset'] = str(withdrawal['coin'])
                    withdrawal['withdrawOrderId'] = ''
                    withdrawal['amount'] = float(withdrawal['amount'])
                    withdrawal['transactionFee'] = float(withdrawal['transactionFee'])

                if type(withdrawal['applyTime']) is str:
                    applyTime = time.strptime(withdrawal['applyTime'], '%Y-%m-%d %H:%M:%S')
                    applyTime = 1000 * timegm(applyTime)
                    withdrawal['applyTime'] = int(applyTime)

            deposits = [d for d in deposits if d['asset'] in {asset, base}]
            withdrawals = [w for w in withdrawals if w['asset'] in {asset, base}]

            d_complete = list()
            w_complete = list()

            # add new pending trades to pending set
            # check for and process old pending trades that were filled
            for deposit in deposits:
                id = deposit['txId']
                if deposit['status'] < 1:
                    self.deposits_pending[id] = deposit['insertTime']
                    continue
                d_obj = dict()
                d_obj['asset'] = deposit['asset']
                d_obj['amt'] = float(deposit['amount'])
                d_obj['fee'] = 0.0
                if id not in self.deposits_pending.keys():
                    if deposit['insertTime'] > ts_last: d_complete.append(d_obj)
                    continue
                d_complete.append(d_obj)
                self.deposits_pending.pop(id)
            for withdrawal in withdrawals:
                id = withdrawal['id']
                if withdrawal['status'] < 0:
                    self.withdrawals_pending[id] = withdrawal['applyTime']
                    continue
                w_obj = dict()
                w_obj['asset'] = withdrawal['asset']
                w_obj['amt'] = float(withdrawal['amount'])
                w_obj['fee'] = float(withdrawal['transactionFee'])
                if id not in self.withdrawals_pending.keys():
                    if withdrawal['applyTime'] > ts_last: w_complete.append(w_obj)
                    continue
                w_complete.append(w_obj)
                self.withdrawals_pending.pop(id)

            data.append(d_complete)
            data.append(w_complete)
        elif self.name == "kucoin":
            account_activity = self.client.get_account_activity()['items']
            d_complete = list()
            w_complete = list()

            for i in range(len(account_activity)):
                item = account_activity[i]
                if item['accountType'] != 'TRADE': continue
                if item['bizType'] != 'Transfer': continue
                if item['createdAt'] < self.last_acc_check_cache: continue
                if item['createdAt'] > self.last_acc_check: continue
                item_asset = self.tickers[item['currency']]
                if item_asset not in {asset, base}: continue
                item_obj = dict()
                item_obj['asset'] = item_asset
                item_obj['amt'] = float(item['amount'])
                item_obj['fee'] = 0.0
                if item['direction'] == 'in': d_complete.append(item_obj)
                if item['direction'] == 'out': w_complete.append(item_obj)

            data.append(d_complete)
            data.append(w_complete)

        return data

    def get_trades(self, asset, base, max_num):
        data = list()

        if self.name == "binance":
            pair = f'{asset}{base}'
            trades = reversed(self.client.get_my_trades(symbol = pair, limit = max_num))
            for trade in trades:
                if trade['time'] < self.last_acc_check_cache: continue
                if trade['time'] > self.last_acc_check: continue
                tr = dict()
                tr['amt_asset'] = trade['qty']
                tr['amt_base'] = trade['quoteQty']
                tr['amt_fee'] = trade['commission']
                tr['fee_currency'] = trade['commissionAsset']
                tr['side'] = 'sell'
                if trade['isBuyer']: tr['side'] = 'buy'
                data.append(tr)
        elif self.name == "kucoin":
            account_activity = self.client.get_account_activity()['items']
            trades = dict()
            account_activity = [item for item in account_activity if item['accountType'] == 'TRADE']
            account_activity = [item for item in account_activity if item['createdAt'] > self.last_acc_check_cache]
            account_activity = [item for item in account_activity if item['createdAt'] < self.last_acc_check]

            for i in range(len(account_activity)):
                item = account_activity[i]
                if item['bizType'] != 'Exchange': continue
                item_asset = self.tickers[item['currency']]
                if item_asset not in {asset, base}: continue
                context = json.loads(item['context'])
                id = context['orderId']
                if id not in trades:
                    trades[id] = dict()
                    trades[id]['amt_asset'] = 0.0
                    trades[id]['amt_base'] = 0.0
                    trades[id]['amt_fee'] = 0.0
                    trades[id]['side'] = 'buy'
                if item_asset == base:
                    trades[id]['amt_base'] += float(item['amount'])
                    trades[id]['amt_fee'] += float(item['fee'])
                    trades[id]['fee_currency'] = base
                    if item['direction'] == 'in': trades[id]['amt_base'] += float(item['fee'])
                    if item['direction'] == 'out': trades[id]['amt_base'] -= float(item['fee'])
                elif item_asset == asset:
                    trades[id]['amt_asset'] += float(item['amount'])
                    if item['direction'] == 'out': trades[id]['side'] = 'sell'
            for i in range(len(account_activity)):
                item = account_activity[i]
                if item['bizType'] != 'KCS Pay Fees': continue
                item_asset = self.tickers[item['currency']]
                context = json.loads(item['context'])
                id = context['orderId']
                if id in trades:
                    trades[id]['amt_fee'] = float(item['amount'])
                    trades[id]['fee_currency'] = item_asset
            for id in trades:
                if trades[id]['amt_asset'] == 0: continue
                if trades[id]['amt_base'] == 0: continue
                for key in trades[id]:
                    if key in {'fee_currency', 'side'}: continue
                    trades[id][key] = f"{trades[id][key]:.8f}"
                data.append(trades[id])

        return data

    def get_pair_info(self, asset, base):
        data = dict()

        if self.name == "binance":
            pair = f'{asset}{base}'
            pair_info = self.client.get_symbol_info(pair)['filters']
            data['asset_min_qty'] = pair_info[2]['minQty']
            data['base_min_qty'] = pair_info[3]['minNotional']
            data['asset_precision'] = pair_info[2]['stepSize']
            data['price_precision'] = pair_info[0]['tickSize']
        elif self.name == "kucoin":
            pair = f'{asset}-{base}'
            pair_info = self.client.get_symbols()
            pair_info = [item for item in pair_info if item['name'] == pair][0]
            data['asset_min_qty'] = pair_info['baseMinSize']
            data['base_min_qty'] = pair_info['quoteMinSize']
            data['asset_precision'] = pair_info['baseIncrement']
            data['price_precision'] = pair_info['priceIncrement']

        return data

    def get_historical_candles(self, asset, base, n_candles):
        data = list()

        if self.name == "binance":
            pair = f'{asset}{base}'
            start_str = f"{n_candles + 5} minutes ago UTC"

            tries = 0
            while True:
                candles, err = list(), str()
                try: candles = self.client.get_historical_klines(pair, "1m", start_str)
                except Exception as e: err = e
                tries += 1
                if len(candles) == 0:
                    if tries <= 3:
                        err_msg = "Error getting historical candle data. Retrying in 5 seconds..."
                        if err != "": err_msg += "\n'{}'".format(err)
                        logger.error(err_msg)
                    if tries == 3:
                        logger.error("(Future repeats of this error hidden to avoid spam.)")
                    time.sleep(5)
                else: break
            if tries > 3: logger.error("Failed to get historical candle data {} times.".format(tries - 1))
            ts_data_end = int(time.time())

            for i in range(len(candles)):
                candle = candles[i]
                data.append({
                    "ts_start": int(candle[0]),
                    "open": round(float(candle[1]), 8),
                    "high": round(float(candle[2]), 8),
                    "low": round(float(candle[3]), 8),
                    "close": round(float(candle[4]), 8),
                    "volume": round(float(candle[7]), 8),
                    "ts_end": int(candle[6])})

            if data[-1]['ts_end'] > ts_data_end: data.pop()
            data = data[-n_candles:]
        elif self.name == "kucoin":
            for key in self.tickers:
                if self.tickers[key] == asset: asset = key
            pair = f'{asset}-{base}'
            ts_data_start = int(time.time() - 1500*60)

            # Find the latest candles with volume
            klines = list()
            while len(klines) == 0:
                try: klines = self.client.get_kline_data(symbol = pair, kline_type = "1min", start = ts_data_start)
                except Exception as err:
                    err_msg = "Error getting historical candle data. Retrying in 5 seconds..."
                    if err != "": err_msg += "\n'{}'".format(err)
                    logger.error(err_msg)
                    time.sleep(5)
                    continue
                ts_data_end = int(time.time())
                ts_data_start = int(ts_data_start - 1500*60)

            # fill in missing candles with zero volume candles
            candles = list(klines)
            while int(candles[0][0]) + 60 < ts_data_end:
                candles = [[str(int(candles[0][0]) + 60), candles[0][1], candles[0][2], candles[0][3], candles[0][4], '0', '0']] + candles

            # get enough candles to fulfill the request
            while len(candles) < n_candles + 5:
                ts_data_start = int(candles[-1][0]) - 1500*60
                try: klines = self.client.get_kline_data(symbol = pair, kline_type = "1min", start = ts_data_start, end = candles[-1][0])
                except Exception as err:
                    err_msg = "Error getting historical candle data. Retrying in 5 seconds..."
                    if err != "": err_msg += "\n'{}'".format(err)
                    logger.error(err_msg)
                    time.sleep(5)
                    continue
                candles = candles + klines

            # Put the list in the desired format
            candles = candles[::-1]
            for i in range(len(candles)):
                candle = candles[i]
                data.append({
                    "ts_start": int(candle[0]) * 1000,
                    "open": round(float(candle[1]), 8),
                    "high": round(float(candle[3]), 8),
                    "low": round(float(candle[4]), 8),
                    "close": round(float(candle[2]), 8),
                    "volume": round(float(candle[6]), 8),
                    "ts_end": (int(candle[0]) + 60) * 1000 - 1})

            if data[-1]['ts_end'] > 1000 * ts_data_end: data.pop()
            data = data[-n_candles:]

        return data

    def get_open_orders(self, asset, base):
        data = list()

        if self.name == "binance":
            pair = f'{asset}{base}'
            open_orders = self.client.get_open_orders(symbol = pair)
            data = [{"order_id": order["orderId"]} for order in open_orders]
        elif self.name == "kucoin":
            for key in self.tickers:
                if self.tickers[key] == asset: asset = key
            pair = f'{asset}-{base}'
            open_orders = self.client.get_orders(symbol = pair, status = 'active')['items']
            data = [{"order_id": order["id"]} for order in open_orders]

        return data

    def cancel_order(self, asset, base, order_id):
        if self.name == "binance":
            pair = f'{asset}{base}'
            self.client.cancel_order(symbol = pair, orderId = order_id)
        elif self.name == "kucoin":
            self.client.cancel_order(order_id)

    def order_limit_buy(self, asset, base, amt, pt):
        if self.name == "binance":
            pair = f'{asset}{base}'
            self.client.order_limit_buy(symbol = pair, quantity = "{:.8f}".format(amt), price = "{:.8f}".format(pt))
        elif self.name == "kucoin":
            for key in self.tickers:
                if self.tickers[key] == asset: asset = key
            pair = f'{asset}-{base}'
            self.client.create_limit_order(symbol = pair, size = "{:.8f}".format(amt), price = "{:.8f}".format(pt), side = 'buy')

    def order_limit_sell(self, asset, base, amt, pt):
        if self.name == "binance":
            pair = f'{asset}{base}'
            self.client.order_limit_sell(symbol = pair, quantity = "{:.8f}".format(amt), price = "{:.8f}".format(pt))
        elif self.name == "kucoin":
            for key in self.tickers:
                if self.tickers[key] == asset: asset = key
            pair = f'{asset}-{base}'
            self.client.create_limit_order(symbol = pair, size = "{:.8f}".format(amt), price = "{:.8f}".format(pt), side = 'sell')

class Portfolio:
    def __init__(self, candle, positions, funds):
        self.ts = candle['ts_end']
        self.asset = positions['asset'][1]
        self.base = positions['base'][1]
        self.price = candle['close']
        self.positionValue = self.price * self.asset
        self.size = self.base + self.positionValue
        self.funds = float(funds)
        if self.funds > self.size or self.funds == 0: self.funds = float(self.size)
        self.sizeT = float(self.funds)
        self.rin = self.price * self.asset / self.size
        self.rinT = self.price * self.asset / self.sizeT

class Instance:
    def __init__(self, asset, base, interval_mins):
        self.next_log = 0
        self.ticks = 0; self.days = 0; self.trades = 0
        self.exchange = exchange
        self.base = str(base)
        self.asset = str(asset)
        self.pair = self.asset + self.base
        self.interval = int(interval_mins)
        logger.info("New trader instance started on {} {} {}m.".format(self.exchange.title(), self.pair, self.interval))
        self.get_params()

        self.candles_raw = client.get_historical_candles(self.asset, self.base, 600 * self.interval)
        self.candles_raw_unused = 0
        if (time.time() - self.candles_raw[-1]["ts_end"]/1000) > 60: self.topoff_candles_raw()
        self.candles = self._get_candles()
        self.topoff_candles()

        self.candle_start = None
        self.positions_start = None
        self.positions = self.get_positions()
        self.positions_f = {'asset': list(self.positions['asset'])}
        self.positions_f['base'] = list(self.positions['base'])
        self.positions_t = {'asset': list(self.positions['asset'])}
        self.positions_t['base'] = list(self.positions['base'])
        p = Portfolio(self.candles[-1], self.positions, self.params['funds'])
        self.last_order = {"type": "none", "amt": 0, "pt": self.candles[-1]['close']}
        self.signal = {"rinTarget": p.rinT, "rinTargetLast": p.rinT, "position": "none", "status": 0, "apc": p.price, "target": p.price, "stop": p.price}
        self.performance = {"bh": 0, "change": 0, "W": 0, "L": 0, "wSum": 0, "lSum": 0, "w": 0, "l": 0, "be": 0, "aProfits": 0, "bProfits": 0, "cProfits": 0}
        self.init(p)

    def _get_candles(self):
        # convert historical 1m candles into historical candles
        candles = list(); candle_new = dict()
        candles_raw_clone = list(self.candles_raw)
        for i in range(self.interval-2): candles_raw_clone.pop()
        for i in range(len(candles_raw_clone)):
            order = i % self.interval
            candle_raw = dict(candles_raw_clone[-1 - i])

            if order == 0:
                candle_new = dict(candle_raw)
                continue

            if candle_raw["high"] > candle_new["high"]:
                candle_new["high"] = candle_raw["high"]
            if candle_raw["low"] < candle_new["low"]:
                candle_new["low"] = candle_raw["low"]
            candle_new["volume"] += candle_raw["volume"]

            if order == self.interval - 1:
                candle_new["open"] = candle_raw["open"]
                candle_new["ts_start"] = candle_raw["ts_start"]
                candles.append(candle_new)

        self.candles_raw = self.candles_raw[-2*self.interval:]
        return candles[::-1]

    def limit_buy(self, amt, pt):
        try:
            logger.warning("Trying to buy {} {} for {} {}. (price: {})".format(fix_dec(amt), self.asset, fix_dec(round(amt * pt, self.pt_dec)), self.base, fix_dec(pt)))
            self.last_order = {"type": "buy", "amt": amt, "pt": pt}
            client.order_limit_buy(self.asset, self.base, amt, pt)
        except Exception as err:
            logger.error("Error buying.\n'{}'".format(err))

    def limit_sell(self, amt, pt):
        try:
            logger.warning("Trying to sell {} {} for {} {}. (price: {})".format(fix_dec(amt), self.asset, fix_dec(round(amt * pt, self.pt_dec)), self.base, fix_dec(pt)))
            self.last_order = {"type": "sell", "amt": amt, "pt": pt}
            client.order_limit_sell(self.asset, self.base, amt, pt)
        except Exception as err:
            logger.error("Error selling.\n'{}'".format(err))

    def bso(self, p):
        # buy/sell/other
        s = self.signal

        rbuy = s['rinTarget'] - s['rinTargetLast']
        order_size = 0
        if rbuy * p.asset >= 0:
            order_size = abs(rbuy * p.funds)
            if order_size > p.base: order_size = p.base
        if rbuy * p.asset < 0:
            rbuy_asset = rbuy / s['rinTargetLast']
            order_size = abs(rbuy_asset * p.asset * p.price)
        if order_size < self.min_order: order_size = 0

        if order_size > 0:
            if rbuy > 0: pt = (1 + 0.0015) * p.price
            else: pt = (1 - 0.0015) * p.price
            pt = round(pt, self.pt_dec)
            if rbuy > 0: amt = order_size / pt
            else: amt = order_size / p.price
            amt = round(0.995 * amt * 10**self.amt_dec - 2) / 10**self.amt_dec
            if rbuy > 0: self.limit_buy(amt, pt)
            if rbuy < 0: self.limit_sell(amt, pt)
        if rbuy == 0: order_size = 0
        if order_size == 0:
            if self.ticks == 1: logger.info("Waiting for a signal to trade...")
            self.last_order = {"type": "none", "amt": 0, "pt": p.price}

    def close_orders(self):
        # close open orders
        try:
            orders = client.get_open_orders(self.asset, self.base)
            for order in orders: client.cancel_order(self.asset, self.base, order['order_id'])
        except Exception as err:
            logger.error("Error closing open orders.\n'{}'".format(err))

    def update_vars(self):
        # Get preliminary vars
        self.ticks += 1
        self.days = (self.ticks - 1) * self.interval / (60 * 24)

        try: pair_info = client.get_pair_info(self.asset, self.base)
        except Exception as err:
            logger.error("Error getting pair info.\n'{}'".format(err))
            return

        min_order = float(pair_info['asset_min_qty']) * self.candles[-1]['close']
        self.min_order = 3 * max(min_order, float(pair_info['base_min_qty']))

        amt_dec = len(pair_info['asset_precision'].split('.')[1])
        for char in reversed(pair_info['asset_precision']):
            if char == "0": amt_dec -= 1
            else: break
        if amt_dec > 8:
            logger.error(f"Error: Asset precision is too high. Changing amt_dec from {amt_dec} to 8.")
            amt_dec = 8
        self.amt_dec = amt_dec

        pt_dec = len(pair_info['price_precision'].split('.')[1])
        for char in reversed(pair_info['price_precision']):
            if char == "0": pt_dec -= 1
            else: break
        if pt_dec > 8:
            logger.error(f"Error: Price precision is too high. Changing pt_dec from {pt_dec} to 8.")
            pt_dec = 8
        self.pt_dec = pt_dec

    def get_params(self):
        # import and process params from config.txt
        params = dict()
        with open("config.txt") as cfg:
            par = [l.split()[0] for l in cfg.read().split("\n")[2:-1]]
            for p in par:
                p = p.split("=")
                if len(p) != 2: continue
                params[str(p[0])] = str(p[1])

        # check values
        funds = float(params['funds'])
        if funds < 0:
            logger.warning("Warning! Maximum amount to invest should be zero or greater.")
            params['funds'] = "0"

        logs_per_day = float(params['logs_per_day'])
        if logs_per_day < 0:
            logger.warning("Warning! Logs per day should be zero or greater.")
            params['logs_per_day'] = "1"

        log_dws = str(params['log_dws'])
        if log_dws not in {"yes", "no"}:
            logger.warning("Warning! Log deposits and withdrawals set to 'yes'.")
            params['log_dws'] = "yes"

        # check for additions and removals
        if self.ticks == 0: self.params = dict()

        keys_old = {key for key in self.params}
        keys_new = {key for key in params}

        keys_added = {key for key in keys_new if key not in keys_old}
        keys_removed = {key for key in keys_old if key not in keys_new}

        if len(keys_added) > 0:
            logger.info("{} parameter(s) added.".format(len(keys_added)))
            for key in keys_added: logger.info("    \"{}\": {}".format(key, params[key]))
        if len(keys_removed) > 0:
            logger.info("{} parameter(s) removed.".format(len(keys_removed)))
            for key in keys_removed: logger.info("    \"{}\"".format(key))

        # check for changes
        keys_remaining = {key for key in keys_old if key in keys_new}
        keys_changed = set()

        for key in keys_remaining:
            if params[key] != self.params[key]: keys_changed.add(key)

        if self.ticks == 0:
            keys_changed.add('funds'); keys_changed.add('logs_per_day'); keys_changed.add('log_dws')

        if "funds" in keys_changed:
            if params['funds'] == "0": logger.info("No maximum investment amount specified.")
            else: logger.info("Maximum investment amount set to {} {}.".format(params['funds'], self.base))
            self.params['funds'] = params['funds']
            keys_changed.remove('funds')
        if "logs_per_day" in keys_changed:
            if params['logs_per_day'] == "0": logger.info("Log updates turned off.")
            elif params['logs_per_day'] == "1": logger.info("Logs updating once per day.")
            else: logger.info("Logs updating {} times per day".format(params['logs_per_day']))
            self.params['logs_per_day'] = params['logs_per_day']
            keys_changed.remove('logs_per_day')
        if "log_dws" in keys_changed:
            if params['log_dws'] == "yes": logger.info("Deposit and withdrawal logs enabled.")
            else: logger.info("Deposit and withdrawal logs disabled.")
            self.params['log_dws'] = params['log_dws']
            keys_changed.remove('log_dws')

        if len(keys_changed) > 0:
            logger.info("{} parameter(s) changed.".format(len(keys_changed)))
            for key in keys_changed:
                logger.info("    \"{}\": {} -> {}".format(key, self.params[key], params[key]))
                self.params[key] = params[key]

    def topoff_candles_raw(self):
        # check if there are new raw candles..
        # add the new raw candles..
        # return the number of new raw candles
        n_new = 0

        topoff = client.get_historical_candles(self.asset, self.base, 120)
        for i in range(len(topoff)):
            if topoff[-1]['ts_end'] == self.candles_raw[-1]['ts_end']: break
            elif topoff[i]['ts_end'] <= self.candles_raw[-1]['ts_end']: continue
            else: self.candles_raw.append(topoff[i]); n_new += 1

        return n_new

    def topoff_candles(self):
        # check if there are enough raw candles to create new candles..
        # add the new candles..
        # return the number of new candles
        n_new = 0
        raw_unused = 0

        for i in range(len(self.candles_raw)):
            candle_raw = self.candles_raw[i]
            if candle_raw['ts_end'] <= self.candles[-1]['ts_end']: continue
            else: raw_unused += 1

        self.candles_raw_unused = raw_unused % self.interval
        if raw_unused < self.interval: return 0

        candles_raw_clone = list(self.candles_raw)
        if self.candles_raw_unused == 0: candles_raw_clone = candles_raw_clone[-raw_unused:]
        else: candles_raw_clone = candles_raw_clone[-raw_unused:-self.candles_raw_unused]

        candle_new = dict()
        for i in range(len(candles_raw_clone)):
            candle_raw = dict(candles_raw_clone[i])
            order = (i + 1) % self.interval

            if order == 1:
                candle_new = dict(candle_raw)
                continue

            if candle_raw['high'] > candle_new['high']:
                candle_new['high'] = candle_raw['high']
            if candle_raw['low'] < candle_new['low']:
                candle_new['low'] = candle_raw['low']
            candle_new['volume'] += candle_raw['volume']

            if order == 0:
                candle_new['close'] = candle_raw['close']
                candle_new['ts_end'] = candle_raw['ts_end']
                candle_new['volume'] = round(candle_new['volume'], 8)
                self.candles.append(candle_new)
                candle_new = dict()
                n_new += 1

        if len(self.candles_raw) > 2*self.interval: self.candles_raw = self.candles_raw[-2*self.interval:]
        if len(self.candles) > 5000: self.candles = self.candles[-5000:]

        return n_new

    def get_positions(self):
        try: positions = client.get_account(self.asset, self.base)
        except Exception as err:
            logger.error("Error getting account balances.\n'{}'".format(err))
            return self.positions

        return positions

    def update_f(self, p, apc):
        if apc == 0:
            if self.ticks != 1: return
            apc = p.price
        r = self.performance
        s = self.signal
        pos_f = self.positions_f
        pos_t = self.positions_t

        size = p.base + apc * p.asset
        rin = apc * p.asset / size
        sizeT = p.funds * (1 - s['rinTargetLast']) + apc * p.asset
        rinT = apc * p.asset / sizeT

        if self.ticks == 1: size_f = 1; size_t = 1
        else:
            size_f = pos_f['base'][1] + apc * pos_f['asset'][1]
            size_t = pos_t['base'][1] + apc * pos_t['asset'][1]
            if s['rinTarget'] == 0 and p.positionValue < self.min_order:
                profit = size_t - 1
                if profit >= 0: r['wSum'] += profit; r['W'] += 1; self.trades += 1
                if profit < 0: r['lSum'] += profit; r['L'] += 1; self.trades += 1
                if r['W'] != 0: r['w'] = r['wSum'] / r['W']
                if r['L'] != 0: r['l'] = r['lSum'] / r['L']
                size_t = 1

        base_f = (1 - rin) * size_f; base_t = (1 - rinT) * size_t
        asset_f = (rin / apc) * size_f; asset_t = (rinT / apc) * size_t

        pos_f['base'][1] = base_f; pos_t['base'][1] = base_t
        pos_f['asset'][1] = asset_f; pos_t['asset'][1] = asset_t

    def process_dws(self):
        diffasset_dw = 0; diffbase_dw = 0

        try:
            deposits, withdrawals = client.get_dws(self.asset, self.base)
        except Exception as err:
            logger.error("Error getting deposits and withdrawals.\n'{}'".format(err))
            return 0, 0

        for deposit in deposits:
            diffasset = 0; diffbase = 0
            asset = deposit['asset']
            amt = deposit['amt']
            if self.params['log_dws'] == "yes":
                logger.warning("Deposit of {} {} detected.".format(fix_dec(amt), asset))
            if asset == self.base: diffbase += amt
            else: diffasset += amt
            diffasset_dw += diffasset; diffbase_dw += diffbase
        for withdrawal in withdrawals:
            diffasset = 0; diffbase = 0
            asset = withdrawal['asset']
            amt = withdrawal['amt'] + withdrawal['fee']
            if self.params['log_dws'] == "yes":
                logger.warning("Withdrawal of {} {} detected.".format(fix_dec(amt), asset))
            if asset == self.base: diffbase -= amt
            else: diffasset -= amt
            diffasset_dw += diffasset; diffbase_dw += diffbase

        return diffasset_dw, diffbase_dw

    def process_trades(self, p):
        diffasset_trad = 0; diffbase_trad = 0
        l = self.last_order
        s = self.signal

        # Get trades
        try: trades = client.get_trades(self.asset, self.base, 20)
        except Exception as err:
            logger.error("Error getting trade info.\n'{}'".format(err))
            return 0, 0, p.price

        # process trades
        if len(trades) > 0:
            #str_out = "{} new trade(s) found.".format(len(trades))
            for trade in trades:
                #str_out += "\n    {}".format(trade)
                amt_asset = float(trade['amt_asset'])
                amt_base = float(trade['amt_base'])
                price = amt_base / amt_asset
                if trade['side'] != 'buy': amt_asset *= -1
                diffasset_trad += amt_asset
                diffbase_trad -= amt_asset * price

        rbuy = s['rinTarget'] - s['rinTargetLast']
        rTrade = 0
        apc = 0
        if diffasset_trad != 0: apc = -diffbase_trad / diffasset_trad
        if l['amt'] != 0: rTrade = abs(diffasset_trad / l['amt'])
        if diffasset_trad > 0:
            log_amt = "{} {}".format(fix_dec(diffasset_trad), self.asset)
            log_size = "{} {}".format(fix_dec(diffasset_trad * apc), self.base)
            if l['type'] != "buy":
                logger.info("Manual buy detected.")
                rTrade = 0
            elif abs(rTrade - 1) > 0.1:
                logger.info("Buy order partially filled.")
            logger.warning("{} bought for {}.".format(log_amt, log_size))
        elif diffasset_trad < 0:
            log_amt = "{} {}".format(fix_dec(-diffasset_trad), self.asset)
            log_size = "{} {}".format(fix_dec(-diffasset_trad * apc), self.base)
            if l['type'] != "sell":
                logger.info("Manual sell detected")
                rTrade = 0
            elif abs(rTrade - 1) > 0.1:
                logger.info("Sell order partially filled.")
            logger.warning("{} sold for {}.".format(log_amt, log_size))

        if self.ticks == 1 or diffasset_trad != 0:
            s['rinTargetLast'] += rTrade * rbuy
            self.update_f(p, apc)

        return diffasset_trad, diffbase_trad, apc

    def process_dwts(self, p):
        # get deposits, withdrawals, and trades
        s = self.signal
        diffasset = round(self.positions['asset'][1] - self.positions_last['asset'][1], 8)
        diffbase = round(self.positions['base'][1] - self.positions_last['base'][1], 8)

        # get dws and trades
        diffasset_dw, diffbase_dw = self.process_dws()
        diffasset_trad, diffbase_trad, apc = self.process_trades(p)

        # get unknown changes
        diffasset_expt = round(diffasset_dw + diffasset_trad, 8)
        diffbase_expt = round(diffbase_dw + diffbase_trad, 8)
        diffasset_unkn = diffasset - diffasset_expt
        diffbase_unkn = diffbase - diffbase_expt

        # process unknown changes
        if self.params['log_dws'] == "yes":
            if diffasset_unkn > 0: logger.info("{} {} has become available.".format(fix_dec(diffasset_unkn), self.asset))
            elif diffasset_unkn < 0: logger.info("{} {} has become unavailable.".format(fix_dec(-diffasset_unkn), self.asset))
            if diffbase_unkn > 0: logger.info("{} {} has become available.".format(fix_dec(diffbase_unkn), self.base))
            elif diffbase_unkn < 0: logger.info("{} {} has become unavailable.".format(fix_dec(-diffbase_unkn), self.base))

        # set position and average price
        if apc == 0: apc = p.price
        if p.positionValue > self.min_order:
            if s['position'] != "long":
                s['position'] = "long"; s['apc'] = apc
        elif s['position'] != "none":
            s['position'] = "none"; s['apc'] = apc

        return

    def get_performance(self, p):
        if self.ticks == 1:
            self.candle_start = dict(self.candles[-1])
            self.positions_start = {key: value[:] for key, value in self.positions.items()}
        r = self.performance
        s = self.signal
        pos_f = self.positions_f
        pos_t = self.positions_t
        c_start = self.candle_start
        p_start = self.positions_start
        p_start_size = p_start['base'][1] + c_start['close'] * p_start['asset'][1]

        pfake_size = pos_f['base'][1] + p.price * pos_f['asset'][1]
        ptrade_size = pos_t['base'][1] + p.price * pos_t['asset'][1]

        r['bh'] = (p.price - c_start['close']) / c_start['close']
        r['change'] = (p.size - p_start_size) / p_start_size
        r['bProfits'] = pfake_size - 1
        r['aProfits'] = (1 + r['bProfits']) / (1 + r['bh']) - 1
        r['cProfits'] = ptrade_size - 1

        W = int(r['W']); w = float(r['w'])
        L = int(r['L']); l = float(r['l'])
        if r['cProfits'] >= 0: W += 1; wSum = r['wSum'] + r['cProfits']; w = wSum / W
        if r['cProfits'] < 0: L += 1; lSum = r['lSum'] + r['cProfits']; l = lSum / L
        r['be'] = W * w + L * l

    def log_update(self, p):
        r = self.performance

        hr = "~~~~~"
        tpd = float()
        if self.days != 0: tpd = self.trades / self.days
        winrate = float()
        if r['W'] + r['L'] != 0: winrate = r['W'] / (r['W'] + r['L'])
        header_1 = "{} {} {} {}".format(3 * hr, self.bot_name, self.version, 9 * hr)[:50]
        header_2 = "{} {} {} {}m {}".format(3 * hr, self.exchange.title(), self.pair, self.interval, 9 * hr)[:50]
        trades = "{} trades ({} per day)".format(int(self.trades), round(tpd, 2))
        currency = "{} {}".format(fix_dec(p.base), self.base)
        price = "{} {}".format(fix_dec(p.price), self.base)
        assets = "{} {}".format(fix_dec(p.asset), self.asset)
        assetvalue = "{} {}".format(fix_dec(p.positionValue), self.base)
        accountvalue = "{} {}".format(fix_dec(p.size), self.base)
        boteff = "{}% {},".format(round(100 * r['be'], 2), self.base)
        boteff += " {}% {}".format(round(100 * ((1 + r['be']) / (1 + r['bh'])) - 100, 2), self.asset)
        botprof = "{}% {},".format(round(100 * r['bProfits'], 2), self.base)
        botprof += " {}% {}".format(round(100 * ((1 + r['bProfits']) / (1 + r['bh'])) - 100, 2), self.asset)

        logger.info(header_1)
        logger.info(header_2)
        logger.info("Days running: {} | Trades completed: {}".format(round(self.days, 2), trades))
        logger.info("Currency: {} | Current price: {}".format(currency, price))
        logger.info("Assets: {} | Value of assets: {}".format(assets, assetvalue))
        logger.info("Value of account: {}".format(accountvalue))
        logger.info("    Win rate: {}%".format(round(100 * winrate, 2)))
        logger.info("    Wins: {} | Average win: {}%".format(r['W'], round(100 * r['w'], 2)))
        logger.info("    Losses: {} | Average loss: {}%".format(r['L'], round(100 * r['l'], 2)))
        logger.info("    Current profits: {}%".format(round(100 * r['cProfits'], 2)))
        logger.info("    Bot efficiency: {}".format(boteff))
        logger.info("Bot profits: {}".format(botprof))
        logger.info("Buy and hold: {}%".format(round(100 * r['bh'], 2)))

    def init(self, p):
        # KuCoin Pybot 20/100 SXS
        self.bot_name = "KuCoin Pybot"
        self.version = "1.1"
        logger.info("Analyzing the market...")

        # vars
        ma_lens = [[20, 100]]
        storage["ma_lens"] = ma_lens
        logger.info("Ready to start trading...")

    def strat(self, p):
        """ strategy / trading algorithm
        - Use talib for indicators
        - Talib objects require numpy.array objects as input
        - s stands for signal, rinTarget stands for 'ratio invested target'
        - Set s['rinTarget'] between 0 and 1. 0 is 0%, 1 is 100% invested
        """
        # vars
        s = self.signal
        ma_lens = storage["ma_lens"]
        num_sigs = len(ma_lens)
        close_data = numpy.array([c['close'] for c in self.candles])

        # get sigs
        SMAs = dict()
        class SMA:
            def __init__(self, ma_len):
                self.ma_len = ma_len
                self.arr = talib.SMA(close_data, timeperiod = ma_len)
                self.price = self.arr[-1]
                SMAs[ma_len] = self.price

        class SXS:
            def __init__(self, ma_len_pair):
                self.pair = ma_len_pair
                self.trend = "bear"
                if ma_len_pair[0] not in SMAs:
                    mas = SMA(ma_len_pair[0])
                self.mas = SMAs[ma_len_pair[0]]
                if ma_len_pair[1] not in SMAs:
                    mal = SMA(ma_len_pair[1])
                self.mal = SMAs[ma_len_pair[1]]
                if self.mas > self.mal: self.trend = "bull"

        sigs = list()
        for i in range(num_sigs):
            sig = SXS(ma_lens[i])
            sigs.append(sig)
        num_sigs = len(sigs)

        # set rinTarget
        nBulls = 0; nBears = 0
        s['rinTarget'] = 0
        for i in range(num_sigs):
            if sigs[i].trend == "bear": nBears += 1
            if sigs[i].trend == "bull":
                nBulls += 1
                s['rinTarget'] += 1 / num_sigs

    def ping(self):
        # check if its time for a new tick
        if (time.time() - self.candles_raw[-1]["ts_end"]/1000) < 60: return
        n_new_candles_raw = self.topoff_candles_raw()
        n_new_candles = self.topoff_candles()

        # New tick?
        if n_new_candles > 0:
            # Preliminary setup
            set_log_file()
            self.close_orders()
            self.update_vars()
            self.get_params()

            self.positions_last = {key: value[:] for key, value in self.positions.items()}
            self.positions = self.get_positions()
            p = Portfolio(self.candles[-1], self.positions, self.params['funds'])
            self.min_order = max(self.min_order, round(0.05*p.funds, 8))
            self.process_dwts(p)
            self.get_performance(p)

            # Log output
            if self.params['logs_per_day'] == "0": self.next_log = self.days + 1
            if self.days >= self.next_log:
                self.log_update(p)
                self.next_log += 1 / float(self.params['logs_per_day'])

            # Trading strategy, buy/sell/other
            self.strat(p)
            self.bso(p)

api = [api_key, api_secret, api_passphrase]
client = Exchange(exchange, api)
ins = Instance(asset, base, interval_mins)
while True:
    ins.ping()
    time.sleep(0.5)
