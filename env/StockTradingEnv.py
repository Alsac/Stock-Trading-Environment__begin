import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np

MAX_ACCOUNT_BALANCE = 2147483647
MAX_NUM_SHARES = 2147483647
MAX_SHARE_PRICE = 5000
MAX_OPEN_POSITIONS = 5
MAX_STEPS = 20000

INITIAL_ACCOUNT_BALANCE = 10000


class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df):
        super(StockTradingEnv, self).__init__()

        self.df = df
        self.reward_range = (0, MAX_ACCOUNT_BALANCE)

        # Actions of the format Buy x%, Sell x%, Hold, etc.
        self.action_space = spaces.Box(
            low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float16)

        # Prices contains the OHCL values for the last five prices
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(6, 6), dtype=np.float16)

    def _next_observation(self):
        # Get the stock data points for the last 5 days and scale to between 0-1
        frame = np.array([
            self.df.loc[self.current_step: self.current_step +
                        5, 'Open'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                        5, 'High'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                        5, 'Low'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                        5, 'Close'].values / MAX_SHARE_PRICE,
            self.df.loc[self.current_step: self.current_step +
                        5, 'Volume'].values / MAX_NUM_SHARES,
        ])

        # Append additional data and scale each value to between 0-1
        obs = np.append(frame, [[
            self.balance / MAX_ACCOUNT_BALANCE,
            self.max_net_worth / MAX_ACCOUNT_BALANCE,
            self.shares_held / MAX_NUM_SHARES,
            self.cost_basis / MAX_SHARE_PRICE,
            self.total_shares_sold / MAX_NUM_SHARES,
            self.total_sales_value / (MAX_NUM_SHARES * MAX_SHARE_PRICE),
        ]], axis=0)

        return obs

    def _take_action(self, action): #action format: [行动类型， 数量]
        # Set the current price to a random price within the time step
        current_price = random.uniform(
            self.df.loc[self.current_step, "Open"], self.df.loc[self.current_step, "Close"]) #开盘价到收盘价间的随机值

        action_type = action[0] #行动类型：买/卖/持有
        amount = action[1] #数量

        if action_type < 1: #买入操作的相关账户计算问题
            # Buy amount % of balance in shares
            total_possible = int(self.balance / current_price) #现金余额可买总数量
            shares_bought = int(total_possible * amount) #购买股票数量=总数量*买入比例
            prev_cost = self.cost_basis * self.shares_held #前持仓金额=平均每股持仓价格 * 股票持有数量
            additional_cost = shares_bought * current_price #新增持仓成本（金额）=购买数量*价格

            self.balance -= additional_cost #现金余额 = 现金余额-新增持仓金额
            self.cost_basis = (
                prev_cost + additional_cost) / (self.shares_held + shares_bought) #平均每股持仓价格 =（前持仓成本（金额）+新增持仓成本（金额）/（股票持有总数量+新购买股票数量）
            self.shares_held += shares_bought #股票持有数量 = 股票持有数量 + 新购买股票数量

        elif action_type < 2: #卖出操作的相关账户计算问题
            # Sell amount % of shares held
            shares_sold = int(self.shares_held * amount) #新卖出股票数量 = 股票持有总数*卖出比例   #卖出比例的数值是正值
            self.balance += shares_sold * current_price #账户余额 = 账户余额 + 新卖出股票数量*当前价格  #账户余额增加
            self.shares_held -= shares_sold #股票持有总数 = 股票持有总数 - 新卖出股票数量
            self.total_shares_sold += shares_sold #股票卖出总量 = 股票卖出总量 +新卖出股票量
            self.total_sales_value += shares_sold * current_price #股票卖出总金额 = 股票卖出总量*当前价格

        self.net_worth = self.balance + self.shares_held * current_price #账户净值 = 现金余额 + 股票持有总数 * 当前价格

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth #历史最大账户净值

        if self.shares_held == 0: #现金持有总量为0，则平均每股持仓价格为0
            self.cost_basis = 0

    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)

        self.current_step += 1

        if self.current_step > len(self.df.loc[:, 'Open'].values) - 6:
            self.current_step = 0

        delay_modifier = (self.current_step / MAX_STEPS)

        reward = self.balance * delay_modifier
        done = self.net_worth <= 0

        obs = self._next_observation()

        return obs, reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0

        # Set the current step to a random point within the data frame
        self.current_step = random.randint(
            0, len(self.df.loc[:, 'Open'].values) - 6)

        return self._next_observation()

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE

        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(
            f'Shares held: {self.shares_held} (Total sold: {self.total_shares_sold})')
        print(
            f'Avg cost for held shares: {self.cost_basis} (Total sales value: {self.total_sales_value})')
        print(
            f'Net worth: {self.net_worth} (Max net worth: {self.max_net_worth})')
        print(f'Profit: {profit}')
