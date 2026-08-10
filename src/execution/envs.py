import gym
from gym import spaces
import numpy as np

class OptionsTradingEnv(gym.Env):
    def __init__(self, data, delta, gamma, theta, vega, rho,initial_cash=100000,):
        super(OptionsTradingEnv, self).__init__()
        self.delta = delta     # option delta
        self.gamma = gamma    # gamma
        self.theta = theta     # per day decay
        self.vega = vega      # per 1% IV change
        self.rho = rho        # per 1% rate change

        self.dS = 0           # Nifty moves up by 50 points
        self.d_sigma = 0    # IV increases by 1%
        self.d_t = 1/365       # one day passes
        self.d_r = 0           # no rate change

        self.data = data.reset_index()
        self.initial_cash = initial_cash
        self.current_step = 0
        self.done = False

        # Action space: 0 = Hold, 1 = Buy Call, 2 = Sell Call
        self.action_space = spaces.Discrete(3)

        # Observation space: [option_price, position, cash]
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(3,), dtype=np.float32)

        self.reset()

    def reset(self):
        self.current_step = 0
        self.position = 0  # number of option contracts held
        self.cash = self.initial_cash
        self.done = False
        return self._get_observation()

    def _get_observation(self):
        price = self.data.loc[self.current_step, 'price']
        return np.array([price, self.position, self.cash], dtype=np.float32)

    def step(self, action):
        price = self.data.loc[self.current_step, 'price']
        reward = 0

        if action == 1:  # Buy Call
            if self.cash >= price * 100:
                self.position += 1
                self.cash -= price * 100

        elif action == 2:  # Sell Call
            if self.position > 0:
                self.position -= 1
                self.cash += price * 100

        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            self.done = True

        next_price = self.data.loc[self.current_step, 'price']
        portfolio_value = self.cash + self.position * next_price * 100
        reward = portfolio_value - (self.cash + self.position * price * 100)

        obs = self._get_observation()
        info = {
            "step": self.current_step,
            "portfolio_value": portfolio_value,
            "option_price": next_price
        }

        return obs, reward, self.done, info

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Position: {self.position}, Cash: {self.cash:.2f}")

    def option_value_change(self,dS):
        dV = self.delta * dS + 0.5 * self.gamma * dS**2 + self.vega * self.d_sigma + self.theta * self.d_t + self.rho * self.d_r
        return dV
    


from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env


env = OptionsTradingEnv()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)

# 
if __name__=="__main__":
    delta = 0.55      # option delta
    gamma = 0.03      # gamma
    theta = -0.12     # per day decay
    vega = 0.10       # per 1% IV change
    rho = 0.02        # per 1% rate change

    dS = 50           # Nifty moves up by 50 points
    d_sigma = 0.01    # IV increases by 1%
    d_t = 1/365       # one day passes
    d_r = 0           # no rate change

    data = None
        
    env = OptionsTradingEnv(data,delta, gamma, theta, vega, rho)
    dV = env.option_value_change(dS)


    
    import yfinance as yf

    # Download a sample option's price history
    ticker = yf.Ticker("AAPL")
    expiry = ticker.options[-1]
    chain = ticker.option_chain(expiry)
    option_symbol = chain.calls.iloc[0]['contractSymbol']

    option = yf.Ticker(option_symbol)
    data = option.history(period="6mo")[['Close']]
    data.columns = ['price']
    data.dropna(inplace=True)


    obs = env.reset()

    for _ in range(50):
        action = env.action_space.sample()  # random action
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            break
    print(f"Approximate change in option value: ₹{dV:.2f}")

