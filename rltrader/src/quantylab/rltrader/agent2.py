import numpy as np
from quantylab.rltrader import utils


class Agent:
    # 에이전트 상태가 구성하는 값 개수
    # 주식 보유 비율, 손익률, 주당 매수 단가 대비 주가 등락률
    STATE_DIM = 3

    # 매매 수수료 및 세금
    TRADING_CHARGE = 0.00015  # 거래 수수료 0.015%
    # TRADING_CHARGE = 0.00011  # 거래 수수료 0.011%
    # TRADING_CHARGE = 0  # 거래 수수료 미적용
    TRADING_TAX = 0.002  # 거래세 0.2%
    # TRADING_TAX = 0  # 거래세 미적용

    # 행동
    ACTION_BUY = 0  # 매수
    ACTION_SELL = 1  # 매도
    ACTION_HOLD = 2  # 관망
    # 인공 신경망에서 확률을 구할 행동들
    ACTIONS = [ACTION_BUY, ACTION_SELL, ACTION_HOLD] # Action Space
    NUM_ACTIONS = len(ACTIONS)  # 인공 신경망에서 고려할 출력값의 개수

    # 교재
    def __init__(self, environment, min_trading_unit=1, max_trading_unit=10, delayed_reward_threshold=.05):

    # def __init__(self, environment, initial_balance, min_trading_price, max_trading_price):
        # 현재 주식 가격을 가져오기 위해 환경 참조
        self.environment = environment
        # self.initial_balance = initial_balance  # 초기 자본금

        # 최소 단일 매매 금액, 최대 단일 매매 금액
        # self.min_trading_price = min_trading_price
        # self.max_trading_price = max_trading_price
        
        # 교재
        self.min_trading_unit = min_trading_unit # 최소 단일 거래 단위
        self.max_trading_unit = max_trading_unit # 최대 단일 거래 단위
        self.delayed_reward_threshold = delayed_reward_threshold # 지연보상 임계치

        # Agent 클래스의 속성
        # self.balance = initial_balance  # 현재 현금 잔고
        self.num_stocks = 0  # 보유 주식 수
        # 포트폴리오 가치: balance + num_stocks * {현재 주식 가격}
        self.portfolio_value = 0
        self.num_buy = 0  # 매수 횟수
        self.num_sell = 0  # 매도 횟수
        self.num_hold = 0  # 관망 횟수

        # Agent 클래스의 상태
        self.ratio_hold = 0  # 주식 보유 비율

        self.profitloss = 0  # 손익률 (현재 손익)

        self.avg_buy_price = 0  # 주당 매수 단가

        # 교재 코드
        self.initial_balance = 0
        self.balance = 0
        self.base_portfolio_value = 0 # 직전 학습 시점의 PV
        self.immediate_reward = 0  # 즉시 보상
        self.base_profitloss = 0 # 직전 지연 보상 이후 손익
        self.exploration_base = 0 # 탐험 행동 결정 기준
        self.ratio_portfolio_value = 0 # 포트폴리오 가치 비율

    def reset(self):
        # epoch마다 agent를 초기화
        self.balance = self.initial_balance
        self.num_stocks = 0
        self.portfolio_value = self.initial_balance
        self.base_portfolio_value = self.initial_balance
        self.num_buy = 0
        self.num_sell = 0
        self.num_hold = 0
        self.immediate_reward = 0
        self.ratio_hold = 0 # 주식 보유 비율 = 보유 주식 수 / (PV/현재주가) : 0이면 주식을 보유하지 않은 상태, 1이면 최대로 주식을 보유한 상태
                            # 1에 가까울수록 > 정책 신경망에서 매도의 관점에서 투자에 임하도록 유도
                            # 0에 가까울수록 > 정책 신경망에서 매수의 관점으로 투자에 임하도록 유도
        self.profitloss = 0
        self.avg_buy_price = 0
        self.ratio_portfolio_value = 0 # 포트폴리오 가치 비율 = PV / base_PV
                            # base_PV는 직전 PV이고 PV는 현재 PV이다. 0에 가까울수록 손실이 큰 것이고 1보다 크면 수익 발생을 의미한다.
                            # ratio_PV가 목표 수익률에 가까우면 정책 신경망에서 매도의 관점으로 투자한다.

        # 정책 신경망 입력값 : ratio_hold, ratio_portfolio_value

    def reset_exploration(self):
        # 초기에 매수 탐험을 더 선호하도록 50%의 매수 탐험 확률을 부여
        # exploration_base가 0에 가까우면 매도 탐험을 더 많이 함, 1에 가까우면 매수 탐험을 더 많이 함.
        # >> 이를 통해 [bus,sell,buy,sell,...]과 같은 비효과적 탐험을 줄임. 지속적인 상승세라면 [buy,buy,buy,sell]이 더 효과적
        self.exploration_base = 0.5 + np.random.rand() / 2
    
    def set_balance(self, balance):
        self.initial_balance = balance

    def get_states(self):
        self.ratio_hold = self.num_stocks / int(
            self.portfolio_value / self.environment.get_price()
        )
        self.ratio_portfolio_value = (
            self.portfolio_value / self.base_portfolio_value
        )
        return self.ratio_hold, self.ratio_portfolio_value

        # self.ratio_hold = self.num_stocks * self.environment.get_price() \
        #     / self.portfolio_value
        # return (
        #     self.ratio_hold,
        #     self.profitloss,
        #     (self.environment.get_price() / self.avg_buy_price) - 1 \
        #         if self.avg_buy_price > 0 else 0
        # )

    def decide_action(self, pred_value, pred_policy, epsilon):
        # epsilon-greedy policy : epsilon의 확률로 무작위 행동을 수행, 그렇지 않은 경우 정책 신경망을 통해 행동을 결정
        confidence = 0.

        # 정책 신경망 모델에 따라 policy 또는 value가 전달되므로 값이 있는 것으로 pred 설정
        pred = pred_policy
        if pred is None:
            pred = pred_value

        # 만약 정책이 없거나, 정책값이 모두 같은 경우 무작위 탐험을 진행해야 하므로 epsilon을 1로 설정
        # 그렇지 않은 경우 stochastic policy대로 행동 결정
        if pred is None:
            # 예측 값이 없을 경우 탐험
            epsilon = 1
        else:
            # 값이 모두 같은 경우 탐험
            maxpred = np.max(pred)
            if (pred == maxpred).all():
                epsilon = 1

            # if pred_policy is not None:
            #     if np.max(pred_policy) - np.min(pred_policy) < 0.05:
            #         epsilon = 1

        # 탐험 결정
        if np.random.rand() < epsilon: # 무작위로 생성한 값이 epsilon보다 작은 경우 무작위 탐험 수행
            exploration = True
            # action = np.random.randint(self.NUM_ACTIONS)
            if np.random.rand() < self.exploration_base: # exploration_base를 조절해서 무작위 행동의 기조를 매수/매도로 조절
                action = self.ACTION_BUY
            else:
                action = np.random.randint(self.NUM_ACTIONS - 1) + 1
        else:
            exploration = False
            action = np.argmax(pred)

        confidence = .5
        if pred_policy is not None:
            confidence = pred[action]
        elif pred_value is not None:
            confidence = utils.sigmoid(pred[action])

        return action, confidence, exploration

    def validate_action(self, action):
        # 매수/매도를 결정했을 때 그 행동을 실제로 수행할 수 있는지 확인
        if action == Agent.ACTION_BUY:
            # 최소 매수 단위 이상의 자금이 있는지 확인
            # if self.balance < self.environment.get_price() * (1 + self.TRADING_CHARGE):
            #     return False
            if self.balance < self.environment.get_price() * (1 + self.TRADING_CHARGE) * self.min_trading_unit:
                return False

        elif action == Agent.ACTION_SELL:
            # 주식 잔고가 있는지 확인
            if self.num_stocks <= 0:
                return False
        return True

    def decide_trading_unit(self, confidence):
        # 매수/매도 단위를 결정
        # 정책 신경망의 confidence를 전달받아, 신뢰도가 높으면 더 confidence*(최대-최소 거래 단위)만큼 주식을 매수/매도한다.
        # confidence는 정책 신경망을 통해 결정된 행동에 대한 softmax값이다.
        if np.isnan(confidence):
            return self.min_trading_price
        added_trading = max(min(
            # int(confidence * (self.max_trading_price - self.min_trading_price)),
            # self.max_trading_price-self.min_trading_price), 0)
            int(confidence * (self.max_trading_unit - self.min_trading_unit)),
            self.max_trading_unit-self.min_trading_unit
            ), 0
        )
        # trading_price = self.min_trading_price + added_trading_price

        # return max(int(trading_price / self.environment.get_price()), 1)
        return self.min_trading_unit + added_trading

    def act(self, action, confidence):
        # 결정된 행동을 수행한다.
        if not self.validate_action(action):
            action = Agent.ACTION_HOLD

        # 환경에서 현재 가격 얻기
        curr_price = self.environment.get_price()

        # 즉시 보상 초기화
        self.immediate_reward = 0

        # 매수
        if action == Agent.ACTION_BUY:
            # 매수할 단위를 판단
            trading_unit = self.decide_trading_unit(confidence)
            balance = ( # trading_unit만큼의 가격을 보유 현금에서 뺌.
                self.balance - curr_price *
                (1 + self.TRADING_CHARGE) * trading_unit
            )
            # 보유 현금이 모자랄 경우(위의 코드로 인해 balance<0이 됨) 보유 현금으로 가능한 만큼 최대한 매수
            if balance < 0:
                # trading_unit = min(
                #     int(self.balance / (curr_price * (1 + self.TRADING_CHARGE))),
                #     int(self.max_trading_price / curr_price)
                # )
                trading_unit = max( # trading_unit을 min_trading_unit이상~구매할 수 있는 범위 내로 재설정함.
                    min(
                        int(self.balance / (
                            curr_price * (1+self.TRADING_CHARGE)
                        )),
                        self.max_trading_unit
                    ),
                    self.min_trading_unit
                )

            # 수수료를 적용하여 총 매수 금액 산정
            invest_amount = curr_price * (1 + self.TRADING_CHARGE) * trading_unit # 총 매수 금액
            if invest_amount > 0:
                self.avg_buy_price = \
                    (self.avg_buy_price * self.num_stocks + curr_price * trading_unit) \
                        / (self.num_stocks + trading_unit)  # 주당 매수 단가 갱신
                self.balance -= invest_amount  # 보유 현금을 갱신
                self.num_stocks += trading_unit  # 보유 주식 수를 갱신
                self.num_buy += 1  # 매수 횟수 증가

        # 매도
        elif action == Agent.ACTION_SELL:
            # 매도할 단위를 판단
            trading_unit = self.decide_trading_unit(confidence)
            # 보유 주식이 모자랄 경우 가능한 만큼 최대한 매도
            trading_unit = min(trading_unit, self.num_stocks)
            # 매도
            invest_amount = curr_price * (
                1 - (self.TRADING_TAX + self.TRADING_CHARGE)) * trading_unit
            if invest_amount > 0:
                # 주당 매수 단가 갱신
                self.avg_buy_price = \
                    (self.avg_buy_price * self.num_stocks - curr_price * trading_unit) \
                        / (self.num_stocks - trading_unit) \
                            if self.num_stocks > trading_unit else 0
                self.num_stocks -= trading_unit  # 보유 주식 수를 갱신
                self.balance += invest_amount  # 보유 현금을 갱신
                self.num_sell += 1  # 매도 횟수 증가

        # 관망
        elif action == Agent.ACTION_HOLD:
            self.num_hold += 1  # 관망 횟수 증가

        # 포트폴리오 가치 갱신
        self.portfolio_value = self.balance + curr_price * self.num_stocks
        self.profitloss = self.portfolio_value / self.initial_balance - 1

        self.immediate_reward = self.profitloss
        delayed_reward = 0

        self.base_profitloss = (
            (self.portfolio_value - self.base_portfolio_value) / self.base_portfolio_value
        )
        if self.base_profitloss > self.delayed_reward_threshold or self.base_profitloss < -self.delayed_reward_threshold:
            # 목표 수익률 달성하여 기준 포트폴리오 가치 갱신
            # 또는 손실 기준치를 초과하여 기준 포트폴리오 가치 갱신
            self.base_portfolio_value = self.portfolio_value
            delayed_reward = self.immediate_reward
        else:
            delayed_reward = 0
        """
        기준 PV(과거 PV) 보다 현재 PV가 상승 또는 하락했을 때 기준 PV를 현재 PV로 업데이트해준다.
        기준 PV를 업데이트한 경우, delayed_reward를 immediate_reward로 업데이트하여 
            self.base_profitloss > self.delayed_reward_threshold(=이익) 경우는 positive로 학습하고,
            self.base_profitloss < -self.delayed_reward_threshold(=손실) 경우는 negative로 학습하도록 한다.
        """

        # return self.profitloss
        return self.immediate_reward, delayed_reward