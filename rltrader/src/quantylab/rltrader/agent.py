import numpy as np
from quantylab.rltrader import utils


class Agent:
    ###📢 요일을 나타내는 flag 설정
    # DAYS = [0,1,2,3,4] # 월~금

    # 에이전트 상태가 구성하는 값 개수
    # 주식 보유 비율, 손익률, 주당 매수 단가 대비 주가 등락률
    STATE_DIM = 3

    # 매매 수수료 및 세금
    # TRADING_CHARGE = 0.00015  # 거래 수수료 0.015%
    # TRADING_CHARGE = 0.00011  # 거래 수수료 0.011%
    TRADING_CHARGE = 0  # 거래 수수료 미적용
    TRADING_TAX = 0.002  # 거래세 0.2%
    # TRADING_TAX = 0  # 거래세 미적용
    HANTU_TAX = 0.18 # 한국투자증권 코스피 매도 세금

    # 행동
    ACTION_BUY = 0  # 매수
    ACTION_SELL = 1  # 매도
    ACTION_HOLD = 2  # 관망
    # 인공 신경망에서 확률을 구할 행동들
    ACTIONS = [ACTION_BUY, ACTION_SELL, ACTION_HOLD] # Action Space
    NUM_ACTIONS = len(ACTIONS)  # 인공 신경망에서 고려할 출력값의 개수

    # 교재
    # def __init__(self, environment, min_trading_unit=1, max_trading_unit=2, delayed_reward_threshold=.05):

    def __init__(self, environment, initial_balance, min_trading_price, max_trading_price, stock_code):
        # 현재 주식 가격을 가져오기 위해 환경 참조
        self.environment = environment
        self.initial_balance = initial_balance  # 초기 자본금

        # 최소 단일 매매 금액, 최대 단일 매매 금액
        self.min_trading_price = min_trading_price
        self.max_trading_price = max_trading_price
        
        # 교재
        # self.min_trading_unit = min_trading_unit # 최소 단일 거래 단위
        # self.max_trading_unit = max_trading_unit # 최대 단일 거래 단위
        # self.delayed_reward_threshold = delayed_reward_threshold # 지연보상 임계치

        # Agent 클래스의 속성
        self.balance = initial_balance  # 현재 현금 잔고
        self.num_stocks = 0  # 보유 주식 수
        # 포트폴리오 가치: balance + num_stocks * {현재 주식 가격}
        self.portfolio_value = 0
        self.num_buy = 0  # 매수 횟수
        self.num_sell = 0  # 매도 횟수
        self.num_hold = 0  # 관망 횟수

        # Agent 클래스의 상태
        self.ratio_hold = 0  # PV 대비 주식 보유 비율

        self.profitloss = 0  # 손익률 (현재 손익)

        self.avg_buy_price = 0  # 주당 매수 단가

        ### 📢실시간 트레이딩을 위한 변수
        self.stock_code = stock_code

    def reset(self):
        self.balance = self.initial_balance
        self.num_stocks = 0
        self.portfolio_value = self.initial_balance
        self.num_buy = 0
        self.num_sell = 0
        self.num_hold = 0
        self.ratio_hold = 0
        self.profitloss = 0
        self.avg_buy_price = 0

    ### *⚠️코드 수정 시작⚠️*  ###
    def preset(self, INITIAL_BALANCE, BALANCE, NUM_STOCKS, PORTFOLIO_VALUE, NUM_BUY, NUM_SELL, 
         NUM_HOLD, RATIO_HOLD, PROFITLOSS, AVG_BUY_PRICE): 
        self.initial_balance = INITIAL_BALANCE
        self.balance = BALANCE
        self.num_stocks = NUM_STOCKS
        self.portfolio_value = PORTFOLIO_VALUE
        self.num_buy = NUM_BUY
        self.num_sell = NUM_SELL
        self.num_hold = NUM_HOLD 
        self.ratio_hold = RATIO_HOLD 
        self.profitloss = PROFITLOSS
        self.avg_buy_price = AVG_BUY_PRICE 
    ### *⚠️코드 수정 끝⚠️*  ###

    def set_balance(self, balance):
        self.initial_balance = balance

    def get_states(self):
        self.ratio_hold = self.num_stocks * self.environment.get_price() \
            / self.portfolio_value
        return (
            self.ratio_hold,
            self.profitloss,
            (self.environment.get_price() / self.avg_buy_price) - 1 \
                if self.avg_buy_price > 0 else 0
        )

    def decide_action(self, pred_value, pred_policy, epsilon):
        confidence = 0.

        pred = pred_policy
        if pred is None:
            pred = pred_value

        if pred is None:
            # pred_value, pred_policy 둘 다 없을 경우 탐험
            epsilon = 1
        else:
            # 값이 모두 같은 경우 탐험
            maxpred = np.max(pred)
            if (pred == maxpred).all(): # pred배열의 모든 값이 maxpred와 같다면
                epsilon = 1

            # if pred_policy is not None:
            if np.max(pred_policy) - np.min(pred_policy) < 0.05: # pred_policy의 최대/최솟값의 차이가 5%p보다 작아도 탐험을 한다.
                epsilon = 1

        # 탐험 결정
        if np.random.rand() < epsilon:
            exploration = True
            action = np.random.randint(self.NUM_ACTIONS)
        else:
            exploration = False
            action = np.argmax(pred)

        confidence = .5 # confidence는 정책 신경망을 통해 결정된 행동에 대한 softmax값이다.
        if pred_policy is not None:
            confidence = pred[action]
        elif pred_value is not None:
            confidence = utils.sigmoid(pred[action])

        return action, confidence, exploration

    def validate_action(self, action):
        # 매수/매도를 결정했을 때 그 행동을 실제로 수행할 수 있는지 확인
        if action == Agent.ACTION_BUY:
            # 적어도 1주를 살 수 있는지 확인
            if self.balance < self.environment.get_price() * (1 + self.TRADING_CHARGE):
                return False
        elif action == Agent.ACTION_SELL:
            # 주식 잔고가 있는지 확인
            if self.num_stocks <= 0:
                return False
        return True
    
    def pvalidate_action(self, action):
        charge, add_price = utils.get_charge(self.environment.get_price(), 1)
        # 매수/매도를 결정했을 때 그 행동을 실제로 수행할 수 있는지 확인
        if action == Agent.ACTION_BUY:
            # 적어도 1주를 살 수 있는지 확인
            if self.balance < (self.environment.get_price()+add_price) * (1 + charge):
                return False
        elif action == Agent.ACTION_SELL:
            # 주식 잔고가 있는지 확인
            if self.num_stocks <= 0:
                return False
        return True

    def decide_trading_unit(self, confidence):
        # 매수/매도 단위를 결정
        # 정책 신경망의 confidence를 전달받아, action에 대한 자신감 높으면 더 
        # int(confidence*(최대-최소 거래 금액)/현재주가)만큼 주식을 매수/매도한다.
        # confidence는 정책 신경망을 통해 결정된 행동에 대한 softmax값이다.
        if np.isnan(confidence):
            # return self.min_trading_price ### ??? 아래처럼 코드 써야하는 거 아닌가?
            return max(int(self.min_trading_price / self.environment.get_price()), 1)
        
        added_trading_price = max(
            min(
                int(confidence * (self.max_trading_price - self.min_trading_price)),
                self.max_trading_price-self.min_trading_price
                ), 
            0)
        trading_price = self.min_trading_price + added_trading_price
        return max(int(trading_price / self.environment.get_price()), 1)

    def act(self, action, confidence):
        if not self.validate_action(action):
            action = Agent.ACTION_HOLD

        # 환경에서 현재 가격 얻기
        curr_price = self.environment.get_price()

        # 매수
        if action == Agent.ACTION_BUY:
            # 매수할 단위를 판단
            trading_unit = self.decide_trading_unit(confidence)
            balance = (
                self.balance - curr_price *
                (1 + self.TRADING_CHARGE) * trading_unit
            )
            # 보유 현금이 모자랄 경우 보유 현금으로 가능한 만큼 최대한 매수
            if balance < 0:
                trading_unit = min(
                    int(self.balance / (curr_price * (1 + self.TRADING_CHARGE))),
                    int(self.max_trading_price / curr_price)
                )

            # 수수료를 적용하여 총 매수 금액 산정
            invest_amount = curr_price * (1 + self.TRADING_CHARGE) * trading_unit
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
        return self.profitloss
