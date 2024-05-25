class ReinforcementLearner: #이 클래스는 DQN, PolicyGradient, ActorCritic, A2CLearner 클래스가 상속하는 상위 클래스임. -> 환경, 에이전츠, 신경망 인스턴스들과 학습 데이터를 속성으로 가짐
    __metaclass__ = abc.ABCMeta
    lock = threading.Lock()

    def __init__(self, rl_method='rl', stock_code=None,  #rl_method는 어떤 강화학습 쓰냐에 따라 달라짐(DQN -> dqn, ac, a2c, a3c)
                ###📢chart_data, training_data 고려
                chart_data=None,
                training_data=None,
                #chart_data는 주식 일봉 차트 데이터, training_data는 학습을 위한 전처리된 학습 데이터
                min_trading_price=100000, max_trading_price=10000000, 
                net='dnn', num_steps=1, lr=0.0005, # net인자는 dnn, lstm, cnn 등의 값이 될 수 있으며 이 값에 따라 사용할 신경망 클래스 달라짐
                discount_factor=0.9, num_epoches=1000,
                ### 📢balance 고려 
                balance=100000000,
                start_epsilon=1, #balance.. RLTrader에서 신용거래와 같이 보유 현금 넘어서는 투자는 고려하지 않음. 보유 현금이 부족하면 정책 신경망 결과 매수가 좋아도 관망함.
                #start_epsilon은 초기탐험 비율을 말하는데, 전혀 학습되지 않은 초기에는 탐험비율 크게 해서 무작위 투자 하게 해야함. 
                value_network=None, policy_network=None,
                output_path='', reuse_models=True):
        # 인자 확인
        assert min_trading_price > 0
        assert max_trading_price > 0
        assert max_trading_price >= min_trading_price
        assert num_steps > 0
        assert lr > 0
        # 강화학습 설정
        self.rl_method = rl_method
        self.discount_factor = discount_factor
        self.num_epoches = num_epoches
        self.start_epsilon = start_epsilon
        # 환경 설정
        self.stock_code = stock_code
        self.chart_data = chart_data
        self.environment = Environment(chart_data) #환경은 차트데이터를 순서대로 읽으면서 주가, 거래량 등의 데이터를 제공함. 
        # 에이전트 설정
        self.agent = Agent(self.environment, balance, min_trading_price, max_trading_price) #강화학습 환경을 인자로 Agent 클래스의 인스턴스를 생성함.
       
        # 학습 데이터
        self.training_data = training_data
        self.sample = None
        self.training_data_idx = -1
        # 벡터 크기 = 학습 데이터 벡터 크기 + 에이전트 상태 크기 (47개의 특징 + 에이전트의 상태 3개(매수,매도,관망)를 더해서 50개!)
        self.num_features = self.agent.STATE_DIM
        if self.training_data is not None:
            self.num_features += self.training_data.shape[1]
        # 신경망 설정
        self.net = net
        self.num_steps = num_steps
        self.lr = lr
        self.value_network = value_network
        self.policy_network = policy_network
        self.reuse_models = reuse_models
        # 가시화 모듈
        self.visualizer = Visualizer()
        
        # 메모리 (강화학습 과정에서 발생하는 각종 데이터 쌓아두기 위해)
        self.memory_sample = []
        self.memory_action = []
        self.memory_reward = []
        self.memory_value = []
        self.memory_policy = []
        self.memory_pv = []
        self.memory_num_stocks = []
        self.memory_exp_idx = [] # 탐험 위치
        # 이렇게 저장한 샘플, 보상 등의 데이터로 학습을 진행함
        # 에포크 관련 정보
        self.loss = 0. # 발생한 손실
        self.itr_cnt = 0 # 수익 발생 횟수
        self.exploration_cnt = 0
        self.batch_size = 0
        # 로그 등 출력 경로
        self.output_path = output_path
        
    # 가치 신경망 생성 함수 -> net에 지정된 신경망 종류에 맞게 가치 신경망 생성함. -> 가치 신경망은 손익률을 회귀분석하는 모델 
    def init_value_network(self, shared_network=None, activation='linear', loss='mse'): #회귀분석이라 activation은 선형함수로, 손실함수는 MSE로..
        if self.net == 'dnn':
            self.value_network = DNN(
                input_dim=self.num_features, 
                output_dim=self.agent.NUM_ACTIONS, 
                lr=self.lr, shared_network=shared_network,
                activation=activation, loss=loss)
        elif self.net == 'lstm':
            self.value_network = LSTMNetwork(
                input_dim=self.num_features, 
                output_dim=self.agent.NUM_ACTIONS, 
                lr=self.lr, num_steps=self.num_steps, 
                shared_network=shared_network,
                activation=activation, loss=loss)
        elif self.net == 'cnn':
            self.value_network = CNN(
                input_dim=self.num_features, 
                output_dim=self.agent.NUM_ACTIONS, 
                lr=self.lr, num_steps=self.num_steps, 
                shared_network=shared_network,
                activation=activation, loss=loss)
        if self.reuse_models and os.path.exists(self.value_network_path):
            self.value_network.load_model(model_path=self.value_network_path)

    # 정책 신경망 생성 함수
    def init_policy_network(self, shared_network=None, activation='sigmoid', #얜 activation으로 sigmoid 씀! -> 샘플에 대해 PV를 높이기 위해 취하기 좋은 행동에 대한 분류 모델.
                            loss='binary_crossentropy'):
        if self.net == 'dnn':
            self.policy_network = DNN(
                input_dim=self.num_features, 
                output_dim=self.agent.NUM_ACTIONS, 
                lr=self.lr, shared_network=shared_network,
                activation=activation, loss=loss)
        elif self.net == 'lstm':
            self.policy_network = LSTMNetwork(
                input_dim=self.num_features, 
                output_dim=self.agent.NUM_ACTIONS, 
                lr=self.lr, num_steps=self.num_steps, 
                shared_network=shared_network,
                activation=activation, loss=loss)
        elif self.net == 'cnn':
            self.policy_network = CNN(
                input_dim=self.num_features, 
                output_dim=self.agent.NUM_ACTIONS, 
                lr=self.lr, num_steps=self.num_steps, 
                shared_network=shared_network,
                activation=activation, loss=loss)
        if self.reuse_models and os.path.exists(self.policy_network_path):
            self.policy_network.load_model(model_path=self.policy_network_path)

            
    # 에포크 초기화 함수 -> 에포크마다 새로 데이터가 쌓이는 변수들을 초기화하는 reset() 함수.
    def reset(self):
        self.sample = None #읽어온 데이터는 sample에 저장되는데 초기화 단계에는 읽어온 학습데이터가 없으므로 None
        self.training_data_idx = -1 # 학습 데이터를 다시 처음부터 읽기 위해 이걸 -1로 재설정함. -> 학습 데이터 읽어가며 1씩 증가.
        # 환경 초기화
        self.environment.reset()
        # 에이전트 초기화
        self.agent.reset()
        # 가시화 초기화
        self.visualizer.clear([0, len(self.chart_data)])
        # 메모리 초기화
        self.memory_sample = [] #읽어온 데이터는 sample에 저장됨. 
        self.memory_action = []
        self.memory_reward = []
        self.memory_value = []
        self.memory_policy = []
        self.memory_pv = []
        self.memory_num_stocks = []
        self.memory_exp_idx = []
        # 에포크 관련 정보 초기화
        self.loss = 0.
        self.itr_cnt = 0 # 수행한 에포크 수 저장.
        self.exploration_cnt = 0 # 무작위 투자를 수행한 횟수 저장 (epsilon 0.1이고 100번의 투자 결정 있으면 10번 무작위 투자함.)
        self.batch_size = 0

    def build_sample(self): #학습데이터를 구성하는 샘플 하나 생성
        self.environment.observe() # 환경 객체의 observe() 함수 호출해서 차트 데이터의 현재 인덱스에서 다음 인덱스 데이터를 읽게함. 
        if len(self.training_data) > self.training_data_idx + 1: # 그리고 학습 데이터의 다음 인덱스가 존재하는지 확인 
            self.training_data_idx += 1 # -> 학습 데이터에 다음 인덱스 데이터가 존재하면 변수 1만큼 증가시키고, 
            self.sample = self.training_data.iloc[self.training_data_idx].tolist() # training data 배열에서 idx 인덱스 데이터 받아와서 sample에 저장함. (v3는 47개로 구성. )
            self.sample.extend(self.agent.get_states())  # sample에 에이전트 상태 추가해 sample의 50개 값으로 구성.
            return self.sample
        return None

    @abc.abstractmethod # 추상 메서드.. 하위 클래스들은 반드시 이 함수 구현해야함.ReinforcementLearner 클래스 상속하고도 추상 메서드 구현 안하면 NotImplemented 예외가 발생함. 
    def get_batch(self): #신경망을 학습하기 위해 배치 학습 데이터를 생성
        pass

    # 신경망을 학습하는 fit() 함수 보여줌. 
    def fit(self):
        # 배치 학습 데이터 생성
        x, y_value, y_policy = self.get_batch()
        # 손실 초기화
        self.loss = None
        if len(x) > 0:
            loss = 0
            if y_value is not None:
                # 가치 신경망 갱신
                loss += self.value_network.train_on_batch(x, y_value) # 가치 신경망 학습하기 위해. DQNLearner, ActorCriticLearner, A2CLearner에서 학습.
            if y_policy is not None:
                # 정책 신경망 갱신
                loss += self.policy_network.train_on_batch(x, y_policy) # 정책 신경망 학습하기 위해. PolicyGradientLearner, ActorCriticLearner, A2cLearner에서 학습함. 
            self.loss = loss # 학습 후 발생하는 손실을 인스턴스 속성으로 저장함. 가치 신경망과 정책 신경망 학습하는 경우 두 학습 손실을 합산해 반환함. 

    # 하나의 에포크가 완료되어 에포크 관련 정보 가시화하는 부분 
    def visualize(self, epoch_str, num_epoches, epsilon):
        self.memory_action = [Agent.ACTION_HOLD] * (self.num_steps - 1) + self.memory_action
        self.memory_num_stocks = [0] * (self.num_steps - 1) + self.memory_num_stocks
        if self.value_network is not None: #LSTM과 CNN 신경망 사용하면 에이전트 행동, 보유 주식 수 등등 환경의 일봉수보다 (num_steps - 1) 만큼 부족하므로 의미 없는값을 첫부분에 채워줌
            self.memory_value = [np.array([np.nan] * len(Agent.ACTIONS))] \
                                * (self.num_steps - 1) + self.memory_value
        if self.policy_network is not None:
            self.memory_policy = [np.array([np.nan] * len(Agent.ACTIONS))] \
                                * (self.num_steps - 1) + self.memory_policy
        self.memory_pv = [self.agent.initial_balance] * (self.num_steps - 1) + self.memory_pv 
       
        # 가시화 시작
        self.visualizer.plot(
            epoch_str=epoch_str, num_epoches=num_epoches, 
            epsilon=epsilon, action_list=Agent.ACTIONS,  #에이전트 행동, 보유 주식수, 가치신경망출력, 정책신경망출력, 포폴가치, 탐험 위치 등..
            actions=self.memory_action, 
            num_stocks=self.memory_num_stocks, 
            outvals_value=self.memory_value, 
            outvals_policy=self.memory_policy,
            exps=self.memory_exp_idx, 
            initial_balance=self.agent.initial_balance, 
            pvs=self.memory_pv,
        )
        self.visualizer.save(os.path.join(self.epoch_summary_dir, f'epoch_summary_{epoch_str}.png'))

    def run(self, learning=True): #학습해서 신경망 모델 만들고 싶다면 learning을 true로, 학습된 모델 가지고 투자 시뮬레이션만 하려면 learning을 false로.
        info = (
            f'[{self.stock_code}] RL:{self.rl_method} NET:{self.net} '
            f'LR:{self.lr} DF:{self.discount_factor} '
        )
        with self.lock:
            logger.debug(info) #강화 학습 설정을 로그로 기록함.

        # 시작 시간
        time_start = time.time()

        # 가시화 준비
        # 차트 데이터는 변하지 않으므로 미리 가시화
        self.visualizer.prepare(self.environment.chart_data, info) #가시화 준비 -> 에포크가 진행돼도 변하지 않는 주식투자 환경인 차트 데이터를 미리 가시화함. 

        # 가시화 결과 저장할 폴더 준비 -> 결과는 output_path 경로 하위의 epoch_summary_* 폴더에 저장됨. 
        self.epoch_summary_dir = os.path.join(self.output_path, f'epoch_summary_{self.stock_code}')
        if not os.path.isdir(self.epoch_summary_dir):
            os.makedirs(self.epoch_summary_dir)
        else:
            for f in os.listdir(self.epoch_summary_dir):
                os.remove(os.path.join(self.epoch_summary_dir, f))

        # 학습에 대한 정보 초기화
        max_portfolio_value = 0 #여기엔 수행한 에포크 중에서 가장 높은 포트폴리오 가치가 저장됨.
        epoch_win_cnt = 0 # 여기엔 수행한 에포크 중에서 수익이 발생한 에포크 수를 저장함. (포트폴리오 가치가 초기 자본금보다 높아진 에포크 수)

        # 에포크 반복 #-> 지정된 에포크 수만큼 주식투자 시뮬레이션 반복하며 학습
        for epoch in tqdm(range(self.num_epoches)):
            time_start_epoch = time.time()

            # step 샘플을 만들기 위한 큐 (num_step만큼 샘플을 담아둘 큐 초기화)
            q_sample = collections.deque(maxlen=self.num_steps)
            
            # 환경, 에이전트, 신경망, 가시화, 메모리 초기화
            self.reset()

            # 학습을 진행할 수록 탐험 비율 감소 -> epsilon 값 정할 때는 최초 무작위 투자 비율인 start_epsilon 값에 현재 epoch 수에 학습진행률을 곱해서 정함. 
            if learning:
                epsilon = self.start_epsilon * (1 - (epoch / (self.num_epoches - 1)))
            else:
                epsilon = self.start_epsilon

            # 학습 데이터 반복의 초반부
            for i in tqdm(range(len(self.training_data)), leave=False):
                # 샘플 생성
                next_sample = self.build_sample() #build_sample() 함수를 호출해 환경 객체로부터 하나의 샘플을 읽어옴. 
                if next_sample is None: # next_sample이 None이라면 마지막까지 데이터를 다 읽은 것이므로 학습 데이터 반복을 종료함. 
                    break
                
                ### *⚠️코드 수정 시작⚠️*  ###
                # 마지막 데이터 포인트 확인
                is_last_step = (i == len(self.training_data) - 1)
                # num_steps만큼 샘플 저장 
                q_sample.append(next_sample)
                if len(q_sample) < self.num_steps and not is_last_step: #num_steps 개수만큼 샘플이 준비돼야 행동을 결정할 수 있으므로 샘플 큐에 샘플이 모두 찰 때까지 continue를 통해 이후 로직을 건너뜀.
                    continue
                          
                # 마지막 스텝이면 모든 주식 매도
                if is_last_step:
                    action = Agent.ACTION_SELL  # 매도 행동 강제 설정
                    confidence = 1.0  # 매도에 대한 확신 100%
                    reward = self.agent.act(action, confidence)  # 매도 실행
                    exploration = False  # 탐험 없음
                    
                else:
                    # 신경망 예측 및 행동 결정 로직
                    # 가치 신경망과 정책 신경망으로 예측 행동 가치와 예측 행동 확률을 구하는 부분
                    # 가치, 정책 신경망 예측
                    pred_value, pred_policy = None, None
                    if self.value_network:
                        pred_value = self.value_network.predict(list(q_sample))
                    if self.policy_network:
                        pred_policy = self.policy_network.predict(list(q_sample))
                    # 신경망 또는 탐험에 의한 행동 결정 (위에서 구한 예측 가치와 확률로 투자 행동을 결정함. -> 여기선 매수와 매도 중 하나를 결정함)
                    action, confidence, exploration = self.agent.decide_action(pred_value, pred_policy, epsilon)
                    # decide_action() 함수가 반환하는 값은 세 가지임. -> 결정한 행동인 action, 결정에 대한 확신도인 confidence, 무작위 추자 유무인 exploration
                   
                    # 결정한 행동을 수행하고 보상 획득
                    reward = self.agent.act(action, confidence) # 결정한 행동을 수행하도록 에이전트의 act() 함수를 호출함. act() 함수는 행동을 수행하고 행동 수행 후의 손익(reward)을 반환함.
                    ### *⚠️코드 수정 끝⚠️*  ###
                    
                    
                # 행동 및 행동에 대한 결과를 기억 -> 메모리 변수들은 1) 학습에서 배치 학습 데이터로 사용 2) 가시화기에서 차트 그릴 때 사용
                self.memory_sample.append(list(q_sample))
                self.memory_action.append(action)
                self.memory_reward.append(reward)
                if self.value_network is not None:
                    self.memory_value.append(pred_value)
                if self.policy_network is not None:
                    self.memory_policy.append(pred_policy)
                self.memory_pv.append(self.agent.portfolio_value)
                self.memory_num_stocks.append(self.agent.num_stocks)
                if exploration:
                    self.memory_exp_idx.append(self.training_data_idx)

                # 반복에 대한 정보 갱신
                self.batch_size += 1 # 배치 크기
                self.itr_cnt += 1 # 반복 카운팅 횟수
                self.exploration_cnt += 1 if exploration else 0 # 무작위 투자 횟수(탐험한 경우에만 1 증가시키고 그렇지 않으면 0 더해서 변화 없게.)

            # 에포크 종료 후 학습
            if learning: # 더 이상 샘플이 없는 경우 for 블록을 빠져나옴.
                self.fit() # 학습 모드인 경우 for 블록을 빠져나온 후에 신경망 학습 함수인 fit() 호출함. 

                
            # 에포크 관련 정보 로그 기록(정보 로깅하고 가시화하는 부분)
            num_epoches_digit = len(str(self.num_epoches)) #총 에포크 수의 문자열 길이 확인 (총 에포크 수가 1,000이면 길이는 4가됨. )
            epoch_str = str(epoch + 1).rjust(num_epoches_digit, '0') # 현재 에포크 수를 num_epochs_digit 길이의 문자열로 만들어 epoch_str에 저장함. ex) 네 자리 문자열 만든다 할 때, 첫번째 에포크의 경우 epcoh가 0이므로 1을 더하고 앞에 0을 채워서 0001로.
            time_end_epoch = time.time() # 현재 시간
            elapsed_time_epoch = time_end_epoch - time_start_epoch #현재 시간에서 시작시간 빼서 에포크 수행 소요 시간을 저장함.
            logger.debug(f'[{self.stock_code}][Epoch {epoch_str}/{self.num_epoches}] '
                f'Epsilon:{epsilon:.4f} #Expl.:{self.exploration_cnt}/{self.itr_cnt} '
                f'#Buy:{self.agent.num_buy} #Sell:{self.agent.num_sell} #Hold:{self.agent.num_hold} '
                f'#Stocks:{self.agent.num_stocks} PV:{self.agent.portfolio_value:,.0f} '
                f'Loss:{self.loss:.6f} ET:{elapsed_time_epoch:.4f}')

            # 에포크 관련 정보 가시화
            if self.num_epoches == 1 or (epoch + 1) % int(self.num_epoches / 10) == 0:
                self.visualize(epoch_str, self.num_epoches, epsilon) #가시화기 객체를 이용해 에포크 정보를 하나의 그림으로 가시화한 후 파일로 저장함

            # 학습 관련 정보 갱신
            max_portfolio_value = max(
                max_portfolio_value, self.agent.portfolio_value) #에포크를 수행하는 동안 최대 포트폴리오 가치를 갱신하고
            if self.agent.portfolio_value > self.agent.initial_balance:
                epoch_win_cnt += 1  #해당 에포크에서 포트폴리오 가치가 자본금보다 높으면 epcch_win_cnt를 증가시킴. 

        # 종료 시간
        time_end = time.time()
        elapsed_time = time_end - time_start # 전체 에포크 수행 소요 시간 기록

        # 학습 관련 정보 로그 기록
        with self.lock:
            logger.debug(f'[{self.stock_code}] Elapsed Time:{elapsed_time:.4f} ' #전체소요시간, 최대 포트폴리오 가치, 포폴가치가 자본금보다 높았던 에포크 수를 로그로 남김.
                f'Max PV:{max_portfolio_value:,.0f} #Win:{epoch_win_cnt}')