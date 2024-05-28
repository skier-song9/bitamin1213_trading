import os
import sys
import logging
import argparse
import json
import schedule
import time
import keyring
import requests
import pprint
import traceback
# import pykis
import pandas as pd

from quantylab.rltrader import settings
from quantylab.rltrader import utils
from quantylab.rltrader.utils import *
from quantylab.rltrader import data_manager_3
from quantylab.rltrader.environment import Environment
from quantylab.rltrader.agent import Agent

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'test', 'update', 'predict'], default='train')
    parser.add_argument('--ver', choices=['v1', 'v2', 'v3', 'v4', 'v4.1', 'v4.2'], default='v4.1')
    parser.add_argument('--name', default=utils.get_time_str())
    parser.add_argument('--stock_code', nargs='+')
    parser.add_argument('--rl_method', choices=['dqn', 'pg', 'ac', 'a2c', 'a3c', 'ppo', 'monkey'], default='a2c')
    parser.add_argument('--net', choices=['dnn', 'lstm', 'cnn', 'monkey'], default='dnn')
    parser.add_argument('--backend', choices=['pytorch', 'tensorflow', 'plaidml'], default='pytorch')
    parser.add_argument('--start_date', default='202030560901')
    parser.add_argument('--end_date', default='202405241530')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--discount_factor', type=float, default=0.99)
    parser.add_argument('--balance', type=int, default=100000000)
    
    ### 📢실시간 트레이딩용 파라미터
    parser.add_argument('--is_start_end', default=0, type=int) #0=월요일,1=나머지요일,2=금요일
    parser.add_argument('--is_mock', type=int, default=1) # 1=mock_invest, 0=real_invest

    args = parser.parse_args()

    # 학습기 파라미터 설정
    output_name = f'{args.mode}_{args.name}_{args.rl_method}_{args.net}'
    learning = args.mode in ['train', 'update']
    reuse_models = args.mode in ['test', 'update', 'predict']
    value_network_name = f'{args.name}_{args.rl_method}_{args.net}_value.mdl'
    policy_network_name = f'{args.name}_{args.rl_method}_{args.net}_policy.mdl'
    start_epsilon = 1 if args.mode in ['train', 'update'] else 0
    num_epoches = 300 if args.mode in ['train', 'update'] else 1
    num_steps = 120 if args.net in ['lstm', 'cnn'] else 1

    # Backend 설정
    os.environ['RLTRADER_BACKEND'] = args.backend
    if args.backend == 'tensorflow':
        os.environ['KERAS_BACKEND'] = 'tensorflow'
    elif args.backend == 'plaidml':
        os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'

    # 출력 경로 생성
    output_path = os.path.join(settings.BASE_DIR, 'output', output_name)
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    # 파라미터 기록
    params = json.dumps(vars(args))
    with open(os.path.join(output_path, 'params.json'), 'w') as f:
        f.write(params)

    # 모델 경로 준비
    # 모델 포멧은 TensorFlow는 h5, PyTorch는 pickle
    value_network_path = os.path.join(settings.BASE_DIR, 'models', value_network_name)
    policy_network_path = os.path.join(settings.BASE_DIR, 'models', policy_network_name)

    # 로그 기록 설정
    log_path = os.path.join(output_path, f'{output_name}_pred_{get_today_str()}.log') if args.mode == ' predict' else os.path.join(output_path, f'{output_name}.log')
    if os.path.exists(log_path):
        os.remove(log_path)
    logging.basicConfig(format='%(message)s')
    logger = logging.getLogger(settings.LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler(filename=log_path, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.info(params)
    
    # Backend 설정, 로그 설정을 먼저하고 RLTrader 모듈들을 이후에 임포트해야 함
    from quantylab.rltrader.learners import ReinforcementLearner, DQNLearner, \
        PolicyGradientLearner, ActorCriticLearner, A2CLearner, A3CLearner, PPOLearner

    common_params = {}
    list_stock_code = []
    list_chart_data = []
    list_training_data = []
    list_min_trading_price = []
    list_max_trading_price = []

    ### 📢한국투자증권 API 활용을 위한 코드 [시작]
    # 실전계좌
    api_keys = read_json('./quantylab/api.json')
    investment_type = 'mock_invest' if args.is_mock==1 else 'real_invest'
    api_keys = api_keys[investment_type]
    keyring.set_password('app_key','user',api_keys['app_key'])
    keyring.set_password('app_secret','user',api_keys['app_secret'])
    APP_KEY, APP_SECRET, ACCOUNT = api_keys['app_key'], api_keys['app_secret'], api_keys['account']
    # res = get_access_token(APP_KEY, APP_SECRET, investment_type)
    # pprint.pprint(res.json(), indent=4)
    # ACCESS_TOKEN = res.json()['access_token']
    ACCESS_TOKEN = api_keys['access_token']
    key_info = {
        "appkey": APP_KEY,                  
        "appsecret": APP_SECRET 
    }
    account_info ={
        "account_code": ACCOUNT.split('-')[0],   
        "product_code": ACCOUNT.split('-')[1]
    }
    # if investment_type =='mock_invest':
    #     domain = pykis.DomainInfo(kind="virtual")
    #     api = pykis.Api(key_info=key_info,domain_info=domain,account_info=account_info)
    # else:
    #     api = pykis.Api(key_info=key_info,account_info=account_info)
    ### 📢한국투자증권 API 활용을 위한 코드 [끝]

    for stock_code in args.stock_code:
        # 차트 데이터, 학습 데이터 준비
        chart_data, training_data = data_manager_3.load_data(
            stock_code, args.start_date, args.end_date, ver=args.ver)

        assert len(chart_data) >= num_steps
        
        # 최소/최대 단일 매매 금액 설정
        min_trading_price = 500
        max_trading_price = 10000000

        # 공통 파라미터 설정
        common_params = {'rl_method': args.rl_method, 
            'net': args.net, 'num_steps': num_steps, 'lr': args.lr,
            'balance': args.balance, 'num_epoches': num_epoches, 
            'discount_factor': args.discount_factor, 'start_epsilon': start_epsilon,
            'output_path': output_path, 'reuse_models': reuse_models}

        # 강화학습 시작
        learner = None
        if args.rl_method != 'a3c':
            common_params.update({'stock_code': stock_code,
                'chart_data': chart_data, 
                'training_data': training_data,
                'min_trading_price': min_trading_price, 
                'max_trading_price': max_trading_price})
            if args.rl_method == 'dqn':
                learner = DQNLearner(**{**common_params, 
                    'value_network_path': value_network_path})
            elif args.rl_method == 'pg':
                learner = PolicyGradientLearner(**{**common_params, 
                    'policy_network_path': policy_network_path})
            elif args.rl_method == 'ac':
                learner = ActorCriticLearner(**{**common_params, 
                    'value_network_path': value_network_path, 
                    'policy_network_path': policy_network_path})
            elif args.rl_method == 'a2c':
                learner = A2CLearner(**{**common_params, 
                    'value_network_path': value_network_path, 
                    'policy_network_path': policy_network_path})
            elif args.rl_method == 'ppo':
                learner = PPOLearner(**{**common_params, 
                    'value_network_path': value_network_path, 
                    'policy_network_path': policy_network_path})
            elif args.rl_method == 'monkey':
                common_params['net'] = args.rl_method
                common_params['num_epoches'] = 10
                common_params['start_epsilon'] = 1
                learning = False
                learner = ReinforcementLearner(**common_params)
        else:
            list_stock_code.append(stock_code)
            list_chart_data.append(chart_data)
            list_training_data.append(training_data)
            list_min_trading_price.append(min_trading_price)
            list_max_trading_price.append(max_trading_price)

    if args.rl_method == 'a3c':
        learner = A3CLearner(**{
            **common_params, 
            'list_stock_code': list_stock_code, 
            'list_chart_data': list_chart_data, 
            'list_training_data': list_training_data,
            'list_min_trading_price': list_min_trading_price, 
            'list_max_trading_price': list_max_trading_price,
            'value_network_path': value_network_path, 
            'policy_network_path': policy_network_path})
    
    assert learner is not None

    if args.mode in ['train', 'test', 'update']:
        learner.run(learning=learning)
        if args.mode in ['train', 'update']:
            learner.save_models()


### *⚠️코드 수정 시작⚠️*  ###
    elif args.mode == 'predict':
        ### load properties.json
        props = read_json('./quantylab/properties.json')

        ### Load Informations
        '''
        - is_start_end 변수 : 0이면 월요일, 1이면 나머지 요일, 2이면 금요일을 의미

        - 월요일 : initial_balance를 API로 불러와서 json에 업데이트,
                balance도 initial_balance로 업데이트,
                나머지 정보들은 0으로 리셋
        - 모든 요일 공통 : 모든 정보들을 json 파일에서 불러옴
        >> 'INITIAL_BALANCE', 'BALANCE', 'PORTFOLIO_VALUE', 'NUM_BUY',
        'NUM_SELL', 'NUM_HOLD', 'RATIO_HOLD', 'PROFITLOSS' 변수명으로 해당 값들을 저장함.
        
        - 
        '''
        if args.is_start_end == 0: ### 월요일에만 실행되는 코드
            ### API 활용해서 내 계좌에서 잔액 불러오기
            INITIAL_BALANCE = int(get_balance_api(
                ACCOUNT,APP_KEY,APP_SECRET,ACCESS_TOKEN,investment_type
            ))
            props['initial_balance'] = INITIAL_BALANCE
            props['balance'] = INITIAL_BALANCE
            props['num_stocks'] = 0
            props['portfolio_value'] = INITIAL_BALANCE
            props['num_buy'] = 0
            props['num_sell'] = 0
            props['num_hold'] = 0
            props['ratio_hold'] = 0
            props['profitloss'] = 0
            props['avg_buy_price'] = 0

        # 모든 요일 공통 : 정보 불러오기
        props_list = ['INITIAL_BALANCE', 'BALANCE', 'NUM_STOCKS', 'PORTFOLIO_VALUE', 'NUM_BUY', 'NUM_SELL', 'NUM_HOLD', 'RATIO_HOLD', 'PROFITLOSS', 'AVG_BUY_PRICE']
        for pl in props_list:
            globals()[f'{pl}'] = props[f'{pl.lower()}'] # 변수 동적 할당
        
        ### Learner 객체의 변수값을 수정 >> agent의 preset() 호출
        learner.agent.preset(INITIAL_BALANCE, BALANCE, NUM_STOCKS, PORTFOLIO_VALUE, NUM_BUY, NUM_SELL, 
         NUM_HOLD, RATIO_HOLD, PROFITLOSS, AVG_BUY_PRICE)

        ### 현재 한국 시간을 기준으로 오전 8시 59분 0.5초에 while 루프 시작
        while True:
            # 현재 시간을 가져와서 한국 시간으로 변환
            current_time = time.localtime()
            if current_time.tm_hour >= 8 and current_time.tm_min >= 59 and current_time.tm_sec >= 0:
                break
            time.sleep(0.5)

        ### scheduling 설정
        results = [] # 하룻동안의 결과들을 저장할 변수

        ### 1분마다 predict를 호출
        counter = 0 
        pt = time.time()
        try:
            while True:
                while True:
                    ct = time.time()
                    ctl = time.localtime()
                    if counter == 0:
                        break
                    if (ct-pt) >= 60:
                        pt = ct
                        break
                    continue
                counter += 1
                if int(get_dtime_str()) >= 152000:
                    break 
                print(f"{counter} | {ctl.tm_hour}:{ctl.tm_min}:{ctl.tm_sec} | predict started")
                result = learner.predict()

                ### predict의 result로 매수/매도/관망 행동 판단 후 수행
                pred_value, pred_policy = result[0][-2],result[0][-1] 
                action, confidence, exploration = learner.agent.decide_action(
                    pred_value, pred_policy, 0 # epsilon은 0으로 해야 모험을 안 함.
                )
                print(f'action:{action}, confidence:{confidence}')
                
                ### 매도/매수 API 코드 - agent의 act()함수를 predict에 맞게 구현
                '''
                - 한국투자증권은 매수 금액에 따른 차등수수료 >> validate_action 대신 pvalidate_action()함수 사용
                - decide_trading_unit 후에 charge는 utils의 get_charge()로 불러온다. 
                '''
                if not learner.agent.pvalidate_action(action):
                    action = learner.agent.ACTION_HOLD
                curr_price = learner.environment.get_price()
                do_nothing = False
                if action == learner.agent.ACTION_BUY:
                    # 매수할 단위를 판단
                    trading_unit = learner.agent.decide_trading_unit(confidence)
                    # if counter == 1:
                    # recent_volume =  int(int(learner.environment.observation.iloc[5]) * 2.5)
                    # trading_unit = min(trading_unit, recent_volume)
                    '''
                    while True:
                        1. 매수가능조회(get_possible)해서 매수가능수량 조회 >> min(trading_unit, possible)
                        2. 시장가로 trading_unit만큼 매수(buy_stock)
                        3. time.sleep(2.5) : 2.5초 동안 기다린다.
                        4. 매수가 체결되었는지 확인(select_order) > order_price, is_buyed
                            5. is_buyed가 False이면 미체결된 것 >> order_id 주문을 취소하고 다시 1번으로 돌아감
                        6. is_buyed가 True라면 정보들 업데이트 & break
                    '''
                    buy_counter = 0
                    while 1:
                        possible_unit = get_possible(ACCOUNT,APP_KEY,APP_SECRET,ACCESS_TOKEN,investment_type,stock_code)
                        time.sleep(1)
                        trading_unit = min(trading_unit, int(possible_unit))

                        order_id, order_time = buy_stock(ACCOUNT,APP_KEY,APP_SECRET,ACCESS_TOKEN,investment_type,stock_code,trading_unit)
                        print(f"{order_time}, {order_id} 매수 주문, 주문수량: {trading_unit}")
                        time.sleep(5)

                        order_price, is_buyed = select_order(ACCOUNT,APP_KEY,APP_SECRET,ACCESS_TOKEN,investment_type,stock_code, order_id)
                        time.sleep(1)
                        if not is_buyed:
                            ### 주문 취소
                            rt_cd, order_time = cancel_order(ACCOUNT,APP_KEY,APP_SECRET,ACCESS_TOKEN,investment_type,order_id)
                            time.sleep(1)
                            print(f"{order_time}, {order_id} 매수 주문이 5초내에 체결되지 않아 취소되었습니다.")
                            buy_counter+=1
                            if buy_counter == 3:
                                # 더 이상 주문 시도 멈추고 action을 HOLD로 바꿈.
                                order_price, trading_unit = 0,0
                                learner.agent.action = learner.agent.ACTION_HOLD
                                learner.agent.num_hold += 1
                                break
                            continue

                        ### 주문이 체결된 경우
                        # 잔액 업데이트
                        balance = int(get_balance_api(ACCOUNT,APP_KEY,APP_SECRET,ACCESS_TOKEN,investment_type))
                        time.sleep(1)
                        # 매수 성공 시, 수수료를 적용하여 총 매수 금액 산정 및 변수 업데이트
                        order_price, trading_unit = int(order_price), int(trading_unit)
                        hantu_charge, add_price = get_charge(order_price, trading_unit)
                        invest_amount = order_price * (1 + hantu_charge) * trading_unit + add_price
                        if invest_amount > 0:
                            learner.agent.avg_buy_price = \
                                (learner.agent.avg_buy_price * learner.agent.num_stocks + order_price * trading_unit) \
                                    / (learner.agent.num_stocks + trading_unit)  # 주당 매수 단가 갱신
                            learner.agent.balance = balance  # 보유 현금을 갱신
                            learner.agent.num_stocks += trading_unit  # 보유 주식 수를 갱신
                            learner.agent.num_buy += 1  # 매수 횟수 증가
                        break
                    # end inner 매수 while 
                                            

                elif action == learner.agent.ACTION_SELL:

                    ### 매도 Loop
                    buy_counter = 0
                    while 1:
                        trading_unit = learner.agent.decide_trading_unit(confidence)
                        trading_unit = min(trading_unit, learner.agent.num_stocks)
                        
                        order_id, order_time = sell_stock(ACCOUNT,APP_KEY,APP_SECRET,ACCESS_TOKEN,investment_type,stock_code,trading_unit)
                        print(f"{order_time}, {order_id} 매도 주문, 주문수량: {trading_unit}")
                    #     print(get_time_str())
                        time.sleep(15)
                    #     print(get_time_str())
                        order_price, is_buyed = select_order(ACCOUNT,APP_KEY,APP_SECRET,ACCESS_TOKEN,investment_type,stock_code, order_id)
                        time.sleep(1)
                        if not is_buyed:
                            ### 주문 취소
                            rt_cd, order_time = cancel_order(ACCOUNT,APP_KEY,APP_SECRET,ACCESS_TOKEN,investment_type,order_id)
                            time.sleep(1)
                            print(f"{order_time}, {order_id} 매도 주문이 5초내에 체결되지 않아 취소되었습니다.")
                            buy_counter+=1
                            if buy_counter == 2:
                                # 더 이상 주문 시도 멈추고 action을 HOLD로 바꿈.
                                order_price, trading_unit = 0,0
                                learner.agent.action = learner.agent.ACTION_HOLD
                                learner.agent.num_hold += 1
                                break
                            continue

                        ### 매도 주문이 체결된 경우
                        # 잔액 업데이트
                        balance = int(get_balance_api(ACCOUNT,APP_KEY,APP_SECRET,ACCESS_TOKEN,investment_type))
                        time.sleep(1)
                        # 매도 성공 시, 수수료를 적용하여 총 매도 금액 산정 및 변수 업데이트
                        order_price, trading_unit = int(order_price), int(trading_unit)
                        invest_amount = order_price * (1+ learner.agent.HANTU_TAX) * trading_unit
                        if invest_amount > 0:
                            learner.agent.avg_buy_price = \
                                (learner.agent.avg_buy_price * learner.agent.num_stocks - order_price * trading_unit) \
                                    / (learner.agent.num_stocks - trading_unit) \
                                        if learner.agent.num_stocks > trading_unit else 0  # 주당 매도 단가 갱신
                            learner.agent.balance = balance  # 보유 현금을 갱신
                            learner.agent.num_stocks -= trading_unit  # 보유 주식 수를 갱신
                            learner.agent.num_sell += 1  # 매도 횟수 증가
                        break
                    # end inner 매도 while 

                    ### 📢매도 API 호출
                    # 매도 가능 여부를 확인하고 attempt_to_sell 함수 호출
                    # if learner.agent.num_stocks >= trading_unit:
                    #     attempt_to_sell(api, learner, stock_code, trading_unit, curr_price)
                    # else:
                    #     learner.agent.num_hold += 1  # 보유 주식 수가 매도 단위보다 적을 경우
                    #     print("Not enough stocks to sell")

                else:
                    order_price, trading_unit = 0,0
                    learner.agent.num_hold += 1
                

                '''
                num_stocks, num_buy, num_sell, num_hold, 
                portfolio_value, ratio_hold, profitloss, avg_buy_price 계산 후 
                json파일에 저장 
                '''
                ### 계산.
                if counter == 1:
                    continue
                invest_amount = 0 if invest_amount else invest_amount
                learner.agent.portfolio_value = int(get_balance_api(ACCOUNT,APP_KEY,APP_SECRET,ACCESS_TOKEN,investment_type)) + invest_amount
                time.sleep(1)
                learner.agent.profitloss = learner.agent.portfolio_value / learner.agent.initial_balance - 1
                learner.agent.ratio_hold = learner.agent.num_stocks * curr_price \
                    / learner.agent.portfolio_value
                
                ### 여기서 분봉 데이터를 1개 불러와서 chart_data의 마지막 row로 추가.
                time.sleep(1)
                stck_cntg_hour,stck_oprc,stck_hgpr,stck_lwpr,stck_prpr,cntg_vol = get_min_data(APP_KEY,APP_SECRET,ACCESS_TOKEN,investment_type,stock_code)
                time.sleep(1)
                t = time.localtime()
                stck_cntg_hour = f"{t.tm_year}{t.tm_mon:02}{t.tm_mday:02}{stck_cntg_hour[:4]}"
                new_row = pd.DataFrame(
                    [[stck_cntg_hour,stck_oprc,stck_hgpr,stck_lwpr,stck_prpr,cntg_vol]],
                    columns=['date','open','high','low','close','volume']
                )
                chart_data = pd.concat([chart_data,new_row]).reset_index(drop=True)
                # chart_data.to_csv(f"../data/v1/{stock_code}.csv", index=0)
                new_pre_data = data_manager_3.preprocess(chart_data.iloc[-120:,:6].reset_index(drop=True))
                new_tr = new_pre_data[data_manager_3.COLUMNS_TRAINING_DATA_V1].iloc[-1]
                new_tr = pd.DataFrame([new_tr])
                training_data = pd.concat([training_data, new_tr]).reset_index(drop=True)
                # update learner's data
                learner.chart_data = chart_data
                learner.training_data = training_data
                learner.environment.chart_data = chart_data
                learner.agent.environment = learner.environment 

                ### is_start_end 값이 2일 때 (금요일)
                # counter 390번일 때 -> 마지막 실행 시 주식을 여전히 보유하고 있으면 모두 매도하는 코드
                if (args.is_start_end ==2) and (int(get_dtime_str()) >= 152800):
                    # 모두 매도
                    buy_counter = 0
                    while 1:
                        trading_unit = learner.agent.num_stocks                        
                        order_id, order_time = sell_stock(ACCOUNT,APP_KEY,APP_SECRET,ACCESS_TOKEN,investment_type,stock_code,trading_unit)
                        time.sleep(1)
                        print(f"{order_time}, {order_id} 매도 주문")
                    #     print(get_time_str())
                        time.sleep(15)
                    #     print(get_time_str())
                        order_price, is_buyed = select_order(ACCOUNT,APP_KEY,APP_SECRET,ACCESS_TOKEN,investment_type,stock_code, order_id)
                        if not is_buyed:
                            ### 주문 취소
                            rt_cd, order_time = cancel_order(ACCOUNT,APP_KEY,APP_SECRET,ACCESS_TOKEN,investment_type,order_id)
                            time.sleep(1)
                            print(f"{order_time}, {order_id} 매도 주문이 5초내에 체결되지 않아 취소되었습니다.")
                            buy_counter+=1
                            if buy_counter == 2:
                                print(get_time_str(),'잔량 매도 실패')
                                break
                            continue

                        ### 매도 주문이 체결된 경우
                        # 잔액 업데이트
                        balance = int(get_balance_api(ACCOUNT,APP_KEY,APP_SECRET,ACCESS_TOKEN,investment_type))
                        time.sleep(1)
                        # 매도 성공 시, 수수료를 적용하여 총 매도 금액 산정 및 변수 업데이트
                        order_price, trading_unit = int(order_price), int(trading_unit)
                        income = order_price * (1+ learner.agent.HANTU_TAX) * trading_unit
                        if invest_amount > 0:
                            learner.agent.avg_buy_price =  0  # 주당 매도 단가 갱신
                            learner.agent.balance = balance  # 보유 현금을 갱신
                            learner.agent.num_stocks -= trading_unit  # 보유 주식 수를 갱신
                            learner.agent.num_sell += 1  # 매도 횟수 증가
                        break
                # end 금요일 process
                    
                # Record Log
                try:
                    if order_price is None and trading_unit is None:
                        order_price = 0 
                        trading_unit = 0 
                    acts = {0:'BUY', 1:'SELL',2:'HOLD'}
                    logger.debug(
                        f"c{counter} | #TIME:{get_dtime_str(ct)} | #ACTION:{acts[action]} | #CONFIDENCE:{confidence} | #PRICE:{'HOLD' if action == 2 else order_price} | #UNIT:{'HOLD' if action == 2 else trading_unit} | #PV:{learner.agent.portfolio_value} | #STOCKS:{learner.agent.num_stocks} | #LOSS:{learner.agent.profitloss}"
                    )
                except:
                    traceback.print_exc()

            # end while

            ### 여기서 분봉 데이터를 한 번 더 불러옴 >> 15시30분꺼
            time.sleep(1)
            stck_cntg_hour,stck_oprc,stck_hgpr,stck_lwpr,stck_prpr,cntg_vol = get_min_data(APP_KEY,APP_SECRET,ACCESS_TOKEN,investment_type,stock_code)
            stck_cntg_hour = '152000'
            time.sleep(1)
            t = time.localtime()
            stck_cntg_hour = f"{t.tm_year}{t.tm_mon:02}{t.tm_mday:02}{stck_cntg_hour[:4]}"
            new_row = pd.DataFrame(
                [[stck_cntg_hour,stck_oprc,stck_hgpr,stck_lwpr,stck_prpr,cntg_vol]],
                columns=['date','open','high','low','close','volume']
            )
            chart_data = pd.concat([chart_data,new_row]).reset_index(drop=True)
            # chart_data.to_csv(f"../data/v1/{stock_code}.csv", index=0)
            new_pre_data = data_manager_3.preprocess(chart_data.iloc[-120:,:6].reset_index(drop=True))
            new_tr = new_pre_data[data_manager_3.COLUMNS_TRAINING_DATA_V1].iloc[-1]
            new_tr = pd.DataFrame([new_tr])
            training_data = pd.concat([training_data, new_tr]).reset_index(drop=True)
            # update learner's data
            learner.chart_data = chart_data
            learner.training_data = training_data
            learner.environment.chart_data = chart_data
            learner.agent.environment = learner.environment 

        # end try
        except:
            traceback.print_exc()
        finally:
            # chart_data 저장
            cd = pd.read_csv(f'../data/v1/{stock_code}.csv')
            last_idx = int(chart_data[chart_data['date']==args.end_date].index[-1])+1
            # print('last_idx', last_idx)
            cd = pd.concat([cd, chart_data.iloc[last_idx:,:]]).reset_index(drop=True)
            cd.to_csv(f'../data/v1/{stock_code}.csv',index=0)

            ### json에 저장
            datas = {
                "initial_balance" : learner.agent.initial_balance, 
                "balance" : learner.agent.balance,
                "num_stocks" : learner.agent.num_stocks,
                "portfolio_value" : learner.agent.portfolio_value,
                "num_buy" : learner.agent.num_buy,
                "num_sell" : learner.agent.num_sell,
                "num_hold" : learner.agent.num_hold, 
                "ratio_hold" : learner.agent.ratio_hold,  
                "profitloss" : learner.agent.profitloss,
                "avg_buy_price" : learner.agent.avg_buy_price
                }
            write_json(data=datas,filename='./quantylab/properties.json')
            # schedule.clear()
            # pass

            ## access_token 폐기
            try:
                host = "https://openapivts.koreainvestment.com:29443" if investment_type=='mock_inverst' else 'https://openapi.koreainvestment.com:9443'
                endpoint = "/oauth2/revokeP"
                url = host+endpoint
                headers = {"content-type":"application/json; charset=UTF-8"}
                body = {"appkey":APP_KEY, 
                        "appsecret":APP_SECRET,
                        "token":ACCESS_TOKEN
                        }
                resp = requests.post(url, headers=headers, data=json.dumps(body))
                if resp.json()['code'] == 200:
                    print("access token successfully revoked.")
                else:
                    pprint.pprint(resp.json()['message'])
            except:
                print("access token already revoked.")

            print("✅finish",get_time_str())
        
        
### *⚠️코드 수정 끝⚠️*  ###