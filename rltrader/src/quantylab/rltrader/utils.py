import time
import datetime
import numpy as np
import json
import requests

# 날짜, 시간 관련 문자열 형식
FORMAT_DATE = "%Y%m%d"
FORMAT_DATETIME = "%Y%m%d%H%M%S"
FORMAT_HMS = "%H%M%S"

def get_dtime_str(): # 현재 시간을 FORMAT_DATETIME으로 반환
    return datetime.datetime.fromtimestamp(
        int(time.time())).strftime(FORMAT_HMS)

def get_today_str(): # 현재시간을 FORMAT_DATE으로 반환
    today = datetime.datetime.combine(
        datetime.date.today(), datetime.datetime.min.time())
    today_str = today.strftime('%Y%m%d')
    return today_str

def get_time_str(): # 현재 시간을 FORMAT_DATETIME으로 반환
    return datetime.datetime.fromtimestamp(
        int(time.time())).strftime(FORMAT_DATETIME)

def sigmoid(x):
    x = max(min(x, 10), -10)
    return 1. / (1. + np.exp(-x))

def read_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

def write_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

# 보안 인증키 발급
def get_access_token(APP_KEY,APP_SECRET, investment_type):
    domain = "https://openapivts.koreainvestment.com:29443" if investment_type=='mock_inverst' else 'https://openapi.koreainvestment.com:9443' 
    headers = {"content-type":"application/json; charset=UTF-8"}
    body = {"grant_type":"client_credentials",
            "appkey":APP_KEY, 
            "appsecret":APP_SECRET}
    endpoint = "/oauth2/tokenP"
    URL = domain + endpoint
    res = requests.post(URL, headers=headers, data=json.dumps(body))
    return res

# 계좌 잔액 불러오기 > 주신잔고조회
def get_balance_api(ACCOUNT,APP_KEY,APP_SECRET,ACCESS_TOKEN,investment_type):
    ### API 활용해서 내 계좌에서 잔액 불러오기
    domain = 'https://openapivts.koreainvestment.com:29443' if investment_type=='mock_invest' else 'https://openapi.koreainvestment.com:9443'
    endpoint = '/uapi/domestic-stock/v1/trading/inquire-balance'
    url = domain + endpoint
    params = {
        "CANO" : ACCOUNT.split('-')[0],
        "ACNT_PRDT_CD" : ACCOUNT.split('-')[1],
        "AFHR_FLPR_YN" : "N",
        "OFL_YN" : "",
        "INQR_DVSN" : '02',
        "UNPR_DVSN" : '01',
        "FUND_STTL_ICLD_YN" : 'Y',
        "FNCG_AMT_AUTO_RDPT_YN" : 'N',
        "PRCS_DVSN" : '00',
        "CTX_AREA_FK100" : '',
        "CTX_AREA_NK100" : '',
    }
    headers = {
        "authorization": f"Bearer {ACCESS_TOKEN}",
        "appKey": f"{APP_KEY}",
        "appSecret": f"{APP_SECRET}",
        "tr_id": "VTTC8434R" if investment_type=='mock_invest' else "TTTC8434R"
    }
    res = requests.get(url, params=params, headers=headers)
    if res.status_code == 200:
        resdata = res.json()
        balance = resdata['output2'][0]['prvs_rcdl_excc_amt'] # D+2예수금
        return balance
    else :
        print("(get_balance_api) ERROR when call API:",res.status_code,res.text)
        exit(1)
    

def get_charge(price, units):
    '''
    📢온라인매매(수수료)
    50만원 미만 : 0.4971487%
    50만원 이상 ~ 3백만원 미만 : 0.1271487% + 2,000원
    3백만원 이상 ~ 3천만원 미만 : 0.1271487% + 1,500원
    3천만원 이상 ~ 1억원 미만 : 0.1171487%
    1억원 이상 ~ 3억원 미만 : 0.0971487%
    3억원 이상 : 0.0771487%

    📢매도 증권거래세
    코스피시장 = 매도금액의 0.18%
    '''
    p = price * units
    if p < 500000 :
        return [0.4971487, 0]
    elif p < 3000000 :
        return [0.1271487, 2000]
    elif p < 30000000 :
        return [0.1271487, 1500]
    elif p < 100000000 : 
        return [0.1171487, 0]
    elif p < 300000000 : 
        return [0.0971487, 0]
    else:
        return [0.0771487, 0]

def stock_cur_price(ACCOUNT,APP_KEY,APP_SECRET,ACCESS_TOKEN,investment_type,stock_code):
    ### 주식현재가 호가/예상체결
    domain = 'https://openapivts.koreainvestment.com:29443' if investment_type=='mock_invest' else 'https://openapi.koreainvestment.com:9443'
    endpoint = '/uapi/domestic-stock/v1/quotations/inquire-asking-price-exp-ccn'
    url = domain + endpoint
    params = {
        "FID_COND_MRKT_DIV_CODE" : 'J',
        "FID_INPUT_ISCD" : f"{stock_code}",
    }
    headers = {
        "authorization": f"Bearer {ACCESS_TOKEN}",
        "appKey": f"{APP_KEY}",
        "appSecret": f"{APP_SECRET}",
        "tr_id": "FHKST01010200"
    }
    res = requests.get(url, params=params, headers=headers)
    return res

def buy_stock(ACCOUNT,APP_KEY,APP_SECRET,ACCESS_TOKEN,investment_type,stock_code,trading_unit):
    ### 주식 매수
    domain = 'https://openapivts.koreainvestment.com:29443' if investment_type=='mock_invest' else 'https://openapi.koreainvestment.com:9443'
    endpoint = '/uapi/domestic-stock/v1/trading/order-cash'
    url = domain + endpoint
    data = {
        "CANO" : f"{ACCOUNT.split('-')[0]}",
        "ACNT_PRDT_CD" : f"{ACCOUNT.split('-')[1]}",
        "PDNO" : f"{stock_code}",
        "ORD_DVSN" : "01", # 시장가로 매수
        "ORD_QTY" : f"{trading_unit}",
        "ORD_UNPR" : "0"
    }
    headers = {
        "Content-Type": "application/json",
        "authorization": f"Bearer {ACCESS_TOKEN}",
        "appKey": f"{APP_KEY}",
        "appSecret": f"{APP_SECRET}",
        "tr_id": "VTTC0802U" if investment_type=='mock_invest' else "TTTC0802U"
    }
    res = requests.post(url, data=json.dumps(data), headers=headers)
    if res.status_code == 200:
        resp = res.json()
        rt_cd = resp['rt_cd']
        return resp['output']['ODNO'], resp['output']['ORD_TMD']
    else :
        print("(buy_stock1) ERROR when call API:",res.status_code,res.text)
        exit(1)

def get_possible(ACCOUNT,APP_KEY,APP_SECRET,ACCESS_TOKEN,investment_type,stock_code):
    ### API 활용해서 내 계좌에서 잔액 불러오기
    domain = 'https://openapivts.koreainvestment.com:29443' if investment_type=='mock_invest' else 'https://openapi.koreainvestment.com:9443'
    endpoint = '/uapi/domestic-stock/v1/trading/inquire-psbl-order'
    url = domain + endpoint
    params = {
        "CANO" : f"{ACCOUNT.split('-')[0]}",
        "ACNT_PRDT_CD" : f"{ACCOUNT.split('-')[1]}",
        "PDNO" : f"{stock_code}",
        "ORD_UNPR" : "", # 시장가로 조회
        "ORD_DVSN" : "01",
        "CMA_EVLU_AMT_ICLD_YN" : "N",
        "OVRS_ICLD_YN" : "N"
    }
    headers = {
        "authorization": f"Bearer {ACCESS_TOKEN}",
        "appKey": f"{APP_KEY}",
        "appSecret": f"{APP_SECRET}",
        "tr_id": "VTTC8908R" if investment_type=='mock_invest' else "TTTC8908R"
    }
    res = requests.get(url, params=params, headers=headers)
    if res.status_code == 200:
        return res.json()['output']['nrcvb_buy_qty']
    else :
        print("(get_possible) ERROR when call API:",res.status_code,res.text)
        exit(1)

### 주문번호로 당일주문체결조회
def select_order(ACCOUNT,APP_KEY,APP_SECRET,ACCESS_TOKEN,investment_type,stock_code, order_id):
    domain = 'https://openapivts.koreainvestment.com:29443' if investment_type=='mock_invest' else 'https://openapi.koreainvestment.com:9443'
    endpoint = '/uapi/domestic-stock/v1/trading/inquire-daily-ccld'
    url = domain + endpoint
    t = time.localtime()
    params = {
        "CANO" : f"{ACCOUNT.split('-')[0]}",
        "ACNT_PRDT_CD" : f"{ACCOUNT.split('-')[1]}",
        "INQR_STRT_DT" : f"{t.tm_year}{t.tm_mon:02}{t.tm_mday:02}",
        "INQR_END_DT" : f"{t.tm_year}{t.tm_mon:02}{t.tm_mday:02}",
        "SLL_BUY_DVSN_CD" : "02", # 매수
        "INQR_DVSN" : "00", # 역순조회 (최근 체결이 먼저 출력됨)
        "PDNO" : f"{stock_code}",
        "CCLD_DVSN" : "00", # 00=전체, 01=체결, 02=미체결
        "ORD_GNO_BRNO" : "",
        "ODNO" : f"{order_id}",
        "INQR_DVSN_3" : "01", # 현금주문만 조회
        "INQR_DVSN_1" : "",
        "CTX_AREA_FK100" : "",
        "CTX_AREA_NK100" : ""
    }
    headers = {
        "authorization": f"Bearer {ACCESS_TOKEN}",
        "appKey": f"{APP_KEY}",
        "appSecret": f"{APP_SECRET}",
        "tr_id": "VTTC8001R" if investment_type == 'mock_invest' else "TTTC8001R"
    }
    res = requests.get(url, params=params, headers=headers)
    if res.status_code == 200:
        resp = res.json()
        order_id_ = resp['output1'][0]['odno']
        order_price = resp['output1'][0]['avg_prvs'] # 체결 평균단가 = (총체결금액/총체결수량) # 체결이 안 됐으면 order_price는 0
        is_buyed = True if int(resp['output1'][0]['ord_unpr']) == 0 else False # 주문단가가 0으로 나오면 매수주문이 체결된 것임.
        print(f"{order_id}에 대한 체결 조회 {int(order_id_)}, 주문단가: {order_price}, 체결여부: {is_buyed}")
        return (order_price, is_buyed) 
#         return resp
    else :
        print("(select_order) ERROR when call API:",res.status_code,res.text)
        exit(1)

### 매수주문 취소
### 매수 주문 취소
def cancel_order(ACCOUNT,APP_KEY,APP_SECRET,ACCESS_TOKEN,investment_type,order_id):
    domain = 'https://openapivts.koreainvestment.com:29443' if investment_type=='mock_invest' else 'https://openapi.koreainvestment.com:9443'
    endpoint = "/uapi/domestic-stock/v1/trading/order-rvsecncl"
    URL = domain+endpoint
    headers = {"Content-Type":"application/json", 
        "authorization":f"Bearer {ACCESS_TOKEN}",
        "appKey":APP_KEY,
        "appSecret":APP_SECRET,
        "tr_id":"VTTC0803U" if investment_type == 'mock_invest' else "TTTC0803U"
    }
    data = {
        "CANO" : f"{ACCOUNT.split('-')[0]}",
        "ACNT_PRDT_CD" : f"{ACCOUNT.split('-')[1]}",
        "KRX_FWDG_ORD_ORGNO" : "",
        "ORGN_ODNO" : f"{order_id}",
        "ORD_DVSN" : "01", # 주문구분 : 시장가
        "RVSE_CNCL_DVSN_CD" : "02", # 취소
        "ORD_QTY" : "0", # 주문수량 전부취소
        "ORD_UNPR" : "0", # 취소는 0
        "QTY_ALL_ORD_YN" : "Y" # 잔량전부취소
    }
    res = requests.post(URL, headers=headers, data=json.dumps(data))
    if res.status_code == 200:
        resp = res.json()
        rt_cd = resp['rt_cd']
        ord_time = resp['output']['ORD_TMD']
        return rt_cd, ord_time
    else :
        print("(cancel_order) ERROR when call API:",res.status_code,res.text)
        exit(1)

def get_min_data(APP_KEY,APP_SECRET,ACCESS_TOKEN,investment_type,stock_code):
    domain = 'https://openapivts.koreainvestment.com:29443' if investment_type=='mock_invest' else 'https://openapi.koreainvestment.com:9443'
    endpoint = '/uapi/domestic-stock/v1/quotations/inquire-time-itemchartprice'
    url = domain + endpoint
    t = time.localtime()
    params = {
        "FID_ETC_CLS_CODE" : "",
        'FID_COND_MRKT_DIV_CODE' : 'J',
        'FID_INPUT_ISCD' : f"{stock_code}",
        'FID_INPUT_HOUR_1' : f"{t.tm_hour:02}{t.tm_min:02}{t.tm_sec:02}",
        "FID_PW_DATA_INCU_YN" : "N"
    }
    headers = {
        "authorization": f"Bearer {ACCESS_TOKEN}",
        "appKey": f"{APP_KEY}",
        "appSecret": f"{APP_SECRET}",
        "tr_id": "FHKST03010200",
        "custtype" : "P"
    }
    res = requests.get(url, params=params, headers=headers)
    if res.status_code == 200:
        resp = res.json()
        stck_cntg_hour=resp['output2'][0]['stck_cntg_hour']
        stck_oprc=resp['output2'][0]['stck_oprc']
        stck_hgpr=resp['output2'][0]['stck_hgpr']
        stck_lwpr=resp['output2'][0]['stck_lwpr']
        stck_prpr=resp['output2'][0]['stck_prpr']
        cntg_vol=resp['output2'][0]['cntg_vol']
        return (stck_cntg_hour,stck_oprc,stck_hgpr,stck_lwpr,stck_prpr,cntg_vol)
    else :
        print("(get_min_data) ERROR when call API:",res.status_code,res.text)
        exit(1)