# 🍊BitaMin Project

- BitaMin, data analysis &amp; data science assosiation, 12th and 13th joint project (2024.03.06 ~ 2024.06.05.)
- Time Series and Reinforcement Learning for Stock Trading

### Team
|송규헌|송휘종|서영우|이태경|정유진|정준우|
|:---:|:---:|:---:|:---:|:---:|:---:|
|![song9-avatar](https://avatars.githubusercontent.com/u/113088511?v=4)|![shj](https://avatars.githubusercontent.com/u/170795597?v=4)|![uosstat](https://avatars.githubusercontent.com/u/101987656?v=4)|![taekyoung](https://avatars.githubusercontent.com/u/122856705?v=4)|![y8jinn](https://avatars.githubusercontent.com/u/112757321?v=4)|![junwoo](https://avatars.githubusercontent.com/u/104676353?v=4)|
|[Github](https://github.com/skier-song9)|[Github](https://github.com/songhwijong)|[Github](https://github.com/uosstat98)|[Github](https://github.com/taekyounglee1224)|[Github](https://github.com/y8jinn)|[Github](https://github.com/Junwoo2001)|

## ✅Table of Contents
- [💼Project Introduction](#Project_Introduction)
    - [Overview](#Overview)
    - [🔖Reference](#Reference)
- [🤗Environment](#Environment)
- [🦾Training](#Training)
- [🛠️Test (Backtrading)](#Test-(Backtrading))
- [🛠️Test (System trading)](#Test-(SystemTrading))
<br>


<a name='Project_Introduction'></a>
## 💼Project Introduction
<a name='Overview'></a>
#### Overview)

<h6>◾ Project Topic</h6>
Maximizing Portfolio Value by <b>Time Series Forecasting</b> and system trading using <b>Reinforcement Learning</b>.

<h6>◾ Goals</h6>
<ul style='list-style-type:decimal;'>
    <li>Use <b style='background-color: #EDF3EC;'>Time Series Forecasting</b> to predict the stock prices (high, low, maximum fluctuation rate, etc.) for the next 5 days and then select the 6 stocks with the highest (high-low) difference.</li>
    <li>Train <b style='background-color: #EDF3EC;'>Reinforcement Learning</b> on the selected stocks and implement system trading</li>
</ul>

<h6>◾Flowchart</h6>
<img src='https://github.com/skier-song9/bitamin1213_trading/blob/master/ppt/project_flowchart.png' alt='project flowchart image'>

<a name='Reference'></a>
<h4>🔖Reference</h4>
<h6>◾<a href="https://github.com/quantylab">Quantylab</a></h6>
We used baseline code of quantylab's rltrader for training model. Modified for real-time trading. 
<br/>
<br/>

<a name='Environment'></a>
## 🤗Environment

- recommend making a conda virtual environment
- Python 3.7+
- PyTorch 1.13.1

```bash
conda install python==3.7
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install pandas
conda install matplotlib
conda install -c conda-forge ta-lib
pip install -r trading_requirements.txt
```

<br/>

<a name='Training'></a>
## 🦾Training

<ul style='list-style-type:decimal;'>
    <li><code style="background-color: #EDEDEB;color: #EB7979;border-radius: 3px;padding: 0 3px;font-family: consolas;">cd rltrader/src/</code></li>
    <li>refer to <a href="https://github.com/quantylab/rltrader?tab=readme-ov-file#%EC%8B%A4%ED%96%89">quantylab</a> for detailed parameters descriptions.<br>e.g.)</li>
    <pre>
<code class='bash'>python main.py --mode train --ver v1 --name 001470_202404151325_202405241530 --stock_code 001470 --rl_method ppo --net cnn --backend pytorch --balance 500000000 --start_date 202404151325 --end_date 202405241530</code></pre>
    <li style="margin-left:30px;list-style-type:circle;"><code style="background-color: #EDEDEB;color: #EB7979;border-radius: 3px;padding: 0 3px;font-family: consolas;">--mode</code> : set 'train' when training model</li>
    <li style="margin-left:30px;list-style-type:circle;"><code style="background-color: #EDEDEB;color: #EB7979;border-radius: 3px;padding: 0 3px;font-family: consolas;">--ver</code> : leave it with 'v1' for out code</li>
    <li style="margin-left:30px;list-style-type:circle;"><code style="background-color: #EDEDEB;color: #EB7979;border-radius: 3px;padding: 0 3px;font-family: consolas;">--name</code> : set name of output directory and model file</li>
    <li style="margin-left:30px;list-style-type:circle;"><code style="background-color: #EDEDEB;color: #EB7979;border-radius: 3px;padding: 0 3px;font-family: consolas;">--start_date</code> & <code style="background-color: #EDEDEB;color: #EB7979;border-radius: 3px;padding: 0 3px;font-family: consolas;">--end_date</code> : 'year-month-day-hour-minute'(%Y%m%d%H%M) format</li>
</ul>
<p>When training procedure ends model parameters, output images and log are stored in <code style="background-color: #EDEDEB;color: #EB7979;border-radius: 3px;padding: 0 3px;font-family: consolas;">/rltrader/models/</code> and <code style="background-color: #EDEDEB;color: #EB7979;border-radius: 3px;padding: 0 3px;font-family: consolas;">/rltrader/output/</code> 
</p>


<a name='Test-(Backtrading)'></a>
## 🛠️Test (Backtrading)
<p>check the start_date and end_date of stock. start_date should be the 120 time steps before you want to start the test because of input size of CNN.<br>
e.g.) if you want to backtrade from 2024.05.27 09:01 to 2024.05.31 15:30 then you should set the start_date as 202405241322 which is 120 time steps before the 2024.05.27 09:01.</p>
<p>Also you should set --name same as the train name in order to make sure rltrader use trained model for inference.</p>
<pre>
<code class='bash'>
python main.py --mode test --ver v1 --name 001470_202404151325_202405241530 --stock_code 001470 --rl_method ppo --net cnn --backend pytorch --balance 500000000 --start_date 202405241320 --end_date 202405311530
</code></pre>


<a name='Test-(SystemTrading)'></a>
## 🛠️Test (System Trading)

<h5>setting</h5>
You should place 'api.json' in order to use KIS API for trading.
<ul style='list-style-type:decimal;'>
    <li>Make <a href="https://securities.koreainvestment.com/main/Main.jsp" target="blank">KoreaInvestment</a> account.</li>
    <li>Follow the menu ‘트레이딩’ > ‘모의투자’ > ‘주식/선물옵션 모의투자 > 모의투자안내’ > ‘신청/재도전’</li>
    <li>Set initial balance and trading period as you wish.</li>
    <li>Go to API center <a target="blank" href="https://apiportal.koreainvestment.com/intro">KIS API</a>. </li>
    <li>Apply for an API(<a href="https://securities.koreainvestment.com/main/customer/systemdown/RestAPIService.jsp" target="blank">url</a>).</li>
    <li>Set your api.json file as below and place the file under <code style="background-color: #EDEDEB;color: #EB7979;border-radius: 3px;padding: 0 3px;font-family: consolas;">/bitamin1213_trading/rltrader/src/quantylab/rltrader/</code></li>
<pre>
<code class='json'>
{
    "real_invest" : {
        "account" : "실전투자계좌번호8자리-01", 
        "app_key" : "실전투자계좌 APP Key 복사해서 붙여넣기",
        "app_secret" : "실전투자계좌 APP Secret 복사해서 붙여넣기",
        "access_token" : ""
    },
        
        "mock_invest" : {
            "account" : "모의투자계좌번호8자리-01",
            "app_key" : "모의투자계좌 APP Key 복사해서 붙여넣기",
            "app_secret" : "모의투자계좌 APP Secret 복사해서 붙여넣기",
            "access_token" : ""        
        }    
}</code></pre>
<li>Get access token by runnig utils.py</li>
<pre>
<code class='bash'>
cd /bitamin1213_trading/rltrader/src/quantylab/rltrader
python utils.py
</pre></code>
</ul>
<h5>system trading</h5>
For real-time system trading, use 'predict' keyword. Setting start_date is same as backtrading, 120 time steps before.<br>
You should run system trading code day by day. 'predict' method is just for a day trading.
<pre>
<code class='bash'>
python main.py --mode predict --ver v1 --name 001470_202404151325_202405241530 --stock_code 001470 --rl_method ppo --net cnn --backend pytorch --balance 500000000 --start_date 202405241320 --end_date 202405281530 --is_start_end 2 --is_mock 1
</code></pre>
<ul style='list-style-type:circle;'>
<li><code style="background-color: #EDEDEB;color: #EB7979;border-radius: 3px;padding: 0 3px;font-family: consolas;">--is_start_end</code> : 0 for Monday, 1 for Tuesday~Thursday, 2 for Friday.</li>
<li><code style="background-color: #EDEDEB;color: #EB7979;border-radius: 3px;padding: 0 3px;font-family: consolas;">--is_mock</code> : 0 for real investment, 1 for mock investment.</li>
</ul>
Access_token is revoked after the system trading code ends. So you should get access_token before running predict code everyday.
