# 🍊BitaMin Project

- BitaMin, data analysis &amp; data science assosiation, 12th and 13th joint project (2024.03.06 ~ 2024.).
- Time Series and Reinforcement Learning for Stock Trading

## ✅Table of Contents
- [💼Project Introduction](#Project_Introduction)
    - [Overview](#Overview)
    - [🔖Reference](#Reference)
- [🤗Environment](#Environment)
- [🦾Training](#Training)
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
    <li>get api key from <a href="https://apiportal.koreainvestment.com/">Korea Investment</a> and place it under <code style="background-color: #EDEDEB;color: #EB7979;border-radius: 3px;padding: 0 3px;font-family: consolas;">/rltrader/src/quantylab/</code></li>
    <pre>
<code class='json'>{
    "real_invest" : {
        "account" : "",
        "app_key" : "",
        "app_secret" : "",
        "access_token" : ""
    },
    "mock_invest" : {
        "account" : "",
        "app_key" : "",
        "app_secret" : "",
        "access_token" : ""
    }
}</code></pre>
    <li><code style="background-color: #EDEDEB;color: #EB7979;border-radius: 3px;padding: 0 3px;font-family: consolas;">cd rltrader/src/</code></li>
    <li>refer to <a href="https://github.com/quantylab/rltrader?tab=readme-ov-file#%EC%8B%A4%ED%96%89">quantylab</a> for detailed parameters descriptions.<br>e.g.)</li>
    <pre>
<code class='bash'>python main.py --mode train --ver v1 --name 002870_0110_0524 --stock_code 002870 --rl_method a2c --net cnn --backend pytorch --balance 500000000 --start_date 202401101132 --end_date 202405241530</code></pre>
    <li style="margin-left:30px;list-style-type:circle;"><code style="background-color: #EDEDEB;color: #EB7979;border-radius: 3px;padding: 0 3px;font-family: consolas;">--mode</code> : set 'train' when training model</li>
    <li style="margin-left:30px;list-style-type:circle;"><code style="background-color: #EDEDEB;color: #EB7979;border-radius: 3px;padding: 0 3px;font-family: consolas;">--ver</code> : leave it with 'v1' for out code</li>
    <li style="margin-left:30px;list-style-type:circle;"><code style="background-color: #EDEDEB;color: #EB7979;border-radius: 3px;padding: 0 3px;font-family: consolas;">--name</code> : set name of output directory and model file</li>
    <li style="margin-left:30px;list-style-type:circle;"><code style="background-color: #EDEDEB;color: #EB7979;border-radius: 3px;padding: 0 3px;font-family: consolas;">--start_date</code> & <code style="background-color: #EDEDEB;color: #EB7979;border-radius: 3px;padding: 0 3px;font-family: consolas;">--end_date</code> : 'year-month-day-hour-minute'(%Y%m%d%H%M) format</li>
</ul>
<p>When training procedure ends model parameters, output images and log are stored in <code style="background-color: #EDEDEB;color: #EB7979;border-radius: 3px;padding: 0 3px;font-family: consolas;">/rltrader/models/</code> and <code style="background-color: #EDEDEB;color: #EB7979;border-radius: 3px;padding: 0 3px;font-family: consolas;">/rltrader/output/</code> 
</p>


