# Data-Analysis-Capstone
## 공공 데이터를 이용한 농산물 가격 예측 시스템 개발
소프트웨어융합학과 17학번 강성문
> Library : XGBoost, LightGBM, Scikit-learn, Numpy, Pandas, Keras, etc 
> 
> Environment : Windows 10, Python 3.8
## 1. Overview
### 1.1. 과제 선정 배경
- 인류의 생존에 필수적인 농산물은 우리와 떼어 놓을 수 없는 존재이며, 농산물의 가격은 소비하는 개인과 판매하는 농업인, 정부 모두에게 매우 중요하다. 특히, 코로나 사태가 지속되면서 부족한 농작물에 대한 수입이 어려워지면서 농산물 가격이 급등하기도 한다. 이에 따라, 기존의 데이터를 이용하여 미래의 농산물 가격을 예측하는 것은 정부의 농산물 수급 정책 수립에도 중요한 역할을 할 것이며 소비자와 농업인에게도 중요한 지표가 될 것이다.
### 1.2. 과제 주요 내용
- 농산물의 가격은 단편적으로 해석, 추측하기 어려운 경향이 있다. 가격에 영향을 미치는 요인은 기상, 가격, 거래량, 물가 상승률, 유가, 수출입량, 재배 면적 등 다양하지만, 그 중에서도 가장 메인이 되는 기상, 가격, 거래량 데이터를 이용한다. 해당 데이터들을 수집하여 전처리, 이상치 탐지, 정규화 등의 과정을 거친 뒤, 트리기반 기법의 머신러닝 모델 및 LSTM 딥러닝 모델들을 이용해 학습시킨다. 농산물의 가격은 당일의 기상 데이터와 거래량 데이터가 아닌 그 이전 4주간의 거래량, 가격, 기상 데이터를 바탕으로 예측된다.
- 이번 프로젝트에서는 배추, 무, 양파, 건고추, 마늘, 대파, 얼갈이배추, 양배추, 깻잎, 시금치, 미나리, 당근, 파프리카, 새송이, 팽이버섯, 토마토, 청상추, 백다다기, 애호박, 캠벨얼리, 샤인마스캇 총 21개의 작물에 대하여 가격 예측을 진행한다. 
### 1.3. 과제 목표
- 입력받은 일자의 1주 뒤, 2주 뒤, 4주 뒤의 가격을 예측하는 것을 목표로 한다. 예를 들어, 2021년 12월 31일의 데이터를 입력으로 한다면, 1월 7일, 1월 14일, 1월 28일의 가격을 수치형으로 예측한다. 모델의 성능 평가 지표는 농산물마다 가격 범위 차이가 큰 점을 반영하여 NMAE(Normalized Mean Absolute Error)를 사용하며, 해당 값이 0.2가 넘지 않게 하는 것을 목표로 한다.

--------------------------
## 2. Dataset
### 2.1. 데이터 수집
이번 프로젝트에서는 2016년 1월 1일부터 2020년 9월 28일까지의 데이터를 학습용 데이터셋으로 수집하며, 2020년 9월 29일 + 1week부터 2020년 11월 5일 + 4week 까지를 테스트 데이터셋으로 수집한다. 다만, 기상 데이터는 필요에 따라 지난 2015년의 데이터도 추가 수집한다.
#### 2.1.1. 기상 데이터 수집
- 공공데이터 포털에서 API Key를 발급받아 데이터 수집을 진행한다.
- 농촌진흥청 국립농업과학원 농업기상 데이터: https://www.data.go.kr/data/15078057/openapi.do
![image](https://user-images.githubusercontent.com/65675861/145555384-d6e2f853-0525-4fbf-b970-04dd859cd3a5.png)

#### 2.1.2. 농산물 거래량 및 가격 데이터 수집
- 농넷에서 API Key를 발급받아 전국도매시장 거래정보 데이터를 수집한다.  
- 농넷 | 농산물유통종합정보시스템: https://nongnet.or.kr
--------------------------
### 2.2. 데이터 전처리
#### 2.2.1. 농산물별 주산지 추출
- 기상 데이터를 활용하기에 앞서 각 농산물별로 생산량이 가장 많은 주산지를 선정하고, 해당 지역의 기상 정보를 메인으로 사용한다.
- 생산량은 앞서 수집한 농산물 거래정보 데이터에서 2019년의 1년치 생산량을 통해 정렬한다.
- 주산지를 구한 뒤, 카카오맵 API를 이용하여 해당 주산지의 위도와 경도를 구하고 이에 맞는 기상 관측 지점을 찾는다. 
####
![image](https://user-images.githubusercontent.com/65675861/145563877-4a630869-e6b5-40e9-8351-ab9865634201.png)



###

#### 2.2.2. 가격 및 거래량 데이터 전처리
- 앞서 수집한 거래정보 데이터에서 일자별 전국 농산물 평균 가격과 전국 거래량의 합을 추출한다.

![image](https://user-images.githubusercontent.com/65675861/145555630-0d7488fe-de95-4122-8b9e-3b8327dffb2a.png)

###
#### 2.2.3. 농산물별 전처리
- 모델 학습을 위한 데이터셋을 생성하기 위해 농산물별로 1일전부터 28일전까지의 거래량과 가격, 기상 데이터를 시계열 데이터 처리에 적합한 형태의 데이터프레임으로 만든다.
- 앞서 추출했던 농산물별 주산지 기상 관측 지점을 이용하여 농산물별로 다른 위치의 기상 데이터를 받아온다.
- 다음은 배추의 전처리 예시이다.

![image](https://user-images.githubusercontent.com/65675861/145556300-202ad95a-df25-4478-8145-b040c26dec13.png)
###
#### 2.2.4. Feature 추가 및 제거
- 피어슨 상관계수 분석을 진행하여 타겟값과 연관성이 0.02이하로 떨어지는 특성 제거 -> 풍속, 풍향, 토양 온도 
- 농산물 가격은 요일의 영향을 많이 받는 것으로 분석 (ex. 지난 월요일과 이번주 월요일의 가격 연관 多) -> 요일 특성 추가
- 당근과 건고추의 경우, 중국의 생산량이 대부분을 차지하여 국내 기상 데이터 사용 X
-----------------------------
### 2.3. 이상치 탐지 및 데이터 스케일링
이상치 탐지 및 데이터 정규화는 이상치 인스턴스로 인해 모델의 예측 성능이 저하되는 것을 방지하기 위해서 진행한다. 하지만, XGBoost와 LightGBM의 트리 기반 앙상블 기법 모델들의 경우, 이상치와 정규화의 큰 영향을 받지 않는 것이 일반적이므로 해당 과정은 두 모델에 대해서는 진행하지 않는다.
#### 2.3.1. 이상치 탐지
- 이상치를 탐지 및 제거하지 않고 모델링을 진행하면, 해당 이상치로 인해 모델의 성능이 저하될 수 있으며 MinMaxScaling 혹은 StandardScaling을 진행할 때에도 악영향을 미친다. 
- 해당 프로젝트에서는 IQR 이상치 제거 방식을 채택한다.
- 사분위 개념에서 75% 지점과 25% 지점의 차이를 IQR이라고 하며, 해당 값에 1.5를 곱하여 75% 지점의 값에 더하면 최대값, 25% 지점에서 빼면 최소값으로 결정하고, 이 둘의 범위를 벗어나는 것을 이상치로 판정하고 제거한다. 
####
![image](https://user-images.githubusercontent.com/65675861/145557560-282e316d-e6d8-4df4-8b7e-3f9f50e68b7d.png)

#### 2.3.2. 데이터 스케일링
- 이상치를 제거했다면, MinMaxScaling 혹은 StandardScaling을 통해 서로 범주가 천차만별인 특성들을 비슷하게 만들어 준다.
- 해당 프로젝트에서는 두 방법 모두 적용하지만, 주로 MinMaxScaling의 성능이 더 좋게 나타난다.
---------------------------
## 3. Model
> 모델 훈련에서는 마지막 28일의 데이터를 Validation Set으로 설정하고, 나머지 전체 데이터를 Train Set으로 설정하고 진행한다.
### 3.1. XGBoost
![image](https://user-images.githubusercontent.com/65675861/145561136-388750c1-c40c-4304-ab9e-e13471a83cef.png)
####
- GBM과 같은 Decision Tree 기반의 앙상블 모형
- 시스템 최적화와 알고리즘으로 정형 데이터에서 뛰어난 성능을 보이는 모델 중 하나

``` python
'learning_rate': 0.01, 
'max_depth': 6,
'booster': 'gbtree', 
'objective': 'reg:squarederror', 
'max_leaves': 100,
'colsample_bytree': 0.8,
'subsample': 0.8,
'num_boost_round' : 10000, #early_stopping_rounds 설정되어 있으므로 크게 설정
'early_stopping_rounds' = 80, 
'seed':42
```
---------------
### 3.2. LightGBM
![image](https://user-images.githubusercontent.com/65675861/145565361-27bc2075-920b-437a-9585-fe451ef07f9b.png)
####
- Gradient-based One-Side Sampling(GOSS)를 메인기술로 가중치가 작은 개체에 승수를 적용하여 데이터를 증폭
- Leaf-wise 방식을 채택하여 시간과 메모리 측면에서 XGBoost에 비해 효율적
``` python
'learning_rate': 0.01,
'max_depth': 6, 
'boosting': 'gbdt', 
'objective': 'regression',  
'is_training_metric': True, 
'num_leaves': 100, 
'feature_fraction': 0.8, 
'bagging_fraction': 0.8, 
'bagging_freq': 5, 
'seed': 42,
'num_threads': 8
```
-----------------------
### 3.3. LSTM(Long Short-Term Memory)
![image](https://user-images.githubusercontent.com/65675861/145566421-4b91e92f-1843-4565-9093-3b3bcf460cc7.png)
####
- 긴 의존 기간을 필요로 하는 학습 수행 능력을 갖춘 모델
- RNN과 유사하지만, Neural Network Layer 1개의 층 대신에 4개의 layer 존재
- forget gate layer에서는 0과 1사이의 값을 전달받아 어떠한 정보를 잊어버릴지, 보존할지 결정
> loss = mean_absolute_error 
>
> optimizer = SGD
>
> patience = 100
>
> batch_size = 16
> 
> epoch = 10000 # patience 설정되어 있으므로 크게 설정
``` python
model_dict[f'{pum}_model_{week_num}'] = Sequential()
model_dict[f'{pum}_model_{week_num}'].add(LSTM(16, 
               input_shape=(train_feature.shape[1],train_feature.shape[2]), 
               activation='relu', 
               return_sequences=False)
          )
model_dict[f'{pum}_model_{week_num}'].add(Dense(8))
model_dict[f'{pum}_model_{week_num}'].add(Dense(1)) # output -> 1
```
--------------
### 3.4. Ridge & Lasso & ElasticNet
![image](https://user-images.githubusercontent.com/65675861/145705456-857acbc2-e4e5-4762-b6f8-426489add07d.png)
####
![image](https://user-images.githubusercontent.com/65675861/145705531-ef3ca9d1-0d5a-469b-a1c5-475f49e9ccfa.png)
####
![image](https://user-images.githubusercontent.com/65675861/145705549-a1a80b24-a86b-4af7-8161-1877892be430.png)

####
- RSS(Residual Sum of Squares)를 최소화하는 Linear Model에 추가로 L1 & L2 peanlty 부여
- GridSearchCV를 통해 모델별 최적의 파라미터 탐색 진행
``` python
# Ridge
parameters = {'alpha':np.logspace(-4, 0, 4)}
model_dict[f'{pum}_model_{week_num}'] = GridSearchCV(Ridge(), parameters, scoring='neg_mean_absolute_error',cv=10)

# Lasso
parameters = {'alpha':np.logspace(-4, 0, 4)}
model_dict[f'{pum}_model_{week_num}'] = GridSearchCV(Lasso(), parameters, scoring='neg_mean_absolute_error',cv=10)

# ElasticNet
parameters = {'alpha':np.logspace(-4, 0, 4), "l1_ratio" : np.arange(0.0,1.0,0.1)}
model_dict[f'{pum}_model_{week_num}'] = GridSearchCV(ElasticNet(), parameters, scoring='neg_mean_absolute_error',cv=10)
```
--------------
## 4. Result
|Model|Details|NMAE|Note|
|----|-----|---|----|
|LinearRegression|Default|0.297||
|**XGBoost**|**Tuned Hyperparameters**|**0.168**||
|**LigthGBM**|**Tuned Hyperparameters**|**0.184**||
|**LSTM**|**Num of Hidden Layers : 1**|**0.219**|**SGD / batch_size = 16**|
|LSTM|Num of Hidden Layers : 1|0.235|Adam / batch_size = 16|
|LSTM|Stacking LSTM|0.283|SGD / batch_size = 16|
|LSTM|Num of Hidden Layers : 2|0.267|SGD / batch_size = 16|
|Ridge|Default|0.295||
|Ridge|GridSearchCV|0.297|alpha : np.logspace(-4,0,4)|
|Lasso|Default|0.287||
|Lasso|GridSearchCV|0.286|alpha : np.logspace(-4,0,4)|
|ElasticNet|Default|0.305||
|ElasticNet|GridSearchCV|0.285|alpha : np.logspace(-4,0,4) / l1_ratio : np.arange(0.0,1.0,0.1) |

---------------------
## 5. Conclusion
### 5.1. 결론 및 제언
- 기상 관측 데이터와 농산물 거래 정보 데이터를 이용하여 21개의 작물에 대하여 농산물 가격 예측을 진행했다. 데이터 전처리와 모델별 파라미터 튜닝 등의 과정을 거쳤고, 정형 데이터 처리에서 성능이 좋다고 평가받는 XGBoost와 LightGBM의 트리기반 모델들이 가장 성능이 좋음을 확인했다. 
- 다만, 21개 전체 작물에 대한 도메인적 지식이 전반적으로 부족했기에 특정 모델에 대한 세부적인 데이터 전처리는 진행하지 못한 것이 성능 향상에 어려움을 발생시켰다고 생각한다. 예를 들어, 특정 작물은 60일간의 생육 기간을 거치는데, 이번 프로젝트에서는 전체 모델에 대해 28일간의 기상 데이터를 사용했다. 따라서, 작물별로 세부 사항을 반영하여 데이터셋과 모델에 적용한다면 더 좋은 성능의 모델을 개발할 수 있을 것이다. 
- 마지막으로, 시계열 데이터 처리에 가장 많이 사용되는 모델 중 하나인 LSTM 모델의 성능이 생각보다 좋지 못했는데 그 이유와 성능 개선 방안을 모색하면서 프로젝트를 마무리 짓고자 한다.

### 5.2. 활용 방안
- 표적인 활용방안으로 정부의 농산물 수급 정책을 수립하는데 중요한 지표로서 작용할 것이다. 만약, 농산물 가격이 급등하는 것을 예측한다면 해당 농산물의 수입 비중을 늘려 가격 상승을 방지할 수 있을 것이며, 반대로 급락하는 경우에는 수입 비중을 줄여 국내 농업인들에게 경제적 타격이 가지 않도록 대비할 수 있을 것이다.
- 이를 기관에서 활용하는 경우, 유가와 수출입량 및 물가 상승률 등을 복합적으로 고려하여 특정 작물 모델링에 집중한다면 더욱 성능이 좋은 모델 수립이 가능할 것으로 보인다.

---------------
## 6. Reference 
신성호 · 이미경 · 송사광, 「LSTM 네트워크를 활용한 농산물 가격 예측 모델」,『한국콘텐츠학회논문지』 18(11), 한국콘텐츠학회, 2018, pp.416~429. https://doi.org/10.5392/JKCA.2018.18.11.416
