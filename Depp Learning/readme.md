# 딥러닝기초

- A_31 -> is element in row 3, column 1
- A_24 -> is element in row 2, column 2

similar case `$ a^(i) \in \mathbb{R}^n , and i = 1,2,3, ... , m $`
- a_j^(i) is element in row i, column j
- shape of `A -> mxn`

result of outer product = mxn, nxp = mxp 
- mxn의 row, nxp의 col 을 곱해서 mxp를 채움

- Feed Forward Network : 입력에서 출력까지 한 방향으만 정보가 전달되는 신경망.
- Neural Network       : 일정한 형태를 띄는 함수들의 모음.
  - NN은 y = hθ(x)를 정의하고, 최고의 함수 근사치를 도출하는 매개 변수 θ의 값을 학습한다
- a^[l] = ReLU(W[l]a[l-1] + b[l]) -> hθ(x) = W[r]a[r-1] + b[r]

### Machine learning stages
1. Forward pass
  - 우선 자유도가 높은 척도로 무작위 학습을 진행.
2. Compute cost/loss function
  - 모델이 예측한 값 (hθ(x))과 실제 label 값 (y) 값의 차이를 구함. -> J(θ)
  - Cost Function : 거리 계산, hθ와 y값의 차이이니까.
3. Compute Gradients backward pass
  - Compute Gradients -> 미분을 통한 기울기 계산.
    - 미분 값 = 0 , critical points -> θ값이 가장 작아지는 부분.
  - 위에서 구한 값(cost function)의 미분을 통해 역전파 진행. Δ_θ (J(θ))
4. Update parameters Gradient Descent
  - 미분을 통해 계산한 기울기 값에 learning rate를 곱해 파라메타 θ를 업데이트 한다.
  - learning rate는 Gradient값을 얼마나 이동시킬지 사용자가 조절할 수 있는 하이퍼라메타이다.

### How to prevent local minima and saddle point
- Using different optimizers  (SGD, GD, Adam .. cet)
- Use smaller batch size
  - batch size가 작을수록 local minima를 빠져나오기 쉽다. 배치 사이즈가 작아서 통통 튈 수 있기 때문.
  - batch size가 클 수록 전체 샘플에 대한 cost function 비슷해 질 것. 
### Speeding up training process
- Normalize data (sample normalization, Batch Normalization)
- Regularization ( Dropout, L1 norm, L2 norm)
- Randomize initialization points (Xavier initialization) 

### train process
- train mode on 
- train model 
- training mode off
- validation check  (K fold - 모든 데이터가 val 데이터가 되는 것)
- Best model check - Yes -> save, No -> Hyper paramerter tuning

### Overfitting and Underfitting
- Underfitting : 누가봐도 학습 덜됨.. 
  - 모델 잘못쓰거나, 학습 데이터 부족, 혹은 모델의 깊이가 너무 낮다거나, unit이 적다거나, Activation func가 적다거나.. ect
- Overfitting : 학습 결과는 좋은데 새로운 데이터에 대한 예측을 못함.
  - data를 늘리거나, 모델의 깊이 이런걸 좀 낮춘다던가, Regularlizaion, Dropout 

### CrossEntropyLoss
- 입력 데이터 x에 대한 출력이 y일 확률 -> 1 and 모델이 x에 대해 입력받았을 때 출력값이 y일 확률 - y^
-  J(θ) = -1/m ∑ (1~m, i = 1) Pdata(y|x^i) log Pmodel(y|x^i; θ)
  -  입력 데이터 x^i에 대해 θ를 넣었고 예측한 모델의 예측값 y

``` python:
for e in range(epochs):
    print(f"------ Starting epoch: {e} --------")
    running_loss = 0
    running_acc = 0

    for images, labels in trainloader: # iterator를 for문으로 돌림.
        images = images.reshape(images.shape[0], -1)
    
        logits = model(images)
        loss = criterion(logits, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        ps = softmax(logits)
        pred = get_pred(ps)
        running_acc += torch.sum(pred==labels)/labels.shape[0] 
```
