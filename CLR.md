Cyclic Learning Rate 소개
===

## 1. Intro - Learning Rate의 이해

Learning rate란 현재의 가중치에 loss 기울기를 적용하여 loss가 적은 방향으로 이동

`new_weight = existing_weight — learning_rate * gradient`

_Coursera Andrew Ng 강의에서의 Learning Rate 관련 장표_
<img align="center" src="https://cdn-images-1.medium.com/max/1600/0*00BrbBeDrFOjocpK.">

전통적으로 training중 Learning rate를 설정하는 데에는 아래와 같은 두방법이 있다.

#### 모든 파라미티에 대해서 공통 Learning Rate를 적용하는 경우
일반적으로 SGD에서 볼수 있듯이 훈련 시작시 단일 Learning Rate를 설정하고 어떻게 감소시킬것인지 설정합니다. (step, exponential 등).

여기서 기본 가정은 시간이 지나면 원하는 minimum에 도달할 것이라는 가정을 깔고 있습니다. 
단일 Learning rate는 모든 파라미터를 업데이트 하는 데 사용됩니다.

그러나, 이 방식에는 많은 어려움이 있습니다. 

<img align="center" src="https://cdn-images-1.medium.com/max/1600/0*uIa_Dz3czXO5iWyI.">

[**단일 Learning rate 설정의 문제점**]
- 초기 Learning rate를 미리 설정하기 어렵다
- LR Schedule (시간 경과에 따라 감소하는 LR갱신 매커니즘)도 사전 설정이 어렵다
- 데이터에 따라 동적으로 변하지 못한다
- Saddle point에서 벗어나기 어렵다.

#### 각각의 파라미터에 대하여 개별 Learning Rate를 적용하는 경우
기존의 optimizer 계보
<img src="https://image.slidesharecdn.com/random-170910154045/95/-49-638.jpg?cb=1505089848">

<img src="http://teleported.in/post_imgs/15-Beale.gif">

최적화 알고리즘을 비교한 그림


----
## 2.Cyclic Learning Rate 

2015년 Leslie Smith가 제안했습니다.
Learning rate조정에 대한 접근 방식 값이 하한선과 상한선 사이에서 순환합니다.
SGD와 함께 쓰여 위에 있는 다른 optimizer대용으로 사용할 수도 있습니다.
그러나, 매개 변수 업데이트 마다 optimizer와 combine해서 사용할 수 있습니다.

[lr_scheduler] http://pytorch.org/docs/0.3.1/optim.html#how-to-adjust-learning-rate

### 장점

1. 주기적으로 learning rate를 키워줍니다.
+ 직관: 시간에 따라 learning rate를 감소해야 할 것 같음
+ 실제: higher lower 임계치를 두고 주기적으로 learning rate를 바꾸는 것이 유용함 

이유: 주기적으로 learning rate가 높아지면 모델이 local minima나 saddle point에서 빠져나올 수 있습니다.
반대의 경우 빠져 나오기 힘듭니다.

2. 최적의 learning rate를 시간이 지남에 따라 알게 된다.

### Epoch, Iterations, Cycles, Stepsize와의 관계
한 사이클 동안 학습 속도가 base_lr 속도에서 max_lr 속도로 바뀌고 다시 되돌아 가는 것을 반복하는 것으로 정의합니다.

stepsize는 주기의 절반입니다. 

이 경우 cycle이 반드시 epoch 경계에 속할 필요는 없습니다.

<img src="http://teleported.in/post_imgs/15-clr-triangle.png">

### Learning rate 계산방식

```python
def get_triangular_lr(iteration, stepsize, base_lr, max_lr):
    """Given the inputs, calculates the lr that should be applicable for this iteration"""
    cycle = np.floor(1 + iteration/(2  * stepsize))
    x = np.abs(iteration/stepsize - 2 * cycle + 1)
    lr = base_lr + (max_lr - base_lr) * np.maximum(0, (1-x))
    return lr

# Demo of how the LR varies with iterations
num_iterations = 10000
stepsize = 1000
base_lr = 0.0001
max_lr = 0.001
lr_trend = list()

for iteration in range(num_iterations):
    lr = get_triangular_lr(iteration, stepsize, base_lr, max_lr)
    # Update your optimizer to use this learning rate in this iteration
    lr_trend.append(lr)

plt.plot(lr_trend)
```

### base_lr, max_lr 최적화 하기

## 3. 변형 CLR

<img src="http://teleported.in/post_imgs/15-triangular2.png">
<img src="http://teleported.in/post_imgs/15-exp_range.png">

## 4. 얼마나 좋아지는가

<img src="http://teleported.in/post_imgs/15-clr-cifar10.png">
<img src="http://teleported.in/post_imgs/15-clr-adam.png">

## 5. 결론
Cyclic Learning Rate는 학습 속도를 관리하는 새로운 기술입니다.
SGD또는 optimizer 대용 또는 함께 combine하여 사용할 수 있습니다.


