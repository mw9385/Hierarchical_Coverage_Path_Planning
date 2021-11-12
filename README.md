# Hierarchical_Coverage_Path_Planning

### To train the model, use the command `python main.py`.
**To do list:** 
- ~~Change the reward function~~: The sign of the reward function depends on the Loss function design. In wouter kool's paper, the loss function is defined with total distance
- ~~Visualize the model performance in real time~~
- ~~Need to retrain the model with a changed mask matrix~~ 
- ~~Changed the loss function; high loss function and low loss function~~
- ~~Change the value of C in the pointer network. C value controls exploration and exploitation.~~ 별 차이 없음
- __Give loss weight in high_loss value: 이렇게 하면 low model을 초기에 더 잘 학습 할 수 있을것이라 생각함.__ 변경해서 실행 중 
- __High model 과 low model을 나눠서 학습을 하기 시작했는데, low model이 수렴을 안하는 듯 한데 low model update 과정에 문제가 있는건가?__ Low model 쪽에 문제가 있었음
- Number of nodes를 고정해서 학습하니 속도도 빨라지고 디버깅도 용이해졌음

### Glimpse 적용 효과
- Glimpse를 적용한 모델 (V1)의 성능이 적용하지 않은 모델 (V3) 보다 더 성능이 떨어져 보인다
- Glimpse를 적용하지 않고 node의 수를 줄이거나 늘였을 때 성능이 어떻게 나오는지 확인해 보자
- n_node: 512 개로 적용해서 학습한 모델 (V2)
- n_node: 128 개로 적용해서 학습한 모델 (V1)
- n_node: 256 개로 적용해서 학습한 모델 (V4)
