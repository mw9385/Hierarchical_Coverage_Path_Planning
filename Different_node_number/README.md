# Hierarchical_Coverage_Path_Planning

### To train the model, use the command `python main.py`.
**To do list:** 
- ~~Change the reward function~~: The sign of the reward function depends on the Loss function design. In wouter kool's paper, the loss function is defined with total distance
- ~~Visualize the model performance in real time~~
- ~~Need to retrain the model with a changed mask matrix~~ 
- ~~Changed the loss function; high loss function and low loss function~~
- ~~Change the value of C in the pointer network. C value controls exploration and exploitation.~~ 별 차이 없음
- __Give loss weight in high_loss value: 이렇게 하면 low model을 초기에 더 잘 학습 할 수 있을것이라 생각함.__ 변경해서 실행 중 
- High model 과 low model을 나눠서 학습을 하기 시작했는데, low model이 수렴을 안하는 듯 한데 low model update 과정에 문제가 있는건가?
- Hyperparameter of the low model is not learning anything...?
