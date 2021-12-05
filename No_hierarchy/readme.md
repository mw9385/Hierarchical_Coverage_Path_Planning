## code with no hierarchies to check feasibility

학습하는 부분의 코드는 문제가 없었지만, 시각화 하는 부분에서 문제가 발생했음  
계층형으로 학습하는 경우 새로운 진입점을 어디로 할것인지가 중요함 --> 지금 내가 작성한 코드로는 부족함  
high policy의 input으로 low policy의 마지막점에 대한 정보와 cell 정보를 종합적으로 주어야 적절한 입구 위치를 찾을 수 있을것이라 생각됨.  
high policy input: mean value만 들어가는데 이걸 low policy 정보를 넣을 수 없을까..?

local policy 입력값: [이전 cell의 마지막 노드, 현재 cell 정보, 다음 cell의 처음 node 정보]
# local policy의 입력값으로 $ p_{k}^{c_{t}}, p_{1:k}^{c_{t+1}}, p_{1}^{c_{t+2}} $
