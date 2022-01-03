import numpy as np
import pickle
import matplotlib.pyplot as plt

class Node:
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position

# def heuristic(node, goal, D=1, D2=2 ** 0.5):  # Diagonal Distance
#     dx = abs(node.position[0] - goal.position[0])
#     dy = abs(node.position[1] - goal.position[1])
#     return D * (dx + dy) + (D2 - 2 * D) * min(dx, dy)

def aStar(maze, start, end):
    # startNode와 endNode 초기화    
    startNode = Node(None, start)
    endNode = Node(None, end)

    # openList, closedList 초기화
    openList = []
    closedList = []

    # openList에 시작 노드 추가
    openList.append(startNode)

    # endNode를 찾을 때까지 실행
    while openList:

        # 현재 노드 지정
        currentNode = openList[0]
        currentIdx = 0

        # 이미 같은 노드가 openList에 있고, f 값이 더 크면
        # currentNode를 openList안에 있는 값으로 교체
        for index, item in enumerate(openList):
            if item.f < currentNode.f:
                currentNode = item
                currentIdx = index

        # openList에서 제거하고 closedList에 추가
        openList.pop(currentIdx)
        closedList.append(currentNode)

        # 현재 노드가 목적지면 current.position 추가하고
        # current의 부모로 이동
        if currentNode == endNode:
            path = []
            current = currentNode
            while current is not None:
                # maze 길을 표시하려면 주석 해제
                # x, y = current.position
                # maze[x][y] = 7 
                path.append(current.position)
                current = current.parent
            return path[::-1]  # reverse

        children = []
        # 인접한 xy좌표 전부
        for newPosition in [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]:

            # 노드 위치 업데이트
            nodePosition = (
                currentNode.position[0] + newPosition[0],  # X
                currentNode.position[1] + newPosition[1])  # Y
                
            # 미로 maze index 범위 안에 있어야함
            within_range_criteria = [
                nodePosition[0] > (len(maze) - 1),
                nodePosition[0] < 0,
                nodePosition[1] > (len(maze[len(maze) - 1]) - 1),
                nodePosition[1] < 0,
            ]

            if any(within_range_criteria):  # 하나라도 true면 범위 밖임
                continue

            # 장애물이 있으면 다른 위치 불러오기
            if maze[nodePosition[0]][nodePosition[1]] != 0:
                continue

            new_node = Node(currentNode, nodePosition)
            children.append(new_node)

        # 자식들 모두 loop
        for child in children:

            # 자식이 closedList에 있으면 continue
            if child in closedList:
                continue

            # f, g, h값 업데이트
            child.g = currentNode.g + 1
            child.h = ((child.position[0] - endNode.position[0]) **
                       2) + ((child.position[1] - endNode.position[1]) ** 2)
            child.f = child.g + child.h

            # 자식이 openList에 있으고, g값이 더 크면 continue
            if len([openNode for openNode in openList
                    if child == openNode and child.g > openNode.g]) > 0:
                continue
                    
            openList.append(child)

# def main():
#     # load map and convert them into a list
#     map, _, _ = pickle.load(open('./Decomposed data/decomposed_0', 'rb'))
#     point, cost = pickle.load(open('./Points and costs/PNC_0', 'rb'))
#     _temp_start = point[0, 0, :, 0].tolist()
#     _temp_end = point[5, 0 ,: , 1].tolist()
#     start = (int(_temp_start[0]), int(_temp_start[1]))
#     end = (int(_temp_end[0]), int(_temp_end[1]))
    
#     # convert them into a binary image
#     decomposed_image = np.ones([70, 70, 3], dtype = np.uint8)
#     decomposed_image[map > 0, :] = [255, 255, 255] # white
#     decomposed_image[decomposed_image>127] = 0
#     if len(decomposed_image.shape) > 2:
#         decomposed_image = decomposed_image[:, :, 0]
#     plt.figure()
#     plt.imshow(decomposed_image)
#     plt.show()
#     decomposed_image = decomposed_image.tolist()  
#     for _ in range(100): 
#         path = aStar(decomposed_image, start, end)    
#     print(len(path)) # compile에 대략 0.05초 걸리네 

# if __name__ == '__main__':
#     main()