import numpy as np
import matplotlib.pyplot as plt
import pickle

with open('./Data/Scalability/test_scalability_6_28.pkl','rb') as f:
  data = pickle.load(f)

print(data)
print(data['solution_cost'])
print(data['LKH_cost'])
print(data['gap'])

fields = ['solution_cost', 'LKH_cost', 'gap']

columns = [i for i in range(2,11)]
rows = ('RL', 'LKH', 'Gap')

cell_text = []

for F in fields:
  cell_text.append(['%3.3f' % i for i in data[F]])

# the_table = plt.table(cellText=cell_text,
#                       rowLabels=rows,
#                       rowColours=['white','white','cyan'],
#                       colLabels=columns,
#                       fontsize=50,
#                       loc='bottom')


# plt.subplots_adjust(left=0.2,bottom=0.2)

plt.figure(figsize=(15,6))
plt.title("Training Scalability Analysis")
plt.plot([i for i in range(2,11)], data['gap'], marker='o',color='c')
plt.legend()
plt.ylabel("cost gap(%)")
plt.xlabel("# of task per type")
plt.ylim((0.0, 6.1))
plt.xticks([])
plt.xticks(np.array([i for i in range(2,11)]))
plt.yticks(np.arange(0.0,6.0,0.5))

plt.grid()
# plt.tight_layout()
# plt.savefig('./Archive/test2.png', dpi=300)
plt.show()
