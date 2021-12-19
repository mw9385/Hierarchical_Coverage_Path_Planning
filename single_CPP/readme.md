### Single Coverage Path Planning
#### To train the model, implement the main.py by entering the command
`python main.py`  

Detail hyperparameters can be changed in **main.py** arguments. The weights and bias of the trained model will be stored in the **model** directory.
At every 500 learning steps, the reward is stored in the tensorboard log. To see the log in real time, implement the command  
`tensorboard --logdir=YOUR LOG DIRECTORY`

The file **Attention.py** defines the pointer networks. The file **module.py** contains the attention mechanism (multi-head attention, self-attention etc.).
In the **generate**
