### Single Coverage Path Planning
#### Training
To train the model, implement the main.py by entering the command:  
`python main.py`  

#### Hyperparameter Tuning
Detail hyperparameters can be changed in **main.py** arguments. The weights and bias of the trained model will be stored in the **model** directory.
At every 500 learning steps, the reward is stored in the tensorboard log. To see the log in real time, implement the command:  
`tensorboard --logdir=YOUR LOG DIRECTORY`

#### Define Model
The file **Attention.py** defines the pointer networks. The file **module.py** contains the attention mechanism (multi-head attention, self-attention etc.). In the mean of attention mechansim, the action masking needs to be updated. The updates have been done in the **environment.py**.

#### Data generation
In the **generate_data.py** you could generate training and testing dataset. The points are randomly sampled from the distribution with zero mean and one standard deviation. 

#### Testing
In the **test.py** you could implement the trained code by running the command:  
`python test.py`
