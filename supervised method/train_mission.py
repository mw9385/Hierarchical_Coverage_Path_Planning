import argparse
import numpy as np
import random
import pickle
import torch
import torch.optim as optim
import torch.nn.functional as F


from torch.utils.data import DataLoader
from solver import solver_RNN, HetNet_solver
from Environment import Mission
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str, default="ptrnet")
parser.add_argument("--coverage_num", type=int, default=3)
parser.add_argument("--visiting_num", type=int, default=3)
parser.add_argument("--pick_place_num", type=int, default=3)
parser.add_argument("--num_epochs", type=int, default=1000)
parser.add_argument("--num_tr_dataset", type=int, default=12800)
parser.add_argument("--num_te_dataset", type=int, default=256)
parser.add_argument("--embedding_size", type=int, default=128)
parser.add_argument("--hidden_size", type=int, default=128)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--grad_clip", type=float, default=1.5)
parser.add_argument("--beta", type=float, default=0.9)
parser.add_argument("--use_cuda", type=bool, default=True)
parser.add_argument("--log_interval", type=int, default=100)
parser.add_argument("--log_dir", type=str, default='./logs/V3')
parser.add_argument("--model_dir", type=str, default='./models/V3')
args = parser.parse_args()

writer = SummaryWriter(log_dir=args.log_dir)

if __name__ =="__main__":
    if args.use_cuda:
        use_pin_memory = True
    else:
        use_pin_memory = False

    torch.manual_seed(100)
    torch.cuda.manual_seed(100)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    np.random.seed(100)
    random.seed(100)

    print("LOAD TRAINING DATASET")
    with open("input_points", 'rb') as f:
        train_x = pickle.load(f)

    with open("output_solutions", 'rb') as f:
        train_y = pickle.load(f)


    train_dataset = Mission.PointDataset(train_x=train_x, train_y=train_y)
    train_data_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=use_pin_memory)

    if args.model_type == "ptrnet":
        tensorboard_path = 'runs/'+"PtrNet_C%d_V%d_D%d" %(args.coverage_num, args.visiting_num, args.pick_place_num)
        print("RNN model is used")
        model = solver_RNN(
            args.embedding_size,
            args.hidden_size,
            args.visiting_num+args.coverage_num+args.pick_place_num+1,
            2, 10)
    # AM network
    elif args.model_type.startswith("hetnet"):
        tensorboard_path = 'runs/'+"HetNet_C%d_V%d_D%d" %(args.coverage_num, args.visiting_num, args.pick_place_num)
        print("HetNet is used")
        model = HetNet_solver(
            args.embedding_size,
            args.hidden_size,
            args.visiting_num+args.coverage_num+args.pick_place_num+1,
            2, 10)
    else:
        raise
    if args.use_cuda:
        model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=1.0 * 1e-4)

    print("TRAINING START")
    batch_index = np.random.randint(0, 10000, args.batch_size)
    model.train()
    steps = 2500
    global_count = 0
    for epoch in tqdm(range(args.num_epochs)):         
        for batch_idx, (sample_x, sample_y) in enumerate(train_data_loader):
            if global_count % 100 == 0:
                print(global_count) 

            if args.use_cuda:
                sample_x = sample_x.cuda()
                sample_y = sample_y.cuda()                
            
            # get estimated outputs
            # task가 다른 시점에 종료되어 seq_length가 달라져 가장 짧은 sequence length로 맞춰줌
            # raw_probs, probs, actions, R
            log_probs, probs , actions, _, logits = model(sample_x)
            if log_probs.size(1) >= 11:
                logits = torch.softmax(logits[:, :11, :], dim=-1) # log_probs = [batch_size, task_length, num_nodes]
                outputs = torch.reshape(logits, [-1, 10])

                # log_probs = log_probs[:, :11, :] # log_probs = [batch_size, task_length, num_nodes]
                # outputs = torch.reshape(log_probs, [-1, 10])
                targets = sample_y.clone()
                targets = torch.reshape(targets, [-1])                        

                # define loss and optimizer
                loss = F.cross_entropy(outputs, targets.long())            
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                global_count +=1
                
                if global_count % args.log_interval == 0:
                    # save log history
                    print("Save Model")
                    writer.add_scalar('loss', loss, global_count)

                    # save model
                    dir_name = args.model_dir
                    param_path = dir_name + "/" + 'pointerNet' + '.param'
                    torch.save(model.state_dict(), param_path)