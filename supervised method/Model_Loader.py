import argparse
import json

import torch

from solver import solver_RNN, HetNet_solver, Beta_solver_Attention

parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str, default="ptrnet")
parser.add_argument("--coverage_num", type=int, default=3)
parser.add_argument("--visiting_num", type=int, default=3)
parser.add_argument("--pick_place_num", type=int, default=3)
parser.add_argument("--num_epochs", type=int, default=100)
parser.add_argument("--num_tr_dataset", type=int, default=12800)
parser.add_argument("--num_te_dataset", type=int, default=256)
parser.add_argument("--embedding_size", type=int, default=128)
parser.add_argument("--hidden_size", type=int, default=128)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--grad_clip", type=float, default=1.5)
parser.add_argument("--beta", type=float, default=0.9)
parser.add_argument("--use_cuda", type=bool, default=True)
parser.add_argument("--log_interval", type=int, default=100)
parser.add_argument("--log_dir", type=str, default='./logs/V1')
parser.add_argument("--model_dir", type=str, default='./models/V1')
args = parser.parse_args()

def load_model_eval(param_path, config_path=None):
    if config_path is None:
        config_path = param_path[:-6]+'.config'

    print("LOADING MODEL================")

    if args.model_type == "ptrnet":
            print("Pointer network is used")
            model = solver_RNN(
                args.embedding_size,
                args.hidden_size,
                args.visiting_num+args.coverage_num+args.pick_place_num+1,
                2, 10)
    # AM network
    elif args.model_type.startswith("hetnet"):
        print("HetNet is used")
        model = HetNet_solver(
            args.embedding_size,
            args.hidden_size,
            # args.visiting_num+args.coverage_num+args.pick_place_num+1,
            2, 10)
    else:
        raise

    model.load_state_dict(torch.load(param_path))
    model.eval()

    print("FINSHED LOADING===========")

    return model



if __name__ =="__main__":

  parser = argparse.ArgumentParser()
  args = parser.parse_args()
 

  if args.use_cuda:
      use_pin_memory = True
  else:
      use_pin_memory = False

  if args.model_type == "rnn":
        print("RNN model is used")
        model = solver_RNN(
            args.embedding_size,
            args.hidden_size,
            args.visiting_num+args.coverage_num+args.pick_place_num+1,
            2, 10)
  # AM network
  elif args.model_type.startswith("hetnet"):
      print("HetNet is used")
      model = HetNet_solver(
          args.embedding_size,
          args.hidden_size,
          args.visiting_num+args.coverage_num+args.pick_place_num+1,
          2, 10)

  model.load_state_dict(torch.load('/home/keep9oing/Study/Heterogenenous_Task/Models/HetNet/HetNet_C3_V3_D3.param'))
  model.eval()
