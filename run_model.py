from problems import *
from TrainPPOAgent import *
from utils.config import *
from utils.ortool import *
from utils.Misc import *

import warnings
warnings.filterwarnings('error', category=UnicodeWarning)

import torch

import time
import os
ortool_available = True

def run(args):
    device = torch.device("mps" if torch.backends.mps.is_available() and args.gpu else "cpu")
    print(device)

    if args.seed is not None:
        torch.manual_seed(args.seed)

    if args.verbose:
        verbose_print = print
    else:
        def verbose_print(*args, **kwargs):
            pass

    ## load DVRPSR problem

    verbose_print("Uploading data for training {}".format(args.iter_count * args.batch_size), end=" ", flush=True)
    train_data = torch.load("./data/train/{}_{}_{}_{}/train_debug.pth".format(args.Lambda,
                                                                        args.dod,
                                                                        args.vehicle_count,
                                                                        args.horizon))
    verbose_print("Done")

    verbose_print("Uploading data for testing {}".format(args.test_batch_size), end=" ", flush=True)
    # test data is not normalized
    test_data = torch.load("./data/test/{}_{}_{}_{}/test_debug.pth".format(args.Lambda,
                                                                     args.dod,
                                                                     args.vehicle_count,
                                                                     args.horizon))
    verbose_print("Done")

    if ortool_available:
        reference_routes = ortool_solve(test_data)
    else:
        reference_routes = None
        reference_costs = None
        verbose_print(" No reference to calculate optimality gap", end=" ", flush=True)

    test_data.normalize()

    ## Defining Environemnt for DVRPSR
    env = {"DVRPSR": DVRPSR_Environment}.get(args.problem)
    env_params_train = [train_data.vehicle_count,
                        train_data.vehicle_speed,
                        train_data.vehicle_time_budget,
                        args.pending_cost,
                        args.dynamic_reward]

    env_params_test = [args.pending_cost,
                       args.dynamic_reward]
    env_test = env(test_data, None, None, None, pending_cost=args.pending_cost)

    if reference_routes is not None:
        reference_costs = eval_apriori_routes(env_test, reference_routes, 10)
        print("Reference cost on test dataset {:5.2f} +- {:5.2f}".format(reference_costs.mean(),
                                                                         reference_costs.std()))
    env_test.nodes = env_test.nodes.to(device)
    env_test.distance_matrix = env_test.distance_matrix.to(device)
    env_test.edge_attributes = env_test.edge_attributes.to(device)

    ## PPO agent for DVRPSR
    customer_feature = 4 # customer and vehicle features are fixed
    vehicle_feature = 6

    ## customer counts
    if args.customers_count is None:
        args.customers_count = train_data.customer_count+1
    print(args.customers_count)

    trainppo = TrainPPOAgent(customer_feature, vehicle_feature, args.customers_count, args.model_size,
                             args.encoder_layer, args.num_head, args.ff_size_actor, args.ff_size_critic, args.tanh_xplor,
                             args.edge_embedding_dim, args.greedy, args.learning_rate, args.ppo_epoch, args.batch_size,
                             args.entropy_value, args.epsilon_clip, args.epoch_count, args.timestep, args.max_grad_norm)

    ## Checkpoints
    verbose_print("Creating Output directry...", end=" ", flush=True)
    args.output_dir = "./output/gtm_{}_{}_{}_{}".format( args.Lambda,
                                                        args.dod,
                                                        args.vehicle_count,
                                                        time.strftime("%y%m%d")) if args.output_dir is None else args.output_dir

    os.makedirs(args.output_dir, exist_ok=True)
    write_config_file(args, os.path.join(args.output_dir, "args.json"))
    verbose_print("Create Output dir {}".format(args.output_dir), end=" ", flush=True)

    verbose_print('start training of DVRPSR model')
    trainppo.run_train(args, train_data, env, env_params_train, device, env_test, reference_costs)


if __name__ == "__main__":
    run(ParseArguments())
