from nets import *
from agents import *
from problems import *
from utils import *
from TrainPPOAgent import *
from utils.config import *
from utils.ortool import *
from utils.Misc import *
from utils.save_load import *

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.utils import clip_grad_norm_

import time
import os
from itertools import chain
import tqdm
from tqdm import tqdm

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
    train_data = torch.load("./data/train/DVRPSR_{}_{}_{}_{}/normalized_train.pth".format(args.Lambda,
                                                                                     args.dod,
                                                                                     args.vehicle_count,
                                                                                     args.horizon))
    verbose_print("Done")

    verbose_print("Uploading data for testing {}".format(args.test_batch_size), end=" ", flush=True)
    # test data is not normalized
    test_data = torch.load("./data/test/DVRPSR_{}_{}_{}_{}/unnormalized_test.pth".format(args.Lambda,
                                                                                         args.dod,
                                                                                         args.vehicle_count,
                                                                                         args.horizon))
    verbose_print("Done")

    if ortool_available:
        reference_routes = ortool_solve(test_data)
    else:
        reference_routes = None
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
    env_test = env(test_data, None, None, None, *env_params_test)

    if reference_routes is not None:
        reference_costs = eval_apriori_routes(env_test, reference_routes, 100)
        print("Reference cost on test dataset {:5.2f} +- {:5.2f}".format(reference_costs.mean(),
                                                                         reference_costs.std()))

    env_test.nodes = env_test.nodes.to(device)
    env_test.edge_attributes = env_test.edge_attributes.to(device)

    ## PPO agent for DVRPSR
    customer_feature = 4  # customer and vehicle features are fixed
    vehicle_feature = 8

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
    args.output_dir = "./output/{}_{}_{}_{}_{}".format(
                                                        args.problem.upper(),
                                                        args.Lambda,
                                                        args.dod,
                                                        args.vehicle_count,
                                                        time.strftime("%y%m%d")
    ) if args.output_dir is None else args.output_dir

    os.makedirs(args.output_dir, exist_ok=True)
    write_config_file(args, os.path.join(args.output_dir, "args.json"))
    verbose_print("Create Output dir {}".format(args.output_dir), end=" ", flush=True)

    ## Optimizer and LR Scheduler
    verbose_print("Initializing ADAM Optimizer ...", end=" ", flush=True)
    lr_scheduler = None

    optim = Adam([{"params": trainppo.agent.policy.parameters(), 'lr': args.learning_rate}])

    if args.rate_decay is not None:
        lr_scheduler = LambdaLR(optim,
                                lr_lambda=[lambda epoch: args.learning_rate * args.rate_decay ** epoch])

    if args.resume_state is None:
        start_epoch = 0
    else:
        start_epoch = load_checkpoint(args, trainppo.agent.old_policy, optim, lr_scheduler)

    verbose_print("Running PPO models ")
    train_stats = []
    test_stats = []

    try:
        for epoch in range(start_epoch, args.epoch_count):

            #print('running epoch {}'.format(epoch+1))
            train_stats.append(trainppo.run_train(args, train_data, env, env_params_train, optim, lr_scheduler, device, epoch))

            agent = trainppo.agent.old_policy

            if reference_routes is not None:
                test_stats.append(trainppo.test_epoch(args, env_test, agent, reference_costs))
            if args.rate_decay is not None:
                lr_scheduler.step()
            if args.grad_norm_decay is not None:
                args.max_grad_norm *= args.grad_norm_decay
            if (epoch + 1) % args.checkpoint_period == 0:
                save_checkpoint(args, epoch, agent, optim, lr_scheduler)

    except KeyboardInterrupt:
        save_checkpoint(args, epoch, agent, optim, lr_scheduler)
    finally:
        export_train_test_stats(args, start_epoch, train_stats, test_stats)


if __name__ == "__main__":
    run(ParseArguments())
