from agents.ddpg.ddpg import data_func_strat
import argparse
import copy
import dataclasses
import datetime
import os
import time

import gym
import numpy as np
import rsoccer_gym
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim

import wandb
from agents.ddpg import (DDPGStratHP, data_func, DDPGStratRew)
from agents.utils import ReplayBuffer, save_checkpoint, load_checkpoint, unpack_batch, ExperienceFirstLast
import pyvirtualdisplay

if __name__ == "__main__":
    mp.set_start_method('spawn')
    os.environ['OMP_NUM_THREADS'] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False,
                        action="store_true", help="Enable cuda")
    parser.add_argument("-n", "--name", required=True,
                        help="Name of the run")
    parser.add_argument("-e", "--env", required=True,
                        help="Name of the gym environment")

    parser.add_argument("-l", "--load",
                        help="path to a checkpoint")

    args = parser.parse_args()
    device = "cuda" if args.cuda else "cpu"

    # Input Experiment Hyperparameters
    hp = DDPGStratHP(
        EXP_NAME=args.name,
        DEVICE=device,
        ENV_NAME=args.env,
        N_ROLLOUT_PROCESSES=3,
        LEARNING_RATE=0.001,
        EXP_GRAD_RATIO=10,
        BATCH_SIZE=256,
        GAMMA=0.95,
        REWARD_STEPS=3,
        NOISE_SIGMA_INITIAL=0.8,
        NOISE_THETA=0.15,
        NOISE_SIGMA_DECAY=0.99,
        NOISE_SIGMA_MIN=0.15,
        NOISE_SIGMA_GRAD_STEPS=3000,
        REPLAY_SIZE=5000000,
        REPLAY_INITIAL=100000,
        SAVE_FREQUENCY=100000,
        GIF_FREQUENCY=10000,
        TOTAL_GRAD_STEPS=2000000.
    )
    wandb.init(project='RoboCIn-RL', name=hp.EXP_NAME,
               entity='robocin', config=hp.to_dict())
    current_time = datetime.datetime.now().strftime('%b-%d_%H-%M-%S')
    tb_path = os.path.join('runs', current_time + '_'
                           + hp.ENV_NAME + '_' + hp.EXP_NAME)

    ddpg = DDPGStratRew(hp)

    if args.load:
        load_checkpoint(ddpg, args.load)

    # Playing
    ddpg.share_memory()
    exp_queue = mp.Queue(maxsize=hp.EXP_GRAD_RATIO)
    finish_event = mp.Event()
    sigma_m = mp.Value('f', hp.NOISE_SIGMA_INITIAL)
    gif_req_m = mp.Value('i', -1)
    data_proc_list = []
    for _ in range(hp.N_ROLLOUT_PROCESSES):
        data_proc = mp.Process(
            target=data_func,
            args=(
                ddpg,
                device,
                exp_queue,
                finish_event,
                sigma_m,
                gif_req_m,
                hp
            )
        )
        data_proc.start()
        data_proc_list.append(data_proc)

    # Training
    buffer = ReplayBuffer(buffer_size=hp.REPLAY_SIZE,
                          observation_space=hp.observation_space,
                          action_space=hp.action_space,
                          device=hp.DEVICE,
                          strat_size=hp.N_REWS
                          )
    n_grads = 0
    n_samples = 0
    n_episodes = 0
    best_reward = None
    last_gif = None

    try:
        while n_grads < hp.TOTAL_GRAD_STEPS:
            metrics = {}
            ep_infos = list()
            st_time = time.perf_counter()
            # Collect EXP_GRAD_RATIO sample for each grad step
            new_samples = 0
            while new_samples < hp.EXP_GRAD_RATIO:
                exp = exp_queue.get()
                if exp is None:
                    raise Exception  # got None value in queue
                safe_exp = copy.deepcopy(exp)
                del(exp)

                # Dict is returned with end of episode info
                if isinstance(safe_exp, dict):
                    logs = {"ep_info/"+key: value for key,
                            value in safe_exp.items() if 'truncated' not in key}
                    ep_infos.append(logs)
                    n_episodes += 1
                else:
                    buffer.add(
                        obs=safe_exp.state,
                        next_obs=safe_exp.last_state if safe_exp.last_state is not None else safe_exp.state,
                        action=safe_exp.action,
                        reward=safe_exp.reward,
                        done=False if safe_exp.last_state is not None else True
                    )
                    new_samples += 1
            n_samples += new_samples
            sample_time = time.perf_counter()

            # Only start training after buffer is larger than initial value
            if buffer.size() < hp.REPLAY_INITIAL:
                continue

            # Sample a batch and load it as a tensor on device
            batch = buffer.sample(hp.BATCH_SIZE)
            metrics["train/loss_pi"], metrics["train/loss_Q"], rews, q_strat, qf, next_qf = ddpg.update(
                batch)
            names = ['move', 'ball_grad', 'energy', 'goal']
            for i, name in enumerate(names):
                metrics[f"train/rew_{name}"] = rews[i]
                metrics[f"train/Q_loss_{name}"] = q_strat[i]
                metrics[f"train/Q_{name}"] = qf[i]
                metrics[f"train/Q_next_{name}"] = next_qf[i]

            n_grads += 1
            grad_time = time.perf_counter()
            metrics['speed/samples'] = new_samples/(sample_time - st_time)
            metrics['speed/grad'] = 1/(grad_time - sample_time)
            metrics['speed/total'] = 1/(grad_time - st_time)
            metrics['counters/samples'] = n_samples
            metrics['counters/grads'] = n_grads
            metrics['counters/episodes'] = n_episodes
            metrics["counters/buffer_len"] = buffer.size()

            if ep_infos:
                for key in ep_infos[0].keys():
                    metrics[key] = np.mean([info[key] for info in ep_infos])

            # Log metrics
            wandb.log(metrics)

            if hp.NOISE_SIGMA_DECAY and sigma_m.value > hp.NOISE_SIGMA_MIN \
                    and n_grads % hp.NOISE_SIGMA_GRAD_STEPS == 0:
                # This syntax is needed to be process-safe
                # The noise sigma value is accessed by the playing processes
                with sigma_m.get_lock():
                    sigma_m.value *= hp.NOISE_SIGMA_DECAY

            if hp.SAVE_FREQUENCY and n_grads % hp.SAVE_FREQUENCY == 0:
                save_checkpoint(
                    hp=hp,
                    metrics={
                        'noise_sigma': sigma_m.value,
                        'n_samples': n_samples,
                        'n_episodes': n_episodes,
                        'n_grads': n_grads,
                    },
                    pi=ddpg.pi,
                    Q=ddpg.Q,
                    pi_opt=ddpg.pi_opt,
                    Q_opt=ddpg.Q_opt
                )

            if hp.GIF_FREQUENCY and n_grads % hp.GIF_FREQUENCY == 0:
                gif_req_m.value = n_grads

    except KeyboardInterrupt:
        print("...Finishing...")
        finish_event.set()

    finally:
        if exp_queue:
            while exp_queue.qsize() > 0:
                exp_queue.get()

        print('queue is empty')

        print("Waiting for threads to finish...")
        for p in data_proc_list:
            p.terminate()
            p.join()

        del(exp_queue)

        finish_event.set()
