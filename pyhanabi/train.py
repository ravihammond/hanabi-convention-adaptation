import time
import os
import sys
import argparse
import pprint
import json

import torch

from act_group import ActGroup
from create import create_envs, create_threads
from eval import evaluate
import common_utils
import rela
import r2d2
import utils

class Trainer:
    def __init__(self, args):
        self._args = args
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        logger_path = os.path.join(args.save_dir, "train.log")
        sys.stdout = common_utils.Logger(logger_path)
        self._saver = common_utils.TopkSaver(args.save_dir, 5)

        self.load_convention(args.convention)

        common_utils.set_all_seeds(args.seed)
        pprint.pprint(vars(args))

        self._explore_eps = utils.generate_explore_eps(
            args.act_base_eps, args.act_eps_alpha, args.num_t
        )

        self.create_game_and_agents(args)


    def load_convention(self, convention_path):
        if convention_path == "None":
            return []
        convention_file = open(convention_path)
        self._convention = json.load(convention_file)

    def create_game_and_agents(self, args):
        self._games = create_envs(
            args.num_thread * args.num_game_per_thread,
            args.seed,
            args.num_player,
            args.train_bomb,
            args.max_len,
        )

        self._agent = r2d2.R2D2Agent(
            False,
            1,
            args.gamma,
            args.eta,
            args.train_device,
            self._games[0].feature_size(0),
            args.rnn_hid_dim,
            self._games[0].num_action(),
            args.net,
            args.num_lstm_layer,
            0,
            False,  # uniform priority
            0,
        )
        self._agent.sync_target_with_online()

        if args.load_model and args.load_model != "None":
            print("*****loading pretrained model*****")
            print(args.load_model)
            utils.load_weight(
                self._agent.online_net, 
                args.load_model, 
                args.train_device
            )
            print("*****done*****")

        self._agent = self._agent.to(args.train_device)
        self._optim = torch.optim.Adam(
            self._agent.online_net.parameters(), 
            lr=args.lr, 
            eps=args.eps
        )
        self._eval_agent = self._agent.clone(
            args.train_device, 
            {"vdn": False, "boltzmann_act": False}
        )

        self._replay_buffer = rela.RNNPrioritizedReplay(
            args.replay_buffer_size,
            args.seed,
            args.priority_exponent,
            args.priority_weight,
            args.prefetch,
        )

        self._act_group = ActGroup(
            args.act_device,
            self._agent,
            args.partner,
            args.seed,
            args.num_thread,
            args.num_game_per_thread,
            args.num_player,
            self._explore_eps,
            True,  # trinary, 3 bits for aux task
            self._replay_buffer,
            args.max_len,
            args.gamma,
            self._convention,
            args.convention_act_override
        )

        self._context, self._threads = create_threads(
            args.num_thread,
            args.num_game_per_thread,
            self._act_group.actors,
            self._games,
        )

    def warm_up_replay_buffer(self):
        self._act_group.start()
        self._context.start()
        while self._replay_buffer.size() < self._args.burn_in_frames:
            print("warming up replay buffer:", self._replay_buffer.size())
            time.sleep(1)

        print("Success, Done")
        print("=======================")


    def train_loop(self):
        stat = common_utils.MultiCounter(args.save_dir)
        tachometer = utils.Tachometer()
        stopwatch = common_utils.Stopwatch()
        for epoch in range(self._args.num_epoch):
            print("beginning of epoch: ", epoch)
            print(common_utils.get_mem_usage())
            tachometer.start()
            stat.reset()
            stopwatch.reset()

            for batch_idx in range(self._args.epoch_len):
                num_update = batch_idx + epoch * self._args.epoch_len
                if num_update % self._args.num_update_between_sync == 0:
                    self._agent.sync_target_with_online()
                if num_update % self._args.actor_sync_freq == 0:
                    self._act_group.update_model(self._agent)

                torch.cuda.synchronize()
                stopwatch.time("sync and updating")

                batch, weight = self._replay_buffer.sample(
                        self._args.batchsize, self._args.train_device)
                stopwatch.time("sample data")

                loss, priority, online_q = self._agent.loss(
                        batch, 0, stat)
                loss = (loss * weight).mean()
                loss.backward()

                torch.cuda.synchronize()
                stopwatch.time("forward & backward")

                g_norm = torch.nn.utils.clip_grad_norm_(
                    self._agent.online_net.parameters(), self._args.grad_clip
                )
                self._optim.step()
                self._optim.zero_grad()

                torch.cuda.synchronize()
                stopwatch.time("update model")

                self._replay_buffer.update_priority(priority)
                stopwatch.time("updating priority")

                stat["loss"].feed(loss.detach().item())
                stat["grad_norm"].feed(g_norm)
                stat["boltzmann_t"].feed(batch.obs["temperature"][0].mean())

            count_factor = 1
            print("epoch: %d" % epoch)
            tachometer.lap(self._replay_buffer, self._args.epoch_len * self._args.batchsize, count_factor)
            stopwatch.summary()
            stat.summary(epoch)

            eval_seed = (9917 + epoch * 999999) % 7777777
            self._eval_agent.load_state_dict(self._agent.state_dict())
            score, perfect, *_ = evaluate(
                [self._eval_agent for _ in range(self._args.num_player)],
                1000,
                eval_seed,
                self._args.eval_bomb,
                0,  # explore eps
                0,
                0,
            )

            force_save_name = None
            if epoch > 0 and epoch % self._args.save_checkpoints == 0:
                force_save_name = "model_epoch%d" % epoch
            model_saved = self._saver.save(
                None, 
                self._agent.online_net.state_dict(), 
                score, 
                force_save_name=force_save_name
            )
            print(
                "epoch %d, eval score: %.4f, perfect: %.2f, model saved: %s"
                % (epoch, score, perfect * 100, model_saved)
            )

            print("==========")



def parse_args():
    parser = argparse.ArgumentParser(description="train dqn on hanabi")
    parser.add_argument("--save_dir", type=str, default="exps/exp1")
    parser.add_argument("--min_t", type=float, default=1e-3)
    parser.add_argument("--max_t", type=float, default=1e-1)
    parser.add_argument("--num_t", type=int, default=80)

    parser.add_argument("--load_model", type=str, default="")

    parser.add_argument("--seed", type=int, default=10001)
    parser.add_argument("--gamma", type=float, default=0.999, help="discount factor")
    parser.add_argument(
        "--eta", type=float, default=0.9, help="eta for aggregate priority"
    )
    parser.add_argument("--train_bomb", type=int, default=0)
    parser.add_argument("--eval_bomb", type=int, default=0)
    parser.add_argument("--num_player", type=int, default=2)

    # optimization/training settings
    parser.add_argument("--lr", type=float, default=6.25e-5, help="Learning rate")
    parser.add_argument("--eps", type=float, default=1.5e-5, help="Adam epsilon")
    parser.add_argument("--grad_clip", type=float, default=5, help="max grad norm")
    parser.add_argument("--num_lstm_layer", type=int, default=2)
    parser.add_argument("--rnn_hid_dim", type=int, default=512)
    parser.add_argument(
        "--net", type=str, default="publ-lstm", help="publ-lstm/ffwd/lstm"
    )

    parser.add_argument("--train_device", type=str, default="cuda:0")
    parser.add_argument("--batchsize", type=int, default=128)
    parser.add_argument("--num_epoch", type=int, default=5000)
    parser.add_argument("--epoch_len", type=int, default=1000)
    parser.add_argument("--num_update_between_sync", type=int, default=2500)
    parser.add_argument("--save_checkpoints", type=int, default=100)

    # replay buffer settings
    parser.add_argument("--burn_in_frames", type=int, default=10000)
    parser.add_argument("--replay_buffer_size", type=int, default=100000)
    parser.add_argument(
        "--priority_exponent", type=float, default=0.9, help="alpha in p-replay"
    )
    parser.add_argument(
        "--priority_weight", type=float, default=0.6, help="beta in p-replay"
    )
    parser.add_argument("--max_len", type=int, default=80, help="max seq len")
    parser.add_argument("--prefetch", type=int, default=3, help="#prefetch batch")

    # thread setting
    parser.add_argument("--num_thread", type=int, default=10, help="#thread_loop")
    parser.add_argument("--num_game_per_thread", type=int, default=40)

    # actor setting
    parser.add_argument("--act_base_eps", type=float, default=0.1)
    parser.add_argument("--act_eps_alpha", type=float, default=7)
    parser.add_argument("--act_device", type=str, default="cuda:1")
    parser.add_argument("--actor_sync_freq", type=int, default=10)

    # convention setting
    parser.add_argument("--convention", type=str, default="None")
    parser.add_argument("--convention_act_override", type=int, default=0)
    parser.add_argument("--partner", type=str, default="None")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    args = parse_args()

    trainer = Trainer(args)
    trainer.warm_up_replay_buffer()
    trainer.train_loop()

