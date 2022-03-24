import set_path
import sys
import torch

set_path.append_sys_path()

import rela
import hanalearn
import utils

assert rela.__file__.endswith(".so")
assert hanalearn.__file__.endswith(".so")


class ActGroup:
    def __init__(
        self,
        devices,
        agent,
        partner_weight,
        seed,
        num_thread,
        num_game_per_thread,
        num_player,
        explore_eps,
        trinary,
        replay_buffer,
        max_len,
        gamma,
        convention,
        convention_act_override,
    ):
        self.devices = devices.split(",")
        self.seed = seed
        self.num_thread = num_thread
        self.num_player = num_player
        self.num_game_per_thread = num_game_per_thread
        self.explore_eps = explore_eps
        self.trinary = trinary
        self.replay_buffer = replay_buffer
        self.max_len = max_len
        self.gamma = gamma

        self.load_partner_model(partner_weight)

        self.model_runners = []
        for dev in self.devices:
            runner = rela.BatchRunner(agent.clone(dev), dev)
            runner.add_method("act", 5000)
            runner.add_method("compute_priority", 100)
            runner.add_method("compute_target", 5000)

            partner_runner = rela.BatchRunner(
                    self._partner_agent.clone(dev), dev)
            partner_runner.add_method("act", 5000)
            self.model_runners.append([runner, partner_runner])
        self.num_runners = len(self.model_runners)

        self.convention = convention
        self.convention_act_override = convention_act_override

        self.create_r2d2_actors()

    def load_partner_model(self, weight_file):
        try: 
            state_dict = torch.load(weight_file)
        except:
            sys.exit(f"weight_file {weight_file} can't be loaded")
        overwrite = {}
        overwrite["vdn"] = False
        overwrite["device"] = "cuda:0"
        overwrite["boltzmann_act"] = False

        if "fc_v.weight" in state_dict.keys():
            agent, cfg = utils.load_agent(weight_file, overwrite)
            self._partner_sad = cfg["sad"] if "sad" in cfg else cfg["greedy_extra"]
            self._partner_hide_action = bool(cfg["hide_action"])
        else:
            agent = utils.load_supervised_agent(weight_file, "cuda:0")
            self._partner_sad = False
            self._partner_hide_action = False

        agent.train(False)
        self._partner_agent = agent

    def create_r2d2_actors(self):
        convention_act_override = [0, 0]
        convention_sender = [1, 0]
        if self.convention_act_override:
            convention_act_override = [0, 1]
            convention_sender = [1, 0]

        actors = []
        for i in range(self.num_thread):
            thread_actors = []
            for j in range(self.num_game_per_thread):
                game_actors = []
                actor = hanalearn.R2D2Actor(
                    self.model_runners[i % self.num_runners][0],
                    self.seed,
                    self.num_player,
                    0,
                    self.explore_eps,
                    [0], # boltzmann_act
                    False,
                    0, # sad
                    0, # shuffle_color
                    0, # hide_action
                    self.trinary,
                    self.replay_buffer,
                    1, # multi-step
                    self.max_len,
                    self.gamma,
                    self.convention,
                    1,
                    0,
                    True, # convention_fict_act_override
                    True, # use_experience
                )
                game_actors.append(actor)
                self.seed += 1

                actor = hanalearn.R2D2Actor(
                    self.model_runners[i % self.num_runners][1], # runner
                    self.num_player, # numPlayer
                    1, # playerIdx
                    False, # vdn
                    self._partner_sad, # sad
                    self._partner_hide_action, # hideAction
                    self.convention, # convention
                    0, # conventionSender
                    1) # conventionOverride
                game_actors.append(actor)

                for k in range(self.num_player):
                    partners = game_actors[:]
                    partners[k] = None
                    game_actors[k].set_partners(partners)
                thread_actors.append(game_actors)
            actors.append(thread_actors)
        self.actors = actors
        print("ActGroup created")

    def start(self):
        for runners in self.model_runners:
            for runner in runners:
                runner.start()

    def update_model(self, agent):
        for runner in self.model_runners:
            runner[0].update_model(agent)
