import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ['LAZY_LEGACY_OP'] = '0'
os.environ['TORCHDYNAMO_INLINE_INBUILT_NN_MODULES'] = "1"
os.environ['TORCH_LOGS'] = "+recompiles"
import warnings
warnings.filterwarnings('ignore')
import torch

import hydra
from termcolor import colored

from common.parser import parse_cfg
from common.seed import set_seed
from common.buffer import Buffer
from envs import make_env
from tdmpc2 import TDMPC2
from trainer.offline_trainer import OfflineTrainer
from trainer.online_trainer import OnlineTrainer
from common.logger import Logger

from utils import prompt_if_file_exists
from trainer.dagger_trainer import DaggerTrainer
from copy import deepcopy

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')


@hydra.main(config_name='config', config_path='.')
def train(cfg: dict):
	"""
	Script for training single-task / multi-task TD-MPC2 agents.

	Most relevant args:
		`task`: task name (or mt30/mt80 for multi-task training)
		`model_size`: model size, must be one of `[1, 5, 19, 48, 317]` (default: 5)
		`steps`: number of training/environment steps (default: 10M)
		`seed`: random seed (default: 1)

	See config.yaml for a full list of args.

	Example usage:
	```
		$ python train.py task=mt80 model_size=48
		$ python train.py task=mt30 model_size=317
		$ python train.py task=dog-run steps=7000000
	```
	"""
	assert torch.cuda.is_available()
	assert cfg.steps > 0, 'Must train for at least 1 step.'
	cfg = parse_cfg(cfg)
	set_seed(cfg.seed)
	print(colored('Work dir:', 'yellow', attrs=['bold']), cfg.work_dir)

	trainer_cls = OfflineTrainer if cfg.multitask else OnlineTrainer
	trainer = trainer_cls(
		cfg=cfg,
		env=make_env(cfg),
		agent=TDMPC2(cfg),
		buffer=Buffer(cfg),
		logger=Logger(cfg),
	)
	trainer.train()
	print('\nTraining completed successfully')


# cfg:
#   - base_model_path: A path to the model we're starting from.
#   - end_model_path: A path to the location we save our final model.
#   - student_base_model_path: Start with an existing student model.
@hydra.main(config_name='config', config_path='.')
def train_dagger(cfg: dict):
	assert torch.cuda.is_available()
	
	if not prompt_if_file_exists(cfg.end_model_path):
		print("Stopped.")
		return
	if not prompt_if_file_exists(cfg.results_csv):
		print("Stopped.")
		return
	
	set_seed(100)
	cfg = parse_cfg(cfg)

	expert = TDMPC2(cfg)
	expert.load(cfg.base_model_path)

	agent_cfg = deepcopy(cfg)
	agent_cfg.model_size = cfg.student_model_size
	agent_cfg.mpc = False
	agent = TDMPC2(agent_cfg)
	if 'student_base_model_path' in agent_cfg:
		agent.load(agent_cfg.student_base_model_path)

	# not sure if this actually works
	trainer = DaggerTrainer(
		cfg=cfg,
		env=make_env(cfg),
		agent=agent,
		expert=expert,
		buffer=Buffer(cfg),
		logger=Logger(cfg),
	)
	trainer.train()

	agent.save(cfg.end_model_path)

	print('\nTraining completed successfully')


if __name__ == '__main__':
	train_dagger()
