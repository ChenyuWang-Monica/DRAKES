# Classifier-free Guidance baseline
import os
import fsspec
import hydra
import lightning as L
import omegaconf
import rich.syntax
import rich.tree
import torch
import dataloader_gosai
import diffusion_gosai_cfg
import utils
import random
import string
import datetime
import wandb
omegaconf.OmegaConf.register_new_resolver("uuid", lambda: ''.join(random.choice(string.ascii_letters) for _ in range(10))+'_'+str(datetime.datetime.now().strftime("%Y%m%d_%H%M%S")), use_cache=False)
omegaconf.OmegaConf.register_new_resolver('cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver('device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver('eval', eval)
omegaconf.OmegaConf.register_new_resolver('div_up', lambda x, y: (x + y - 1) // y)


def _load_from_checkpoint(config):
  if 'hf' in config.backbone:
    return diffusion_gosai_cfg.Diffusion(
      config, 
      ).to('cuda')
  
  return diffusion_gosai_cfg.Diffusion.load_from_checkpoint(
    config.eval.checkpoint_path,
    config=config)


@L.pytorch.utilities.rank_zero_only
def _print_config(
  config: omegaconf.DictConfig,
  resolve: bool = True,
  save_cfg: bool = True) -> None:
  """Prints content of DictConfig using Rich library and its tree structure.
  
  Args:
    config (DictConfig): Configuration composed by Hydra.
    resolve (bool): Whether to resolve reference fields of DictConfig.
    save_cfg (bool): Whether to save the configuration tree to a file.
  """

  style = 'dim'
  tree = rich.tree.Tree('CONFIG', style=style, guide_style=style)

  fields = config.keys()
  for field in fields:
    branch = tree.add(field, style=style, guide_style=style)

    config_section = config.get(field)
    branch_content = str(config_section)
    if isinstance(config_section, omegaconf.DictConfig):
      branch_content = omegaconf.OmegaConf.to_yaml(
        config_section, resolve=resolve)

    branch.add(rich.syntax.Syntax(branch_content, 'yaml'))
  rich.print(tree)
  if save_cfg:
    with fsspec.open(
      '{}/config_tree.txt'.format(
        config.checkpointing.save_dir), 'w') as fp:
      rich.print(tree, file=fp)


@L.pytorch.utilities.rank_zero_only
def _print_batch(train_ds, valid_ds, test_ds):
  for dl_type, dl in [
    ('train', train_ds), ('valid', valid_ds), ('test', test_ds)]:
    print(f'Printing {dl_type} dataloader batch.')
    batch = next(iter(dl))
    print('Batch seqs.shape', batch['seqs'].shape)
    print(f'tokens:', dataloader_gosai.dna_detokenize(batch['seqs'][0]))
    print('ids:', batch['seqs'][0])


def _train(config, logger):
  logger.info('Starting Training.')
  wandb_logger = None
  wandb_settings = wandb.Settings(
      base_url='https://api.wandb.ai'  # Specify your wandb host URL here
  )
  if config.get('wandb', None) is not None and not config.debug_mode:
    wandb_logger = L.pytorch.loggers.WandbLogger(
      config=omegaconf.OmegaConf.to_object(config),
      settings=wandb_settings,
      ** config.wandb)

  if (config.checkpointing.resume_from_ckpt
      and config.checkpointing.resume_ckpt_path is not None
      and utils.fsspec_exists(
        config.checkpointing.resume_ckpt_path)):
    ckpt_path = config.checkpointing.resume_ckpt_path
  else:
    ckpt_path = None

  # Lightning callbacks
  callbacks = []
  if 'callbacks' in config:
    for _, callback in config.callbacks.items():
      callbacks.append(hydra.utils.instantiate(callback))

  train_ds, valid_ds, test_ds = dataloader_gosai.get_dataloaders_gosai(config)

  model = diffusion_gosai_cfg.Diffusion(
    config, 
    )

  trainer = hydra.utils.instantiate(
    config.trainer,
    default_root_dir=os.getcwd(),
    callbacks=callbacks,
    strategy=hydra.utils.instantiate(config.strategy),
    logger=wandb_logger)
  print('Start training...')
  trainer.fit(model, train_ds, valid_ds, ckpt_path=ckpt_path)


@hydra.main(version_base=None, config_path='configs_gosai',
            config_name='config_gosai')
def main(config):
  """Main entry point for training."""
  L.seed_everything(config.seed)
  _print_config(config, resolve=True, save_cfg=True)
  
  logger = utils.get_logger(__name__)
  assert config.mode == 'train'
  _train(config, logger)


if __name__ == '__main__':
  main()
