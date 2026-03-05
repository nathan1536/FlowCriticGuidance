if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import hydra
import pathlib
from omegaconf import OmegaConf

from train_maniflow_metaworld_multitask_workspace_2d import TrainManiFlowMetaWorldMultiTaskWorkspace2D

OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(version_base=None, config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")))
def main(cfg):
    workspace = TrainManiFlowMetaWorldMultiTaskWorkspace2D(cfg)
    eval_mode = getattr(cfg, "eval_mode", "best")
    workspace.eval(mode=eval_mode)


if __name__ == "__main__":
    main()


