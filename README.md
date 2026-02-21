
# CheapDrone-VLA
## Final Individual Project for ICL Applied Machine Learning

## Tested Compatibilities:
- [x] Ubuntu 22.02+
- [ ] WSL2 (missing nvidia EGL dependencies for habitat-sim)
- [ ] Windows (habitat-sim does not support)

## How to install:
- Pull Repository
- Install OpenPi Environments
- Replace paths in config/pi0/config.py
- Replace OpenPi configs with updated config/pi0/config.py
- Replace paths in all files of: online_eval/vla_eval 
- Install habitat-sim with bullet physics in an conda environment
- Inference using vla_eval.sh

## Common Errors:
- Habitat-sim built in headless mode
- File paths incorrectly set
- Insufficient VRAM

## Results
| Model | Peak Mem (GB) | SR Full (%) | NDTW Full (%) | SR Easy (%) | NDTW Easy (%) | SR Med (%) | NDTW Med (%) | SR Hard (%) | NDTW Hard (%) |
|-------|:-------------:|:-----------:|:-------------:|:-----------:|:-------------:|:----------:|:------------:|:-----------:|:-------------:|
| IndoorUAV-Agent | 6.641 | 26.55 | 0.79 | 47.05 | 1.54 | 22.33 | 0.50 | 6.09 | 0.53 |

## Acknowledgments:
Models:     [IndoorUAV-Agent](https://github.com/valyentinee/IndoorUAV-Agent/tree/main)

Benchmark:  [IndoorUAV-Dataset](https://www.modelscope.cn/datasets/valyentine/Indoor_UAV)

openpi:     [pi0 (openpi)](https://github.com/Physical-Intelligence/openpi)
  
Habitat:    [Habitat-Sim](https://github.com/facebookresearch/habitat-sim/tree/main)
