
# CheapDrone-VLA
## Final Individual Project for MSc Applied Machine Learning ICL

## Tested Compatibilities:
- [x] True Linux
- [ ] WSL2 (Habitat incompatible, missing lib_nvidiaEGL)
- [ ] Windows

## How to install:
- Pull Repository
- Install OpenPi Environments
- Replace paths in config/pi0/config.py
- Replace OpenPi configs with updated config/pi0/config.py
- Replace paths in all files of: online_eval/vla_eval 
- Install habitat-sim in an conda environment.
- Inference using vla_eval.sh

## Acknowledgments:
Models:     [IndoorUAV-Agent](https://github.com/valyentinee/IndoorUAV-Agent/tree/main)

Benchmark:  [IndoorUAV-Agent](https://www.modelscope.cn/datasets/valyentine/Indoor_UAV)

openpi:     [pi0 (openpi)](https://github.com/Physical-Intelligence/openpi)
  
Habitat:    [Habitat-Sim](https://github.com/facebookresearch/habitat-sim/tree/main)
