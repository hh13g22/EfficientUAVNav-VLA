#!/bin/bash

gnome-terminal -- bash -ic "conda activate openpi && python online_eval/vla_eval/model_runner.py; exec bash"

sleep 10

gnome-terminal -- bash -ic "conda activate habitat && python online_eval/vla_eval/sim_runner.py; exec bash"

sleep 6

gnome-terminal -- bash -ic "conda activate habitat && python online_eval/vla_eval/vla_controller.py; exec bash"
