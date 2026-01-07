#!/bin/bash

gnome-terminal -- bash -c "conda activate uv && python online_eval/vln_eval/model_runner.py; exec bash"

sleep 12

gnome-terminal -- bash -c "conda activate habitat && python online_eval/vln_eval/sim_runner.py; exec bash"

sleep 1

gnome-terminal -- bash -c "conda activate habitat && python online_eval/vln_eval/vln_controller.py; exec bash"
