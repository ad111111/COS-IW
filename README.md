# COS-IW
The code and dataset for our COS IW Final Project.

### Code
Below is a description of the code files used in this project:

- 🐍 `simulation.py`: Mininet simulation script to run virtual network experiments.
- $${\color{blue} C}$$ `bbrs.c`: BBR-S Code was taken from the BBR-Bufferbloat-Video-Project owned by Dr. Santiago Vargas, PhD. BBR-S code can be compiled into a linux kernel module.

### Data
The data folder in this repository contains both the Goodput and RTT data as .txt files:
- 📄 `Goodput_BBR-S.txt`: Goodput results for each BBR-S percentile.
- 📄 `RTT_BBR-S.txt`: RTT results for each BBR-S percentile.
- 📄 `Goodput_TCP_CC.txt`: Goodput results for each TCP Congestion Control algorithm.
- 📄 `RTT_TCP_CC.txt`: RTT results for each TCP Congestion Control algorithm.
