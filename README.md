This is the code for replicating the simulation results in paper [Online Learning in Bandits with Predicted Context](https://arxiv.org/abs/2307.13916).

Two environments are implemented: Gaussian linear bandit and a simulated environment based on HeartStep V2 data.

To run Gaussian linear bandit for 5000 steps with default algorithms: TS, UCB, MEB, MEB_naive, oracle
```bash
python RunExp.py -T 5000
```

To run the simulated environment:
```bash
python RunExp_Real.py -T 5000
```