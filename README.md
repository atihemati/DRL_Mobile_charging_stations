# Using mobile charging stations as probes to discover latent EV charging demand in stochastic environments: A deep reinforcement learning approach

This repository contains the implementation of a **deep reinforcement learning agent** and a **rolling-horizon optimization baseline** for dispatching mobile charging stations (MCSs) to serve stochastic EV charging demand. The agent is an Advantage Actor-Critic with an LSTM encoder; the baseline is an MILP that gets to see future demand perfectly and serves as an idealized upper bound.

It reproduces the toy experiment from:

> **Hemmati Golsefidi, A., Boe Hüttel, F., Samaranayake, S., Câmara Pereira, F. (2026).**
> *Using mobile charging stations as probes to discover latent EV charging demand in stochastic environments: A deep reinforcement learning approach.*
> *Expert Systems with Applications*, **312**, 131433.
> [https://doi.org/10.1016/j.eswa.2026.131433](https://doi.org/10.1016/j.eswa.2026.131433)

Everything you need to run the experiment is in **[`test.ipynb`](test.ipynb)** — open it and follow the cells in order.

The notebook runs two scenarios that differ only in the coverage radius `Dis`:

### Non-intersecting supply coverage (`Dis = 30`)

The coverage radius is small, so MCSs and fixed stations cover disjoint regions. Each high-demand location is served only when an MCS is physically present there, and the agent has to learn to *be in the right place at the right time* across the day/night cycle.

![Non-intersecting scenario](animation/animation_RL/non_intersecting/animation_RL_1.gif)

### Intersecting supply coverage (`Dis = 100`)

The coverage radius is large enough that a single MCS at an intermediate location can serve several demand clusters at once. The agent learns to occupy a few well-chosen hubs instead of chasing every demand peak, and overall rewards are higher than in the non-intersecting case.

![Intersecting scenario](animation/animation_RL/intersecting/animation_RL_1.gif)

---

## Output files

Some experiment output files are included in the repository to support
reproducibility of `test.ipynb`.

Very large generated scenario files (`.pkl`) are excluded from GitHub because they exceed GitHub file-size limits. These files can be
regenerated automatically by running the notebook.


---

## Citation

If you use this code, please cite the paper:

```bibtex
@article{golsefidi2026using,
  title   = {Using Mobile Charging Stations as Probes to Discover Latent 
            EV Charging Demand in Stochastic Environments: 
            A Deep Reinforcement Learning Approach},
  author  = {Hemmati Golsefidi, Atefeh and Boe H{\"u}ttel, Frederik and
             Samaranayake, Samitha and C{\^a}mara Pereira, Francisco},
  journal = {Expert Systems with Applications},
  volume  = {312},
  pages   = {131433},
  year    = {2026},
  publisher = {Elsevier},
  doi     = {10.1016/j.eswa.2026.131433}
}
```

The same citation is also available in [`CITATION.cff`](CITATION.cff), which GitHub surfaces automatically under the *“Cite this repository”* button.

## License

MIT — see [`LICENSE`](LICENSE).
