# Numerical experiments

## Comparison models

The retained workflow evaluates all policies on common states:

| Label | Method | Safety representation |
|---|---|---|
| M0 | Deterministic forecast-point DC-OLS | Forecast-point limits |
| M1 | Empirical HMCR | Eventwise |
| M2 | KDE-HMCR | Eventwise |
| M3 | KDE-HMCR with Pearson phi-divergence | Eventwise |
| M3-ClassMax | M3 aggregation ablation | Six class maxima |
| M4 | Wasserstein-CVaR | Native deficit-side joint set |
| M5 | Moment-based distributionally robust chance constraints | Native eventwise set |
| M6 | KDE-Pearson-phi-HMCR with robust satisficing | Six class maxima and two target relaxations |

`M6-Tradeoff` and `M6-Safety` are validation-selected parameter settings of M6, not separate models.

## Retained pipeline

1. `new_model_experiment/run_resource_pressure_screen.jl` selects a validation-only resource-pressure setting.
2. `run_resource_validation_shard.jl` evaluates the predeclared parameter grid on validation states.
3. `lock_resource_formal_experiment.jl` writes the immutable model and resource lock.
4. `generate_resource_component_state_draws.jl` generates the shared component-state stream.
5. `run_resource_formal_replay_shard.jl` solves every listed model on its assigned stress-test states.
6. `generate_unconditional_draws.jl` and `run_unconditional_replay_shard.jl` produce the equal-hour adequacy replay.
7. `postprocess/` applies the common acceptance rule and generates the manuscript summaries and figures without changing optimizer decisions.

The two formal protocols are:

- `new_model_experiment/protocols/plan_rbts_resource_pressure_screen.toml`
- `new_model_experiment/protocols/plan_rts79_resource_pressure_screen.toml`

The frozen locks and component draws used by the published runs are under `results/artifacts/`. Formal replay programs accept the protocol, lock, and draw paths on the command line; invoking a program with no arguments prints its complete usage signature.

## Data and outputs

- `RBTS/` and `RTS_79/` contain input network, load, and wind data.
- `results/` contains only the formal RBTS/RTS79 stress and adequacy records retained for the manuscript.
- Generated pipeline caches belong in `data_cache*` and new runs belong in `new_model_experiment/results/`; both are ignored by Git.

All headline comparisons use common-state replays. Validation data select pressure and M6 settings; test states are reserved for reporting. Nonaccepted solves remain in comparison denominators through the conservative penalty rule described in the manuscript.

