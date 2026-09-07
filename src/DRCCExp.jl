# =====================================================================
#  DRCCExp.jl  —  总入口模块。按顺序 include 各子模块(只一次), 避免
#  Cases 被重复 include 导致的类型不一致。脚本统一 `using .DRCCExp`。
# =====================================================================
module DRCCExp

include("Cases.jl")
include("CaseInterface.jl")
include("Bandwidth.jl")
include("ReliabilityStates.jl")
include("DataPipeline.jl")
include("ModelCore.jl")
include("ScalarMWModel.jl")
include("Baselines.jl")
include("Audit.jl")
include("Driver.jl")

using .Cases, .CaseInterface, .Bandwidth, .ReliabilityStates, .DataPipeline, .ModelCore, .ScalarMWModel, .Baselines, .Audit, .Driver

# 便捷再导出
export Cases, CaseInterface, DataPipeline, ModelCore, Audit
export ReliabilityStates
export parse_matpower, build_ptdf
export CaseData, ScenarioData, Snapshot, build_casedata, build_scenariodata, snapshot_at
export sample_safe_curtailment_cap
export FixedBandwidthProfile, build_fixed_bandwidth_profile,
       build_target_specific_bandwidth_profile
export apply_generator_availability
export generator_unavailability, sample_generator_states, apply_generator_state
export build_ols, calibrate_Z0, calibrate_targets, retarget_targets,
       MODEL_CONFIGS, ModelResult
export ScalarMWModel, ScalarMWBandwidthProfile, raw_loss_samples,
       solve_bandwidth_pilot, build_raw_profile,
       calibrate_scalar_targets, solve_scalar_mw,
       scalar_event_count, scalar_event_labels
export build_wdro, build_moment_dro, WdroResult
export run_pipeline
export audit_snapshot, aggregate_audit, AuditAgg
export run_fixed_experiment, run_nsmc_experiment, load_or_run_pipeline

end # module
