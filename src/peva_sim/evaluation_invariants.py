"""Independent invariants for the single-target baseline validation records."""
import math


def verify_episode(record,scenario,config):
    for key in ("episode_id","global_index","replay_index","track_id","environment_seed","start_utc","end_utc"):
        if record[key]!=scenario[key]: raise ValueError("scenario identity mismatch: "+key)
    steps=record["steps"];dt=config["dt_s"];horizon=config["horizon_s"]
    if not isinstance(steps,int) or not 1<=steps<=round(horizon/dt):
        raise ValueError("invalid episode steps")
    if record["time_s"]!=steps*dt: raise ValueError("time/steps mismatch")
    success=record["success_rate"]
    if success not in (0.,1.): raise ValueError("single-target success must be binary")
    if not success and record["time_s"]!=horizon: raise ValueError("failed mission ended early")
    expected_ttd=record["time_s"]/horizon if success else 1.
    expected_survival=math.exp(-record["time_s"]/config["survival_tau_s"]) if success else 0.
    for key,expected in (("ttd_all",expected_ttd),("survival_score",expected_survival)):
        if not math.isclose(record[key],expected,rel_tol=1e-9,abs_tol=1e-12):
            raise ValueError("inconsistent metric: "+key)
    if config["step_cost"]==0 and not math.isclose(record["return_value"],expected_survival,abs_tol=1e-12):
        raise ValueError("return/survival mismatch")
    expected_bytes={
        "peer_payload":0,"peer_header":0,
        "shared_uplink":steps*config["n_uavs"]*(16+config["header_bytes"])+int(success)*4,
        "shared_downlink":steps*(config["grid_side"]**2*4+8+config["header_bytes"])}
    if record["bytes"]!=expected_bytes:
        raise ValueError("baseline communication accounting mismatch")
    return sum(expected_bytes.values())
