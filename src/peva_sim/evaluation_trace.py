"""Recompute simulated episode metrics from persisted post-hoc traces."""
import math
import numpy as np

def audit_trace(trace, record, config):
    times=np.asarray(trace["time_s"])
    found=np.asarray(trace["found"])
    terminal=np.asarray(trace["terminal_found"])
    actions=np.asarray(trace["actions_mps"])
    rewards=np.asarray(trace["rewards"])
    steps=len(times);n=config["n_targets"];u=config["n_uavs"]
    dt=config["dt_s"];horizon=config["horizon_s"]
    if not 1<=steps<=round(horizon/dt):
        raise ValueError("Invalid episode length")
    shapes={"found":(steps,n),"terminal_found":(n,),
            "actions_mps":(steps,u,2),"rewards":(steps,),
            "uavs_m":(steps,u,2),"targets_m":(steps,n,2)}
    for key,shape in shapes.items():
        value=np.asarray(trace[key])
        if value.shape!=shape or not np.isfinite(value).all():
            raise ValueError("Invalid trace array: "+key)
    if not np.array_equal(times,np.arange(steps)*dt):
        raise ValueError("Invalid trace times")
    if found.dtype!=np.bool_ or terminal.dtype!=np.bool_ or found[0].any():
        raise ValueError("Invalid found flags")
    full=np.concatenate((found,terminal[None]),axis=0)
    changes=np.diff(full.astype(int),axis=0)
    if (changes<0).any():
        raise ValueError("Target resurrected")
    if np.linalg.norm(actions,axis=-1).max()>config["speed_mps"]+1e-5:
        raise ValueError("Speed bound violated")
    if not terminal.all() and steps*dt!=horizon:
        raise ValueError("Premature termination")
    for key in ("uavs_m","targets_m"):
        if (trace[key]<-1e-5).any() or (trace[key]>config["region_m"]+1e-5).any():
            raise ValueError("Position outside domain")
    detected=np.sum(changes,axis=1)
    event_times=(np.arange(steps)+1)*dt
    expected_rewards=detected*np.exp(-event_times/config["survival_tau_s"])
    if config["step_cost"]!=0:
        raise ValueError("Nonzero step cost needs explicit audit formula")
    if not np.allclose(rewards,expected_rewards,rtol=1e-9,atol=1e-12):
        raise ValueError("Discovery reward mismatch")
    count=int(terminal.sum())
    expected=dict(success_rate=count/n,
                  survival_score=float(expected_rewards.sum()/n),
                  ttd_all=float((detected@event_times+(n-count)*horizon)/(n*horizon)),
                  time_s=steps*dt)
    for key,value in expected.items():
        if not math.isclose(record["info"][key],value,rel_tol=1e-9,abs_tol=1e-12):
            raise ValueError("Metric mismatch: "+key)
    if record["steps"]!=steps or not math.isclose(record["return_value"],float(rewards.sum()),rel_tol=1e-9,abs_tol=1e-12):
        raise ValueError("Episode totals mismatch")
    return expected
