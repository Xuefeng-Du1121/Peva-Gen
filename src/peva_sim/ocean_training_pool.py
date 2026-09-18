"""Training-only scenario pool with recomputed environmental split checks."""
import json
import os
from pathlib import Path
import numpy as np
from .env import Config
from .ocean_field import OceanField
from .ocean_forced_env import OceanForcedSAR
from .real_env import LocalFrame
from .forcing_scenarios import support,validate_split_support
from .protocol import file_sha
ROOT=Path(__file__).resolve().parents[2]
DATA_ROOT=Path(os.environ.get("PEVA_DATA_ROOT", ROOT / "data")).expanduser().resolve()

class OceanTrainingPool(OceanForcedSAR):
    def __init__(self,inventory_path):
        path=Path(inventory_path).resolve()
        if not path.is_relative_to(ROOT): raise ValueError("Inventory outside project")
        inventory=json.loads(path.read_text())
        schema=inventory.get("schema")
        if schema not in {"ocean-forcing-inventory-v1","ocean-forcing-inventory-v2"}:
            raise ValueError("Unknown inventory schema")
        if schema=="ocean-forcing-inventory-v2" and inventory.get("status")!="frozen formal protocol":
            raise ValueError("Formal inventory is not frozen")
        field_path=Path(inventory["field_path"]).resolve()
        if not field_path.is_relative_to(DATA_ROOT): raise ValueError("Field outside allowed data root")
        if file_sha(field_path)!=inventory["field_sha256"]: raise ValueError("Forcing hash mismatch")
        field=OceanField(field_path);config=Config(**inventory["config"])
        for row in inventory["scenarios"]:
            expected=support(field,row["origin_lonlat"],row["start_utc"],config.region_m,config.horizon_s)
            if expected!=row["support"]: raise ValueError("Tampered scenario support")
        validate_split_support(inventory["scenarios"])
        self.training_scenarios=[r for r in inventory["scenarios"] if r["split"]=="train"]
        if not self.training_scenarios: raise ValueError("No training scenarios")
        self.inventory_sha256=file_sha(path)
        self.inventory_path=str(path)
        self.inventory_status=inventory["status"]
        first=self.training_scenarios[0]
        super().__init__(field,first["origin_lonlat"],first["start_utc"],config)
    def reset_for_training_scenario(self,seed,scenario_id):
        matches=[row for row in self.training_scenarios if row["id"]==scenario_id]
        if len(matches)!=1: raise ValueError("Requested scenario is not in training partition")
        row=matches[0]
        self.active_scenario_id=row["id"]
        self.start_utc=float(row["start_utc"])
        self.frame=LocalFrame(row["origin_lonlat"],self.cfg.region_m)
        return super().reset(seed)

    def reset(self,seed=0):
        scenario_rng=np.random.default_rng(np.random.SeedSequence([seed,991206]))
        row=self.training_scenarios[int(scenario_rng.integers(len(self.training_scenarios)))]
        self.active_scenario_id=row["id"]
        self.start_utc=float(row["start_utc"])
        self.frame=LocalFrame(row["origin_lonlat"],self.cfg.region_m)
        return super().reset(seed)
    def forcing_metadata(self):
        meta=super().forcing_metadata()
        meta.update(inventory_path=self.inventory_path,inventory_sha256=self.inventory_sha256,
                    inventory_status=self.inventory_status,allowed_split="train",
                    training_scenario_ids=[r["id"] for r in self.training_scenarios])
        return meta
