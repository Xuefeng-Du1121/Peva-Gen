"""Build a development-only ocean scenario inventory from an explicit JSON specification."""
import argparse,json
from pathlib import Path
import numpy as np
from .env import Config
from .ocean_field import OceanField
from .forcing_scenarios import support,validate_split_support
from .protocol import file_sha
ROOT=Path(__file__).resolve().parents[2]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--field",required=True);ap.add_argument("--spec",required=True);ap.add_argument("--out",required=True)
    a=ap.parse_args()
    def resolve_input(value):
        path = Path(value).expanduser()
        return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    field_path, spec_path = resolve_input(a.field), resolve_input(a.spec)
    out = (ROOT / Path(a.out)).resolve() if not Path(a.out).is_absolute() else Path(a.out).resolve()
    if not out.is_relative_to(ROOT):
        ap.error("output must remain inside the code repository")
    if out.exists(): ap.error("Refusing overwrite")
    spec=json.loads(spec_path.read_text())
    config=Config(**spec["config"])
    field=OceanField(field_path)
    rows=[]
    for row in spec["scenarios"]:
        region=support(field,row["origin_lonlat"],row["start_utc"],config.region_m,config.horizon_s)
        slices=tuple(slice(region[k+"_indices"][0],region[k+"_indices"][1]+1) for k in ("time","latitude","longitude"))
        if not np.isfinite(field.u[slices]).all() or not np.isfinite(field.v[slices]).all():
            raise ValueError("Missing/land node in complete episode support: "+row["id"])
        rows.append(dict(row,support=region))
    audit=validate_split_support(rows)
    report=dict(schema="ocean-forcing-inventory-v1",status="development candidate; not frozen formal protocol",
                field_path=str(field_path),field_sha256=file_sha(field_path),spec_sha256=file_sha(spec_path),
                source_sha256=file_sha(__file__),config=vars(config),scenarios=rows,audit=audit,
                limitations=["Historical analysis, not issue-time forecasts","No recorded test targets supplied by this inventory",
                             "Environment support disjointness does not establish unseen trajectory labels"])
    with out.open("x") as f: json.dump(report,f,indent=2)
    print(json.dumps(audit),flush=True)

if __name__=="__main__": main()
