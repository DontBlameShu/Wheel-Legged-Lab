from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

from pathlib import Path
from pxr import Usd, UsdGeom, UsdPhysics, Gf

MANAGER = Path(r"C:\Users\shuha\Desktop\Wheel-Legged-Lab\source\wheel_legged_robot\wheel_legged_robot\tasks\manager_based")
ASSET_DIR = MANAGER / "wheel_legged_robot" / "assets" / "terrain"
SRC = ASSET_DIR / "RMUC2026_V2.0.0.usd"
DST = ASSET_DIR / "RMUC2026_V2.0.0_PLAY.usd"

stage = Usd.Stage.Open(str(SRC))
if stage is None:
    raise RuntimeError(f"Cannot open: {SRC}")

xcache = UsdGeom.XformCache(Usd.TimeCode.Default())
points = []
indices = []
counts = []
offset = 0
mesh_count = 0

for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
    if not prim.IsA(UsdGeom.Mesh):
        continue
    mesh = UsdGeom.Mesh(prim)
    p = mesh.GetPointsAttr().Get() or []
    fc = mesh.GetFaceVertexCountsAttr().Get() or []
    fi = mesh.GetFaceVertexIndicesAttr().Get() or []
    xf = xcache.GetLocalToWorldTransform(prim)

    for v in p:
        points.append(xf.Transform(Gf.Vec3d(float(v[0]), float(v[1]), float(v[2]))))

    pos = 0
    for c in fc:
        c = int(c)
        face = fi[pos:pos + c]
        pos += c
        if c < 3:
            continue
        for j in range(1, c - 1):
            counts.append(3)
            indices.extend((offset + int(face[0]), offset + int(face[j]), offset + int(face[j + 1])))

    offset += len(p)
    mesh_count += 1
    if mesh_count % 250 == 0:
        print(f"[RMUC] processed meshes: {mesh_count}")

out = Usd.Stage.CreateNew(str(DST))
UsdGeom.SetStageUpAxis(out, UsdGeom.GetStageUpAxis(stage))
UsdGeom.SetStageMetersPerUnit(out, UsdGeom.GetStageMetersPerUnit(stage))

mesh = UsdGeom.Mesh.Define(out, "/RMUC")
out.SetDefaultPrim(mesh.GetPrim())
mesh.CreatePointsAttr(points)
mesh.CreateFaceVertexCountsAttr(counts)
mesh.CreateFaceVertexIndicesAttr(indices)
mesh.CreateSubdivisionSchemeAttr("none")

collision = UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
collision.CreateCollisionEnabledAttr(True)
mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
mesh_collision.CreateApproximationAttr("none")

out.GetRootLayer().Save()

print("[RMUC] MERGED MESHES:", mesh_count)
print("[RMUC] VERTICES:", len(points))
print("[RMUC] TRIANGLES:", len(counts))
print("[RMUC] PLAY USD:", DST)

cfg_path = MANAGER / "wisc_stairs" / "wisc_stairs_env_cfg.py"
cfg_text = cfg_path.read_text(encoding="utf-8")
marker = "# --- RMUC PLAY OVERRIDE ---"
cfg_text = cfg_text.split(marker)[0].rstrip() + "\n\n"
cfg_text += '''# --- RMUC PLAY OVERRIDE ---
import os as _os

if _os.environ.get("WISC_RMUC_PLAY") == "1":
    _rmuc_original_init = WiscStairsEnvCfg.__init__

    def _rmuc_play_init(self, *args, **kwargs):
        _rmuc_original_init(self, *args, **kwargs)
        _rmuc_play_usd = (
            Path(__file__).resolve().parent.parent
            / "wheel_legged_robot"
            / "assets"
            / "terrain"
            / "RMUC2026_V2.0.0_PLAY.usd"
        )
        self.terrain_cfg = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="usd",
            usd_path=str(_rmuc_play_usd),
            env_spacing=1.0,
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=0.5,
                dynamic_friction=0.5,
                restitution=0.0,
            ),
            debug_vis=False,
        )
        if hasattr(self, "scene") and hasattr(self.scene, "num_envs"):
            self.scene.num_envs = 1

    WiscStairsEnvCfg.__init__ = _rmuc_play_init
'''
cfg_path.write_text(cfg_text, encoding="utf-8")
print("[RMUC] patched play-only terrain override:", cfg_path)

play_path = MANAGER / "wisc_stairs" / "play.py"
play_text = play_path.read_text(encoding="utf-8")
if 'os.environ["WISC_RMUC_PLAY"] = "1"' not in play_text:
    anchor = "from isaaclab.app import AppLauncher"
    if anchor not in play_text:
        raise RuntimeError("Could not find AppLauncher import in play.py")
    play_text = play_text.replace(
        anchor,
        'import os\nos.environ["WISC_RMUC_PLAY"] = "1"\n\n' + anchor,
        1,
    )
    play_path.write_text(play_text, encoding="utf-8")
print("[RMUC] play.py now always uses RMUC map:", play_path)

app.close()
