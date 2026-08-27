from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

from pxr import Usd, UsdGeom, UsdPhysics, Gf
import os

src = r"C:\Users\shuha\Desktop\Wheel-Legged-Lab\source\wheel_legged_robot\wheel_legged_robot\tasks\manager_based\wheel_legged_robot\assets\terrain\RMUC2026_V2.0.0.usd"
dst = r"C:\Users\shuha\Desktop\Wheel-Legged-Lab\source\wheel_legged_robot\wheel_legged_robot\tasks\manager_based\wheel_legged_robot\assets\terrain\RMUC2026_V2.0.0_PLAY.usd"

s = Usd.Stage.Open(src)
if s is None:
    raise RuntimeError(f"Cannot open source USD: {src}")

xc = UsdGeom.XformCache(Usd.TimeCode.Default())
pts = []
inds = []
counts = []
off = 0
nm = 0

for prim in s.Traverse():
    if not prim.IsA(UsdGeom.Mesh):
        continue
    m = UsdGeom.Mesh(prim)
    p = m.GetPointsAttr().Get() or []
    fc = m.GetFaceVertexCountsAttr().Get() or []
    fi = m.GetFaceVertexIndicesAttr().Get() or []
    xf = xc.GetLocalToWorldTransform(prim)
    pts.extend(xf.Transform(Gf.Vec3d(float(v[0]), float(v[1]), float(v[2]))) for v in p)
    pos = 0
    for c in fc:
        c = int(c)
        face = fi[pos:pos+c]
        pos += c
        if c < 3:
            continue
        for j in range(1, c - 1):
            counts.append(3)
            inds.extend((off + int(face[0]), off + int(face[j]), off + int(face[j+1])))
    off += len(p)
    nm += 1
    if nm % 250 == 0:
        print(f"processed meshes: {nm}")

o = Usd.Stage.CreateNew(dst)
UsdGeom.SetStageUpAxis(o, UsdGeom.GetStageUpAxis(s))
UsdGeom.SetStageMetersPerUnit(o, UsdGeom.GetStageMetersPerUnit(s))
root = UsdGeom.Xform.Define(o, "/RMUC")
o.SetDefaultPrim(root.GetPrim())

visual = UsdGeom.Xform.Define(o, "/RMUC/Visual")
visual.GetPrim().GetReferences().AddReference(os.path.basename(src))

mesh = UsdGeom.Mesh.Define(o, "/RMUC/CollisionMesh")
mesh.CreatePointsAttr(pts)
mesh.CreateFaceVertexCountsAttr(counts)
mesh.CreateFaceVertexIndicesAttr(inds)
mesh.CreateSubdivisionSchemeAttr("none")
mesh.CreateVisibilityAttr("invisible")

collision = UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
collision.CreateCollisionEnabledAttr(True)
mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
mesh_collision.CreateApproximationAttr("none")

o.GetRootLayer().Save()

print("MERGED MESHES:", nm)
print("VERTICES:", len(pts))
print("TRIANGLES:", len(counts))
print("PLAY USD:", dst)

app.close()
