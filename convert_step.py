from pathlib import Path

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

try:
    import omni.kit.app

    app = omni.kit.app.get_app()
    ext_manager = app.get_extension_manager()

    print("Enabling omni.kit.converter.hoops_core ...")
    ext_manager.set_extension_enabled_immediate(
        "omni.kit.converter.hoops_core", True
    )

    for _ in range(30):
        app.update()

    import omni.converter.hoops
    from omni.kit.converter.hoops_core import HoopsOptions

    root = Path(__file__).resolve().parent
    input_path = (
        root
        / "source"
        / "wheel_legged_robot"
        / "wheel_legged_robot"
        / "assets"
        / "terrain"
        / "RMUC2026_V2.0.0.stp"
    )
    output_path = input_path.with_suffix(".usd")

    if not input_path.exists():
        raise FileNotFoundError(f"STEP file not found: {input_path}")

    print(f"Input : {input_path}")
    print(f"Output: {output_path}")

    options = HoopsOptions()
    options.tessLOD = 1
    options.upAxis = omni.converter.hoops.UpAxis.eZup
    options.filterStyle = omni.converter.hoops.FilterStyle.eOmit
    options.compositionStyle = omni.converter.hoops.CompositionStyle.eNone
    options.useMaterials = False
    options.useNormals = True
    options.convertCurves = False
    options.convertMetadata = False
    options.convertPhysicsData = False
    options.dedup = True
    options.reportProgress = True

    print("Starting STEP -> USD conversion...")
    print("The STEP file is very large; this can take a while.")

    converter = omni.converter.hoops.Converter(options)
    result_code, error_msg = converter.convert(
        str(input_path),
        str(output_path),
        {},
    )

    print(f"Converter result: {result_code}")
    print(f"Converter message: {error_msg}")

    if not output_path.exists():
        raise RuntimeError(
            f"Conversion did not create the USD file. "
            f"result={result_code}, message={error_msg}"
        )

    size_mb = output_path.stat().st_size / 1024 / 1024
    print("STEP -> USD SUCCESS")
    print(f"USD : {output_path}")
    print(f"Size: {size_mb:.1f} MB")

finally:
    simulation_app.close()
