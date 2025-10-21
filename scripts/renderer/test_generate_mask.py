import subprocess

def main():
    envs = [
        "../../test_input/abandoned_church/abandoned_church_original.exr",
        "../../test_input/brick_factory/brick_factory_original.exr",
        "../../test_input/cape_hill/cape_hill_original.exr",
        "../../test_input/immenstadter_horn/immenstadter_horn_original.exr",
        "../../test_input/industrial_sunset/industrial_sunset_original.exr",
        "../../test_input/kloofendal_48d_partly_cloudy/kloofendal_48d_partly_cloudy_original.exr",
        "../../test_input/park_bench/park_bench_original.exr",
        "../../test_input/piazza_martin_lutero/piazza_martin_lutero_original.exr",
        "../../test_input/street_lamp/street_lamp_original.exr",
        "../../test_input/table_mountain/table_mountain_original.exr",
    ]

    names = []
    for env in envs:
        name = env.split("/")[-1].split("_original")[0]
        names.append(name)

    deltas = [0.01, 0.001, 0.0008, 0.0005]

    for delta in deltas:
        for (name, env) in zip(names, envs):
            print(f"Processing {name} with delta {delta}")
            subprocess.run([
                "python", "generate_mask.py",
                "--input_bsr_path", "../../T_bsr.npz",
                "--input_env_path", env,
                "--input_stroke_path", "../../resources/stroke.exr",
                "--input_albedo_path", "../../resources/albedo.exr",
                "--output_mask_path", f"out/{name}_mask_{delta}.exr",
                "--output_env_path", f"out/{name}_env_{delta}.exr",
                "--mult_constant", "1e-4",
                "--delta", f"{delta}",
                "--resize_env",
            ], check=True)

if __name__ == "__main__":
    main()