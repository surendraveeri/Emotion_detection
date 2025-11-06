import h5py

file_path = "model_emotion.h5"        # 👈 change to the exact filename

with h5py.File(file_path, "r") as f:
    print("\n🔹 Top-level keys:", list(f.keys()))

    if "model_weights" in f:
        print("\n🔹 Layers inside 'model_weights':")
        for name in f["model_weights"].keys():
            print("  -", name)

    if "optimizer_weights" in f:
        print("\n🔹 Optimizer weights found:", list(f["optimizer_weights"].keys()))
