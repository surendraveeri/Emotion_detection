import h5py

with h5py.File("model_emotion_full.h5", "r") as f:
    print("\n🔹 Top-level groups in model file:")
    for key in f.keys():
        print("  ", key)

    print("\n🔹 Listing layers:")
    for i, layer_name in enumerate(f.keys()):
        try:
            weights = f[layer_name][layer_name]
            print(f"{i+1:02d}. {layer_name} ->", [w for w in weights.keys()])
        except:
            pass

    print("\n✅ File analyzed successfully.")
