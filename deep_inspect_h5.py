# deep_inspect_h5.py
import h5py
import json
import sys

file_path = "model_emotion_full.h5"   # change if your filename is different

try:
    f = h5py.File(file_path, 'r')
except Exception as e:
    print("ERROR opening file:", e)
    sys.exit(1)

def print_weights(group, prefix=""):
    for name in sorted(group.keys()):
        obj = group[name]
        if isinstance(obj, h5py.Group):
            # dive in
            print_weights(obj, prefix + name + "/")
        else:
            # dataset: print shape and dtype
            try:
                print(f"{prefix}{name} -> shape={obj.shape}, dtype={obj.dtype}")
            except Exception as e:
                print(f"{prefix}{name} -> <could not read shape: {e}>")

print("Top-level keys:", list(f.keys()))
if "model_weights" in f:
    print("\n--- model_weights contents ---")
    mw = f["model_weights"]
    # list top-level layer names inside model_weights
    for layer_name in mw.keys():
        print("\nLayer group:", layer_name)
        sub = mw[layer_name]
        # print datasets in this layer group
        print_weights(sub, prefix=layer_name + "/")
else:
    # fallback: print everything
    print("\n--- full file tree ---")
    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"{name} -> shape={obj.shape} dtype={obj.dtype}")
    f.visititems(visitor)

f.close()
print("\nDone.")
