import os, sys, glob
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "2"
import numpy as np
from multiprocessing import Pool

R = r"C:\DEV\ai-voice-conversion"
D = os.path.join(R, r"runs\E1-base\diagnosis")
C = os.path.join(D, "_cache")
sys.path.insert(0, D)


def job(args):
    name, path, sal, crepe = args
    out = os.path.join(C, f"feat_{name}.npz")
    if os.path.exists(out):
        return name
    import features
    y = features.load_mono(path, 44100)
    f = features.compute(y, with_salience=sal, with_crepe=crepe)
    np.savez_compressed(out, **f)
    return name


if __name__ == "__main__":
    jobs = [
        ("src", os.path.join(C, "src_vocals.wav"), True, True),
        ("c300", os.path.join(C, "conv_300e_est.wav"), True, True),
        ("c500", os.path.join(C, "conv_500e_est.wav"), True, True),
        ("ref", glob.glob(R + r"\dataset\private\separated\reference\*(vocals)*.wav")[0], False, True),
    ]
    for p in sorted(glob.glob(R + r"\dataset\private\train\*.wav")):
        jobs.append(("train__" + os.path.basename(p)[:-4], p, False, False))
    with Pool(12) as pool:
        for n in pool.imap_unordered(job, jobs):
            print("done", n, flush=True)
