#!/usr/bin/env python3
"""Independently verify completed P88 fold predictions against the original P69 CPU reference.

Only original unlabelled JPEGs and detector outputs are read; no GPU, GT or Kaggle activity.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import pathlib
import pickle
import numpy as np

def sha(path):
    """Stream a complete file SHA-256 digest."""
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1048576),b""): h.update(block)
    return h.hexdigest()

def same(a,b):
    """Require identical ordered float scores, boxes and crop mask bytes."""
    if len(a)!=len(b): return False
    return all(len(x)==len(y)==6 and x[:5]==y[:5] and np.array_equal(x[5],y[5]) for x,y in zip(a,b))

def selftest():
    """Reject score, box, mask and count corruption."""
    a=(0.5,1,2,3,4,np.ones((2,2),dtype=np.uint8))
    assert same([a],[a])
    assert not same([a],[])
    assert not same([a],[(0.6,1,2,3,4,a[5])])
    assert not same([a],[(0.5,2,2,3,4,a[5])])
    assert not same([a],[(0.5,1,2,3,4,np.zeros_like(a[5]))])

def audit(root,ref,images,fold):
    """Verify every original input and both historical reference detector views."""
    folder=root/fold
    m=folder/"manifest.json"
    r=ref/fold/"manifest.json"
    meta=json.loads(m.read_text())
    old=json.loads(r.read_text())
    inventory=sorted(images.glob("*.jpeg"))
    entries=meta.get("entries",{})
    originals=old.get("image_outputs",{})
    if meta.get("status")!="PUBLIC_SOURCE_CPU_EXACT_PASS" or old.get("status")!="CPU_BYTE_EXACT_PASS":
        raise ValueError("Unapproved completion state")
    if meta.get("checkpoint_sha256")!=old.get("original_weights_sha256"):
        raise ValueError("Checkpoint changed")
    if len(inventory)!=180 or len(entries)!=180 or len(list(folder.glob("*.pkl")))!=180:
        raise ValueError("Unexpected file count")
    if set(entries)!={p.name for p in inventory} or set(originals)!=set(entries):
        raise ValueError("Image inventory mismatch")
    n_instances=0
    for image in inventory:
        e=entries[image.name]
        a=folder/(image.stem+".pkl")
        b=ref/fold/(image.stem+".pkl")
        if e.get("both_views_bit_exact") is not True or sha(image)!=e["input_sha256"]:
            raise ValueError("Image hash mismatch: "+image.name)
        if sha(a)!=e["sha256"] or sha(b)!=originals[image.name]["sha256"]:
            raise ValueError("Prediction hash mismatch: "+image.name)
        with a.open("rb") as fh: pa=pickle.load(fh)
        with b.open("rb") as fh: pb=pickle.load(fh)
        if len(pa)!=2 or len(pb)!=2: raise ValueError("View count mismatch")
        for x,y in zip(pa,pb):
            if not same(x,y): raise ValueError("Bit-exact failure: "+image.name)
            n_instances+=len(x)
    if meta.get("bit_exact_images")!=180 or meta.get("bit_exact_views")!=360 or meta.get("exact_instances")!=n_instances:
        raise ValueError("Manifest summary mismatch")
    return dict(status=f"P88_{fold}_INDEPENDENT_FULL_DOMAIN_PASS",fold=fold,images=180,views=360,
                exact_instances=n_instances,manifest_sha256=sha(m),reference_manifest_sha256=sha(r),
                source_sha256=sha(pathlib.Path(__file__)),
                utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                cpu_only=True,no_test_labels=True,no_new_kaggle_submission=True,
                end_to_end_csv_reproduced=False)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=pathlib.Path)
    parser.add_argument("--reference",type=pathlib.Path)
    parser.add_argument("--images",type=pathlib.Path)
    parser.add_argument("--fold",choices=("F1","F2","F3","F4"))
    parser.add_argument("--self-test",action="store_true")
    a=parser.parse_args()
    selftest()
    if a.self_test:
        print("SELFTEST_OK")
        return
    if not all((a.root,a.reference,a.images,a.fold)): parser.error("Missing audit inputs")
    print(json.dumps(audit(a.root,a.reference,a.images,a.fold),indent=2,sort_keys=True))
if __name__=="__main__":
    main()
