"""Complete evaluation of a frozen best checkpoint without repeating training."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from .data import data_root, load_cache, load_records, split_from_manifest
from .evaluate import GroupScorer, SEMANTIC_GRID
from .train_unet import build_model, predict_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', required=True, type=Path)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--record', required=True, type=Path)
    parser.add_argument('--data', required=True)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--amp-dtype', choices=('float16', 'bfloat16'), default='float16',
                        help='Match the original T4 evaluation precision when recovering on Ampere')
    args = parser.parse_args()
    os.environ['SOLARFIL_DATA'] = args.data
    os.environ['SOLARFIL_AMP_DTYPE'] = args.amp_dtype
    assert not args.out.exists(), 'Preserve existing evaluation; select a new output path'
    assert torch.cuda.is_available(), 'Use the managed compatible GPU queue'
    device = torch.device('cuda')
    record = json.loads(args.record.read_text())
    model = build_model(record['args']['arch'], record['args']['encoder'], None).to('cuda')
    model.load_state_dict(torch.load(args.checkpoint, map_location='cuda', weights_only=True))
    images, labels, metadata = load_cache(args.cache)
    groups = split_from_manifest(metadata, args.manifest)
    _, full = load_records(data_root())
    file_index = {name: i for i, name in enumerate(metadata['files'])}
    calibration = GroupScorer(groups['calibration'], metadata, labels, full)
    test = GroupScorer(groups['test'], metadata, labels, full)
    cal_predictions = predict_files(model, images, file_index, calibration.files(), device, tta=False)
    config, cal_score, table = calibration.select(cal_predictions, SEMANTIC_GRID)
    predictions = predict_files(model, images, file_index, test.files(), device, tta=False)
    test_score = test.score(predictions, config, native=True)
    result = dict(checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                  seed=record['args']['seed'], kind='frozen_checkpoint_evaluation_recovery',
                  selected_on='calibration only', selected_cfg=config,
                  gpu=torch.cuda.get_device_name(0), amp_dtype=args.amp_dtype,
                  calibration=cal_score, test=test_score, native_table=table,
                  note='No training repeated; original incomplete run record preserved')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
