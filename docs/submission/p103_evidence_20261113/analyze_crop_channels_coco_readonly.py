import csv,json,hashlib,statistics,collections,datetime
from pathlib import Path
import numpy as np
from pycocotools import mask as coco
p=Path('/srv/almacen/nas-2tb-1/solar-filament-2026/paired-small-20261112')
cur=p/'p103_full_public_stage_coupling_20261113'
hist=p.parent/'final-release-staging-20261112/h9/official/crops'
def rows(n):
 with (p/n).open() as f:
  return list(csv.DictReader(f))
names={'reconstructed':'p103_full_public_stage_coupling_20261113/p103_180_reconstructed.csv','accepted':'p74_accepted_artifact_audit/accepted_h9b_nosite.csv'}
data={k:rows(n) for k,n in names.items()}
group={}
for k,records in data.items():
 group[k]=collections.defaultdict(list)
 for r in records:group[k][r['filament_id'].rsplit('_',1)[0]].append(r)
matched=[];bijective=True;exact_count=0
for image,pred in group['reconstructed'].items():
 ref=group['accepted'][image]
 a=[{'size':[2048,2048],'counts':r['segmentation_rle'].encode()} for r in pred]
 b=[{'size':[2048,2048],'counts':r['segmentation_rle'].encode()} for r in ref]
 ia=coco.area(a).astype(np.int64);ib=coco.area(b).astype(np.int64)
 matrix=coco.iou(a,b,[0]*len(b))
 best=np.argmax(matrix,axis=1)
 bijective=bijective and len(set(map(int,best)))==len(b)
 for i,j in enumerate(best):
  iou=float(matrix[i,j]);total=int(ia[i])+int(ib[j]);intersection=round(iou*total/(1+iou))
  sym=total-2*intersection
  exact_count+=sym==0
  if sym:matched.append({'id':pred[i]['filament_id'],'best_accepted_id':ref[j]['filament_id'],'iou':iou,'symmetric_difference_pixels':sym})
channels=[{'changed_elements':0,'total_elements':0,'max_absolute_byte_difference':0,'sum_absolute_byte_difference':0,'image_mismatches':0} for _ in range(4)]
crop_records={}
for f in sorted((cur/'crops').glob('*.npz')):
 with np.load(f,allow_pickle=False) as a,np.load(hist/f.name,allow_pickle=False) as b:
  x=a['X'];y=b['X'];assert x.shape==y.shape and x.dtype==np.uint8
  assert x.shape[1:]==(4,128,384)
  row={}
  for k in range(4):
   d=np.abs(x[:,k].astype(np.int16)-y[:,k].astype(np.int16))
   changed=int(np.count_nonzero(d));mx=int(d.max());sm=int(d.sum())
   row[str(k)]={'changed_elements':changed,'max_absolute_byte_difference':mx,'sum_absolute_byte_difference':sm}
   z=channels[k];z['changed_elements']+=changed;z['total_elements']+=int(d.size);z['max_absolute_byte_difference']=max(z['max_absolute_byte_difference'],mx);z['sum_absolute_byte_difference']+=sm;z['image_mismatches']+=changed>0
  crop_records[f.stem]=row
out={'analysis':'Independent pycocotools RLE geometry crosscheck plus read-only NumPy crop channel comparison; no models, raw JPEGs, test labels or remote writes.','generated_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'input_sha256':{k:hashlib.sha256((p/n).read_bytes()).hexdigest() for k,n in names.items()},'numpy_version':np.__version__,'pycocotools_bijective':bool(bijective),'exact_masks':exact_count,'changed_masks':len(matched),'changed_masks_symmetric_difference_sum':sum(r['symmetric_difference_pixels'] for r in matched),'changed_masks_max_symmetric_difference':max(r['symmetric_difference_pixels'] for r in matched),'changed_masks_min_iou':min(r['iou'] for r in matched),'changed_masks_median_iou':statistics.median(r['iou'] for r in matched),'channel_names':['flat_field_intensity','fused_probability','candidate_seed','other_candidates'],'crop_channels':channels,'crop_channel_records':crop_records}
print(json.dumps(out))

