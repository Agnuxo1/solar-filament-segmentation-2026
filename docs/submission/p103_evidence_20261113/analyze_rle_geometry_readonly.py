import csv,json,hashlib,statistics,collections
from pathlib import Path
p=Path('/srv/almacen/nas-2tb-1/solar-filament-2026/paired-small-20261112')
names={'reconstructed':'p103_full_public_stage_coupling_20261113/p103_180_reconstructed.csv','accepted':'p74_accepted_artifact_audit/accepted_h9b_nosite.csv'}
def rows(n):
 with (p/n).open() as f:
  return list(csv.DictReader(f))
data={k:rows(n) for k,n in names.items()}
assert [x['filament_id'] for x in data['accepted']]==[x['filament_id'] for x in data['reconstructed']]
def runs(s):
 counts=[];pos=0
 while pos<len(s):
  value=0;shift=0
  while True:
   c=ord(s[pos])-48;pos+=1
   value |= (c&31)<<shift
   shift+=5
   if not c&32:
    if c&16:value|=(-1)<<shift
    break
  if len(counts)>2:value+=counts[-2]
  assert value>=0
  counts.append(value)
 assert sum(counts)==2048*2048
 out=[];offset=0
 for i,n in enumerate(counts):
  if i%2 and n:out.append((offset,offset+n))
  offset+=n
 return out
def area(r):return sum(b-a for a,b in r)
def inter(a,b):
 i=j=out=0
 while i<len(a) and j<len(b):
  out+=max(0,min(a[i][1],b[j][1])-max(a[i][0],b[j][0]))
  if a[i][1]<b[j][1]:i+=1
  else:j+=1
 return out
parsed={k:{r['filament_id']:runs(r['segmentation_rle']) for r in rows} for k,rows in data.items()}
byimage=collections.defaultdict(list)
for r in data['accepted']:byimage[r['filament_id'].rsplit('_',1)[0]].append(r['filament_id'])
changed=[]
for pred,ref in zip(data['reconstructed'],data['accepted']):
 if pred['segmentation_rle']==ref['segmentation_rle']:continue
 ident=pred['filament_id']; a=parsed['reconstructed'][ident];b=parsed['accepted'][ident]
 aa=area(a);bb=area(b);ii=inter(a,b);uu=aa+bb-ii
 best=(-1,None)
 for rid in byimage[ident.rsplit('_',1)[0]]:
  c=parsed['accepted'][rid];ic=inter(a,c);union=aa+area(c)-ic
  score=ic/union if union else 1.0
  if score>best[0]:best=(score,rid)
 changed.append({'id':ident,'reconstructed_area':aa,'accepted_area':bb,'intersection':ii,'union':uu,'iou':ii/uu if uu else 1.0,'symmetric_difference_pixels':aa+bb-2*ii,'best_accepted_id':best[1],'best_accepted_iou':best[0]})
def stats(vals):
 vals=sorted(vals)
 return {'count':len(vals),'min':min(vals),'median':statistics.median(vals),'max':max(vals),'sum':sum(vals)}
out={'analysis':'Read-only RLE geometry comparison; masks are own predictions, not ground truth. No images or models loaded. Row ordering checked by filament_id. COCO runs decoded with nonnegative counts and exactly 2048x2048 total pixels.','input_sha256':{k:hashlib.sha256((p/n).read_bytes()).hexdigest() for k,n in names.items()},'rows':len(data['accepted']),'changed_rows':len(changed),'changed_rows_decoding_to_identical_mask':sum(x['symmetric_difference_pixels']==0 for x in changed),'changed_rows_best_match_different_position':sum(x['id']!=x['best_accepted_id'] for x in changed),'changed_rows_iou':stats([x['iou'] for x in changed]),'changed_rows_symmetric_difference_pixels':stats([x['symmetric_difference_pixels'] for x in changed]),'changed_row_details':changed}
print(json.dumps(out))

