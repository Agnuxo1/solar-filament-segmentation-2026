import ast,struct,zipfile,json,hashlib
from pathlib import Path
p=Path('/srv/almacen/nas-2tb-1/solar-filament-2026/paired-small-20261112')
cur=p/'p103_full_public_stage_coupling_20261113'
ref=p.parent/'final-release-staging-20261112/h9/official/crops'
def array(path,key):
 with zipfile.ZipFile(path) as z:
  b=z.read(key+'.npy')
 assert b[:6]==b'\x93NUMPY'
 version=b[6]
 if version==1:
  n=struct.unpack('<H',b[8:10])[0];offset=10
 else:
  n=struct.unpack('<I',b[8:12])[0];offset=12
 header=ast.literal_eval(b[offset:offset+n].decode('latin1').strip())
 dtype=header['descr']; assert not header['fortran_order']
 spec={'<f4':'f','<f8':'d','<i4':'i','<i8':'q','|u1':'B','<u4':'I','<u8':'Q'}[dtype]
 raw=b[offset+n:]
 values=struct.unpack('<'+str(len(raw)//struct.calcsize(spec))+spec,raw)
 return {'shape':header['shape'],'dtype':dtype,'values':values}
out={}
for receipt in sorted((cur/'crop_receipts').glob('*.json')):
 r=json.loads(receipt.read_text());c=r['comparison']
 if c['M']['exact'] and c['score']['exact']:continue
 name=receipt.stem
 row={}
 for field in ['M','score']:
  a=array(cur/'crops'/f'{name}.npz',field);b=array(ref/f'{name}.npz',field)
  assert a['shape']==b['shape'] and a['dtype']==b['dtype']
  dif=[{'index':i,'reconstructed':x,'historical':y,'absolute_difference':abs(x-y)} for i,(x,y) in enumerate(zip(a['values'],b['values'])) if x!=y]
  row[field]={'shape':a['shape'],'dtype':a['dtype'],'different_elements':dif}
  if field=='score':
   row['score_sort_reconstructed']=sorted(range(len(a['values'])),key=lambda i:-a['values'][i])
   row['score_sort_historical']=sorted(range(len(b['values'])),key=lambda i:-b['values'][i])
 row['rank_changed']=row['score_sort_reconstructed']!=row['score_sort_historical']
 out[name]=row
print(json.dumps({'scope':'Read-only archived NPY M/score arrays; no images, models, GPU, labels or remote writes.','images_with_M_or_score_difference':len(out),'images_with_ranking_difference':sum(r['rank_changed'] for r in out.values()),'records':out}))

