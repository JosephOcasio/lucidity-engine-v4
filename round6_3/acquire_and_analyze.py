import hashlib, json, os, zipfile, io, time
from pathlib import Path
import requests
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
from shapely.geometry import box, mapping

OUT=Path('round6_3_output'); OUT.mkdir(exist_ok=True)
BBOX={'west':-5.9982,'south':35.8904,'east':-5.9534,'north':35.9282}
PAD={'west':-6.02,'south':35.87,'east':-5.93,'north':35.95}
urls={
 'gebco_tid':'https://dap.ceda.ac.uk/bodc/gebco/global/gebco_2026/type_identifier_grid/geotiff/gebco_2026_tid_geotiff.zip?download=1',
 'gmrt_mask':f"https://www.gmrt.org/services/GridServer?north={BBOX['north']}&west={BBOX['west']}&east={BBOX['east']}&south={BBOX['south']}&layer=topo-mask&format=geotiff&resolution=max",
 'gmrt_meta':f"https://www.gmrt.org/services/GridServer/metadata?north={PAD['north']}&west={PAD['west']}&east={PAD['east']}&south={PAD['south']}&format=geotiff&mformat=json&resolution=max"
}

def get(url,path,timeout=240):
 r=requests.get(url,timeout=timeout,headers={'User-Agent':'ACE-HCR-Round6.3/1.0'}); r.raise_for_status(); path.write_bytes(r.content); return {'status':r.status_code,'bytes':len(r.content),'sha256':hashlib.sha256(r.content).hexdigest(),'content_type':r.headers.get('content-type')}
receipt={'bbox':BBOX,'urls':urls,'downloads':{},'generated_utc':pd.Timestamp.utcnow().isoformat()}
# GEBCO TID global tiled zip
zpath=OUT/'gebco_2026_tid_geotiff.zip'; receipt['downloads']['gebco_tid']=get(urls['gebco_tid'],zpath)
with zipfile.ZipFile(zpath) as z:
 names=z.namelist(); receipt['gebco_zip_members']=names
 # tile containing lon -6, lat 36 should be N0.0_W90.0 or equivalent; inspect all tif bounds and crop first intersecting.
 z.extractall(OUT/'gebco_tid_tiles')
geom=box(BBOX['west'],BBOX['south'],BBOX['east'],BBOX['north'])
selected=[]
for fp in (OUT/'gebco_tid_tiles').rglob('*.tif'):
 with rasterio.open(fp) as ds:
  b=box(*ds.bounds)
  if b.intersects(geom):
   arr,tr=mask(ds,[mapping(geom)],crop=True,all_touched=False)
   profile=ds.profile.copy(); profile.update(height=arr.shape[1],width=arr.shape[2],transform=tr,compress='deflate')
   out=OUT/'SPT_GEO_001_GEBCO_2026_TID.tif'
   with rasterio.open(out,'w',**profile) as dst: dst.write(arr)
   selected.append({'source':str(fp),'bounds':list(ds.bounds),'shape':arr.shape})
   break
receipt['gebco_selected']=selected
# GMRT mask and metadata
mpath=OUT/'SPT_GEO_001_GMRT_topo_mask.tif'; receipt['downloads']['gmrt_mask']=get(urls['gmrt_mask'],mpath)
metap=OUT/'SPT_GEO_001_GMRT_metadata.json'; receipt['downloads']['gmrt_meta']=get(urls['gmrt_meta'],metap)
# summarize TID
TID_LABELS={0:'Land',10:'Singlebeam',11:'Multibeam',12:'Seismic',13:'Isolated sounding',14:'ENC sounding',15:'Lidar',16:'Depth contour',17:'Other direct measurement',40:'Predicted',41:'Interpolated',42:'Digital elevation model',43:'Satellite-derived',44:'Modelled',45:'Unknown indirect',70:'Pre-generated grid',71:'Mixed/other',72:'Unknown'}
with rasterio.open(OUT/'SPT_GEO_001_GEBCO_2026_TID.tif') as ds:
 a=ds.read(1,masked=True); vals=a.compressed().astype(int); u,c=np.unique(vals,return_counts=True); total=c.sum(); rows=[]
 for v,n in zip(u,c): rows.append({'tid':int(v),'label':TID_LABELS.get(int(v),'Other'), 'cells':int(n),'percent':100*n/total})
 pd.DataFrame(rows).to_csv(OUT/'SPT_GEO_001_GEBCO_TID_coverage.csv',index=False)
 direct=sum(r['cells'] for r in rows if 10<=r['tid']<=17); receipt['gebco_tid_summary']={'valid_cells':int(total),'direct_cells':int(direct),'direct_percent':100*direct/total if total else None,'classes':rows}
# summarize GMRT high-res mask; finite ocean cells are high-res by service definition
with rasterio.open(mpath) as ds:
 m=ds.read(1,masked=True); valid=np.isfinite(m.filled(np.nan)); receipt['gmrt_mask_summary']={'width':ds.width,'height':ds.height,'valid_cells':int(valid.sum()),'total_cells':int(valid.size),'valid_percent':100*valid.sum()/valid.size if valid.size else None,'bounds':list(ds.bounds),'crs':str(ds.crs)}
# hashes
for fp in OUT.rglob('*'):
 if fp.is_file() and fp.name!='round6_3_execution_receipt.json':
  receipt.setdefault('output_hashes',{})[str(fp.relative_to(OUT))]=hashlib.sha256(fp.read_bytes()).hexdigest()
(OUT/'round6_3_execution_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
print(json.dumps(receipt,indent=2))
