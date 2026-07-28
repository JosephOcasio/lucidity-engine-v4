#!/usr/bin/env python3
"""ACE HCR Round 6.6 open-data acquisition and provenance analysis.

Downloads only openly accessible sources. Request-gated TV-GIB material is
registered but never represented as downloaded. Designed for macOS/Linux.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, re, zipfile
from pathlib import Path
from urllib.parse import urljoin, urlparse

import numpy as np
import requests
import rasterio
from bs4 import BeautifulSoup
from pyproj import Geod
from rasterio.features import geometry_mask
from rasterio.mask import mask
from shapely.geometry import shape, mapping

UA = "ACE-HCR-Round6.6/1.0 (research acquisition; contact: founder@ocasioconvergence.com)"
GEOD = Geod(ellps="WGS84")
DOIS = {
    "seanoe_dtm": "10.17882/109440",
    "seanoe_sparker": "10.17882/109442",
    "tv_gib": "10.17600/3020070",
    "ras_spartel_paper": "10.1016/j.geomorph.2026.110387",
    "pangaea_msm36": "10.1594/PANGAEA.893200",
}
GEBCO_TID_URL = "https://dap.ceda.ac.uk/bodc/gebco/global/gebco_2026/type_identifier_grid/geotiff/gebco_2026_tid_geotiff.zip?download=1"
TID_LABELS = {
    0:"Land",10:"Singlebeam",11:"Multibeam",12:"Seismic",13:"Isolated sounding",
    14:"ENC sounding",15:"Lidar",16:"Depth contour",17:"Other direct measurement",
    40:"Predicted",41:"Interpolated",42:"DEM",43:"Satellite-derived",44:"Modelled",
    45:"Unknown indirect",70:"Pre-generated grid",71:"Mixed/other",72:"Unknown"
}
RELEVANT_EXT = (".tif",".tiff",".nc",".grd",".xyz",".csv",".zip",".sgy",".segy",".pdf",".json")

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()

def safe_name(url: str, fallback: str) -> str:
    name=Path(urlparse(url).path).name or fallback
    name=re.sub(r"[^A-Za-z0-9._-]+","_",name)
    return name[:180]

def get(session, url, path: Path, timeout=300, max_bytes=None):
    path.parent.mkdir(parents=True,exist_ok=True)
    with session.get(url,stream=True,timeout=timeout,allow_redirects=True) as r:
        r.raise_for_status()
        total=int(r.headers.get("content-length") or 0)
        if max_bytes and total and total>max_bytes:
            return {"status":"SKIPPED_SIZE","url":url,"final_url":r.url,"content_length":total}
        written=0
        with path.open("wb") as f:
            for chunk in r.iter_content(1024*1024):
                if not chunk: continue
                written += len(chunk)
                if max_bytes and written>max_bytes:
                    f.close(); path.unlink(missing_ok=True)
                    return {"status":"SKIPPED_STREAM_LIMIT","url":url,"final_url":r.url,"bytes":written}
                f.write(chunk)
    return {"status":"DOWNLOADED","url":url,"final_url":r.url,"bytes":path.stat().st_size,"sha256":sha256(path),"content_type":r.headers.get("content-type")}

def row_areas(ds):
    arr=np.empty(ds.height,dtype=float); t=ds.transform
    for row in range(ds.height):
        lat0=t.f+row*t.e; lat1=t.f+(row+1)*t.e; lon0=t.c; lon1=t.c+t.a
        a,_=GEOD.polygon_area_perimeter([lon0,lon1,lon1,lon0],[lat0,lat0,lat1,lat1])
        arr[row]=abs(a)/1e6
    return arr

def area(mask_arr, row_area):
    return float(np.dot(np.count_nonzero(mask_arr,axis=1),row_area))

def datacite(session, doi, outdir):
    outdir.mkdir(parents=True,exist_ok=True)
    key=doi.replace("/","_").replace(".","_")
    record={"doi":doi}
    for suffix,label in [("","metadata"),("/media","media")]:
        url=f"https://api.datacite.org/dois/{doi}{suffix}"
        try:
            r=session.get(url,timeout=60); record[label+"_http"]=r.status_code
            p=outdir/f"datacite_{key}_{label}.json"; p.write_text(r.text,encoding="utf-8")
            if r.ok and label=="metadata":
                obj=r.json(); attrs=obj.get("data",{}).get("attributes",{})
                titles=attrs.get("titles") or [{}]
                record.update({"title":titles[0].get("title"),"publisher":attrs.get("publisher"),"landing_url":attrs.get("url"),"registered":attrs.get("registered"),"published":attrs.get("published")})
        except Exception as e: record[label+"_error"]=repr(e)
    return record

def inspect_landing(session, name, url, outdir, download_cap=450_000_000):
    outdir.mkdir(parents=True,exist_ok=True)
    rec={"name":name,"url":url,"candidates":[]}
    if not url: rec["status"]="NO_URL"; return rec
    try:
        r=session.get(url,timeout=90,allow_redirects=True); rec.update({"status":"FETCHED","http":r.status_code,"final_url":r.url,"content_type":r.headers.get("content-type")})
        p=outdir/f"landing_{name}.html"; p.write_bytes(r.content)
        soup=BeautifulSoup(r.text,"html.parser")
        links=[]
        for a in soup.find_all("a",href=True):
            u=urljoin(r.url,a["href"]); low=u.lower(); text=" ".join(a.get_text(" ",strip=True).split())
            if any(ext in low for ext in RELEVANT_EXT) or "download" in low or "file" in text.lower(): links.append((u,text))
        seen=set(); total_dl=0
        for idx,(u,text) in enumerate(links):
            if u in seen: continue
            seen.add(u); item={"url":u,"text":text[:250]}
            try:
                h=session.head(u,timeout=45,allow_redirects=True); item.update({"http":h.status_code,"final_url":h.url,"content_type":h.headers.get("content-type"),"content_length":int(h.headers.get("content-length") or 0)})
            except Exception as e: item["head_error"]=repr(e)
            low=(item.get("final_url") or u).lower(); size=item.get("content_length",0)
            if any(low.endswith(ext) or ext+"?" in low for ext in RELEVANT_EXT) and (not size or size<=download_cap) and total_dl<750_000_000:
                dest=outdir/"downloads"/name/safe_name(item.get("final_url") or u,f"file_{idx}")
                try:
                    dl=get(session,u,dest,timeout=600,max_bytes=download_cap)
                    item["download"]=dl
                    if dl.get("status")=="DOWNLOADED": total_dl += dl["bytes"]
                except Exception as e: item["download_error"]=repr(e)
            rec["candidates"].append(item)
        (outdir/f"landing_{name}_links.json").write_text(json.dumps(rec,indent=2),encoding="utf-8")
    except Exception as e: rec.update({"status":"ERROR","error":repr(e)})
    return rec

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out",default="round6_6_output"); ap.add_argument("--geometry",default="round6_6/SPT_GEO_001_candidate_zones.geojson"); ap.add_argument("--dry-run",action="store_true")
    args=ap.parse_args(); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    geom_doc=json.loads(Path(args.geometry).read_text()); feat=next(f for f in geom_doc["features"] if f["properties"]["zone_id"]=="SPT-Z001"); geom=shape(feat["geometry"]); w,s,e,n=geom.bounds
    session=requests.Session(); session.headers.update({"User-Agent":UA})
    receipt={"mode":"DRY_RUN" if args.dry_run else "LIVE","spt_z001_bbox":{"west":w,"south":s,"east":e,"north":n},"geometry_sha256":sha256(Path(args.geometry)),"requests":{},"doi_records":{},"landing_inspection":{}}
    gmrt_url=f"https://www.gmrt.org/services/GridServer?west={w}&east={e}&south={s}&north={n}&layer=topo-mask&format=geotiff&resolution=max"
    gmrt_meta=f"https://www.gmrt.org/services/GridServer/metadata?west={w}&east={e}&south={s}&north={n}&layer=topo-mask&format=geotiff&mformat=json&resolution=max"
    receipt["requests"].update({"gmrt_topo_mask":gmrt_url,"gmrt_metadata":gmrt_meta,"gebco_tid_global":GEBCO_TID_URL})
    if not args.dry_run:
        for label,url,path in [("gmrt_topo_mask",gmrt_url,out/"SPT_Z001_GMRT_topo_mask.tif"),("gmrt_metadata",gmrt_meta,out/"SPT_Z001_GMRT_metadata.json")]:
            try: receipt[label]=get(session,url,path,timeout=300,max_bytes=1_000_000_000)
            except Exception as ex: receipt[label]={"status":"ERROR","error":repr(ex),"url":url}
        zpath=out/"gebco_2026_tid_geotiff.zip"
        try:
            receipt["gebco_tid_download"]=get(session,GEBCO_TID_URL,zpath,timeout=900,max_bytes=500_000_000)
            if receipt["gebco_tid_download"].get("status")=="DOWNLOADED":
                ext=out/"gebco_tid_tiles"; ext.mkdir(exist_ok=True)
                with zipfile.ZipFile(zpath) as z: z.extractall(ext); receipt["gebco_zip_members"]=z.namelist()
                selected=None
                for fp in ext.rglob("*.tif"):
                    with rasterio.open(fp) as ds:
                        if not (ds.bounds.right<w or ds.bounds.left>e or ds.bounds.top<s or ds.bounds.bottom>n):
                            arr,tr=mask(ds,[mapping(geom)],crop=True,filled=False)
                            profile=ds.profile.copy(); profile.update(height=arr.shape[1],width=arr.shape[2],transform=tr,compress="deflate",nodata=255)
                            dest=out/"SPT_Z001_GEBCO_2026_TID.tif"
                            with rasterio.open(dest,"w",**profile) as dst: dst.write(arr.filled(255))
                            selected=dest; break
                if selected:
                    with rasterio.open(selected) as ds:
                        vals=ds.read(1); poly_mask=geometry_mask([mapping(geom)],out_shape=(ds.height,ds.width),transform=ds.transform,invert=True)
                        valid=poly_mask&(vals!=255); ra=row_areas(ds); total=area(valid,ra); rows=[]
                        for v in np.unique(vals[valid]).astype(int):
                            m=valid&(vals==v); ar=area(m,ra); rows.append({"tid":v,"label":TID_LABELS.get(v,"Other"),"area_km2":ar,"percent":100*ar/total if total else None,"cells":int(m.sum())})
                        if rows:
                            with (out/"SPT_Z001_GEBCO_TID_coverage.csv").open("w",newline="") as f:
                                cw=csv.DictWriter(f,fieldnames=rows[0].keys()); cw.writeheader(); cw.writerows(rows)
                        receipt["gebco_tid_summary"]={"total_area_km2":total,"classes":rows,"direct_area_km2":sum(r["area_km2"] for r in rows if 10<=r["tid"]<=17)}
        except Exception as ex: receipt["gebco_tid_download"]={"status":"ERROR","error":repr(ex),"url":GEBCO_TID_URL}
        gm=out/"SPT_Z001_GMRT_topo_mask.tif"
        if gm.exists():
            try:
                with rasterio.open(gm) as ds:
                    a=ds.read(1,masked=True); pmsk=geometry_mask([mapping(geom)],out_shape=(ds.height,ds.width),transform=ds.transform,invert=True)
                    valid=pmsk&~np.ma.getmaskarray(a); ra=row_areas(ds); tot=area(pmsk,ra); support=area(valid,ra)
                    receipt["gmrt_mask_summary"]={"polygon_area_km2":tot,"supported_area_km2":support,"supported_percent":100*support/tot if tot else None,"width":ds.width,"height":ds.height,"crs":str(ds.crs),"bounds":list(ds.bounds)}
            except Exception as ex: receipt["gmrt_mask_analysis_error"]=repr(ex)
    for name,doi in DOIS.items():
        if args.dry_run: receipt["doi_records"][name]={"doi":doi,"status":"PLANNED"}; continue
        rec=datacite(session,doi,out/"metadata"); receipt["doi_records"][name]=rec
        receipt["landing_inspection"][name]=inspect_landing(session,name,rec.get("landing_url") or f"https://doi.org/{doi}",out/"landing",download_cap=450_000_000)
    files=[]
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.name not in {"Round6_6_execution_receipt.json","SHA256SUMS.txt"}: files.append({"file":str(p.relative_to(out)),"bytes":p.stat().st_size,"sha256":sha256(p)})
    receipt["files"]=files
    (out/"Round6_6_execution_receipt.json").write_text(json.dumps(receipt,indent=2),encoding="utf-8")
    with (out/"SHA256SUMS.txt").open("w") as f:
        for x in files: f.write(f"{x['sha256']}  {x['file']}\n")
    print(json.dumps({"status":"PASS","mode":receipt["mode"],"output":str(out),"files":len(files)},indent=2))
if __name__=="__main__": main()
