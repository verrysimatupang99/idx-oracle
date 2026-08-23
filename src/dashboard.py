#!/usr/bin/env python3
"""
idx-oracle dashboard v2 — mobile-first, zero-dependency stdlib server.
Serves:
  /            → responsive dashboard (cards stack, table scrolls horizontally)
  /api/results → v1+v2 comparison JSON (v2 falls back to v1 sharpe if missing)
Port 3911.
"""
import json
import os
import csv
import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(BASE, "results")
PORT = 3911


def load_results():
    out = {"v1": [], "v2": [], "v3": [], "meta": {"generated": datetime.datetime.now().isoformat(timespec="seconds")}}
    for tag, fn in (("v1", "baseline_lgbm.csv"), ("v2", "baseline_v2.csv"), ("v3", "baseline_v3_meta.csv")):
        p = os.path.join(RESULTS, fn)
        if os.path.exists(p):
            with open(p) as f:
                out[tag] = list(csv.DictReader(f))
    return out


PAGE = """<!DOCTYPE html>
<html lang="id"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>idx-oracle — Market Prediction Research</title>
<style>
:root{--bg:#0b0e14;--card:#131722;--line:#232a3b;--text:#dbe2ef;--dim:#8b98b3;
--green:#22c55e;--red:#ef4444;--gold:#f59e0b}
*{margin:0;padding:0;box-sizing:border-box;-webkit-text-size-adjust:100%}
body{background:var(--bg);color:var(--text);font-family:-apple-system,'Segoe UI',system-ui,sans-serif;
padding:clamp(12px,3vw,24px)}
.wrap{max-width:1100px;margin:0 auto}
h1{font-size:clamp(1.25rem,5vw,1.6rem)} h1 span{color:var(--gold)}
.sub{color:var(--dim);font-size:clamp(.72rem,2.8vw,.85rem);margin-bottom:16px;line-height:1.4}
.cards{display:grid;grid-template-columns:repeat(2,1fr);gap:8px;margin-bottom:16px}
@media(min-width:640px){.cards{grid-template-columns:repeat(4,1fr)}}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px}
.card .k{color:var(--dim);font-size:.62rem;text-transform:uppercase;letter-spacing:.04em}
.card .v{font-size:clamp(1.05rem,4.5vw,1.45rem);font-weight:700;margin-top:3px}
h2{font-size:clamp(.85rem,3.5vw,1rem);margin-bottom:10px;color:var(--text)}
.tablewrap{overflow-x:auto;-webkit-overflow-scrolling:touch;background:var(--card);
border-radius:12px;border:1px solid var(--line)}
table{width:100%;border-collapse:collapse;font-size:clamp(.7rem,2.9vw,.85rem);min-width:520px}
th,td{padding:9px 10px;text-align:left;border-bottom:1px solid var(--line);white-space:nowrap}
th{color:var(--dim);font-size:.62rem;text-transform:uppercase;letter-spacing:.05em;
background:#0f1320;position:sticky;top:0}
tr:last-child td{border-bottom:none}
.win{color:var(--green);font-weight:600}.lose{color:var(--red)}.up{color:var(--green);font-weight:700}
.badge{display:inline-block;padding:2px 8px;border-radius:99px;font-size:.6rem;font-weight:700}
.b-g{background:#14321f;color:var(--green)}.b-r{background:#3a1518;color:var(--red)}
.b-y{background:#332a12;color:var(--gold)}
.note{color:var(--dim);font-size:clamp(.68rem,2.7vw,.78rem);margin-top:14px;line-height:1.55}
footer{margin-top:20px;color:var(--dim);font-size:.68rem;text-align:center}
.empty{color:var(--dim);text-align:center;padding:28px 10px;font-size:.8rem}
@media(max-width:640px){
  /* table → stacked cards on phones */
  .tablewrap{overflow-x:visible;border:none;background:none;border-radius:0}
  table{min-width:0}
  thead{display:none}
  table,tbody,tr,td{display:block;width:100%}
  tr{background:var(--card);border:1px solid var(--line);border-radius:12px;
     margin-bottom:8px;padding:10px 12px;display:grid;
     grid-template-columns:1fr 1fr;gap:4px 10px}
  td{border:none;padding:2px 0;white-space:normal;display:flex;
     justify-content:space-between;align-items:center;font-size:.8rem}
  td::before{content:attr(data-label);color:var(--dim);font-size:.6rem;
     text-transform:uppercase;letter-spacing:.04em;margin-right:8px}
  td:nth-child(1){grid-column:1/-1;font-weight:700;font-size:.92rem;
     border-bottom:1px solid var(--line);padding-bottom:6px;margin-bottom:4px}
  td:nth-child(1)::before{content:none}
}
</style></head><body><div class="wrap">
<h1>🔮 idx<span>-oracle</span></h1>
<div class="sub">Multi-market direction research · walk-forward honest backtest · open-source (MIT)</div>

<div class="cards" id="cards"></div>

<h2 id="tblTitle">Model v1 vs v2 — Out-of-Sample Sharpe</h2>
<div class="tablewrap">
<table id="tbl"><thead><tr>
<th>Symbol</th><th>v1 LGBM</th><th>v2 Ens+Regime</th><th>v3 Meta</th><th>B&H</th><th>Akurasi v3</th><th>Best</th>
</tr></thead><tbody></tbody></table></div>
<div id="emptyMsg" class="empty" style="display:none">⏳ v2 training masih jalan — tabel keisi otomatis begitu selesai.</div>

<div class="note">⚠️ Semua angka = <b>out-of-sample walk-forward</b> (expanding window, embargo 5 hari, refit tahunan).
Akurasi 50-53% itu NORMAL untuk pasar likuid — edge kecil yang jujur &gt; angka besar bohongan.
Bukan saran finansial.</div>
<footer>idx-oracle · Yahoo Finance + Binance public API · updated <span id="gen"></span></footer>
</div>
<script>
async function main(){
  const r=await fetch('api/results');if(!r.ok)throw new Error('API '+r.status);
  const d=await r.json();
  const v1={};d.v1.forEach(x=>v1[x.symbol]=x);
  const tb=document.querySelector('#tbl tbody');
  let bhWins=0,best3=0,n=0,sumAcc=0;
  const v3={};d.v3.forEach(x=>v3[x.symbol]=x);
  const src=(d.v3.length?d.v3:(d.v2.length?d.v2:d.v1)).map(x=>({...x}));
  const rows=src.map(row=>{
    const a=v1[row.symbol]||{};
    const hasV3=row.strat_sharpe!==''&&!!v3[row.symbol];
    const s1=parseFloat(a.strat_sharpe||0);
    const s2=parseFloat((d.v2.find(x=>x.symbol===row.symbol)||{}).strat_sharpe);
    const s3=hasV3?parseFloat(row.strat_sharpe):NaN;
    const bh=parseFloat(row.bh_sharpe||a.bh_sharpe||0);
    const acc=parseFloat(row.acc||0);
    let best='v1';
    if(isFinite(s2)&&s2>s1)best='v2';
    if(isFinite(s3)&&s3>=Math.max(s1,isFinite(s2)?s2:-99))best='v3';
    if(hasV3){n++;sumAcc+=acc;if(s3>bh)bhWins++;if(best==='v3')best3++;}
    return {sym:row.symbol,s1,s2,s3,bh,acc,best,hasV3};
  });
  rows.sort((a,b)=>(b.hasV3-b.hasV3)||(b.s3||-99)-(a.s3||-99));
  for(const r of rows){
    const cls=r.best==='v3'?'b-g':(r.s3<r.s1?'b-r':'b-y');
    tb.insertAdjacentHTML('beforeend','<tr>'+
      '<td>'+r.sym+'</td>'+
      '<td data-label="v1 LGBM">'+(isFinite(r.s1)?r.s1.toFixed(2):'-')+'</td>'+
      '<td data-label="v2 Ens+Regime">'+(isFinite(r.s2)?r.s2.toFixed(2):'-')+'</td>'+
      '<td data-label="v3 Meta" class="'+(r.hasV3?'up':'')+'">'+(isFinite(r.s3)?('<b>'+r.s3.toFixed(2)+'</b>'):'⏳')+'</td>'+
      '<td data-label="Buy&Hold" class="'+(r.bh<0?'lose':'')+'">'+r.bh.toFixed(2)+'</td>'+
      '<td data-label="Akurasi v3">'+(r.hasV3?(r.acc*100).toFixed(1)+'%':'-')+'</td>'+
      '<td data-label="Best"><span class="badge '+(r.best==='v3'?cls:'')+'">'+r.best+'</span></td></tr>');
  }
  document.getElementById('cards').innerHTML=
    '<div class="card"><div class="k">Markets tracked</div><div class="v">'+d.v1.length+'</div></div>'+
    '<div class="card"><div class="k">v3 beats B&amp;H</div><div class="v" style="color:var(--green)">'+bhWins+'/'+n+'</div></div>'+
    '<div class="card"><div class="k">v3 best model</div><div class="v">'+best3+'/'+n+'</div></div>'+
    '<div class="card"><div class="k">Avg accuracy</div><div class="v">'+(n?(sumAcc/n*100).toFixed(1)+'%':'—')+'</div></div>';
  document.getElementById('gen').textContent=(d.meta.generated||'').replace('T',' ');
  if(!d.v3.length){document.getElementById('emptyMsg').style.display='block';}
}
main().catch(e=>{
  document.getElementById('emptyMsg').style.display='block';
  document.getElementById('emptyMsg').textContent='⚠️ '+e.message+' — coba refresh';
});
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?", 1)[0].split("#", 1)[0]
        if path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store, max-age=0")
            self.send_header("Content-Length", str(len(PAGE.encode())))
            self.end_headers()
            self.wfile.write(PAGE.encode())
        elif self.path == "/api/results":
            self._send(200, json.dumps(load_results()).encode(), "application/json")
        elif self.path == "/api/health":
            self._send(200, b'{"ok":true}', "application/json")
        else:
            self._send(404, b'{"error":"not found"}', "application/json")

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    print(f"idx-oracle dashboard on :{PORT}")
    HTTPServer(("127.0.0.1", PORT), H).serve_forever()
