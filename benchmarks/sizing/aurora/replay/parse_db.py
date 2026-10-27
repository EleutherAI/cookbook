import re, sys, json
src=open(sys.argv[1]).read()
lines=[l for l in src.splitlines() if l.startswith("{{'")]
num=r"-?[\d.eE+-]+"
out=[]
for l in lines:
    m=re.match(r"\{\{'(\w)', \"(\w+)\", \{([^}]*)\}, \{([^}]*)\}\}, \{(.*?)\}, \"([^\"]*)\", \{(.*)\}, \{'(\w)', (\d+), \{([^}]*)\}\}\},?$", l)
    if not m: continue
    hw,kt,prec,lay,restr,strat,drv,mid,pc,params=m.groups()
    prec=[p.strip().strip('"') for p in prec.split(",")]; lay=[p.strip().strip('"') for p in lay.split(",")]
    # restrictions: steppingMin, steppingMax, {acc min}, {acc max}, {allowed min}, {allowed max}, {align}, "tags"
    r=re.match(r"(-?\d+), (-?\d+), \{([^}]*)\}, \{([^}]*)\}, \{([^}]*)\}, \{([^}]*)\}, \{([^}]*)\}, \"([^\"]*)\"",restr)
    ints=lambda s:[int(x) for x in s.split(",")]
    R=dict(stepMin=int(r[1]),stepMax=int(r[2]),accMin=ints(r[3]),accMax=ints(r[4]),allowMin=ints(r[5]),allowMax=ints(r[6]),align=ints(r[7]),tags=r[8])
    d=re.sub(r"\(\w+\) ","",drv)
    d=d.replace("true","1").replace("false","0")
    toks=re.findall(r"\{[^}]*\}|-?\d+",d)
    D={}
    keys=["sg","fusedLoop","grf","loopOrder","blocking","blockingAlt","unroll","wg","wgExpand","wgUpdate","flags","slm","perKSLM","alignment","support4GB"]
    for k_,t in zip(keys,toks):
        D[k_]=[int(x) for x in t.strip("{}").split(",")] if t.startswith("{") else int(t)
    out.append(dict(hw=hw,kt=kt,prec=prec,lay=lay,R=R,strategy=strat,D=D,model=mid,params=[float(x) for x in params.split(",")]))
json.dump(out,open(sys.argv[2],"w"),indent=0)
print(len(lines),"lines",len(out),"parsed")
