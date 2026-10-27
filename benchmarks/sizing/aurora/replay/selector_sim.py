"""oneDNN v3.10.2 gemmstone cost model (kernel_evaluator.cpp) for one Max 1550 stack (448 EUs)."""
import json, math, os
CAT=json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"catalog.json")))
def divUp(x,y): return (x+y-1)//y
def alignUp(x,y): return divUp(x,y)*y
def rdsp2(x):  # roundDownSmallPow2
    if x<=1: return 1
    return 1<<int(math.floor(math.log2(x)))
def rusp2(x):  # roundUpSmallPow2 (uint8)
    x=int(x)&0xFF
    if x<=1: return 1
    return 1<<int(math.ceil(math.log2(x)))
F_KR,F_KP,F_KPL,F_KV,F_FB,F_FPO,F_TC,F_AFB,F_AA,F_SWK=1,2,4,8,0x10,0x20,0x40,0x80,0x100,0x200
F_FIXK=0x1000
def has(e,f): return (e['D']['flags'] & f)!=0
def derived(e,m,n,k,eu=448,batch=1):
    D=e['D']; uM,uN,uK=D['unroll']; wgM,wgN,wgK=D['wg']
    tM,tN=wgM*uM,wgN*uN
    d=dict(m=m,n=n,k=k,batch=batch,wgCountM=divUp(m,tM),wgCountN=divUp(n,tN),wgCountK=1)
    if D['wgUpdate']!=1:
        if m<tM: wgM=max(divUp(m,uM),1); tM=wgM*uM; d['wgCountM']=1
        if n<tN: wgN=max(divUp(n,uN),1); tN=wgN*uN; d['wgCountN']=1
    tpw=wgM*wgN*D['wgExpand']
    if not has(e,F_KV): tpw*=wgK
    d['threadCount']=float(d['wgCountM']*d['wgCountN']*tpw*batch)
    tpe=4 if D['grf']>128 else 8
    ss=eu>>3
    d['cap']=tpe*eu; d['cap1']=tpw*ss; d['pwc']=divUp(d['cap'],d['cap1'])
    d['tpw']=tpw; d['tileM']=tM; d['tileN']=tN
    return d
P=dict(C0=0,C1=1,Ck0=2,Ck1=3,Cb0=4,Cb1=5,Cp0=5,Ma=6,Mb=7,Mc=8,Mcu=9,Ef=10,Ep0=11,Ep1=12,Em=13,Fp=14,Fr0=15,Fr1=16)
def ecore(e,d,aux,noKR=False,beta=0.0,cConvert=True,postOps=False):
    p=lambda s:e['params'][P[s]]; D=e['D']
    threads=d['threadCount']; m,n,k=d['m'],d['n'],d['k']; kthread=k
    cap=d['cap']; cap1=d['cap1']; pwc=d['pwc']
    wgKmax=D['wg'][2]
    wgK=1
    if has(e,F_FIXK): wgK=wgKmax
    elif aux.get('kParallel'): wgK=max(1,divUp(k,aux['k0']))
    elif aux.get('kv'):
        wgc=d['wgCountM']*d['wgCountN']*d['batch']
        if wgc<pwc: wgK=rusp2(divUp(pwc,wgc))
    elif has(e,F_KV):
        if threads<cap and not noKR: wgK=rdsp2(int(math.floor(cap/threads)) if threads else 1)
    elif has(e,F_SWK):
        raise NotImplementedError
    else: wgK=wgKmax
    wgK=min(wgK,wgKmax); aux['wgK']=wgK
    if has(e,F_KPL) and not aux.get('kv'):
        kthread=alignUp(divUp(k,wgK),D['unroll'][2]); kthread=max(kthread,2*D['unroll'][2]); aux['k0']=kthread
    Mc=p('Mc') if beta==0 else p('Mcu')
    C0,C1=p('C0'),p('C1')
    if has(e,F_KPL): C0+=wgK*p('Ck0'); C1+=wgK*p('Ck1')
    if aux.get('kParallel') or aux.get('kv'):
        fpo=has(e,F_FPO) and (postOps or cConvert)
        if fpo:
            if has(e,F_AFB): C0+=p('Cp0')
            else: Mc+=p('Mcu')
        if has(e,F_FB) and (beta!=1 or fpo):
            if has(e,F_AFB): C0+=p('Cb0')
            else: C0=p('Cb0'); C1=p('Cb1'); Mc+=p('Mc')
    if has(e,F_KV):
        threads*=wgK; cap1*=wgK; pwc//=wgK
    tfull=math.floor(threads/cap)*cap; tpart=threads-tfull
    pw=math.ceil(tpart/cap1); npart=math.ceil(threads/cap1)
    ctime=C0+npart*C1+Mc*m*n*d['batch']
    mtime=(p('Ma')*m+p('Mb')*n)*k*d['batch']
    Ef=p('Ef'); eF=Ef*tfull; Ep=max(Ef,p('Ep0')+p('Ep1')*pwc/max(pw,1)); eP=Ep*pw*cap1
    eLB=eF+eP; eNo=Ef*threads
    Em=p('Em')
    if threads<cap: Em=1.0
    if aux.get('kv'): Em=0.0
    et=((1-Em)*eNo+Em*eLB)*D['unroll'][0]*D['unroll'][1]*kthread/D['wgExpand']
    F=max(1.0,min(p('Fp'),p('Fr0')+m*n*k*p('Fr1'))); et*=F
    return ctime+max(mtime,et), dict(ctime=ctime,mtime=mtime,etime=et,threads=threads,waves=threads/cap)
def evaluate(e,m,n,k,batch=1):
    d=derived(e,m,n,k,batch=batch)
    aux=dict(kParallel=has(e,F_KP),kv=False)
    if e['model']!='E': raise NotImplementedError(e['model'])
    if has(e,F_KP): raise NotImplementedError
    if has(e,F_KV):
        s,info=ecore(e,d,aux); best=(s,dict(aux),info)
        if e['D']['wg'][2]>1:
            a2=dict(aux); s2,i2=ecore(e,d,a2,noKR=True)
            if s2<best[0]: best=(s2,a2,i2)
        if batch>1: return best[0],best[1],best[2],d
        tc=d['threadCount']; tryKV=(int(tc)!=tc) or (int(tc)%d['cap']!=0)
        if tryKV:
            a3=dict(kParallel=False,kv=True); s3,i3=ecore(e,d,a3)
            if s3<best[0]:
                best=(s3,a3,i3)
                approxK0=d['threadCount']*k/d['cap']
                if approxK0<=32: best[1]['switchToKP']=True
        return best[0],best[1],best[2],d
    s,info=ecore(e,d,aux); return s,aux,info,d
