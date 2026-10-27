"""Predict oneDNN v3.10.2 kernel choice, k-slicing and wave efficiency for a bf16 torch matmul on one Max 1550 stack.
C[M_t x N_t] = A[M_t x K_t] @ B[K_t x N_t]; oneDNN sees the column-major problem m = N_t, n = M_t, k = K_t."""
import functools, os, sys
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import selector_sim as S
def p2(x): return x & -x
def gem_problem(M_t,N_t,K_t,transA_t=False,transB_t=False,ldA_t=None,ldB_t=None,ldC_t=None,esz=2):
    # torch A is (M_t x K_t) logical; stored row-major unless transA_t (then stored K_t x M_t)
    ldA_t = ldA_t or (M_t if transA_t else K_t)
    ldB_t = ldB_t or (K_t if transB_t else N_t)
    ldC_t = ldC_t or N_t
    m,n,k=N_t,M_t,K_t
    layA = 'T' if transB_t else 'N'   # gemmstone A = torch B
    layB = 'T' if transA_t else 'N'   # gemmstone B = torch A
    alA=max(esz,min(128,p2(ldB_t*esz))); alB=max(esz,min(128,p2(ldA_t*esz))); alC=max(esz,min(128,p2(ldC_t*esz)))
    return dict(m=m,n=n,k=k,lay=(layA,layB,'N'),align=(alA,alB,alC),ld=(ldB_t,ldA_t,ldC_t))
def copy_based(prob):
    # xe_hp_systolic_t::use_nocopy() for bf16
    lda,ldb,_=prob['ld']; m,n,k=prob['m'],prob['n'],prob['k']
    if m<=1 or n<=1 or k<=1: return False
    if (m<32 and n<32) or (m<32 and k<32) or (n<32 and k<32): return False
    bad=((lda*2)|(ldb*2))&3 != 0
    ta=1 if prob['lay'][0]=='T' else 0; tb=1 if prob['lay'][1]=='T' else 0
    mn=[[0,12800],[0,0]] if not bad else [[1024,1024],[1024,1024]]
    kl=[[0,0],[0,0]] if not bad else [[1024,1024],[1024,1024]]
    mnl=mn[ta][tb]; kk=kl[ta][tb]
    nocopy=(mnl==0 or m*n<mnl*mnl) and (kk==0 or k<kk)
    return not nocopy
def candidates(prob, tags="IABC"):
    out=[]
    for i,e in enumerate(S.CAT):
        if e is None: continue
        if e['hw']!='F' or e['kt']!='gemm' or tuple(e['lay'])!=prob['lay'] or e['prec'][2]!='S': continue
        if tuple(e['prec'][:2]) not in (("B","B"),("H","H"),("H","B"),("B","H")): continue
        R=e['R']
        if any(prob['align'][j] % max(R['align'][j],1) for j in range(3)): continue
        ok=True
        for c in R['tags']:
            cu=chr(ord(c)&~0x20); want=(ord(c)&0x20)==0
            if (cu in tags)!=want: ok=False
        mnk=[prob['m'],prob['n'],prob['k']]
        for j in range(3):
            if R['allowMin'][j]>=0 and mnk[j]<R['allowMin'][j]: ok=False
            if R['allowMax'][j]>=0 and mnk[j]>R['allowMax'][j]: ok=False
        if ok and e['model']=='E' and not S.has(e,S.F_KP): out.append(i)
    return out
def select(prob):
    m,n,k=prob['m'],prob['n'],prob['k']; B=prob.get('batch',1)
    c=candidates(prob, tags='IABC'+('V' if B>1 else ''))
    if not c: return None
    sc={i:S.evaluate(S.CAT[i],m,n,k,batch=B) for i in c}
    def less(i,j):
        a,b=S.CAT[i],S.CAT[j]
        fa=a['R']['tags'][:1]=='#'; fb=b['R']['tags'][:1]=='#'
        aA,aB=max(a['R']['align'][0],4),max(a['R']['align'][1],4)
        bA,bB=max(b['R']['align'][0],4),max(b['R']['align'][1],4)
        la=lambda A1,B1,A2,B2:(A1<=A2) and (B1<=B2) and (A1+B1<B1+B2)
        if fb and la(bA,bB,aA,aB): return -1
        if fa and la(aA,aB,bA,bB): return 1
        return -1 if sc[i][0]<sc[j][0] else (1 if sc[i][0]>sc[j][0] else (i>j)-(i<j))
    best=sorted(c,key=functools.cmp_to_key(less))[0]
    s,aux,info,d=sc[best]; D=S.CAT[best]['D']
    Tm=D['unroll'][0]*D['wg'][0]; Tn=D['unroll'][1]*D['wg'][1]
    thr=D['wg'][0]*D['wg'][1]*D['wg'][2]; slots=((4 if D['grf']>128 else 8)*448)//thr
    tm,tn=-(-m//Tm),-(-n//Tn); W=tm*tn*B; waves=-(-W//slots)
    pad=m*n/(tm*Tm*tn*Tn); weff=W/(waves*slots)
    block2d=('A' in S.CAT[best]['R']['tags'])
    return dict(entry=best,Tm=Tm,Tn=Tn,slots=slots,W=W,waves=waves,pad_eff=pad,wave_eff=weff,
                eff=pad*(1.0 if aux.get('kv') else weff),kv=bool(aux.get('kv')),block2d=block2d,
                strategy=S.CAT[best]['strategy'])
def predict(M_t,N_t,K_t,transA_t=False,transB_t=False,batch=1,**kw):
    prob=gem_problem(M_t,N_t,K_t,transA_t,transB_t,**kw); prob['batch']=batch
    r={'prob':prob,'copy_based':copy_based(prob)}
    if not r['copy_based']: r.update(select(prob) or {})
    return r
if __name__=="__main__":
    for kk in [41218,41986,4096,24576,8192]:
        r=predict(2048,kk,2048); print(kk, r['entry'], r['Tm'],r['Tn'],r['kv'],round(r['eff'],3))
    print(predict(2048,4097,2048)['copy_based'])
