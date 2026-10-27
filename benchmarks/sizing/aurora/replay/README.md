# oneDNN v3.10.2 GEMM kernel selection, offline

- `parse_db.py kernel.db catalog.json` builds the catalog from oneDNN's `src/gpu/intel/gemm/jit/selector/db/kernel.db`
  (`fetch_catalog.sh` downloads it).
- `selector_sim.py` is the cost model `evaluateE`, including variable k-slicing (split-K).
- `gemm_predict.py` applies the eligibility rules and sort order and returns the selected entry, tile, waves, wave and padding
  efficiency and k-slicing flag, or `copy_based=True`.

```python
from gemm_predict import predict
predict(2048, 98562, 2048)                 # torch.mm(A[2048x2048], B[2048x98562])
predict(8192, 4096, 4096, transB_t=True)   # F.linear forward
```

Scope: bf16 in, fp32 accumulation, no post-ops, one stack. The copy path is detected but not modelled.
`catalog.json` is derived from oneDNN (Apache License 2.0).
