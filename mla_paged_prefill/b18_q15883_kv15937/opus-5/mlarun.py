import torch, ctypes, subprocess, os, math
_dir = os.path.dirname(os.path.abspath(__file__))
_lib = None
def build(force=False):
    global _lib
    so = os.path.join(_dir,"mlak_%s_%s_%s_%s_%s_%s.so"%(os.environ.get("MLA_ABL","0"),os.environ.get("MLA_LAG","1"),os.environ.get("MLA_NS","4"),os.environ.get("MLA_NV","2"),os.environ.get("MLA_NP","4"),os.environ.get("MLA_TIME","0")+os.environ.get("MLA_QRUN","1")+os.environ.get("MLA_PAIR","2"))) ; src=os.path.join(_dir,"mla.cu")
    if force or not os.path.exists(so) or os.path.getmtime(src)>os.path.getmtime(so):
        cmd=["nvcc","-O3","-gencode","arch=compute_100a,code=sm_100a","-std=c++17","--shared",
             "-Xcompiler","-fPIC","-I"+os.path.join(_dir,"cutlass/include"),"--expt-relaxed-constexpr",*(["-DDBG"] if os.environ.get("MLA_DBG") else []),"-DABL="+os.environ.get("MLA_ABL","0"),"-DLAG="+os.environ.get("MLA_LAG","1"),"-DNS="+os.environ.get("MLA_NS","4"),"-DNV="+os.environ.get("MLA_NV","2"),"-DNP="+os.environ.get("MLA_NP","4"),"-DQRUN="+os.environ.get("MLA_QRUN","1"),"-DPAIR="+os.environ.get("MLA_PAIR","2"),*(["-DTIMING"] if os.environ.get("MLA_TIME") else []),
             "-lcuda","-o",so,src]
        r=subprocess.run(cmd,capture_output=True,text=True)
        if r.returncode!=0:
            print(r.stdout[-4000:]); print(r.stderr[-12000:]); raise SystemExit("compile failed")
        print("compiled",flush=True)
    _lib = ctypes.CDLL(so)
    _lib.mla_run.restype=ctypes.c_int
    _lib.mla_smem.restype=ctypes.c_int
    return _lib

class Runner:
    def __init__(self, d, Q_LENS, KV_LENS):
        self.d=d
        self.B=len(Q_LENS)
        self.total_q=sum(Q_LENS); self.total_kv=sum(KV_LENS)
        self.kvpad=sum(((k+63)//64)*64 for k in KV_LENS)
        dev=d["q_nope"].device
        self.ckv_c=torch.zeros(self.kvpad,512,dtype=torch.bfloat16,device=dev)
        self.kpe_c=torch.zeros(self.kvpad,64,dtype=torch.bfloat16,device=dev)
        self.kvbase=torch.zeros(self.B+1,dtype=torch.int32,device=dev)
        maxtiles=self.total_q//8+self.B+1
        self.sched=torch.zeros(2*maxtiles,dtype=torch.int32,device=dev)
        self.nt=torch.zeros(1,dtype=torch.int32,device=dev)
        self.out=torch.zeros(self.total_q,16,512,dtype=torch.bfloat16,device=dev)
        self._lsebuf=torch.zeros(self.total_q*16+4096,dtype=torch.float32,device=dev)
        self.lse=self._lsebuf[:self.total_q*16].view(self.total_q,16)
    def run(self,check=0,stages=7):
        d=self.d
        return _lib.mla_run(
            ctypes.c_void_p(d["q_nope"].data_ptr()), ctypes.c_void_p(d["q_pe"].data_ptr()),
            ctypes.c_void_p(d["ckv"].data_ptr()), ctypes.c_void_p(d["kpe"].data_ptr()),
            ctypes.cast(d["qo_indptr"].data_ptr(),ctypes.POINTER(ctypes.c_int)),
            ctypes.cast(d["kv_indptr"].data_ptr(),ctypes.POINTER(ctypes.c_int)),
            ctypes.cast(d["kv_indices"].data_ptr(),ctypes.POINTER(ctypes.c_int)),
            ctypes.c_void_p(self.out.data_ptr()),
            ctypes.cast(self.lse.data_ptr(),ctypes.POINTER(ctypes.c_float)),
            ctypes.c_void_p(self.ckv_c.data_ptr()), ctypes.c_void_p(self.kpe_c.data_ptr()),
            ctypes.cast(self.kvbase.data_ptr(),ctypes.POINTER(ctypes.c_int)),
            ctypes.cast(self.sched.data_ptr(),ctypes.POINTER(ctypes.c_int)),
            ctypes.cast(self.nt.data_ptr(),ctypes.POINTER(ctypes.c_int)),
            ctypes.c_float(d["sm_scale"]), ctypes.c_int(self.B), ctypes.c_int(self.total_q),
            ctypes.c_int(self.total_kv), ctypes.c_int(self.kvpad), ctypes.c_int(check), ctypes.c_int(stages))
