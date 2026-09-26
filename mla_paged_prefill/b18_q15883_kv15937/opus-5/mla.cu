// MLA paged prefill (causal) for B200 / sm_100a  -- tcgen05 2SM UMMA
#include <cute/tensor.hpp>
#include <cute/arch/mma_sm100_umma.hpp>
#include <cute/arch/copy_sm100.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/atom/mma_traits_sm100.hpp>
#include <cutlass/numeric_types.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cstdio>

#ifndef ABL
#define ABL 0
#endif
#ifndef LAG
#define LAG 1
#endif
#ifndef NS
#define NS 2
#endif
#define NSSH (NS==2?1:2)
#ifndef NV
#define NV 2
#endif
#define NVSH (NV==1?0:1)
#ifndef NP
#define NP 2
#endif
#ifndef QRUN
#define QRUN 2
#endif
#ifndef PAIR
#define PAIR 2
#endif
#define NPSH (NP==2?1:2)
using namespace cute;
using bf16 = cutlass::bfloat16_t;
#define DEV __device__ __forceinline__

// ------------------------- problem constants -------------------------
static constexpr int H      = 16;
static constexpr int DCKV   = 512;
static constexpr int DKPE   = 64;
static constexpr int QTOK   = 8;     // q tokens per cluster tile
static constexpr int MCTA   = 64;    // rows per CTA  (= 4 q tokens * 16 heads)
static constexpr int T      = 64;    // kv tile
static constexpr int THALF  = T/2;   // kv rows per CTA for the QK B operand
static constexpr int NSTG   = 2;   // K stages
static constexpr int KVALIGN= 64;
static constexpr float DELTA_LOG2 = 8.0f;

// ------------------------- smem layouts -------------------------
using LQn = decltype(tile_to_shape(UMMA::Layout_K_SW128_Atom<bf16>{}, Shape<Int<MCTA>, Int<DCKV>>{}));
using LQp = decltype(tile_to_shape(UMMA::Layout_K_SW128_Atom<bf16>{}, Shape<Int<MCTA>, Int<DKPE>>{}));
using LKn = decltype(tile_to_shape(UMMA::Layout_K_SW128_Atom<bf16>{}, Shape<Int<THALF>,_256>{}));
using LKp = decltype(tile_to_shape(UMMA::Layout_K_SW128_Atom<bf16>{}, Shape<Int<THALF>,Int<DKPE>>{}));
using LV  = decltype(tile_to_shape(UMMA::Layout_MN_SW128_Atom<bf16>{},Shape<_64, Int<T>>{}));   // (dv=64, kv=T)
using LP  = decltype(tile_to_shape(UMMA::Layout_K_SW128_Atom<bf16>{}, Shape<Int<MCTA>, Int<T>>{}));

static constexpr int SZ_Qn = MCTA*DCKV*2;
static constexpr int SZ_Qp = MCTA*DKPE*2;
static constexpr int SZ_Kn = THALF*256*2;
static constexpr int NKH  = 3;
static constexpr int SZ_Kp = THALF*DKPE*2;
static constexpr int SZ_V  = 64*T*2;
static constexpr int SZ_P  = MCTA*T*2;

struct SmemStore {
  alignas(1024) uint8_t qn[SZ_Qn];
  alignas(1024) uint8_t qp[SZ_Qp];
  alignas(1024) uint8_t kn[NKH][SZ_Kn];
  alignas(1024) uint8_t kp[NSTG][SZ_Kp];
  alignas(1024) uint8_t vv[NV][4][SZ_V];
  alignas(1024) uint8_t pp[NP][SZ_P];
  alignas(128)  uint64_t bQl;           // local Q tma
  uint64_t bKl[NKH];
  uint64_t bKpl[NSTG];
  uint64_t bVl[NV];                     // local V tma
  uint64_t bQ;                          // cluster Q ready (CTA0, 2 arrivals)
  uint64_t bK[NKH];
  uint64_t bKp[NSTG];
  uint64_t bV[NV];                      // cluster V ready (CTA0, 2)
  uint64_t bSdone[4];                   // QK mma done (both CTAs)
  uint64_t bPVdone[NP];                 // PV mma done (both CTAs)
  uint64_t bP[NP];                      // P ready (CTA0: 2 arrivals)
  uint64_t bSfree[4];                   // softmax released S buf (in CTA0: 2 arrivals)
  uint64_t bKfree[NKH];
  uint64_t bKpfree[NSTG];
  uint64_t bVfree[NV];                  // PV consumed V  (both CTAs)
  alignas(128) uint32_t tmem_slot[4];
  alignas(128) float    red[512];
  alignas(128) int      fl[8];
  alignas(128) int      flag[4];
};

struct Params {
  const bf16* q_nope; const bf16* q_pe;
  const bf16* ckv_cache; const bf16* kpe_cache;
  const int* qo_indptr; const int* kv_indptr; const int* kv_indices;
  bf16* out; float* lse;
  bf16* ckv_c; bf16* kpe_c;
  int* kvbase; int* sched; int* nt;
  float sc;            // sm_scale * log2(e)
  int B, total_q, total_kv, kvpad, maxtiles;
  CUtensorMap tmQn, tmQp, tmKn, tmKp, tmV;
};

// ------------------------- ptx helpers -------------------------
DEV void mbar_init(uint64_t* b,int c){uint32_t a=cast_smem_ptr_to_uint(b);asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"::"r"(a),"r"(c));}
DEV void mbar_wait(uint64_t* b,uint32_t ph){uint32_t a=cast_smem_ptr_to_uint(b);
  asm volatile("{\n.reg .pred P;\nLW%=: mbarrier.try_wait.parity.shared::cta.b64 P,[%0],%1;\n@P bra LD%=;\nbra LW%=;\nLD%=:\n}"::"r"(a),"r"(ph));}
DEV void mbar_arrive(uint64_t* b){uint32_t a=cast_smem_ptr_to_uint(b);
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"::"r"(a));}
DEV void mbar_arrive_remote(uint32_t a){
  asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];"::"r"(a));}
DEV void mbar_arrive_expect_remote(uint32_t a,uint32_t bytes){
  asm volatile("mbarrier.arrive.expect_tx.shared::cluster.b64 _, [%0], %1;"::"r"(a),"r"(bytes));}
DEV uint32_t map_remote(void* p,uint32_t cta){
  uint32_t a=cast_smem_ptr_to_uint(p),r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;":"=r"(r):"r"(a),"r"(cta));
  return r;}
DEV void commit2(uint64_t* b){uint32_t a=cast_smem_ptr_to_uint(b);
  asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;"::"r"(a),"h"((uint16_t)0x3));}
DEV void fence_before(){asm volatile("tcgen05.fence::before_thread_sync;":::"memory");}
DEV void fence_after(){asm volatile("tcgen05.fence::after_thread_sync;":::"memory");}
DEV void nbar(int id,int cnt){asm volatile("bar.sync %0, %1;"::"r"(id),"r"(cnt):"memory");}
DEV float ex2(float x){float r;asm volatile("ex2.approx.ftz.f32 %0, %1;":"=f"(r):"f"(x));return r;}
DEV uint32_t cvt2bf(float a,float b){uint32_t r;asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;":"=r"(r):"f"(b),"f"(a));return r;}
DEV void tma2d(const CUtensorMap* d,uint64_t* mb,void* sp,int c0,int c1){
  uint64_t gd=reinterpret_cast<uint64_t>(d);
  uint32_t mba=cast_smem_ptr_to_uint(mb), spa=cast_smem_ptr_to_uint(sp);
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
    ::"r"(spa),"l"(gd),"r"(mba),"r"(c0),"r"(c1):"memory");}
DEV void tma2d_r(const CUtensorMap* d,uint32_t mba,void* sp,int c0,int c1){
  uint64_t gd=reinterpret_cast<uint64_t>(d);
  uint32_t spa=cast_smem_ptr_to_uint(sp);
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
    ::"r"(spa),"l"(gd),"r"(mba),"r"(c0),"r"(c1):"memory");}

DEV void mma2(uint32_t acc,uint64_t da,uint64_t db,uint32_t idesc,uint32_t sc){
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p,%4,0;\ntcgen05.mma.cta_group::2.kind::f16 [%0],%1,%2,%3,p;\n}"
    ::"r"(acc),"l"(da),"l"(db),"r"(idesc),"r"(sc));}

// ------------------------- prep kernel -------------------------
__global__ void k_prep(Params p)      // parallel: one thread per tile (level-major order)
{
  __shared__ int qo[64], kvs[64], ntb[64], sh[4];
  int tid=threadIdx.x;
  if(tid<=p.B && tid<64){ qo[tid]=p.qo_indptr[tid]; kvs[tid]=p.kv_indptr[tid]; }
  __syncthreads();
  if(tid<p.B) ntb[tid]=(qo[tid+1]-qo[tid]+QTOK-1)/QTOK;
  __syncthreads();
  if(tid==0){
    int NT=0,Lmax=0;
    for(int b=0;b<p.B;++b){ NT+=ntb[b]; if(ntb[b]>Lmax) Lmax=ntb[b]; }
    sh[0]=NT; sh[1]=Lmax;
    if(blockIdx.x==0){
      *p.nt=NT;
      int acc=0;
      for(int b=0;b<p.B;++b){ p.kvbase[b]=acc; int kl=kvs[b+1]-kvs[b]; acc+=((kl+KVALIGN-1)/KVALIGN)*KVALIGN; }
    }
  }
  __syncthreads();
  int NT=sh[0], Lmax=sh[1];
  for(int tile=blockIdx.x*blockDim.x+tid; tile<NT; tile+=gridDim.x*blockDim.x){
    int lo=0, hi=Lmax-1;
    while(lo<hi){
      int mid=(lo+hi)>>1, c=0;
      for(int b=0;b<p.B;++b){ int v=ntb[b]-mid-1; if(v>0) c+=v; }
      if(c<=tile) hi=mid; else lo=mid+1;
    }
    int L=lo, c=0;
    for(int b=0;b<p.B;++b){ int v=ntb[b]-L-1; if(v>0) c+=v; }
    int idx=tile-c, bb=0;
    for(int b=0;b<p.B;++b){ if(ntb[b]>L){ if(idx==0){ bb=b; break; } --idx; } }
    p.sched[2*tile]=bb; p.sched[2*tile+1]=L*QTOK;
  }
}

// ------------------------- gather kernel -------------------------
__global__ void k_gather(Params p)
{
  int g = blockIdx.x*blockDim.y + threadIdx.y;      // kv element index
  if(g >= p.total_kv) return;
  // find sequence (B is small)
  int b=0;
  while(b+1<p.B && p.kv_indptr[b+1]<=g) ++b;
  int j = g - p.kv_indptr[b];
  int dst = p.kvbase[b] + j;
  int page = p.kv_indices[g];
  const int4* src = reinterpret_cast<const int4*>(p.ckv_cache + (long)page*DCKV);
  int4* dd = reinterpret_cast<int4*>(p.ckv_c + (long)dst*DCKV);
  for(int i=threadIdx.x;i<DCKV/8;i+=32) dd[i]=src[i];
  const int4* src2 = reinterpret_cast<const int4*>(p.kpe_cache + (long)page*DKPE);
  int4* dd2 = reinterpret_cast<int4*>(p.kpe_c + (long)dst*DKPE);
  for(int i=threadIdx.x;i<DKPE/8;i+=32) dd2[i]=src2[i];
}

// ------------------------- main kernel -------------------------
extern "C" __global__ void __cluster_dims__(2,1,1) __launch_bounds__(384)
k_mla(__grid_constant__ const Params p)
{
  extern __shared__ char smem_raw[];
  SmemStore& S = *reinterpret_cast<SmemStore*>(smem_raw);
  const int tid  = threadIdx.x;
  const int warp = tid>>5;
  const uint32_t cta = cute::block_rank_in_cluster();
  const int tile = blockIdx.x>>1;
  if(tile >= *p.nt) { return; }
  const int b   = p.sched[2*tile];
  const int qt0 = p.sched[2*tile+1];
  const int q0  = p.qo_indptr[b];
  const int ql  = p.qo_indptr[b+1]-q0;
  const int kl  = p.kv_indptr[b+1]-p.kv_indptr[b];
  const int off = kl-ql;
  const int kvb = p.kvbase[b];
  int kend = qt0+QTOK+off; if(kend>kl) kend=kl; if(kend<1) kend=1;
  const int niter = (kend+T-1)/T;
  const int qrow0 = (q0+qt0)*H + cta*MCTA;

  // ---- barrier init ----
  if(tid==0){
    mbar_init(&S.bQ,2); mbar_init(&S.bQl,1);
    for(int s=0;s<NKH;++s){ mbar_init(&S.bK[s],2); mbar_init(&S.bKl[s],1); mbar_init(&S.bKfree[s],1); }
    for(int s=0;s<NSTG;++s){ mbar_init(&S.bKp[s],2); mbar_init(&S.bKpl[s],1); mbar_init(&S.bKpfree[s],1); }
    for(int s=0;s<NV;++s){ mbar_init(&S.bV[s],2); mbar_init(&S.bVl[s],1); mbar_init(&S.bVfree[s],1); }
    for(int s=0;s<NS;++s){ mbar_init(&S.bSdone[s],1); mbar_init(&S.bSfree[s],4); }
    for(int s=0;s<NP;++s){ mbar_init(&S.bPVdone[s],1); mbar_init(&S.bP[s],4); }
    for(int k=0;k<8;++k) S.fl[k]=0;
  }
  __syncthreads();
  cute::cluster_sync();

  // ---- TMEM ----
  cute::TMEM::Allocator2Sm alloc;
  if(warp==0){ alloc.allocate(512, S.tmem_slot); alloc.release_allocation_lock(); }
  __syncthreads();
  const uint32_t tmO = S.tmem_slot[0];
  const uint32_t tmSb = S.tmem_slot[0]+256;   // S buffer b at tmSb + 32*b

  // ---- cute tensors / descriptors ----
  Tensor sQn=make_tensor(make_smem_ptr((bf16*)S.qn),LQn{});
  Tensor sQp=make_tensor(make_smem_ptr((bf16*)S.qp),LQp{});
  Tensor dQn=make_tensor<UMMA::smem_desc<UMMA::Major::K>>(zipped_divide(sQn,make_shape(Int<MCTA>{},_16{})));
  Tensor dQp=make_tensor<UMMA::smem_desc<UMMA::Major::K>>(zipped_divide(sQp,make_shape(Int<MCTA>{},_16{})));
  Tensor tKn=make_tensor(make_smem_ptr((bf16*)S.kn[0]),LKn{});
  Tensor dKnT=make_tensor<UMMA::smem_desc<UMMA::Major::K>>(zipped_divide(tKn,make_shape(Int<THALF>{},_16{})));
  Tensor tKp=make_tensor(make_smem_ptr((bf16*)S.kp[0]),LKp{});
  Tensor dKpT=make_tensor<UMMA::smem_desc<UMMA::Major::K>>(zipped_divide(tKp,make_shape(Int<THALF>{},_16{})));
  Tensor tVv=make_tensor(make_smem_ptr((bf16*)S.vv[0][0]),LV{});
  Tensor dVvT=make_tensor<UMMA::smem_desc<UMMA::Major::MN>>(zipped_divide(tVv,make_shape(_64{},_16{})));
  Tensor tPp=make_tensor(make_smem_ptr((bf16*)S.pp[0]),LP{});
  Tensor dPpT=make_tensor<UMMA::smem_desc<UMMA::Major::K>>(zipped_divide(tPp,make_shape(Int<MCTA>{},_16{})));
  constexpr uint64_t STR_Kn = SZ_Kn/16, STR_Kp = SZ_Kp/16, STR_V = SZ_V/16, STR_P = SZ_P/16;
  const uint32_t idQK = uint32_t(UMMA::make_runtime_instr_desc<bf16,bf16,float,128,T,UMMA::Major::K,UMMA::Major::K>()>>32);
  const uint32_t idPV = uint32_t(UMMA::make_runtime_instr_desc<bf16,bf16,float,128,128,UMMA::Major::K,UMMA::Major::MN>()>>32);

#ifdef DBG
  if(tid==0) printf("cta%d tile%d b%d qt0%d niter%d nt%d qrow0%d kvb%d\n",(int)cta,tile,b,qt0,niter,*p.nt,qrow0,kvb);
#endif
  // =========================== PRODUCER (warp 4) ===========================
  if(warp==8 && ABL<5){
    const bool el = cute::elect_one_sync();
    uint32_t bQr = cast_smem_ptr_to_uint(&S.bQl);
    const uint32_t bKr0=cast_smem_ptr_to_uint(&S.bKl[0]);
    const uint32_t bKpr0=cast_smem_ptr_to_uint(&S.bKpl[0]);
    if(el){
      mbar_arrive_expect_remote(bQr, SZ_Qn+SZ_Qp);
      #pragma unroll
      for(int j=0;j<8;++j) tma2d_r(&p.tmQn,bQr,(bf16*)S.qn + j*(MCTA*64), j*64, qrow0);
      tma2d_r(&p.tmQp,bQr,(bf16*)S.qp, 0, qrow0);
    }
    for(int i=0;i<niter;++i){
      int kvrow = kvb + i*T + cta*THALF;
      #pragma unroll
      for(int hh=0;hh<2;++hh){
        int h=2*i+hh, cyc=h/3, sl=h-3*cyc;
        if(h>=NKH) mbar_wait(&S.bKfree[sl], (uint32_t)((cyc-1)&1));
        if(el){
          uint32_t bb=bKr0+8u*sl;
          mbar_arrive_expect_remote(bb, SZ_Kn);
          #pragma unroll
          for(int j=0;j<4;++j) tma2d_r(&p.tmKn,bb,(bf16*)S.kn[sl] + j*(THALF*64), hh*256 + j*64, kvrow);
        }
      }
      { int s=i&1;
        if(i>=NSTG) mbar_wait(&S.bKpfree[s], (uint32_t)(((i-NSTG)>>1)&1));
        if(el){
          uint32_t bb=bKpr0+8u*s;
          mbar_arrive_expect_remote(bb, SZ_Kp);
          tma2d_r(&p.tmKp,bb,(bf16*)S.kp[s], 0, kvrow);
        }
      }
    }
  }
  // =========================== V PRODUCER (warp 11) ===========================
  else if(warp==11 && ABL<5){
    const bool el = cute::elect_one_sync();
    const uint32_t bVl0=cast_smem_ptr_to_uint(&S.bVl[0]);
    const uint32_t bVr0=map_remote(&S.bV[0],0);
    for(int i=0;i<niter;++i){
      int vs=i&(NV-1);
      if(i>=NV) mbar_wait(&S.bVfree[vs], (uint32_t)(((i-NV)>>NVSH)&1));
      if(el){
        int kvrow2 = kvb + i*T;
        mbar_arrive_expect_remote(bVl0+8u*vs, 4*SZ_V);
        #pragma unroll
        for(int j=0;j<4;++j) tma2d_r(&p.tmV,bVl0+8u*vs,(bf16*)S.vv[vs][j], 128*j + 64*cta, kvrow2);
      }
      if(i>=1){ int ps=(i-1)&(NV-1);
        mbar_wait(&S.bVl[ps], (uint32_t)(((i-1)>>NVSH)&1)); if(el) mbar_arrive_remote(bVr0+8u*ps); }
    }
    { int ps=(niter-1)&(NV-1);
      mbar_wait(&S.bVl[ps], (uint32_t)(((niter-1)>>NVSH)&1)); if(el) mbar_arrive_remote(bVr0+8u*ps); }
  }
  // =========================== RELAY (warp 6) ===========================
  else if(warp==10 && ABL<5){
    const bool el = cute::elect_one_sync();
    uint32_t bQr=map_remote(&S.bQ,0);
    const uint32_t bKr0=map_remote(&S.bK[0],0);
    const uint32_t bKpr0=map_remote(&S.bKp[0],0);
    mbar_wait(&S.bQl,0); if(el) mbar_arrive_remote(bQr);
    for(int i=0;i<niter;++i){
      #pragma unroll
      for(int hh=0;hh<2;++hh){
        int h=2*i+hh, cyc=h/3, sl=h-3*cyc;
        mbar_wait(&S.bKl[sl], (uint32_t)(cyc&1)); if(el) mbar_arrive_remote(bKr0+8u*sl);
      }
      { int s=i&1; mbar_wait(&S.bKpl[s], (uint32_t)((i>>1)&1)); if(el) mbar_arrive_remote(bKpr0+8u*s); }
    }
  }
  // =========================== MMA (warp 5, cta 0) ===========================
  else if(warp==9 && cta==0){
    const bool el = cute::elect_one_sync();
#if ABL<5
    mbar_wait(&S.bQ,0);
#endif
    fence_before();
    if(el){
      auto issue_QK = [&](int j){
        int sb=j&(NS-1);
        uint32_t acc=tmSb + 32u*sb;
        #pragma unroll
        for(int hh=0;hh<2;++hh){
          int h=2*j+hh, cyc=h/3, sl=h-3*cyc;
#if ABL<5
          mbar_wait(&S.bK[sl], (uint32_t)(cyc&1));
#endif
          #pragma unroll
          for(int k=0;k<16;++k) mma2(acc, uint64_t(dQn(0,make_coord(0,hh*16+k))), uint64_t(dKnT(0,make_coord(0,k)))+sl*STR_Kn, idQK, (hh==0&&k==0)?0:1);
#if ABL<5
          commit2(&S.bKfree[sl]);
#endif
        }
        { int sk=j&1;
#if ABL<5
          mbar_wait(&S.bKp[sk], (uint32_t)((j>>1)&1));
#endif
          #pragma unroll
          for(int k=0;k<DKPE/16;++k) mma2(acc, uint64_t(dQp(0,make_coord(0,k))), uint64_t(dKpT(0,make_coord(0,k)))+sk*STR_Kp, idQK, 1);
#if ABL<5
          commit2(&S.bKpfree[sk]);
#endif
        }
      };
      auto issue_PV = [&](int j){
        int ps=j&(NP-1), vs=j&(NV-1);
#if ABL<5
        mbar_wait(&S.bV[vs], (uint32_t)((j>>NVSH)&1));
#endif
        #pragma unroll
        for(int jj=0;jj<4;++jj){
          uint32_t oacc=tmO + 64*jj;
          #pragma unroll
          for(int k=0;k<T/16;++k) mma2(oacc, uint64_t(dPpT(0,make_coord(0,k)))+ps*STR_P, uint64_t(dVvT(0,make_coord(0,k)))+(vs*4+jj)*STR_V, idPV, (j==0&&k==0)?0:1);
        }
#if ABL<5
        commit2(&S.bVfree[vs]);
#endif
      };
      const int nr = (niter+PAIR-1)/PAIR;
      for(int r=0;r<nr;++r){
#ifdef DBG
        printf("M r%d begin\n",r);
#endif
#if ABL<5
        if(r>=2) mbar_wait(&S.bSfree[r&1], (uint32_t)(((r-2)>>1)&1));
#endif
        #pragma unroll
        for(int t=0;t<PAIR;++t){ int j=r*PAIR+t; if(j<niter) issue_QK(j); }
#if ABL<5
        commit2(&S.bSdone[r&1]);
#endif
#ifdef DBG
        printf("M r%d QK done\n",r);
#endif
        if(r>=1){
          int rp=r-1;
#if ABL<5 && ABL!=11
          mbar_wait(&S.bP[rp&1], (uint32_t)((rp>>1)&1));
#endif
          #pragma unroll
          for(int t=0;t<PAIR;++t){ int j=rp*PAIR+t; if(j<niter) issue_PV(j); }
#if ABL<5
          commit2(&S.bPVdone[rp&1]);
#endif
        }
      }
      { int rp=nr-1;
#if ABL<5 && ABL!=11
        mbar_wait(&S.bP[rp&1], (uint32_t)((rp>>1)&1));
#endif
        #pragma unroll
        for(int t=0;t<PAIR;++t){ int j=rp*PAIR+t; if(j<niter) issue_PV(j); }
#if ABL<5
        commit2(&S.bPVdone[rp&1]);
#endif
      }
    }
  }
  // =========================== COMPUTE (warps 0-7) ===========================
  if(warp<8 && ABL<5){
    constexpr int NCOL = THALF/2;          // 16 columns per thread
    const int dp  = 32*(warp&3) + (tid&31);
    const int grp = warp&1;
    const int gbar= 1+grp;
    const bool glead = (tid==0)||(tid==32);
    const int cg  = warp>>2;               // column group 0..3
    const int cb  = cg*NCOL;
    const int r0  = dp & 63;
    const int hi  = dp >> 6;
    const int qtok_local = qt0 + cta*4 + (r0>>4);
    const int head = r0 & 15;
    const int jmax = (qtok_local + off) < (kl-1) ? (qtok_local+off) : (kl-1);

    float mref2 = 0.f;
    float lpart = 0.f;
    const uint32_t bPr0=map_remote(&S.bP[0],0);
    const uint32_t bFr0=map_remote(&S.bSfree[0],0);

    Tensor sPQ=make_tensor(make_smem_ptr((bf16*)S.pp[0]),LP{});
    long long acc0=0,acc1=0,acc2=0,acc3=0,acc4=0,tprev=0;
    const float THR = 256.0f;   // 2^8 == DELTA_LOG2
#if ABL==3
    for(int i=0;i<niter;++i){
      int sb=i&(NS-1);
#ifdef TIMING
      long long tA=clock64();
#endif
      mbar_wait(&S.bSdone[sb], (uint32_t)((i>>NSSH)&1));
#ifdef TIMING
      long long tB=clock64(); acc0 += tB-tA;
#endif
      if(i>=2) mbar_wait(&S.bPVdone[i&1], (uint32_t)(((i-2)>>1)&1));
      nbar(gbar,128);
      if(tid==0){ mbar_arrive_remote(bPr0+8u*(i&1)); mbar_arrive_remote(bFr0+8u*sb); }
    }
    mbar_wait(&S.bPVdone[(nr-1)&1], (uint32_t)(((nr-1)>>1)&1));
    if(tid==0) p.lse[0]=mref2+lpart;
#else
    const int nr = (niter+PAIR-1)/PAIR;
    auto loadS = [&](int i, uint32_t* v){
      uint32_t addr = tmSb + 32u*(i&(NS-1)) + (uint32_t(dp)<<16) + cb;
      cute::SM100_TMEM_LOAD_32dp32b8x::copy(addr,
        v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7]);
      asm volatile("tcgen05.wait::ld.sync.aligned;":::"memory");
    };
    for(int r=0;r<nr;++r){
      mbar_wait(&S.bSdone[r&1], (uint32_t)((r>>1)&1));
      if(r>=2) mbar_wait(&S.bPVdone[r&1], (uint32_t)(((r-2)>>1)&1));
      fence_after();
      int ntile = niter - r*PAIR; if(ntile>PAIR) ntile=PAIR;
      float pmax=0.f, lsum=0.f;
      const float nm = -mref2;
      #pragma unroll
      for(int t=0;t<PAIR;++t){
        if(t>=ntile) continue;
        const int i = r*PAIR+t;
        uint32_t v[NCOL]; loadS(i, v);
        const int jbase = i*T + hi*THALF + cb;
        float pp[NCOL];
        if(jbase+NCOL-1 <= jmax){
          #pragma unroll
          for(int c=0;c<NCOL;++c){ float q=ex2(fmaf(__int_as_float(v[c]),p.sc,nm)); pp[c]=q; lsum+=q; pmax=fmaxf(pmax,q); }
        } else {
          #pragma unroll
          for(int c=0;c<NCOL;++c){
            float raw=((jbase+c)<=jmax)?__int_as_float(v[c]):-3.0e30f;
            float q=ex2(fmaf(raw,p.sc,nm)); pp[c]=q; lsum+=q; pmax=fmaxf(pmax,q);
          }
        }
        bf16* pbase = (bf16*)S.pp[0] + (i&(NP-1))*(SZ_P/2);
        #pragma unroll
        for(int c=0;c<NCOL/8;++c){
          int4 w4; w4.x=(int)cvt2bf(pp[8*c+0],pp[8*c+1]); w4.y=(int)cvt2bf(pp[8*c+2],pp[8*c+3]);
          w4.z=(int)cvt2bf(pp[8*c+4],pp[8*c+5]); w4.w=(int)cvt2bf(pp[8*c+6],pp[8*c+7]);
          *reinterpret_cast<int4*>(pbase + (&sPQ(r0, hi*THALF+cb+8*c) - (bf16*)S.pp[0])) = w4;
        }
      }
      bool bad = !(pmax <= THR) || (r==0);
      if(__any_sync(0xffffffffu, bad) && (tid&31)==0) atomicOr(&S.fl[(r&3)*2+grp],1);
      asm volatile("fence.proxy.async.shared::cta;":::"memory");
      nbar(gbar,128);
      int f = S.fl[(r&3)*2+grp];
      if(f){
        float mx=-INFINITY;
        #pragma unroll
        for(int t=0;t<PAIR;++t){
          if(t>=ntile) continue;
          const int i=r*PAIR+t; uint32_t v[NCOL]; loadS(i,v);
          const int jbase = i*T + hi*THALF + cb;
          #pragma unroll
          for(int c=0;c<NCOL;++c){
            float raw=((jbase+c)<=jmax)?__int_as_float(v[c]):-3.0e30f;
            mx=fmaxf(mx, raw*p.sc);
          }
        }
        S.red[cg*128+dp]=mx; nbar(gbar,128);
        float mtile=fmaxf(fmaxf(S.red[r0],S.red[r0+64]),fmaxf(S.red[128+r0],S.red[128+r0+64]));
        float newm = (r==0) ? mtile : fmaxf(mref2, mtile);
        float fix = ex2(mref2-newm);
        mref2 = newm;
        if(r>0){
          mbar_wait(&S.bPVdone[(r-1)&1], (uint32_t)(((r-1)>>1)&1));
          fence_after();
          #pragma unroll
          for(int blk=0;blk<8;++blk){
            uint32_t oa = tmO + cg*128 + blk*16 + (uint32_t(dp)<<16);
            uint32_t o[16];
            cute::SM100_TMEM_LOAD_32dp32b16x::copy(oa,
              o[0],o[1],o[2],o[3],o[4],o[5],o[6],o[7],o[8],o[9],o[10],o[11],o[12],o[13],o[14],o[15]);
            asm volatile("tcgen05.wait::ld.sync.aligned;":::"memory");
            #pragma unroll
            for(int c=0;c<16;++c) o[c]=__float_as_int(__int_as_float(o[c])*fix);
            cute::SM100_TMEM_STORE_32dp32b16x::copy(oa,
              o[0],o[1],o[2],o[3],o[4],o[5],o[6],o[7],o[8],o[9],o[10],o[11],o[12],o[13],o[14],o[15]);
            asm volatile("tcgen05.wait::st.sync.aligned;":::"memory");
          }
          lpart *= fix;
        }
        const float nm2 = -mref2;
        lsum = 0.f;
        #pragma unroll
        for(int t=0;t<PAIR;++t){
          if(t>=ntile) continue;
          const int i=r*PAIR+t; uint32_t v[NCOL]; loadS(i,v);
          const int jbase = i*T + hi*THALF + cb;
          float pp[NCOL];
          #pragma unroll
          for(int c=0;c<NCOL;++c){
            float raw=((jbase+c)<=jmax)?__int_as_float(v[c]):-3.0e30f;
            float q=ex2(fmaf(raw,p.sc,nm2)); pp[c]=q; lsum+=q;
          }
          bf16* pbase = (bf16*)S.pp[0] + (i&(NP-1))*(SZ_P/2);
          #pragma unroll
          for(int c=0;c<NCOL/8;++c){
            int4 w4; w4.x=(int)cvt2bf(pp[8*c+0],pp[8*c+1]); w4.y=(int)cvt2bf(pp[8*c+2],pp[8*c+3]);
            w4.z=(int)cvt2bf(pp[8*c+4],pp[8*c+5]); w4.w=(int)cvt2bf(pp[8*c+6],pp[8*c+7]);
            *reinterpret_cast<int4*>(pbase + (&sPQ(r0, hi*THALF+cb+8*c) - (bf16*)S.pp[0])) = w4;
          }
        }
        asm volatile("fence.proxy.async.shared::cta;":::"memory");
        nbar(gbar,128);
      }
      lpart += lsum;
      if(tid<2 || (tid>=32 && tid<34)){
        uint32_t aadr = ((tid&31)==0) ? (bPr0+8u*(r&1)) : (bFr0+8u*(r&1));
        mbar_arrive_remote(aadr);
      }
      if(glead) S.fl[((r+1)&3)*2+grp]=0;
    }
    // ---------------- epilogue ----------------
    mbar_wait(&S.bPVdone[(nr-1)&1], (uint32_t)(((nr-1)>>1)&1));
    fence_after();
#ifdef TIMING
    if(tid==0 && tile<64){ p.lse[(long)p.total_q*H+4*tile+0]=(float)acc0/niter; p.lse[(long)p.total_q*H+4*tile+1]=(float)acc1/niter; p.lse[(long)p.total_q*H+4*tile+2]=(float)acc2/niter; p.lse[(long)p.total_q*H+4*tile+3]=(float)acc3/niter; p.lse[(long)p.total_q*H+256+tile]=(float)acc4/niter; }
#endif
    nbar(gbar,128);
    S.red[cg*128+dp]=lpart; nbar(gbar,128);
    float ltot = S.red[r0]+S.red[r0+64]+S.red[128+r0]+S.red[128+r0+64];
    float inv = 1.0f/ltot;
    const int token = q0 + qtok_local;
    const bool valid = (qtok_local < ql);
    if(valid && hi==0 && cg==0) p.lse[(long)token*H+head] = (__log2f(ltot)+mref2)*0.6931471805599453f;
    #pragma unroll
    for(int jj=0;jj<2;++jj){
      int j = cg*2+jj;
      #pragma unroll
      for(int half=0;half<2;++half){
        uint32_t o[32];
        uint32_t oa = tmO + 64*j + 32*half + (uint32_t(dp)<<16);
        cute::SM100_TMEM_LOAD_32dp32b32x::copy(oa,
          o[0],o[1],o[2],o[3],o[4],o[5],o[6],o[7],o[8],o[9],o[10],o[11],o[12],o[13],o[14],o[15],
          o[16],o[17],o[18],o[19],o[20],o[21],o[22],o[23],o[24],o[25],o[26],o[27],o[28],o[29],o[30],o[31]);
        asm volatile("tcgen05.wait::ld.sync.aligned;":::"memory");
        if(valid){
          bf16* dst = p.out + ((long)token*H+head)*DCKV + 128*j + 64*hi + 32*half;
          int4* d4=reinterpret_cast<int4*>(dst);
          #pragma unroll
          for(int c=0;c<4;++c){
            int4 w; w.x=(int)cvt2bf(__int_as_float(o[8*c+0])*inv,__int_as_float(o[8*c+1])*inv);
            w.y=(int)cvt2bf(__int_as_float(o[8*c+2])*inv,__int_as_float(o[8*c+3])*inv);
            w.z=(int)cvt2bf(__int_as_float(o[8*c+4])*inv,__int_as_float(o[8*c+5])*inv);
            w.w=(int)cvt2bf(__int_as_float(o[8*c+6])*inv,__int_as_float(o[8*c+7])*inv);
            d4[c]=w;
          }
        }
      }
    }
#endif
  }
#if ABL>=5
  if(warp==9 && cta==0 && cute::elect_one_sync()) commit2(&S.bSdone[0]);
  if(warp<8) mbar_wait(&S.bSdone[0],0);
#endif
  cute::cluster_sync();
  if(warp==0) alloc.free(S.tmem_slot[0],512);
}

// ============================ host ============================
#include <map>
static void mk_tm(CUtensorMap* tm, void* base, uint64_t d0, uint64_t d1, uint32_t b0, uint32_t b1)
{
  uint64_t gdim[2]={d0,d1};
  uint64_t gstr[1]={d0*2};
  uint32_t bdim[2]={b0,b1};
  uint32_t estr[2]={1,1};
  CUresult r = cuTensorMapEncodeTiled(tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, base,
      gdim, gstr, bdim, estr, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if(r!=CUDA_SUCCESS){ const char* s; cuGetErrorString(r,&s); printf("tma encode fail %d %s\n",(int)r,s); }
}

extern "C" int mla_run(
  const void* q_nope, const void* q_pe, const void* ckv_cache, const void* kpe_cache,
  const int* qo_indptr, const int* kv_indptr, const int* kv_indices,
  void* out, float* lse,
  void* ckv_c, void* kpe_c, int* kvbase, int* sched, int* nt,
  float sm_scale, int B, int total_q, int total_kv, int kvpad, int check, int stages)
{
  Params p{};
  p.q_nope=(const bf16*)q_nope; p.q_pe=(const bf16*)q_pe;
  p.ckv_cache=(const bf16*)ckv_cache; p.kpe_cache=(const bf16*)kpe_cache;
  p.qo_indptr=qo_indptr; p.kv_indptr=kv_indptr; p.kv_indices=kv_indices;
  p.out=(bf16*)out; p.lse=lse;
  p.ckv_c=(bf16*)ckv_c; p.kpe_c=(bf16*)kpe_c;
  p.kvbase=kvbase; p.sched=sched; p.nt=nt;
  p.sc = sm_scale * 1.4426950408889634f;
  p.B=B; p.total_q=total_q; p.total_kv=total_kv; p.kvpad=kvpad;
  p.maxtiles = total_q/QTOK + B + 1;

  mk_tm(&p.tmQn,(void*)q_nope,   DCKV, (uint64_t)total_q*H, 64, MCTA);
  mk_tm(&p.tmQp,(void*)q_pe,     DKPE, (uint64_t)total_q*H, 64, MCTA);
  mk_tm(&p.tmKn,ckv_c,           DCKV, (uint64_t)kvpad,     64, THALF);
  mk_tm(&p.tmKp,kpe_c,           DKPE, (uint64_t)kvpad,     64, THALF);
  mk_tm(&p.tmV ,ckv_c,           DCKV, (uint64_t)kvpad,     64, T);

  if(stages&1) k_prep<<<16,256>>>(p);
  if(stages&2){
    dim3 blk(32,4);
    int nb=(total_kv+3)/4;
    k_gather<<<nb,blk>>>(p);
  }
  static int smem_set=0;
  int smem = (int)sizeof(SmemStore);
  if(!smem_set){ cudaFuncSetAttribute((void*)k_mla, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); smem_set=1; }
  if(stages&4) k_mla<<<2*p.maxtiles, 384, smem>>>(p);
  if(check){
    cudaError_t e=cudaDeviceSynchronize();
    if(e!=cudaSuccess){ printf("k_mla err: %s (smem=%d)\n", cudaGetErrorString(e), smem); return (int)e; }
  }
  return 0;
}
extern "C" int mla_smem(){ return (int)sizeof(SmemStore); }
