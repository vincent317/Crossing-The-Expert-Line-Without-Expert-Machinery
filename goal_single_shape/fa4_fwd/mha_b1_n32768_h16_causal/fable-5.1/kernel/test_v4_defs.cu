// v4 test wrapper
#include "fa_fwd4.cu"

int main() {
  printf("v4 QTILE=%d KTILE=%d KSTAGES=%d VSTAGES=%d NSOFT_WARPS=%d\n",
         QTILE, KTILE, KSTAGES, VSTAGES, NSOFT_WARPS);
  printf("smem: Q=%d K=%d V=%d P=%d bar=1024 total=%d\n",
         Q_BYTES, K_BYTES*KSTAGES, V_BYTES*VSTAGES, P_BYTES, SM_TOTAL);
  printf("tmem: TM_COLS=%d\n", TM_COLS);
  return 0;
}
