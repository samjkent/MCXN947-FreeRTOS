#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

int init_model();

float run_inference(void *ptr, uint32_t len);

#ifdef __cplusplus
}
#endif
