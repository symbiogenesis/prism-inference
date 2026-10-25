#ifndef PRISM_PUBLIC_C_ABI_H
#define PRISM_PUBLIC_C_ABI_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
struct prism_public_c_abi {
    uint64_t model_params_size, model_params_alignment;
    uint64_t context_params_size, context_params_alignment;
    uint64_t batch_size, batch_alignment;
    uint64_t token_size, position_size, sequence_id_size, c_standard;
};
struct prism_public_c_abi prism_public_c_abi_observe(void);
#ifdef __cplusplus
}
#endif
#endif
