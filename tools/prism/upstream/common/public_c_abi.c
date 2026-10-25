/* C compilation of the public production API and raw layout observations only. */
#include "public_c_abi.h"
#include "llama.h"
struct prism_public_c_abi prism_public_c_abi_observe(void) {
    struct prism_public_c_abi value = {
        sizeof(struct llama_model_params), _Alignof(struct llama_model_params),
        sizeof(struct llama_context_params), _Alignof(struct llama_context_params),
        sizeof(struct llama_batch), _Alignof(struct llama_batch),
        sizeof(llama_token), sizeof(llama_pos), sizeof(llama_seq_id),
#ifdef __STDC_VERSION__
        __STDC_VERSION__
#else
        0
#endif
    };
    return value;
}
