#pragma once

#ifdef __cplusplus
extern "C"
{
#endif // __cplusplus
    void* call_tensor(unsigned int nd, unsigned int* dimensions, const void* data);
    void delete_tensor(void* t);
    const char* to_string(void* t);
    void* add_tensor(const void* a, const void* b);
#ifdef __cplusplus
}
#endif // __cplusplus