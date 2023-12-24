#pragma once

#ifdef __cplusplus
extern "C"
{
#endif // __cplusplus
    void* call_tensor(unsigned int nd, unsigned int* dimensions, const void* data);
    void delete_tensor(void* t);
    const char* to_string(void* t);
#ifdef __cplusplus
}
#endif // __cplusplus