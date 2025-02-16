#ifdef __cplusplus
extern "C" {
#endif

bool InitTFLite();
bool RunInference(const float* input_data, int input_length, float* output_data, int output_length);

#ifdef __cplusplus
}
#endif

