// Intentionally no native ONNX implementation.
// ONNX Runtime Android is owned by OnnxBackend.kt, which maintains persistent
// CPU and best-effort NNAPI sessions and uploads the 20,670-value correction
// output through RendererNative.nativeUploadExternalDelta(). The uploaded SSBO
// is consumed by the same no-readback GPU combine/render path as native ZD-ECN.
