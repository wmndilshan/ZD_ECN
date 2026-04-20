-keep class com.graphicsresearch.androidcomputebenchmark.BenchmarkActivity { *; }
-keep class com.graphicsresearch.androidcomputebenchmark.BenchmarkSurfaceView { *; }
-keep class com.graphicsresearch.androidcomputebenchmark.BenchmarkRenderer { *; }
-keep class com.graphicsresearch.androidcomputebenchmark.BenchmarkNative { *; }
-keepclassmembers class com.graphicsresearch.androidcomputebenchmark.BenchmarkNative {
    native <methods>;
}
