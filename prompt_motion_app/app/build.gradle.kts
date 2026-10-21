import java.util.Properties

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

// OPENAI_API_KEY / OPENAI_MODEL come from the repo-root .env (git-ignored) or the
// environment. The key is compiled into BuildConfig: do not distribute this APK.
val dotenv = Properties().apply {
    val f = rootProject.file("../.env")
    if (f.exists()) f.forEachLine { line ->
        val t = line.trim()
        if (t.isNotEmpty() && !t.startsWith("#") && "=" in t) {
            val (k, v) = t.split("=", limit = 2)
            setProperty(k.trim(), v.trim().removeSurrounding("\"").removeSurrounding("'"))
        }
    }
}
fun secret(name: String, default: String = "") =
    (System.getenv(name) ?: dotenv.getProperty(name) ?: default).replace("\"", "")

android {
    namespace   = "com.graphicsresearch.promptmotion"
    compileSdk  = 34
    ndkVersion  = "27.0.12077973"

    defaultConfig {
        applicationId = "com.graphicsresearch.promptmotion"
        minSdk        = 24
        targetSdk     = 34
        versionCode   = 1
        versionName   = "1.0"

        ndk { abiFilters += "arm64-v8a" }
        externalNativeBuild { cmake { cppFlags += "-std=c++17" } }

        buildConfigField("String", "OPENAI_API_KEY", "\"${secret("OPENAI_API_KEY")}\"")
        buildConfigField("String", "OPENAI_MODEL", "\"${secret("OPENAI_MODEL", "gpt-4o-mini")}\"")
    }

    buildFeatures { buildConfig = true }

    buildTypes {
        release {
            isMinifyEnabled = false
            isDebuggable    = true
            signingConfig   = signingConfigs.getByName("debug")
        }
        debug { isMinifyEnabled = false }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
    packaging { jniLibs { useLegacyPackaging = true } }
    androidResources { noCompress += listOf("bin", "onnx") }
    externalNativeBuild {
        cmake {
            path    = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }
}

dependencies {
    implementation("androidx.appcompat:appcompat:1.6.1")
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.18.0")
}
