CXX ?= g++
CXXFLAGS ?= -O3 -Wall -fPIC -std=c++17

TARGET_LIB = libzd_ecn_playback.so
TARGET_BIN = test_playback_native

all: $(TARGET_LIB) $(TARGET_BIN)

$(TARGET_LIB): MotionPlaybackEngine.cpp MotionPlaybackEngine.h
	$(CXX) $(CXXFLAGS) -shared -o $@ MotionPlaybackEngine.cpp

$(TARGET_BIN): main_test.cpp $(TARGET_LIB)
	$(CXX) $(CXXFLAGS) main_test.cpp -L. -lzd_ecn_playback -Wl,-rpath,. -Wl,-rpath,/tmp/zd_ecn_motion/native -o $@

clean:
	rm -f $(TARGET_LIB) $(TARGET_BIN)
