CXX = g++
NVCC = nvcc
TARGET = greyscale

OPENCV_CFLAGS = $(shell pkg-config --cflags opencv4)
OPENCV_LIBS = $(shell pkg-config --libs opencv4)

CUDA_PATH = /usr/local/cuda
CUDA_LIBS = -L$(CUDA_PATH)/lib64 -lcudart -lnppicc -lnppig -lnppc

NVCC_FLAGS = -std=c++14 $(OPENCV_CFLAGS)
CXX_FLAGS = -std=c++14

all: $(TARGET)

greyscale.o: greyscale.cu
	$(NVCC) $(NVCC_FLAGS) -c greyscale.cu -o greyscale.o

$(TARGET): greyscale.o
	$(CXX) -o $(TARGET) greyscale.o $(OPENCV_LIBS) $(CUDA_LIBS)

clean:
	rm -f *.o $(TARGET)