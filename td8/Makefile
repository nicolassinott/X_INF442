CXX = g++
CXXFLAGS = ${INCLUDES} -std=c++11 -O3
INCLUDES = -Igradinglib -Igrading -Idataset -Iconfusion_matrix -Ikernel -Isvm

.PHONY: all clean

all: grader

# TD-specific part

OBJECTS_SPECIFIC = dataset.o confusion_matrix.o kernel.o svm.o
OBJECTS_SPECIFIC = dataset/dataset.cpp confusion_matrix/confusion_matrix.cpp kernel/kernel.cpp svm/svm.cpp

dataset.o: dataset/dataset.cpp
	$(CXX) -c $(INCLUDES) $(CFLAGS) dataset/dataset.cpp

confusion_matrix.o: confusion_matrix/confusion_matrix.cpp
	$(CXX) -c $(INCLUDES) $(CFLAGS) confusion_matrix/confusion_matrix.cpp

kernel.o: kernel/kernel.cpp
	$(CXX) -c $(INCLUDES) $(CFLAGS) kernel/kernel.cpp

svm.o: svm/svm.cpp
	$(CXX) -c $(INCLUDES) $(CFLAGS) svm/svm.cpp

# Common part
SOURCES_COMMON = gradinglib/gradinglib.cpp grading/grading.cpp main.cpp
OBJECTS_COMMON = gradinglib.o grading.o main.o

grader: $(OBJECTS_COMMON) $(OBJECTS_SPECIFIC)
	$(CXX) $(CXXFLAGS) -o grader $(OBJECTS_COMMON) $(OBJECTS_SPECIFIC)

gradinglib.o: gradinglib/gradinglib.cpp gradinglib/gradinglib.hpp
	$(CXX) -c $(CXXFLAGS) -o gradinglib.o gradinglib/gradinglib.cpp

grading.o: grading/grading.cpp gradinglib/gradinglib.hpp
	$(CXX) -c $(CXXFLAGS) -o grading.o grading/grading.cpp 

main.o: main.cpp grading/grading.hpp
	$(CXX) -c $(CXXFLAGS) -o main.o main.cpp

clean:
	rm -f grader *.o
