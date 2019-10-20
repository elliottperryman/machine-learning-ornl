# Copyright 2018.  Aaron Sprow
# This file is part of pulse_generation.


vpath %.cpp src/
vpath %.h inc/

CXX = g++
CPPFLAGS = -I inc/ -std=c++11
LDFLAGS = -lfftw3

MAIN = pulse_generation
SOURCES = simulation.cpp
OBJECTS = $(SOURCES:.cpp=.o) $(MAIN).o 
INCLUDES = $(SOURCES:.cpp=.h)
BUILD = build/
EXECS = bin/

all: $(MAIN)

$(MAIN): $(OBJECTS)
	$(CXX) $(CPPFLAGS) -o $(EXECS)$@ $(BUILD)$(SOURCES:.cpp=.o) $(BUILD)$(MAIN).o $(LDFLAGS)

$(MAIN).o : $(MAIN).cpp $(INCLUDES)
	$(CXX) $(CPPFLAGS) -c $< -o $(BUILD)$@ $(LDFLAGS)

%.o : %.cpp %.h
	$(CXX) $(CPPFLAGS) -c $< -o $(BUILD)$@ $(LDFLAGS)

$(shell mkdir -p $(BUILD) $(EXECS))

clean :
	rm -f *.o *~ $(MAIN) build/*.o

.PHONY : all clean
