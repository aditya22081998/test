CC = g++
CFLAGS = -g -Wall
RM= rm -f
SRCS = main.cpp
PROG = main

OPENCV = `pkg-config opencv --cflags --libs`
LIBS = $(OPENCV)

$(PROG):$(SRCS)
	$(CC) $(CFLAGS) -o $(PROG) $(SRCS) $(LIBS)

clean  :
	$(RM) *o $(PROG)
