CC = g++
CFLAGS = -g -Wall
RM= rm -f
SRCS = findcontour.cpp
PROG = findcontour

OPENCV = `pkg-config --cflags --libs opencv`
LIBS = $(OPENCV)

$(PROG):$(SRCS)
	$(CC) $(CFLAGS) -o $(PROG) $(SRCS) $(LIBS)

clean  :
	$(RM) *o $(PROG)
