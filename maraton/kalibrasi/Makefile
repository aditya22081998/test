CC = g++
CFLAGS = -g -Wall
SRCS = kalibrasi.cpp
PROG = kalibrasi

OPENCV = `pkg-config opencv --cflags --libs`
LIBS = $(OPENCV) -std=c++11

$(PROG):$(SRCS)
		$(CC) $(CFLAGS) -o $(PROG) $(SRCS) $(LIBS)
clean:
	rm -f $(PROG)
compile:
	rm -f $(PROG)
	$(CC) $(CFLAGS) -o $(PROG) $(SRCS) $(LIBS) 
