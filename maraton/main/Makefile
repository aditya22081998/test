TARGET = tes
 
INCLUDE_DIRS = -I../../../include -I../../../../Framework/include 
LIB = `pkg-config opencv --cflags --libs` 
CXX = g++
CXXFLAGS += -O2 -DLINUX -Wall -std=c++11 $(INCLUDE_DIRS)
#CXXFLAGS += -O2 -DDEBUG -DLINUX -Wall $(INCLUDE_DIRS)
LFLAGS += -pthread -ljpeg -lrt 
 
OBJECTS =   main.o
 
all: $(TARGET)
 
clean:
	rm -f *.a *.o $(TARGET) core *~ *.so *.lo
 
libclean:
	make -C ../../../build clean
 
distclean: clean libclean
 
darwin.a:
	make -C ../../../build
 
$(TARGET): darwin.a $(OBJECTS)
	$(CXX) $(CFLAGS) $(OBJECTS) ../../../lib/darwin.a -o $(TARGET) $(LIB)  $(LFLAGS) 
	chmod 755 $(TARGET)
 
# useful to make a backup "make tgz"
tgz: clean
	mkdir -p backups
	tar czvf ./backups/ball_following_`date +"%Y_%m_%d_%H.%M.%S"`.tgz --exclude backups *
