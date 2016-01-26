EXEC := main
#LIBS := `pkg-config gtkmm-2.4 --cflags --libs`


all:
		#echo $(LIBS)
		g++ test_gtk.cpp *.cc -o $(EXEC) `pkg-config gtkmm-2.4 --cflags --libs`

run:
	  ./$(EXEC)
