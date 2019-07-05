VPATH:=opencv control core
SRC:=$(wildcard $(addsuffix /*.cpp,$(VPATH)))
BIN:=bin
CFLAGS := -g -w
EXEC:= $(addprefix $(BIN)/,$(notdir $(SRC:.cpp=)))
LIBS:= `pkg-config opencv --cflags --libs`
CFLAGS:= -Wall -g

all: $(EXEC)

$(BIN)/%:	%.cpp
	@echo "---------------------------------------------------"
	@echo "compiling $@"
	@echo "---------------------------------------------------"
	@g++ $(CFLAGS) -o $@ $< $(LIBS)

clean:
	@echo "cleaning executables:  $(EXEC)"
	@rm -rf $(EXEC)
	@rm -rf $(BIN)/*

info:
	@echo "---------------------------------------------------"
	@echo "SOURCE FILES"
	@echo "---------------------------------------------------"
	@echo $(SRC)
	@echo "---------------------------------------------------"
	@echo "EXEC FILES"
	@echo "---------------------------------------------------"
	@echo $(EXEC)
