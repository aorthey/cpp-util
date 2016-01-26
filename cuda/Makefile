SRCNVCC := $(wildcard *.cu)

CFLAGS := -w -fPIC
EXEC:= cuda_simple
CUDA:=/usr/local/cuda
NVCC := $(CUDA)/bin/nvcc
LIBS:= -L$(CUDA)/lib64 -lcudart
INCL:= -I$(CUDA)/include

all: $(EXEC)

$(EXEC): cuda_simple.o cuda_functions.o
	g++ -fPIC -Wl,-rpath=/usr/local/cuda/lib64 -o $@ $< $(LIBS)

%.o:	%.cu
	@echo "compiling $@"
	@$(NVCC) -o $@ -c $< $(LIBS) $(INCL)

%.cu: %.h

clean:
	@echo "cleaning executable:  $(EXEC)"
	@rm -rf $(EXEC)

run:
	  ./$(EXEC)
