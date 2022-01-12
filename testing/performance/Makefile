CC = /usr/local/Cellar/gcc/11.2.0/bin/gcc
CFLAGS = -O3

INCLUDE = -I../../include
LIB = ../../lib/pidan_gemm.a

OPENBLAS = -I /Users/pidan/guoke/研二上/工作/gemm_opt/OpenBLAS/include/ -L/Users/pidan/guoke/研二上/工作/gemm_opt/OpenBLAS/lib -lopenblas -lpthread -lgfortran

SRCC = $(wildcard *.c)
OBJC =$(subst .c,,$(SRCC))

all: $(SRCC) $(OBJC)

$(OBJC):% : %.c $(LIB)
	$(CC) $(INCLUDE) $(LIB) $(OPENBLAS) $(CFLAGS) $< -o $@

clean:
	rm -rf $(OBJC)