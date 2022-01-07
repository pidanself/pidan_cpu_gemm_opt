CC = /usr/local/Cellar/gcc/11.2.0/bin/gcc
CFLAGS = -O3

INCLUDE = -I../include
LIB = ../lib/pidan_gemm.a

OPENBLAS = -I /Users/pidan/guoke/研二上/工作/gemm_opt/OpenBLAS/include/ -L/Users/pidan/guoke/研二上/工作/gemm_opt/OpenBLAS/lib -lopenblas -lpthread -lgfortran


correctness: correctness.c
	$(CC) $(INCLUDE) $(CFLAGS) $(LIB) $(OPENBLAS) $< -o $@

clean:
	rm -rf correctness