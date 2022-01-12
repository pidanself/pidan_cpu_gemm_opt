CC = /usr/local/Cellar/gcc/11.2.0/bin/gcc
CFLAGS = -O3

INCLUDE = -I../include
LIB = ../lib/pidan_gemm.a

OPENBLAS = -I /Users/pidan/guoke/研二上/工作/gemm_opt/OpenBLAS/include/ -L/Users/pidan/guoke/研二上/工作/gemm_opt/OpenBLAS/lib -lopenblas -lpthread -lgfortran

TARGET = correctness test_6x16 test_mc_nc packA

correctness: correctness.c $(LIB)
	$(CC) $(INCLUDE) $(CFLAGS) $(LIB) $(OPENBLAS) $< -o $@

test_6x16: test_6x16.c $(LIB)
	$(CC) $(INCLUDE) $(CFLAGS) $(LIB) $(OPENBLAS) $< -o $@

test_mc_nc: test_mc_nc.c $(LIB)
	$(CC) $(INCLUDE) $(CFLAGS) $(LIB) $(OPENBLAS) $< -o $@

packA: packA.c $(LIB)
	$(CC) $(INCLUDE) $(CFLAGS) $(LIB) $(OPENBLAS) $< -o $@

clean:
	rm -rf $(TARGET)