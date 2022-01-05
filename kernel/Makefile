CC = /usr/local/Cellar/gcc/11.2.0/bin/gcc


TARGETS = sgemm_4x8 sgemm_6x16

test_sgemm_4x8:
	$(CC) -o sgemm_4x8 main.c sgemm_4x8.S

test_sgemm_6x16:
	$(CC) -o sgemm_6x16 main.c sgemm_6x16.S

clean:
	rm -rf $(TARGETS)