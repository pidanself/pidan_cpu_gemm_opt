.PHONY: all clean

dirs = src
dirs += testing

all:
	$(foreach N,$(dirs),make -C $(N) -j;)
clean:
	$(foreach N,$(dirs),make -C $(N) clean;)
