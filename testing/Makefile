.PHONY: all clean

dirs = performance
dirs += validity

all:
	$(foreach N,$(dirs),make -C $(N) -j;)
clean:
	$(foreach N,$(dirs),make -C $(N) clean;)