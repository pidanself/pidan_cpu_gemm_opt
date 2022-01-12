CC = /usr/local/Cellar/gcc/11.2.0/bin/gcc
CFLAGS = -O3
AR = ar
ARFLAGS = cr

INCLUDE = -I../include -mavx2
LIB = 

TARGET = pidan_gemm.a

SRCC = $(wildcard *.c)
SRCA += $(wildcard *.S)

OBJC =$(subst .c,.o,$(SRCC))
OBJA =$(subst .S,.o,$(SRCA))

all: $(TARGET)

$(TARGET): $(OBJC) $(OBJA)
	$(AR) $(ARFLAGS) ../lib/$(TARGET) $^


$(OBJC):%.o : %.c
	$(CC) -c $(INCLUDE) $(CFLAGS) $< -o $@

$(OBJA):%.o : %.S
	$(CC) -c $(INCLUDE) $(CFLAGS) $< -o $@

clean:
	rm -rf $(OBJC) $(OBJA) ../lib/$(TARGET)