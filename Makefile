CC := clang 
CFLAGS := -Wall -g 

SRC  := $(wildcard *.c)
OBJS := $(SRC:.c=.o)

.PHONY: clean

toynet: $(OBJS)
	$(CC) -o toynet $^ 

%.o: %.c
	$(CC) -o $@ -c $< $(CFLAGS)

clean:
	rm -rf $(OBJS) ./toynet 
