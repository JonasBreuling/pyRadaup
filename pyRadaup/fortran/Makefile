CC = gcc
FC = gfortran
LD = gcc

CCFLAGS = -fPIC
FCFLAGS = -fPIC -fno-underscoring -ffixed-line-length-0
LDFLAGS = -shared

# OBJS = dc_decsol.o decsol.o radau5.o radaup.o integration_radau.o integration_radaup.o
# OBJS = dc_decsol.o decsol.o radaup.o integration_radaup.o
OBJS = dc_decsol.o decsol.o radau.o integration_radau.o

LIB = lib_radau.so

all: $(LIB)

$(LIB): $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $^ -lgfortran

%.o: %.c
	$(CC) $(CCFLAGS) -c $< -o $@

%.o : %.f
	$(FC) $(FCFLAGS) -c $< -o $@

clean:
	rm -f $(LIB) $(OBJS)
