KOKKOS_PATH = ${HOME}/Kokkos/kokkos
SRC = main.cpp
KOKKOS_DEVICES=OpenMPTarget
KOKKOS_ARCH = Volta70

default: build
	echo "Start Build"

CXX = clang++
EXE = test.ompt
KOKKOS_CXX_STANDARD=c++17

CXXFLAGS = -O3 -g -DNDEBUG

# To avoid warnings
CXXFLAGS += -Wno-unknown-cuda-version -Wno-undefined-internal -Wno-pass-failed -Wno-openmp-mapping -Wno-extra-tokens

# This flag avoids the misaligned_address error. 
#CXXFLAGS += -fopenmp-cuda-mode

LINK = ${CXX}
LINKFLAGS =-Wno-unknown-cuda-version

DEPFLAGS = -M

OBJ = $(SRC:.cpp=.o)
LIB =

include $(KOKKOS_PATH)/Makefile.kokkos

build: $(EXE)

$(EXE): $(OBJ) $(KOKKOS_LINK_DEPENDS)
	$(LINK) $(KOKKOS_LDFLAGS) $(LINKFLAGS) $(EXTRA_PATH) $(OBJ) $(KOKKOS_LIBS) $(LIB) -o $(EXE)

clean: kokkos-clean
	rm -f *.o *.cuda *.host *.ompt

# Compilation rules

%.o:%.cpp $(KOKKOS_CPP_DEPENDS)
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(EXTRA_INC) -c $<
