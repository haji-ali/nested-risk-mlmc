################################################################################
# Copied from CUDA examples
# Gencode arguments
SMS ?= 35 # 60

ifeq ($(SMS),)
$(info >>> WARNING - no SM architectures have been specified - waiving sample <<<)
SAMPLE_ENABLED := 0
endif

ifeq ($(GENCODE,FLAGS),)
# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval GENCODE.FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

# Generate PTX code from the highest SM architecture in $(SMS) to guarantee forward-compatibility
HIGHEST_SM := $(lastword $(sort $(SMS)))
ifneq ($(HIGHEST_SM),)
GENCODE.FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)
endif
endif

ifeq ($(SAMPLE_ENABLED),0)
EXEC ?= @echo "[@]"
endif
################################################################################

CXX.OPENMP			?= -fopenmp
CXX.STD				?= -std=c++14
CXX.INC				?=
CXX.LIB				?=
CXX.BIN				?= g++
CXX.COMPILE.FLAGS	?= -fPIC -Wall -Wextra $(CXX.STD) -O3 $(CXX.OPENMP) -Wno-expansion-to-defined
CXX.COMPILE 		?= $(CXX.BIN) -c $(CXX.COMPILE.FLAGS) $(CXX.INC)
CXX.LINK.FLAGS    	?= $(CXX.OPENMP)
CXX.LINK    		?= $(CXX.BIN) $(CXX.LINK.FLAGS) $(CXX.LIB)

NVCC.INC			?=
NVCC.LIB			?= -lcudart
NVCC.BIN    		?= nvcc
NVCC.FLAGS			?= -lineinfo $(GENCODE.FLAGS) --use_fast_math $(CXX.STD) -O3 -Xcompiler "-O3 -fPIC -Wall -Wextra" --x cu -rdc=true # --ptxas-options=-v -g
NVCC.LINK.FLAGS 	?= -lcudadevrt  $(GENCODE.FLAGS)
NVCC.COMPILE 		?= $(NVCC.BIN) -c $(NVCC.FLAGS) $(NVCC.INC)
NVCC.LINK    		?= $(NVCC.BIN) $(NVCC.LINK.FLAGS) $(NVCC.LIB)

OUTDIR     			?= build/
ALL_HEADERS     	?= $(wildcard *.hpp)

$(shell   mkdir -p $(OUTDIR))

all: cxx cuda

clean:
	$(RM) -rf build

cmake:
	mkdir -p build/output/ && \
	cd build && cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=1 .. && make VERBOSE=1

.ONESHELL:

$v.SILENT:

################################################################################
cxx: $(OUTDIR)portfolio_sampler $(OUTDIR)model_sampler # $(OUTDIR)libnested.so

$(OUTDIR)%.o: %.cpp $(ALL_HEADERS)
	$(CXX.COMPILE) -o $@ $<

$(OUTDIR)%_main.o: %.cpp $(ALL_HEADERS)
	$(CXX.COMPILE) -DINCLUDE_MAIN_MLMC -o $@ $<

$(OUTDIR)libnested.so: $(OUTDIR)common.o $(OUTDIR)model_sampler.o $(OUTDIR)portfolio_sampler.o
	$(CXX.LINK) -shared -o $@ $^

$(OUTDIR)portfolio_sampler: $(OUTDIR)common.o $(OUTDIR)portfolio_sampler_main.o
	$(CXX.LINK) -o $@ $^

$(OUTDIR)model_sampler: $(OUTDIR)common.o $(OUTDIR)model_sampler_main.o
	$(CXX.LINK) -o $@ $^

$(OUTDIR)mlmc_test: $(OUTDIR)mlmc_test_main.o
	$(CXX.LINK) -o $@ $^

test: $(OUTDIR)mlmc_test

################################################################################
cuda: $(OUTDIR)portfolio_sampler.cu $(OUTDIR)model_sampler.cu # $(OUTDIR)libnested.cu.so

$(OUTDIR)%.cu.o: %.cpp $(ALL_HEADERS)
	$(NVCC.COMPILE) -o $@ $<

$(OUTDIR)%_main.cu.o: %.cpp $(ALL_HEADERS)
	$(NVCC.COMPILE) -DINCLUDE_MAIN_MLMC -o $@ $<

$(OUTDIR)libnested.cu.so: $(OUTDIR)common.cu.o $(OUTDIR)model_sampler.cu.o $(OUTDIR)portfolio_sampler.cu.o
	$(NVCC.LINK) -shared -o $@ $^

$(OUTDIR)portfolio_sampler.cu: $(OUTDIR)common.cu.o $(OUTDIR)portfolio_sampler_main.cu.o
	$(NVCC.LINK) -o $@ $^

$(OUTDIR)model_sampler.cu: $(OUTDIR)common.cu.o $(OUTDIR)model_sampler_main.cu.o
	$(NVCC.LINK) -o $@ $^

################################################################################
