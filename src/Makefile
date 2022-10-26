sanitize: compile run clean

test: compile run_test clean

profile: compile run_profiler clean

get_gpu_properties: compile_with_properties simple_run simple_clean

compile: primitives.cu
	nvcc -O3 -lineinfo -lcublas -DCHECK -o out32 primitives.cu
	nvcc -O3 -lineinfo -lcublas -DCHECK -DF64 -o out64 primitives.cu

run:
	compute-sanitizer --tool memcheck out32.exe
	compute-sanitizer --tool racecheck out32.exe
	compute-sanitizer --tool initcheck out32.exe
	compute-sanitizer --tool synccheck out32.exe
	compute-sanitizer --tool memcheck out64.exe
	compute-sanitizer --tool racecheck out64.exe
	compute-sanitizer --tool initcheck out64.exe
	compute-sanitizer --tool synccheck out64.exe

run_test:
	./out32.exe
	./out64.exe

run_profiler:
	ncu ./out32.exe

clean:
	-del out32.exe
	-del out32.exp
	-del out32.lib
	-del out64.exe
	-del out64.exp
	-del out64.lib

compile_with_properties: primitives.cu
	nvcc -DGPU_PROPERTIES -o out primitives.cu

simple_run: 
	./out.exe

simple_clean:
	-del out.exe
	-del out.exp
	-del out.lib
