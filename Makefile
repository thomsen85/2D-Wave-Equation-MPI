CC=gcc
PARALLEL_CC=mpicc
CFLAGS+= -std=c99 -O2 -Wall -Wextra
LDLIBS+= -lm
SEQUENTIAL_SRC_FILES=wave_2d_sequential.c argument_utils.c
PARALLEL_SRC_FILES=wave_2d_parallel.c argument_utils.c
IMAGES=$(shell find data -type f | sed s/\\.dat/.png/g | sed s/data/images/g )
.PHONY: all clean dirs plot movie
all: dirs ${TARGETS}
dirs:
	mkdir -p data images
sequential: ${SEQUENTIAL_SRC_FILES}
	$(CC) $^ $(CFLAGS) -o $@ $(LDLIBS)
parallel: ${PARALLEL_SRC_FILES}
	$(PARALLEL_CC) $^ $(CFLAGS) -o $@ $(LDLIBS)
plot: ${IMAGES}
images/%.png: data/%.dat
	./plot_image.sh $<
movie: ${IMAGES}
	ffmpeg -y -an -i images/%5d.png -vcodec libx264 -pix_fmt yuv420p -profile:v baseline -level 3 -r 12 wave.mp4
check: dirs sequential parallel
	mkdir -p data_sequential
	./sequential
	cp -rf ./data/* ./data_sequential
	mpiexec -n 1 --oversubscribe ./parallel
	./compare.sh
	mpiexec -n 4 --oversubscribe ./parallel
	./compare.sh
	rm ./data_sequential/*
	./sequential -m 2048 -n 512
	cp -rf ./data/* ./data_sequential
	mpiexec -n 16 --oversubscribe ./parallel -m 2048 -n 512
	./compare.sh
	rm -rf data_sequential
clean:
	-rm -fr sequential parallel data images wave.mp4
