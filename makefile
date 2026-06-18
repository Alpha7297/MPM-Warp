PYTHON=python
SRC=basicMPM
OUT=outputs

.PHONY: all run generate plot clean train

all: run

plot:
	mkdir -p $(OUT)
	$(PYTHON) NCLaw/plot.py

generate:
	mkdir -p $(OUT)
	$(PYTHON) NCLaw/generate.py

clean:
	rm -rf $(OUT)/*

video:
	mpv outputs/videos/cube_compare.mp4
	mpv outputs/videos/table_compare.mp4

train:
	$(PYTHON) NCLaw/train.py
