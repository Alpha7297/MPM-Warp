PYTHON=python
SRC=basicMPM
OUT=outputs

.PHONY: all run generate plot clean clear

all: run

run: plot

plot:
	mkdir -p $(OUT)
	$(PYTHON) $(SRC)/plot.py

clean:
	rm -rf $(OUT)/*

video:
	mpv outputs/videos/mpm.mp4