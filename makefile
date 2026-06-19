PYTHON?=python
OUT=outputs
VIDEO_OUT=$(OUT)/videos
MPM3D_OUT=$(OUT)/3d_mpm

.PHONY: all \
	3DCube_generate 3DTable_generate \
	3DCube_plot 3DTable_plot \
	3DCube_save 3DTable_save \
	2D_generate 2D_plot \
	NCLaw_generate NCLaw_plot NCLaw_train \
	generate plot train video clean

all: 3DCube_generate

3DCube_generate:
	mkdir -p $(MPM3D_OUT)
	$(PYTHON) 3D/generate.py --model cube

3DTable_generate:
	mkdir -p $(MPM3D_OUT)
	$(PYTHON) 3D/generate.py --model table

3DCube_plot:
	mkdir -p $(MPM3D_OUT)
	$(PYTHON) 3D/plot.py --model cube --render show

3DTable_plot:
	mkdir -p $(MPM3D_OUT)
	$(PYTHON) 3D/plot.py --model table --render show

3DCube_save:
	mkdir -p $(MPM3D_OUT)
	$(PYTHON) 3D/plot.py --model cube --render video

3DTable_save:
	mkdir -p $(MPM3D_OUT)
	$(PYTHON) 3D/plot.py --model table --render video

2D_generate:
	mkdir -p $(OUT)
	$(PYTHON) 2D/generate.py

2D_plot:
	mkdir -p $(VIDEO_OUT)
	$(PYTHON) 2D/plot.py

NCLaw_generate:
	mkdir -p $(OUT)
	$(PYTHON) NCLaw/generate.py

NCLaw_plot:
	mkdir -p $(VIDEO_OUT)
	$(PYTHON) NCLaw/plot.py

NCLaw_train:
	mkdir -p NCLaw/net
	$(PYTHON) NCLaw/train.py

generate: NCLaw_generate

plot: NCLaw_plot

train: NCLaw_train

video:
	mpv outputs/videos/cube_compare.mp4
	mpv outputs/videos/table_compare.mp4

clean:
	rm -rf $(OUT)/*
