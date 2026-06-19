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
	exp11_train exp11_plot \
	exp12_train exp12_plot \
	exp13_train exp13_plot \
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
	mpv outputs/videos/mpm.mp4

NCLaw_generate:
	mkdir -p $(OUT)
	$(PYTHON) NCLaw/generate.py

NCLaw_plot:
	mkdir -p $(VIDEO_OUT)
	$(PYTHON) NCLaw/plot.py

NCLaw_train:
	mkdir -p NCLaw/net
	$(PYTHON) NCLaw/train.py

exp11_train:
	mkdir -p experiment/exp1/net
	$(PYTHON) experiment/exp1/train.py --model 1

exp11_plot:
	mkdir -p $(VIDEO_OUT)
	$(PYTHON) experiment/exp1/plot.py --model 1

exp12_train:
	mkdir -p experiment/exp1/net
	$(PYTHON) experiment/exp1/train.py --model 2

exp12_plot:
	mkdir -p $(VIDEO_OUT)
	$(PYTHON) experiment/exp1/plot.py --model 2

exp13_train:
	mkdir -p experiment/exp1/net
	$(PYTHON) experiment/exp1/train.py --model 3

exp13_plot:
	mkdir -p $(VIDEO_OUT)
	$(PYTHON) experiment/exp1/plot.py --model 3

generate: NCLaw_generate

plot: NCLaw_plot

train: NCLaw_train

video:
	mpv outputs/videos/cube_compare.mp4
	mpv outputs/videos/table_compare.mp4

clean:
	rm -rf $(OUT)/*
