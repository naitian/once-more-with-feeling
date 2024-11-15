CURRENT_REV=$(shell git rev-parse HEAD | head -c 8)
clean:
	rm -rf container.sif


containers/dependencies.sif: dependencies.def environment.yaml
	git diff environment.yaml
	apptainer build -F containers/dependencies.sif dependencies.def


containers/container.sif: containers/dependencies.sif container.def
	git diff-index HEAD -- ':(exclude)notebooks/*'
	apptainer build -F containers/container.sif container.def


container: containers/container.sif


push: container
	scp containers/container.sif dtn.srdc.berkeley.edu:~/container-latest.sif


files: src/ scripts/ models/ slurm/ setup.py
	git diff-index HEAD -- ':(exclude)notebooks/*'
	echo "$(CURRENT_REV)" > version.txt
	rsync -av src scripts models slurm setup.py version.txt dtn.srdc.berkeley.edu:~/feelings/


@PHONY: clean push container
