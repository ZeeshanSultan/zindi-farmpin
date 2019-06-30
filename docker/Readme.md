# Docker stuff

This directory is a good place to put submodules, packages that need to be
installed in docker that are local.

## Creating submodules

Example is of installing ds-utils. Assume that you are in docker dir.
* clone submodule into repo:
`$ git submodule add git@github.com:jumo/ds-utils.git`

If reinitializing a repo from a clone. Go into submodule repo and use:
`$ git submodule update --remote`

## Installing submodule in docker image
All submodules should be able to be put into Docker images $PATH. This is so, 
don't have to use $PYTHONPATH or add paths to docker dir. 

Getting docker to install can be done in a normal bash way. First the files need
to be copied into the docker image. Then the setup script/installation needs
to be executed.

Example with ds-utils:
In DockerFile:
```COPY ds-utils /ds-utils```

To run setup script:
```RUN cd /ds-utils; python setup.py install```
