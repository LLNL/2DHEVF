# Two dimensional topology optimization of heat exchangers with the volume fraction method.

This repository contains the code to reproduce the results based on the volume
fraction method in the paper "Two dimensional topology optimization of heat
exchangers with the volume fraction and level-set method."

## Requirements

* [Firedrake](https://www.firedrakeproject.org/index.html)
* [Gmsh](https://gmsh.info/)
* [pygmsh](https://pypi.org/project/pygmsh/)

## Instructions
To run, generate the mesh with
```
python3 2D_mesh.py && gmsh -2 2D_mesh.geo
```
and run the code with
```
python3 he_volume_frac.py
```
Inspect the options with
```
python3 he_volume_frac.py --help
```

To run the parameter sweep in `project.py`, please download
[signac](https://signac.io/).


LLNL Release Number: LLNL-CODE- 817119
