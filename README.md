ballin-lana
===========

Este proyecto es una multiplicación de matrices en multiples maquinas dividiendo el proceso con una distribución de Poisson la cual nos ayuda a calcular cuanto tiempo es el necesario para saber si un esclavo sigue conectado o no.

Para compilar se necesita correr mpicc mulMat.c -o mulMat -lm 
Nota: -lm es necesario porque se ocupan logaritmos.

Para correr se necesita hacer:
mpirun -np 4 mulMat

Y para correrlo en varias computadoras ocupar: mpirun -np 4 -f hosts mulMat.
el archivo hosts debe de tener las IPs de las diferentes computadoras conectadas.

Consultar https://wiki.mpich.org/mpich/index.php/Using_the_Hydra_Process_Manager para mayor información.
