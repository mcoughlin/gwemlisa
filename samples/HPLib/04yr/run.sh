#!/bin/sh

gb_mcmc --inj ../HPLib.dat --samples 256 --sim-noise --known-source --cheat --no-rj --em-prior ../HPLib_inc.dat --duration 125829120 --steps 10000 --fix-freq 

