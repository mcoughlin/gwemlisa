#!/bin/sh

gb_mcmc --inj ../HPLib.dat --samples 512 --sim-noise --known-source --cheat --no-rj --em-prior ../HPLib_inc.dat --duration 251658240 --steps 10000 --fix-freq 

