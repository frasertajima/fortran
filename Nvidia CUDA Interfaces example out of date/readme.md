Nvidia's 19.12 Using cuSPARSE from CUDA Fortran Host Code example is out of date:
Compiler returns:

! nvfortran sparsematvec.cuf -cudalib=cusparse -o sparsematvec

! /usr/bin/ld: /tmp/nvfortranFmQkD0IGXFx3.o: in function `MAIN_':

! sparsematvec.cuf:51:(.text+0x4a7): undefined reference to `cusparseSdense2csr'

! sparsematvec.cuf:58:(.text+0x50d): undefined reference to `cusparsescsrmv_sethpm_'

! sparsematvec.cuf:66:(.text+0x55e): undefined reference to `cusparsescsrmv_sethpm_'

! pgacclnk: child process exit status 1: /usr/bin/ld

Fortunately, Claude was eventually able to create an updated example. While the Claude example runs, it is difficult to know what the correct code should be given that Nvidia's example does not compile.

<img width="1964" height="788" alt="image" src="https://github.com/user-attachments/assets/1f241383-8085-434c-a2eb-a4118b6f1d12" />

