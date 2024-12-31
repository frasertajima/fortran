! sorting using thrust library (better than my radix sort!!)
! took only 6.652E-03 seconds for 2M elements, 3.64E-02 seconds for N=200M
! custom radix sort took 0.67s, CPU version took 0.25s (thrust sort is 100x faster)
! thrust needs fedora 39 distrobox to get gcc version 13; need to create symbolic link to 13 as well
! as otherwise host system gcc 14 will be called inside the distrobox




! to compile: 
! nvcc -c -o thrust.C.o thrust.cu
! nvfortran thrust.cuf testSort2.cuf thrust.C.o -c++libs -o testSort2
! 800M seems to be the largest array before memory runs out (takes 0.1279s)
! version 3 extends the sort example to add more thrust functions
