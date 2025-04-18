	PROGRAM D14R11
C	Driver for routine ROFUNC
	PARAMETER(NMAX=1000,SPREAD=0.05)
	COMMON /ARRAYS/ NPT,X(NMAX),Y(NMAX),ARR(NMAX),AA,ABDEV
	IDUM=-11
	NPT=100
	DO 11 I=1,NPT
		X(I)=0.1*I
		Y(I)=-2.0*X(I)+1.0+SPREAD*GASDEV(IDUM)
		write(6,*) y(i), x(i), spread, gasdev(idum)
11	CONTINUE
	WRITE(*,'(/1X,T10,A,T20,A,T26,A,T37,A/)') 'B','A','ROFUNC','ABDEV'
	DO 12 I=-5,5
		B=-2.0+0.02*I
		RF=ROFUNC(B)
		WRITE(*,'(1X,4F10.2)') B,AA,ROFUNC(B),ABDEV
12	CONTINUE	
	END

! nvfortran -O3 gasdev.o ran1_refactored.o rofunc.o sort.o rofunc_prog.f -o rofunc_prog
! with refactored ran1, answers are now correct, same as f90 versions
