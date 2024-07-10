program d14r10
    use csv_operations
    implicit none
  
    real, parameter                         :: spread = 0.1
    character(len=*), parameter             :: input_file = 'input_data.csv'          ! could be user modified
    character(len=*), parameter             :: sample_file = 'sample_data.csv'        ! random to get started
    
    real, allocatable                       :: x(:), y(:), sig(:)
    real                                    :: a, b, siga, sigb, chi2, q, abdev
    integer                                 :: npt, mwt
  
    ! Generate sample CSV file
    call write_sample_csv(sample_file, 100, spread)
    print *, "Sample CSV file generated:       ", sample_file
  
    ! Read data from CSV file
    call read_csv(input_file, x, y, sig, npt)
    print *, "Data read from file:             ", input_file
    print *, "Number of data points: ", npt
  
    ! Perform fits
    mwt = 1
    call fit(x, y, npt, sig, mwt, a, b, siga, sigb, chi2, q)
  
    print '(/1X,A)', 'According to routine FIT the result is:'
    print '(1X,T5,A,F8.4,T20,A,F8.4)', 'A = ', a, 'Uncertainty: ', siga
    print '(1X,T5,A,F8.4,T20,A,F8.4)', 'B = ', b, 'Uncertainty: ', sigb
    print '(1X,T5,A,F16.8,A,I4,A)', 'Chi-squared: ', chi2, ' for    ', npt, ' points'
    print '(1X,T5,A,F8.4)', 'Goodness-of-fit(stay > 0.1):', q
  
    print '(/1X,A)', 'According to routine MEDFIT the result is:'
    call medfit(x, y, npt, a, b, abdev)
    print '(1X,T5,A,F8.4)', 'A (intercept) =             ', a
    print '(1X,T5,A,F8.4)', 'B (slope) =                 ', b
    print '(1X,T5,A,F8.4)', 'Absolute deviation (per data point): ', abdev
    print '(1X,T5,A,F8.4,A)', '(note: Gaussian spread is   ', spread, ')'
  
    ! Deallocate memory
    deallocate(x, y, sig)
  
end program d14r10

! subroutines were compiled unaltered with nvfortran -O3
! nvfortran -O3 csv_operations.o gammln.o sort.o gcf.o gser.o gammq.o rofunc.o fit.o medfit.o medfit_prog3.f90 -o medfit_prog3

! This refactored version uses modern Fortran features and best practices. Here are the main changes:
! 1. A new module `data_generation` is created to encapsulate the data generation logic.
! 2. The `GASDEV` function is replaced with a modern Fortran implementation of the Box-Muller transform for 
! generating normally distributed random numbers. This is done in the `random_normal` subroutine.
! 3. The main program is simplified and uses allocatable arrays instead of fixed-size arrays.
! 4. The data generation is performed using the `generate_data` subroutine, which takes a seed for reproducibility.
! 5. Modern Fortran's random number generator is used instead of a custom one.
! 6. The program structure is more modular and easier to read and maintain.
! 7. Explicit array bounds are used in all array operations for clarity and safety.
! 8. Memory is properly allocated and deallocated.
! 9. The printing statements use more modern Fortran formatting.
! Note that this version assumes that the `FIT` and `MEDFIT` routines are available and haven't changed. 

! NB: result is different from medfit_prog, the Fortran 77 version. But the updated one seems to make more sense
! in terms of output: I need to check results; random numbers are likely different
! need to print out data?

! This refactored version does the following:

! It uses a new module csv_operations that contains subroutines for reading from and writing to CSV files.
! The program first generates a sample CSV file using write_sample_csv. This creates a file 
! named 'sample_data.csv' with 100 data points.
! It then reads data from an input CSV file named 'input_data.csv' using read_csv. This subroutine 
! dynamically allocates the arrays based on the number of lines in the input file.
! The rest of the program remains largely the same, performing fits and printing results.

! To use this program:

! First, run it to generate the sample CSV file ('sample_data.csv').
! You can then use this sample file as your input by renaming it to 'input_data.csv', or create your own 
! input file with the same format (x, y, sig on each line).
! Run the program again, and it will read from 'input_data.csv' and perform the fits.

! This approach makes the program more flexible, allowing it to work with different datasets without 
! recompiling, and provides a way to generate sample data for testing.