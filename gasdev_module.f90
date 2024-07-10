module gasdev_module
    use ran1_module, only: ran1
    implicit none
    private
    public :: gasdev

contains
    function gasdev(idum) result(gauss)
        integer, intent(inout) :: idum
        real :: gauss
        real :: v1, v2, r, fac
        real, save :: gset
        logical, save :: iset = .false.

        if (.not. iset) then
            do
                v1 = 2.0 * ran1(idum) - 1.0
                v2 = 2.0 * ran1(idum) - 1.0
                r = v1**2 + v2**2
                ! write (6,*) v1, v2, r
                if (r < 1.0 .and. r /= 0.0) exit
            end do
            fac = sqrt(-2.0 * log(r) / r)
            gset = v1 * fac
            gauss = v2 * fac
            iset = .true.
        else
            gauss = gset
            iset = .false.
        end if
    end function gasdev
end module gasdev_module

! Here are the main changes and improvements made to the code:

! Encapsulated the function in a module (gasdev_module).
! Used lowercase for better readability (Fortran is case-insensitive).
! Added implicit none for better variable control and error detection.
! Changed from fixed-form to free-form format.
! Replaced the DATA statement with parameter initialization using save attribute.
! Used logical instead of integer for the iset flag.
! Replaced GO TO with a do-exit loop for better readability and structure.
! Used intent(inout) for the idum parameter to clearly indicate it's modified.
! Improved indentation and overall code structure.
! Used more descriptive variable names (e.g., gauss instead of gasdev for the result).
! Made use of the ran1 function from the previously refactored ran1_module.

! The gasdev function generates normally distributed random numbers using the Box-Muller transform.
! To use this function in your program, you would:

! Use the module: use gasdev_module
! Call the function: gaussian_number = gasdev(idum)

! Where idum is an integer seed that you provide (use a negative value for initialization of the underlying ran1 function).
! Note that this module depends on the ran1_module we refactored earlier. 
! Make sure you have both modules compiled and available when using gasdev.