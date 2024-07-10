module ran1_module
    implicit none
    private
    public :: ran1

    integer, parameter :: m1 = 259200, ia1 = 7141, ic1 = 54773
    real, parameter :: rm1 = 3.8580247E-6
    integer, parameter :: m2 = 134456, ia2 = 8121, ic2 = 28411
    real, parameter :: rm2 = 7.4373773E-6
    integer, parameter :: m3 = 243000, ia3 = 4561, ic3 = 51349

    logical :: iff = .false.
    integer :: ix1, ix2, ix3
    real :: r(97)

contains
    function ran1(idum) result(rand)
        integer, intent(inout) :: idum
        real :: rand
        integer :: j

        if (idum < 0 .or. .not. iff) then
            call initialize(idum)
        end if

        ix1 = mod(ia1 * ix1 + ic1, m1)
        ix2 = mod(ia2 * ix2 + ic2, m2)
        ix3 = mod(ia3 * ix3 + ic3, m3)
        j = 1 + (97 * ix3) / m3

        if (j > 97 .or. j < 1) then
            error stop "RAN1: j out of bounds"
        end if

        rand = r(j)
        r(j) = (real(ix1) + real(ix2) * rm2) * rm1
    end function ran1

    subroutine initialize(idum)
        integer, intent(inout) :: idum
        integer :: j

        iff = .true.
        ix1 = mod(ic1 - idum, m1)
        ix1 = mod(ia1 * ix1 + ic1, m1)
        ix2 = mod(ix1, m2)
        ix1 = mod(ia1 * ix1 + ic1, m1)
        ix3 = mod(ix1, m3)

        do j = 1, 97
            ix1 = mod(ia1 * ix1 + ic1, m1)
            ix2 = mod(ia2 * ix2 + ic2, m2)
            r(j) = (real(ix1) + real(ix2) * rm2) * rm1
        end do

        idum = 1
    end subroutine initialize
end module ran1_module

! Here are the main changes and improvements made to the code:

! Encapsulated the function and its associated data in a module (ran1_module).
! Used lowercase for better readability (Fortran is case-insensitive).
! Added implicit none for better variable control and error detection.
! Changed from fixed-form to free-form format.
! Replaced the DATA statement with parameter initialization in the module.
! Used logical instead of integer for the iff flag.
! Replaced PAUSE with error stop for better error handling.
! Separated the initialization logic into a separate subroutine for better readability.
! Used intent(inout) for the idum parameter to clearly indicate it's modified.
! Used real instead of FLOAT for type conversion.
! Improved indentation and overall code structure.

! The ran1 function is a random number generator using three linear congruential generators in combination.
! To use this function in your program, you would:

! Use the module: use ran1_module
! Call the function: random_number = ran1(idum)

! Where idum is an integer seed that you provide (use a negative value for initialization).
